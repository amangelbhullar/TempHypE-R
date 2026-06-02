"""
RHGNN Ablation Study
Tests contribution of each component:
1. Full RHGNN
2. w/o ODE (replace flow with identity)
3. w/o H-GRU (replace jump with linear update)
4. w/o Hyperbolic (replace Poincare ball with Euclidean)
5. w/o Message Passing (use zero message)
"""

import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhgnn_end_to_end import (
    load_temporal_kg, build_snapshot_graphs,
    expmap0, logmap0, mobius_add, hyp_distance,
    EPS, poincare_project, ODEFunc,
    TemporalKGData, evaluate
)

from geoopt.optim import RiemannianAdam

Quad = Tuple[int, int, int, int]


class RHGNNAblation(nn.Module):
    """
    RHGNN with ablation flags:
    - use_ode:       True = Neural ODE flow, False = identity
    - use_hgru:      True = H-GRU jump,      False = linear update
    - use_hyperbolic:True = Poincare ball,    False = Euclidean
    - use_messages:  True = neighbour aggregation, False = zero message
    """

    def __init__(
        self,
        num_entities, num_relations, dim=200,
        init_curvature=1.0, dropout=0.1, ode_steps=5,
        use_ode=True, use_hgru=True,
        use_hyperbolic=True, use_messages=True,
    ):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps
        self.use_ode       = use_ode
        self.use_hgru      = use_hgru
        self.use_hyperbolic= use_hyperbolic
        self.use_messages  = use_messages

        self._log_c    = nn.Parameter(torch.tensor(math.log(math.exp(init_curvature) - 1.0)))
        self.entity_emb   = nn.Embedding(num_entities,  dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.rel_lin      = nn.Embedding(num_relations, dim * dim)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        # Euclidean GRU for ablation
        self.Wz_euc = nn.Linear(dim, dim)
        self.Uz_euc = nn.Linear(dim, dim, bias=False)
        self.ode_func = ODEFunc(dim, dropout)
        self.bias     = nn.Parameter(torch.zeros(num_entities))
        self.dropout  = nn.Dropout(dropout)
        nn.init.normal_(self.entity_emb.weight,   std=0.01)
        nn.init.normal_(self.relation_emb.weight, std=0.01)
        nn.init.normal_(self.rel_lin.weight,      std=0.01)

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(1e-1, 3.0)

    def _to_hyp(self, ids):
        raw = self.entity_emb(ids)
        if self.use_hyperbolic:
            return expmap0(raw, self.c)
        return raw  # Euclidean: just return raw

    def _rel_hyp(self, ids):
        raw = self.relation_emb(ids)
        if self.use_hyperbolic:
            return expmap0(raw, self.c)
        return raw

    def build_nbr_index(self, snapshots, max_nbrs=16):
        self._nbr_index = {}
        for tau, ent_dict in snapshots.items():
            for ent, nbrs in ent_dict.items():
                nbrs = nbrs[:max_nbrs]
                self._nbr_index[(tau, ent)] = (
                    torch.tensor([n for n, _ in nbrs], dtype=torch.long),
                    torch.tensor([r for _, r in nbrs], dtype=torch.long),
                )

    def refresh_msg_table(self, device):
        if not self.use_messages:
            self._msg_table = {}
            return
        c = self.c.detach()
        index = self._nbr_index
        with torch.no_grad():
            raw = self.entity_emb.weight.to(device)
            if self.use_hyperbolic:
                all_tan = logmap0(expmap0(raw, c), c)
            else:
                all_tan = raw
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                nbr_tan = all_tan[nbr_ids]
                W = self.rel_lin(rel_ids).view(len(nbr_ids), self.dim, self.dim)
                m = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                self._msg_table[(tau, ent)] = torch.tanh(m.mean(dim=0))

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        device   = entity_ids.device
        B        = entity_ids.size(0)
        msgs     = torch.zeros(B, self.dim, device=device)
        if not self.use_messages:
            return msgs
        table    = getattr(self, "_msg_table", {})
        tau_list = tau_ids.tolist()
        e_list   = entity_ids.tolist()
        for b in range(B):
            key = (tau_list[b], e_list[b])
            if key in table:
                msgs[b] = table[key]
        return msgs

    def _jump(self, h_prev, msg_tan):
        if self.use_hyperbolic and self.use_hgru:
            # Full H-GRU
            c      = self.c
            v_prev = logmap0(h_prev, c)
            z      = torch.sigmoid(self.Wz(v_prev) + self.Uz(msg_tan))
            v_new  = z * msg_tan + (1.0 - z) * v_prev
            return expmap0(v_new, c)
        elif self.use_hgru:
            # Euclidean GRU
            z     = torch.sigmoid(self.Wz_euc(h_prev) + self.Uz_euc(msg_tan))
            return z * msg_tan + (1.0 - z) * h_prev
        else:
            # No GRU — simple linear update
            return h_prev + 0.1 * msg_tan

    def _flow(self, h_hyp, delta_t):
        if not self.use_ode or delta_t <= 0:
            return h_hyp
        if self.use_hyperbolic:
            c     = self.c
            state = logmap0(h_hyp, c)
            dt    = delta_t / max(self.ode_steps, 1)
            for _ in range(self.ode_steps):
                state = state + dt * self.ode_func(state)
            return expmap0(state, c)
        else:
            state = h_hyp
            dt    = delta_t / max(self.ode_steps, 1)
            for _ in range(self.ode_steps):
                state = state + dt * self.ode_func(state)
            return state

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        c       = self.c
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._jump(h_hyp, msg_tan)
        h_flow  = self._flow(h_jump, delta_t)
        r_hyp   = self._rel_hyp(r)
        o_hyp   = self._to_hyp(t)
        if self.use_hyperbolic:
            pred = mobius_add(h_flow, r_hyp, c)
            dist = hyp_distance(pred, o_hyp, c)
        else:
            pred = h_flow + r_hyp
            dist = (pred - o_hyp).norm(dim=-1)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0):
        c       = self.c
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._jump(h_hyp, msg_tan)
        h_flow  = self._flow(h_jump, delta_t)
        r_hyp   = self._rel_hyp(r)
        all_ids = torch.arange(self.num_entities, device=h.device)
        all_hyp = self._to_hyp(all_ids)
        if self.use_hyperbolic:
            pred     = mobius_add(h_flow, r_hyp, c)
            pred_exp = pred.unsqueeze(1).expand(-1, self.num_entities, -1)
            tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist     = hyp_distance(pred_exp, tail_exp, c)
        else:
            pred     = h_flow + r_hyp
            pred_exp = pred.unsqueeze(1).expand(-1, self.num_entities, -1)
            tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist     = (pred_exp - tail_exp).norm(dim=-1)
        return -dist + self.bias.unsqueeze(0)


def train_ablation(model, train_groups, optimizer, device,
                   num_entities, snapshots, ts_to_real,
                   neg_ratio=5, batch_size=512, grad_clip=1.0):
    model.train()
    model.refresh_msg_table(device)
    total_loss = 0.0
    total_n    = 0
    prev_real_ts = None
    for ts_id, quads in train_groups:
        real_ts  = ts_to_real.get(ts_id, ts_id)
        delta_t  = float(real_ts - prev_real_ts) if prev_real_ts is not None else 1.0
        delta_t  = max(delta_t, 1.0)
        prev_real_ts = real_ts
        for start in range(0, len(quads), batch_size):
            mini  = quads[start: start + batch_size]
            batch = torch.tensor(mini, dtype=torch.long, device=device)
            h, r, t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
            pos_score = model.score(h, r, t, tau, snapshots, delta_t)
            pos_loss  = F.softplus(-pos_score).mean()
            neg_losses = []
            for _ in range(neg_ratio):
                neg_t = torch.randint(0, num_entities, (h.size(0),), device=device)
                ns    = model.score(h, r, neg_t, tau, snapshots, delta_t)
                neg_losses.append(F.softplus(ns).mean())
            loss = pos_loss + torch.stack(neg_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(mini)
            total_n    += len(mini)
    return total_loss / max(total_n, 1)


def run_ablation(data_dir, gpu_id=0, epochs=100, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{gpu_id}")
    data   = load_temporal_kg(data_dir)

    snapshots = build_snapshot_graphs(data.train)

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items(), key=lambda x: x[0])

    true_tails_map = defaultdict(list)
    for (hh, rr, tt, ta) in data.all_true:
        true_tails_map[(hh, rr, ta)].append(tt)

    variants = [
        ("Full RHGNN",          dict(use_ode=True,  use_hgru=True,  use_hyperbolic=True,  use_messages=True)),
        ("w/o ODE",             dict(use_ode=False, use_hgru=True,  use_hyperbolic=True,  use_messages=True)),
        ("w/o H-GRU",           dict(use_ode=True,  use_hgru=False, use_hyperbolic=True,  use_messages=True)),
        ("w/o Hyperbolic",      dict(use_ode=True,  use_hgru=True,  use_hyperbolic=False, use_messages=True)),
        ("w/o Message Passing", dict(use_ode=True,  use_hgru=True,  use_hyperbolic=True,  use_messages=False)),
    ]

    print(f"\n{'Variant':25s} | {'MRR':6s} | {'H@1':6s} | {'H@3':6s} | {'H@10':6s} | {'MAR':7s}")
    print("-" * 65)

    results = {}
    for name, flags in variants:
        model = RHGNNAblation(
            num_entities=data.num_entities,
            num_relations=data.num_relations,
            dim=200, init_curvature=1.0,
            dropout=0.1, ode_steps=5, **flags
        ).to(device)
        model.build_nbr_index(snapshots)

        optimizer = RiemannianAdam([
            {"params": [p for n, p in model.named_parameters() if n != "_log_c"], "lr": 1e-3},
            {"params": [model._log_c], "lr": 1e-4},
        ], stabilize=10)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

        best_mrr = -1.0
        best_metrics = {}
        for epoch in range(1, epochs + 1):
            train_ablation(model, train_groups, optimizer, device,
                          data.num_entities, snapshots, ts_to_real,
                          neg_ratio=3, batch_size=512)
            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs:
                metrics = evaluate(
                    model=model, eval_quads=data.valid,
                    all_true=data.all_true, snapshots=snapshots,
                    ts_to_real=ts_to_real, device=device, batch_size=256,
                )
                if metrics["MRR"] > best_mrr:
                    best_mrr     = metrics["MRR"]
                    best_metrics = metrics

        results[name] = best_metrics
        m = best_metrics
        print(f"{name:25s} | {m['MRR']:.4f} | {m['Hits@1']:.4f} | {m['Hits@3']:.4f} | {m['Hits@10']:.4f} | {m['MAR']:.1f}")

    # Delta computation
    full = results["Full RHGNN"]
    print(f"\n{'Variant':25s} | {'ΔMRR':8s} | {'ΔH@10':8s} | {'Drop%':6s}")
    print("-" * 55)
    for name, m in results.items():
        if name == "Full RHGNN":
            continue
        delta_mrr  = m["MRR"]     - full["MRR"]
        delta_h10  = m["Hits@10"] - full["Hits@10"]
        drop_pct   = abs(delta_mrr) / full["MRR"] * 100
        print(f"{name:25s} | {delta_mrr:+.4f}  | {delta_h10:+.4f}  | {drop_pct:.1f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ICEWS14")
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    print(f"Running ablation on {args.data_dir} for {args.epochs} epochs...")
    run_ablation(args.data_dir, args.gpu, args.epochs, args.seed)

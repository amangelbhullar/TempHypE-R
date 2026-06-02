"""
RHGNN v3 — V2 + Historical Vocabulary + Subgraph Reasoning
"""
import argparse, math, os, random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt
from geoopt.optim import RiemannianAdam

import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import (
    set_seed, load_temporal_kg, build_snapshot_graphs, evaluate, Quad
)
from rhgnn_v2 import (
    expmap0, logmap0, mobius_add, hyp_distance,
    TimeEncoding, ODEFunc, adversarial_loss
)

EPS = 1e-6

# ── Historical Vocabulary ─────────────────────────────────────────────────────

def build_history_vocab(train_data):
    """
    For each (subject, relation) pair, track all objects seen in training.
    Used to boost historically recurring entities during scoring.
    """
    vocab = defaultdict(set)
    for h, r, t, _ in train_data:
        vocab[(h, r)].add(t)
    return vocab

def build_history_mask(h_ids, r_ids, history_vocab, num_entities, device):
    """Build binary mask of historically seen entities for each query."""
    B = h_ids.size(0)
    mask = torch.zeros(B, num_entities, device=device)
    for i in range(B):
        hist = history_vocab.get((h_ids[i].item(), r_ids[i].item()), set())
        if hist:
            idx = torch.tensor(list(hist), dtype=torch.long, device=device)
            mask[i, idx] = 1.0
    return mask

# ── Subgraph Encoder (RGCN-style) ─────────────────────────────────────────────

class SubgraphEncoder(nn.Module):
    """
    Multi-layer RGCN for encoding local subgraph around each entity.
    Replaces simple 1-hop mean aggregation.
    """
    def __init__(self, dim, num_relations, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dim        = dim
        # Relation-specific weight matrices per layer
        self.W_rel = nn.ModuleList([
            nn.Embedding(num_relations * 2, dim * dim)  # *2 for inverse
            for _ in range(num_layers)
        ])
        self.W_self = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        for W in self.W_rel:
            nn.init.normal_(W.weight, std=0.01)

    def forward(self, entity_embs, nbr_ids, rel_ids, layer_idx=0):
        """
        entity_embs: (B, dim) — query entity embeddings
        nbr_ids: list of neighbor id tensors
        rel_ids: list of relation id tensors
        """
        if len(nbr_ids) == 0:
            return entity_embs

        W   = self.W_rel[layer_idx]
        Ws  = self.W_self[layer_idx]
        norm= self.norms[layer_idx]

        # Self transformation
        h_self = Ws(entity_embs)

        # Neighbor aggregation with relation-specific transforms
        nbr_emb = entity_embs[nbr_ids] if nbr_ids.max() < entity_embs.size(0) \
                  else entity_embs
        W_mat   = W(rel_ids).view(len(rel_ids), self.dim, self.dim)
        msgs    = torch.bmm(W_mat, nbr_emb.unsqueeze(-1)).squeeze(-1)

        # Attention-weighted aggregation
        attn = F.softmax(
            (msgs * h_self.unsqueeze(0)).sum(-1, keepdim=True), dim=0)
        agg  = (attn * msgs).sum(0)

        out = F.relu(norm(h_self + agg))
        return self.dropout(out)

# ── RHGNN v3 ──────────────────────────────────────────────────────────────────

class RHGNNv3(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5,
                 num_sgcn_layers=2, copy_weight=2.0):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps
        self.copy_weight   = nn.Parameter(torch.tensor(copy_weight))

        # Learnable curvature
        self._log_c = nn.Parameter(
            torch.tensor(math.log(math.exp(init_curvature) - 1.0)))

        # Base embeddings
        self.entity_emb   = nn.Embedding(num_entities,  dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Frequency-aware
        self.freq_emb  = nn.Embedding(num_entities, dim // 4)
        self.freq_gate = nn.Linear(dim + dim // 4, dim)

        # Subgraph encoder (replaces simple message passing)
        self.sgcn = SubgraphEncoder(
            dim, num_relations, num_layers=num_sgcn_layers, dropout=dropout)

        # Relation-aware message passing
        self.rel_W   = nn.Embedding(num_relations, dim * dim)
        self.mp_gate = nn.Linear(dim * 2, dim)
        self.mp_norm = nn.LayerNorm(dim)

        # Full H-GRU with reset gate
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim)
        self.Uh = nn.Linear(dim, dim, bias=False)

        # ODE
        self.ode_func = ODEFunc(dim, dropout)

        # Relation rotation for scoring
        self.rel_rot = nn.Embedding(num_relations, dim * dim)

        # Copy gate — learned weight between generative and copy scores
        self.copy_gate = nn.Linear(dim, 1)
        self.gen_proj  = nn.Linear(dim, num_entities)

        self.bias    = nn.Parameter(torch.zeros(num_entities))
        self.dropout = nn.Dropout(dropout)

        # Init
        for emb in [self.entity_emb, self.relation_emb,
                    self.rel_W, self.rel_rot]:
            nn.init.normal_(emb.weight, std=0.01)

        # History vocab (set during training)
        self.history_vocab = None

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(5e-2, 5.0)

    def set_history_vocab(self, vocab):
        self.history_vocab = vocab

    def _to_hyp(self, ids):
        h = self.entity_emb(ids)
        f = self.freq_emb(ids)
        g = torch.sigmoid(self.freq_gate(torch.cat([h, f], dim=-1)))
        return expmap0(g * h, self.c)

    def _rel_hyp(self, ids):
        return expmap0(self.relation_emb(ids), self.c)

    def build_nbr_index(self, snapshots, max_nbrs=32):
        self._nbr_index = {}
        for tau, ent_dict in snapshots.items():
            for ent, nbrs in ent_dict.items():
                nbrs = nbrs[:max_nbrs]
                self._nbr_index[(tau, ent)] = (
                    torch.tensor([n for n, _ in nbrs], dtype=torch.long),
                    torch.tensor([r for _, r in nbrs], dtype=torch.long),
                )
        print(f"Neighbour index: {len(self._nbr_index):,} entries")

    def refresh_msg_table(self, device):
        c = self.c.detach()
        with torch.no_grad():
            all_emb = self.entity_emb.weight.to(device)
            all_tan = logmap0(expmap0(all_emb, c), c)
            self._msg_table = {}
            self._all_tan   = all_tan  # cache for subgraph encoder

            for (tau, ent), (nbr_ids, rel_ids) in self._nbr_index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)

                if len(nbr_ids) == 0:
                    self._msg_table[(tau, ent)] = torch.zeros(
                        self.dim, device=device)
                    continue

                # Multi-layer subgraph encoding
                nbr_tan = all_tan[nbr_ids]
                h_self  = all_tan[ent].unsqueeze(0).expand(
                    len(nbr_ids), -1)

                # Layer 1 — relation-aware transform
                W    = self.rel_W(rel_ids).view(len(nbr_ids), self.dim, self.dim)
                msgs = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)

                # Attention gate
                attn = torch.sigmoid(
                    self.mp_gate(torch.cat([msgs, h_self], dim=-1)))
                agg  = (attn * msgs).mean(dim=0)
                msg  = self.mp_norm(torch.tanh(agg))

                self._msg_table[(tau, ent)] = msg

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        c     = self.c
        h_hyp = self._to_hyp(entity_ids)
        h_tan = logmap0(h_hyp, c)
        r_mean = self.relation_emb.weight.mean(dim=0)
        msg    = torch.tanh(h_tan + r_mean.unsqueeze(0))
        return msg

    def _hgru_jump(self, h_prev_hyp, msg_tan, delta_t=1.0):
        c      = self.c
        v_prev = logmap0(h_prev_hyp, c)
        r      = torch.sigmoid(self.Wr(v_prev) + self.Ur(msg_tan))
        z      = torch.sigmoid(self.Wz(v_prev) + self.Uz(msg_tan))
        h_cand = torch.tanh(self.Wh(v_prev) + self.Uh(r * msg_tan))
        decay  = torch.exp(-0.1 * torch.tensor(
            delta_t, device=v_prev.device, dtype=v_prev.dtype))
        z      = z * decay
        v_new  = z * h_cand + (1.0 - z) * v_prev
        return expmap0(v_new, c)

    def _ode_flow(self, h_hyp, delta_t):
        if delta_t <= 0.0:
            return h_hyp
        c     = self.c
        state = logmap0(h_hyp, c)
        dt    = delta_t / max(self.ode_steps, 1)
        for _ in range(self.ode_steps):
            state = state + dt * self.ode_func(state)
        return expmap0(state, c)

    def _hyperbolic_rotate(self, h_hyp, r_ids):
        c     = self.c
        h_tan = logmap0(h_hyp, c)
        R     = self.rel_rot(r_ids).view(-1, self.dim, self.dim)
        h_rot = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        return expmap0(h_rot, c)

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._hgru_jump(h_hyp, msg_tan, delta_t)
        h_flow  = self._ode_flow(h_jump, delta_t)
        h_rot   = self._hyperbolic_rotate(h_flow, r)
        return h_rot

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        dist  = hyp_distance(h_rot, self._to_hyp(t), self.c)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0):
        c     = self.c
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        h_tan = logmap0(h_rot, c)

        # Generative score — hyperbolic distance
        all_ids  = torch.arange(self.num_entities, device=h.device)
        all_hyp  = self._to_hyp(all_ids)
        pred_exp = h_rot.unsqueeze(1).expand(-1, self.num_entities, -1)
        tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
        dist     = hyp_distance(pred_exp, tail_exp, c)
        gen_score = -dist + self.bias.unsqueeze(0)

        # Copy score — boost historically seen entities
        if self.history_vocab is not None:
            hist_mask = build_history_mask(
                h, r, self.history_vocab, self.num_entities, h.device)
            copy_score = hist_mask * self.copy_weight.abs()
            # Learned gate between generative and copy
            gate      = torch.sigmoid(self.copy_gate(h_tan))
            final     = gate * gen_score + (1.0 - gate) * (gen_score + copy_score)
        else:
            final = gen_score

        return final

# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch_v3(model, train_groups, optimizer, device,
                        num_entities, snapshots, ts_to_real,
                        neg_ratio=10, batch_size=512, grad_clip=1.0,
                        adv_temp=1.0):
    model.train()
    total_loss, total_n = 0.0, 0
    prev_real_ts = None

    for ts_id, quads in train_groups:
        real_ts  = ts_to_real.get(ts_id, ts_id)
        delta_t  = float(real_ts - prev_real_ts) \
                   if prev_real_ts is not None else 1.0
        delta_t  = max(delta_t, 1.0)
        prev_real_ts = real_ts

        for start in range(0, len(quads), batch_size):
            mini  = quads[start:start+batch_size]
            batch = torch.tensor(mini, dtype=torch.long, device=device)
            h, r, t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]

            pos_score = model.score(h, r, t, tau, snapshots, delta_t)

            neg_scores_list = []
            for _ in range(neg_ratio):
                neg_t = torch.randint(0, num_entities, (h.size(0),),
                                     device=device)
                ns    = model.score(h, r, neg_t, tau, snapshots, delta_t)
                neg_scores_list.append(ns)
            neg_scores = torch.stack(neg_scores_list, dim=-1)

            loss = adversarial_loss(pos_score, neg_scores, adv_temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(mini)
            total_n    += len(mini)

    return total_loss / max(total_n, 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   default="data/ICEWS14")
    parser.add_argument("--epochs",         type=int,   default=500)
    parser.add_argument("--batch_size",     type=int,   default=1024)
    parser.add_argument("--eval_batch",     type=int,   default=512)
    parser.add_argument("--dim",            type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_decay",       type=float, default=0.8)
    parser.add_argument("--lr_decay_every", type=int,   default=50)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--curvature",      type=float, default=1.0)
    parser.add_argument("--ode_steps",      type=int,   default=5)
    parser.add_argument("--neg_ratio",      type=int,   default=10)
    parser.add_argument("--adv_temp",       type=float, default=1.0)
    parser.add_argument("--eval_every",     type=int,   default=10)
    parser.add_argument("--patience",       type=int,   default=60)
    parser.add_argument("--sgcn_layers",    type=int,   default=2)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--save_path",      type=str,
                        default="checkpoints/rhgnn_v3_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data      = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(data.train)
    print(f"Entities: {data.num_entities} | Relations: {data.num_relations}")
    print(f"Train: {len(data.train)} | Valid: {len(data.valid)} | "
          f"Test: {len(data.test)}")

    # Build historical vocabulary from training data
    print("Building historical vocabulary...", flush=True)
    history_vocab = build_history_vocab(data.train)
    print(f"History vocab: {len(history_vocab):,} (subject,relation) pairs")

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items(), key=lambda x: x[0])

    model = RHGNNv3(
        num_entities  = data.num_entities,
        num_relations = data.num_relations,
        dim           = args.dim,
        init_curvature= args.curvature,
        dropout       = args.dropout,
        ode_steps     = args.ode_steps,
        num_sgcn_layers = args.sgcn_layers,
    ).to(device)

    # Set history vocabulary
    model.set_history_vocab(history_vocab)

    total_params = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    best_mrr, no_improve = 0.0, 0

    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_v3(
            model, train_groups, optimizer, device,
            data.num_entities, snapshots, ts_to_real,
            neg_ratio=args.neg_ratio,
            batch_size=args.batch_size,
            adv_temp=args.adv_temp)
        scheduler.step()

        c_val = model.c.item()
        lr    = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | "
              f"c: {c_val:.4f} | LR: {lr:.2e}", flush=True)

        if epoch % args.eval_every == 0:
            model.refresh_msg_table(device)
            vm = evaluate(model, data.valid, data.all_true,
                         snapshots, ts_to_real, device,
                         batch_size=args.eval_batch)
            print(f"  Valid | MRR={vm['MRR']:.4f} | "
                  f"H@1={vm['Hits@1']:.4f} | H@3={vm['Hits@3']:.4f} | "
                  f"H@10={vm['Hits@10']:.4f} | MAR={vm['MAR']:.1f}",
                  flush=True)

            if vm['MRR'] > best_mrr:
                best_mrr   = vm['MRR']
                no_improve = 0
                torch.save({'model_state': model.state_dict(),
                           'epoch': epoch,
                           'mrr': best_mrr}, args.save_path)
                print(f"  [saved] -> {args.save_path}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience // args.eval_every:
                    print(f"Early stop at epoch {epoch}", flush=True)
                    break

    print(f"\nBest valid MRR: {best_mrr:.4f}")
    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device,
                 batch_size=args.eval_batch)
    print(f"  Test  | MRR={tm['MRR']:.4f} | H@1={tm['Hits@1']:.4f} | "
          f"H@3={tm['Hits@3']:.4f} | H@10={tm['Hits@10']:.4f} | "
          f"MAR={tm['MAR']:.1f}")
    print(f"\nBest valid MRR : {best_mrr:.4f}  (epoch {ckpt['epoch']})")
    print("Done.")

if __name__ == '__main__':
    main()

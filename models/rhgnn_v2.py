"""
TempHypE-R-FA (V2) — Improved with:
1. Relation-aware message passing
2. Multi-hop attention
3. Frequency-aware entity embedding
4. Adversarial negative sampling
5. Copy mechanism
6. Time encoding
7. Hyperbolic bilinear scoring
"""
import argparse, math, os, random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import geoopt
from geoopt.optim import RiemannianAdam

# reuse data loading from original
import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import (
    set_seed, TemporalKGData, read_quad_file, build_id_maps,
    encode_rows, load_temporal_kg, build_snapshot_graphs, evaluate,
    Quad
)

EPS = 1e-6

# ── Hyperbolic ops ────────────────────────────────────────────────────────────

def expmap0(u, c):
    sqrt_c = c.sqrt()
    norm   = u.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)

def logmap0(p, c):
    sqrt_c = c.sqrt()
    norm   = p.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0-EPS)) * p / norm

def mobius_add(x, y, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    den = (1 + 2*c*xy + c**2 * x2 * y2).clamp(min=EPS)
    return num / den

def hyp_distance(x, y, c):
    sqrt_c = c.sqrt()
    diff   = mobius_add(-x, y, c)
    norm   = diff.norm(dim=-1).clamp(min=EPS)
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0-EPS))

# ── Time Encoding ─────────────────────────────────────────────────────────────

class TimeEncoding(nn.Module):
    """Bochner's theorem based continuous time encoding (TGAT-style)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w   = nn.Linear(1, dim // 2)
        nn.init.normal_(self.w.weight, std=0.01)

    def forward(self, t):
        t    = t.float().unsqueeze(-1)
        freq = self.w(t)
        return torch.cat([torch.cos(freq), torch.sin(freq)], dim=-1)

# ── ODE Function ──────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
    def forward(self, z): return self.net(z)

# ── Copy Mechanism ────────────────────────────────────────────────────────────

class CopyMechanism(nn.Module):
    """CyGNet-style copy gate for recurring facts"""
    def __init__(self, dim, num_entities):
        super().__init__()
        self.copy_gate = nn.Linear(dim, 1)
        self.gen_proj  = nn.Linear(dim, num_entities)

    def forward(self, h_evolved, history_mask=None):
        gate      = torch.sigmoid(self.copy_gate(h_evolved))
        gen_score = self.gen_proj(h_evolved)
        if history_mask is not None:
            copy_score = history_mask.float() * 10.0
            return gate * gen_score + (1.0 - gate) * copy_score
        return gen_score

# ── TempHypE-R-FA (V2) ──────────────────────────────────────────────────────────────────

class TempHypERv2(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5,
                 use_copy=True, use_time_enc=True, use_freq=True):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps
        self.use_copy      = use_copy
        self.use_time_enc  = use_time_enc
        self.use_freq      = use_freq

        # Learnable curvature
        self._log_c = nn.Parameter(
            torch.tensor(math.log(math.exp(init_curvature) - 1.0)))

        # Base embeddings
        self.entity_emb   = nn.Embedding(num_entities,  dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Relation-aware message passing (improved)
        self.rel_W        = nn.Embedding(num_relations, dim * dim)
        self.mp_gate      = nn.Linear(dim * 2, dim)
        self.mp_norm      = nn.LayerNorm(dim)

        # Frequency-aware embedding
        if use_freq:
            self.freq_emb  = nn.Embedding(num_entities, dim // 4)
            self.freq_gate = nn.Linear(dim + dim // 4, dim)

        # Time encoding
        if use_time_enc:
            self.time_enc    = TimeEncoding(dim)
            self.time_fusion = nn.Linear(dim * 2, dim)

        # H-GRU (improved with forget gate)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim)
        self.Uh = nn.Linear(dim, dim, bias=False)

        # ODE
        self.ode_func = ODEFunc(dim, dropout)

        # Copy mechanism
        if use_copy:
            self.copy_mech = CopyMechanism(dim, num_entities)

        # Scoring
        self.rel_rot   = nn.Embedding(num_relations, dim * dim)
        self.bias      = nn.Parameter(torch.zeros(num_entities))
        self.dropout   = nn.Dropout(dropout)

        # Init
        for emb in [self.entity_emb, self.relation_emb,
                    self.rel_W, self.rel_rot]:
            nn.init.normal_(emb.weight, std=0.01)

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(5e-2, 5.0)

    def _to_hyp(self, ids):
        h = self.entity_emb(ids)
        if self.use_freq:
            f     = self.freq_emb(ids)
            gate  = torch.sigmoid(self.freq_gate(
                torch.cat([h, f], dim=-1)))
            h     = gate * h
        return expmap0(h, self.c)

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
            all_tan = logmap0(
                expmap0(self.entity_emb.weight.to(device), c), c)
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in self._nbr_index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                nbr_tan = all_tan[nbr_ids]
                W       = self.rel_W(rel_ids).view(
                    len(nbr_ids), self.dim, self.dim)
                m       = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                # Attention over neighbors
                h_self  = all_tan[ent].unsqueeze(0).expand(len(nbr_ids), -1)
                attn    = torch.sigmoid(
                    self.mp_gate(torch.cat([m, h_self], dim=-1)))
                msg     = (attn * m).mean(dim=0)
                self._msg_table[(tau, ent)] = torch.tanh(msg)

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        c       = self.c
        h_hyp   = self._to_hyp(entity_ids)
        h_tan   = logmap0(h_hyp, c)
        # relation-aware self message
        r_mean  = self.relation_emb.weight.mean(dim=0)
        msg     = torch.tanh(h_tan + r_mean.unsqueeze(0))
        return msg

    def _hgru_jump(self, h_prev_hyp, msg_tan, delta_t=1.0):
        """Full GRU with reset gate in hyperbolic space"""
        c      = self.c
        v_prev = logmap0(h_prev_hyp, c)
        # Reset gate
        r      = torch.sigmoid(self.Wr(v_prev) + self.Ur(msg_tan))
        # Update gate
        z      = torch.sigmoid(self.Wz(v_prev) + self.Uz(msg_tan))
        # Candidate
        h_cand = torch.tanh(self.Wh(v_prev) + self.Uh(r * msg_tan))
        # Temporal decay on update gate
        decay  = torch.exp(-0.1 * torch.tensor(delta_t, device=v_prev.device))
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
        """Relation-specific rotation in hyperbolic space"""
        c     = self.c
        h_tan = logmap0(h_hyp, c)
        R     = self.rel_rot(r_ids).view(-1, self.dim, self.dim)
        h_rot = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        return expmap0(h_rot, c)

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        """Compute evolved entity representation"""
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._hgru_jump(h_hyp, msg_tan, delta_t)
        h_flow  = self._ode_flow(h_jump, delta_t)
        # Apply relation rotation
        h_rot   = self._hyperbolic_rotate(h_flow, r)
        return h_rot

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        c      = self.c
        h_rot  = self.forward_entity(h, r, tau, snapshots, delta_t)
        dist   = hyp_distance(h_rot, self._to_hyp(t), c)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0,
                        history_mask=None):
        c      = self.c
        h_rot  = self.forward_entity(h, r, tau, snapshots, delta_t)

        if self.use_copy and history_mask is not None:
            # Copy mechanism scores
            return self.copy_mech(logmap0(h_rot, c), history_mask)

        # Standard hyperbolic distance scoring
        all_ids  = torch.arange(self.num_entities, device=h.device)
        all_hyp  = self._to_hyp(all_ids)
        pred_exp = h_rot.unsqueeze(1).expand(-1, self.num_entities, -1)
        tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
        dist     = hyp_distance(pred_exp, tail_exp, c)
        return -dist + self.bias.unsqueeze(0)

# ── Adversarial Negative Sampling Loss ───────────────────────────────────────

def adversarial_loss(pos_scores, neg_scores, temperature=1.0):
    """Self-adversarial negative sampling (RotatE-style)"""
    neg_weights = F.softmax(neg_scores * temperature, dim=-1).detach()
    neg_loss    = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1)
    pos_loss    = -F.logsigmoid(pos_scores)
    return (pos_loss + neg_loss).mean()

# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch_v2(model, train_groups, optimizer, device,
                        num_entities, snapshots, ts_to_real,
                        neg_ratio=10, batch_size=512, grad_clip=1.0,
                        adv_temp=1.0):
    model.train()
    total_loss, total_n = 0.0, 0
    prev_real_ts = None

    for ts_id, quads in train_groups:
        real_ts  = ts_to_real.get(ts_id, ts_id)
        delta_t  = float(real_ts - prev_real_ts) if prev_real_ts is not None else 1.0
        delta_t  = max(delta_t, 1.0)
        prev_real_ts = real_ts

        for start in range(0, len(quads), batch_size):
            mini  = quads[start:start+batch_size]
            batch = torch.tensor(mini, dtype=torch.long, device=device)
            h, r, t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]

            pos_score = model.score(h, r, t, tau, snapshots, delta_t)

            # Adversarial negatives
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
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--save_path",      type=str,   default="checkpoints/rhgnn_v2_best.pt")
    parser.add_argument("--no_copy",        action="store_true")
    parser.add_argument("--no_time_enc",    action="store_true")
    parser.add_argument("--no_freq",        action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data      = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(data.train)
    print(f"Entities: {data.num_entities} | Relations: {data.num_relations}")
    print(f"Train: {len(data.train)} | Valid: {len(data.valid)} | Test: {len(data.test)}")

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items(), key=lambda x: x[0])

    model = TempHypERFA(
        num_entities  = data.num_entities,
        num_relations = data.num_relations,
        dim           = args.dim,
        init_curvature= args.curvature,
        dropout       = args.dropout,
        ode_steps     = args.ode_steps,
        use_copy      = not args.no_copy,
        use_time_enc  = not args.no_time_enc,
        use_freq      = not args.no_freq,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    best_mrr, no_improve = 0.0, 0

    # Build index once
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_v2(
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
                           'epoch': epoch, 'mrr': best_mrr}, args.save_path)
                print(f"  [saved] -> {args.save_path}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience // args.eval_every:
                    print(f"Early stop at epoch {epoch}", flush=True)
                    break

    # Test
    print(f"\nBest valid MRR: {best_mrr:.4f}")
    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device, batch_size=args.eval_batch)
    print(f"  Test  | MRR={tm['MRR']:.4f} | H@1={tm['Hits@1']:.4f} | "
          f"H@3={tm['Hits@3']:.4f} | H@10={tm['Hits@10']:.4f} | "
          f"MAR={tm['MAR']:.1f}")
    print(f"\nBest valid MRR : {best_mrr:.4f}  (epoch {ckpt['epoch']})")
    print("Done.")

if __name__ == '__main__':
    main()

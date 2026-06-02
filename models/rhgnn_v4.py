"""
RHGNN v4 — V3 + Contrastive Learning + Soft Labels + 
           Relation Path + Curvature Per Relation + 
           Temporal Smoothness + Ensemble ready
"""
import argparse, math, os, random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt.optim import RiemannianAdam

import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import (
    set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
)
from rhgnn_v2 import expmap0, logmap0, mobius_add, hyp_distance, ODEFunc, adversarial_loss
from rhgnn_v3 import build_history_vocab, build_history_mask, SubgraphEncoder

EPS = 1e-6

# ── Contrastive Loss ──────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    def __init__(self, dim, temp=0.07):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.temp = temp

    def forward(self, anchor, positives, negatives):
        """
        anchor:    (B, dim) query entity embeddings
        positives: (B, dim) historically seen tail embeddings
        negatives: (B, K, dim) never-seen entity embeddings
        """
        a = F.normalize(self.proj(anchor),    dim=-1)
        p = F.normalize(self.proj(positives), dim=-1)
        n = F.normalize(self.proj(negatives.view(-1, negatives.size(-1))), dim=-1)
        n = n.view(negatives.size(0), negatives.size(1), -1)

        pos_sim = (a * p).sum(-1, keepdim=True) / self.temp
        neg_sim = torch.bmm(a.unsqueeze(1), n.transpose(1,2)).squeeze(1) / self.temp

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
        return F.cross_entropy(logits, labels)

# ── Soft Label Loss ───────────────────────────────────────────────────────────

def soft_label_loss(scores, true_tails, history_freq=None, smooth=0.1):
    """
    KL divergence with soft targets.
    history_freq: (B, num_entities) frequency of historical entities
    """
    B, E = scores.size()
    target = torch.zeros(B, E, device=scores.device)
    target.scatter_(1, true_tails.unsqueeze(1), 1.0)

    if history_freq is not None:
        target = (1 - smooth) * target + smooth * history_freq
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=EPS)

    log_probs = F.log_softmax(scores, dim=-1)
    return F.kl_div(log_probs, target.detach(), reduction='batchmean')

# ── Relation Path Encoder ─────────────────────────────────────────────────────

class RelationPathEncoder(nn.Module):
    def __init__(self, dim, num_relations, max_path_len=3):
        super().__init__()
        self.path_linear = nn.Linear(dim, dim)
        self.rel_emb   = nn.Embedding(num_relations, dim)
        self.path_gate = nn.Linear(dim * 2, dim)
        self.max_len   = max_path_len

    def forward(self, entity_emb, rel_ids, nbr_rel_map):
        """
        For each entity, encode its relation neighborhood as a path.
        rel_ids: (B,) query relation ids
        nbr_rel_map: dict mapping entity_id to list of (nbr, rel) pairs
        """
        r_emb = self.rel_emb(rel_ids)  # (B, dim)
        # Simple: encode query relation + mean neighbor relations
        gate  = torch.sigmoid(self.path_gate(
            torch.cat([entity_emb, r_emb], dim=-1)))
        return gate * entity_emb + (1 - gate) * r_emb

# ── RHGNN v4 ──────────────────────────────────────────────────────────────────

class RHGNNv4(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5,
                 num_sgcn_layers=2, contrastive_temp=0.07,
                 smooth_label=0.1, lambda_smooth=0.01):
        super().__init__()
        self.num_entities     = num_entities
        self.num_relations    = num_relations
        self.dim              = dim
        self.ode_steps        = ode_steps
        self.smooth_label     = smooth_label
        self.lambda_smooth    = lambda_smooth

        # Global + per-relation curvature
        self._log_c     = nn.Parameter(
            torch.tensor(math.log(math.exp(init_curvature) - 1.0)))
        self._log_c_rel = nn.Parameter(torch.zeros(num_relations))

        # Embeddings
        self.entity_emb   = nn.Embedding(num_entities,  dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Frequency embedding
        self.freq_emb  = nn.Embedding(num_entities, dim // 4)
        self.freq_gate = nn.Linear(dim + dim // 4, dim)

        # Subgraph encoder
        self.sgcn = SubgraphEncoder(
            dim, num_relations, num_layers=num_sgcn_layers,
            dropout=dropout)

        # Relation-aware MP
        self.rel_W   = nn.Embedding(num_relations, dim * dim)
        self.mp_gate = nn.Linear(dim * 2, dim)
        self.mp_norm = nn.LayerNorm(dim)

        # Full H-GRU
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim)
        self.Uh = nn.Linear(dim, dim, bias=False)

        # ODE
        self.ode_func = ODEFunc(dim, dropout)

        # Relation path encoder
        self.path_enc = RelationPathEncoder(dim, num_relations)

        # Hyperbolic rotation scoring
        self.rel_rot = nn.Embedding(num_relations, dim * dim)

        # Copy mechanism
        self.copy_weight = nn.Parameter(torch.tensor(2.0))
        self.copy_gate   = nn.Linear(dim, 1)
        self.gen_proj    = nn.Linear(dim, num_entities)

        # Contrastive loss
        self.contrastive = ContrastiveLoss(dim, temp=contrastive_temp)

        # Temporal smoothness — previous entity states
        self._prev_entity_emb = None

        self.bias    = nn.Parameter(torch.zeros(num_entities))
        self.dropout = nn.Dropout(dropout)

        # History vocab
        self.history_vocab = None
        self.history_freq  = None

        # Init
        for emb in [self.entity_emb, self.relation_emb,
                    self.rel_W, self.rel_rot]:
            nn.init.normal_(emb.weight, std=0.01)

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(5e-2, 5.0)

    def c_rel(self, rel_ids):
        """Per-relation curvature."""
        return F.softplus(
            self._log_c + self._log_c_rel[rel_ids]
        ).clamp(5e-2, 5.0)

    def set_history_vocab(self, vocab):
        self.history_vocab = vocab

    def set_history_freq(self, freq):
        """freq: dict mapping (h,r) -> frequency tensor over entities"""
        self.history_freq = freq

    def _to_hyp(self, ids, c=None):
        if c is None:
            c = self.c
        h = self.entity_emb(ids)
        f = self.freq_emb(ids)
        g = torch.sigmoid(self.freq_gate(torch.cat([h, f], dim=-1)))
        return expmap0(g * h, c)

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
                if len(nbr_ids) == 0:
                    self._msg_table[(tau, ent)] = torch.zeros(
                        self.dim, device=device)
                    continue
                nbr_tan = all_tan[nbr_ids]
                h_self  = all_tan[ent].unsqueeze(0).expand(
                    len(nbr_ids), -1)
                W    = self.rel_W(rel_ids).view(
                    len(nbr_ids), self.dim, self.dim)
                msgs = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                attn = torch.sigmoid(
                    self.mp_gate(torch.cat([msgs, h_self], dim=-1)))
                agg  = (attn * msgs).mean(dim=0)
                self._msg_table[(tau, ent)] = self.mp_norm(
                    torch.tanh(agg))

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        c      = self.c
        h_hyp  = self._to_hyp(entity_ids)
        h_tan  = logmap0(h_hyp, c)
        r_mean = self.relation_emb.weight.mean(dim=0)
        return torch.tanh(h_tan + r_mean.unsqueeze(0))

    def _hgru_jump(self, h_prev_hyp, msg_tan, delta_t=1.0):
        c      = self.c
        v_prev = logmap0(h_prev_hyp, c)
        r      = torch.sigmoid(self.Wr(v_prev) + self.Ur(msg_tan))
        z      = torch.sigmoid(self.Wz(v_prev) + self.Uz(msg_tan))
        h_cand = torch.tanh(self.Wh(v_prev) + self.Uh(r * msg_tan))
        decay  = torch.exp(-0.1 * torch.tensor(
            delta_t, device=v_prev.device, dtype=v_prev.dtype))
        v_new  = (z * decay) * h_cand + (1.0 - z * decay) * v_prev
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

    def temporal_smooth_loss(self):
        """Penalize large changes in entity embeddings between epochs."""
        if self._prev_entity_emb is None:
            return torch.tensor(0.0, device=self.entity_emb.weight.device)
        diff = self.entity_emb.weight - self._prev_entity_emb.detach()
        return self.lambda_smooth * (diff ** 2).mean()

    def update_prev_emb(self):
        self._prev_entity_emb = self.entity_emb.weight.detach().clone()

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)

        # Relation path encoding
        h_tan   = logmap0(h_hyp, self.c)
        h_path  = self.path_enc(h_tan, r, {})
        h_hyp   = expmap0(h_path, self.c)

        h_jump  = self._hgru_jump(h_hyp, msg_tan, delta_t)
        h_flow  = self._ode_flow(h_jump, delta_t)

        # Per-relation curvature for rotation
        c_r   = self.c_rel(r)
        h_tan = logmap0(h_flow, self.c)
        R     = self.rel_rot(r).view(-1, self.dim, self.dim)
        h_rot = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        h_rot = expmap0(h_rot, c_r.unsqueeze(-1).expand_as(h_rot))
        return h_rot

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        dist  = hyp_distance(h_rot, self._to_hyp(t), self.c)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0,
                        chunk_size=1000):
        c     = self.c
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        h_tan = logmap0(h_rot, c)

        # Generative score — chunked to avoid OOM on large datasets
        all_ids   = torch.arange(self.num_entities, device=h.device)
        gen_score = torch.zeros(h.size(0), self.num_entities, device=h.device)
        for start in range(0, self.num_entities, chunk_size):
            end      = min(start + chunk_size, self.num_entities)
            chunk    = all_ids[start:end]
            all_hyp  = self._to_hyp(chunk)
            pred_exp = h_rot.unsqueeze(1).expand(-1, len(chunk), -1)
            tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist     = hyp_distance(pred_exp, tail_exp, c)
            gen_score[:, start:end] = -dist
        gen_score = gen_score + self.bias.unsqueeze(0)

        # Copy score with history vocab
        if self.history_vocab is not None:
            hist_mask  = build_history_mask(
                h, r, self.history_vocab,
                self.num_entities, h.device)
            copy_score = hist_mask * self.copy_weight.abs()
            gate       = torch.sigmoid(self.copy_gate(h_tan))
            return gate * gen_score + (1.0 - gate) * (gen_score + copy_score)

        return gen_score

    def compute_contrastive(self, h, r, t, tau, snapshots, delta_t,
                            num_neg=16):
        """Compute contrastive loss between historical and random entities."""
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        anchor_tan = logmap0(h_rot, self.c)

        # Positive: true tail embedding
        pos_hyp = self._to_hyp(t)
        pos_tan = logmap0(pos_hyp, self.c)

        # Negatives: random non-historical entities
        neg_ids = torch.randint(
            0, self.num_entities, (h.size(0), num_neg),
            device=h.device)
        neg_hyp = self._to_hyp(neg_ids.view(-1))
        neg_tan = logmap0(neg_hyp, self.c).view(
            h.size(0), num_neg, self.dim)

        return self.contrastive(anchor_tan, pos_tan, neg_tan)

# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch_v4(model, train_groups, optimizer, device,
                        num_entities, snapshots, ts_to_real,
                        neg_ratio=10, batch_size=512, grad_clip=1.0,
                        adv_temp=1.0, lambda_contrast=0.1,
                        use_soft_label=True):
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

            # Main scoring loss
            if use_soft_label:
                scores = model.score_all_tails(h, r, tau, snapshots, delta_t)
                main_loss = soft_label_loss(
                    scores, t, smooth=model.smooth_label)
            else:
                pos_score = model.score(h, r, t, tau, snapshots, delta_t)
                neg_scores_list = []
                for _ in range(neg_ratio):
                    neg_t = torch.randint(0, num_entities,
                                         (h.size(0),), device=device)
                    ns = model.score(h, r, neg_t, tau, snapshots, delta_t)
                    neg_scores_list.append(ns)
                neg_scores = torch.stack(neg_scores_list, dim=-1)
                main_loss  = adversarial_loss(pos_score, neg_scores, adv_temp)

            # Contrastive loss
            contrast_loss = model.compute_contrastive(
                h, r, t, tau, snapshots, delta_t)

            # Temporal smoothness
            smooth_loss = model.temporal_smooth_loss()

            loss = main_loss + \
                   lambda_contrast * contrast_loss + \
                   smooth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(mini)
            total_n    += len(mini)

    model.update_prev_emb()
    return total_loss / max(total_n, 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         type=str,   default="data/ICEWS14")
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--eval_batch",       type=int,   default=512)
    parser.add_argument("--dim",              type=int,   default=200)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lr_decay",         type=float, default=0.8)
    parser.add_argument("--lr_decay_every",   type=int,   default=50)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--curvature",        type=float, default=1.0)
    parser.add_argument("--ode_steps",        type=int,   default=5)
    parser.add_argument("--neg_ratio",        type=int,   default=10)
    parser.add_argument("--adv_temp",         type=float, default=1.0)
    parser.add_argument("--eval_every",       type=int,   default=10)
    parser.add_argument("--patience",         type=int,   default=60)
    parser.add_argument("--sgcn_layers",      type=int,   default=2)
    parser.add_argument("--lambda_contrast",  type=float, default=0.1)
    parser.add_argument("--smooth_label",     type=float, default=0.1)
    parser.add_argument("--lambda_smooth",    type=float, default=0.01)
    parser.add_argument("--use_soft_label",   action="store_true")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save_path",        type=str,
                        default="checkpoints/rhgnn_v4_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data      = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(data.train)
    print(f"Entities: {data.num_entities} | Relations: {data.num_relations}")
    print(f"Train: {len(data.train)} | Valid: {len(data.valid)} | "
          f"Test: {len(data.test)}")

    print("Building historical vocabulary...", flush=True)
    history_vocab = build_history_vocab(data.train)
    print(f"History vocab: {len(history_vocab):,} (s,r) pairs")

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items(), key=lambda x: x[0])

    model = RHGNNv4(
        num_entities     = data.num_entities,
        num_relations    = data.num_relations,
        dim              = args.dim,
        init_curvature   = args.curvature,
        dropout          = args.dropout,
        ode_steps        = args.ode_steps,
        num_sgcn_layers  = args.sgcn_layers,
        contrastive_temp = 0.07,
        smooth_label     = args.smooth_label,
        lambda_smooth    = args.lambda_smooth,
    ).to(device)

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
        loss = train_one_epoch_v4(
            model, train_groups, optimizer, device,
            data.num_entities, snapshots, ts_to_real,
            neg_ratio    = args.neg_ratio,
            batch_size   = args.batch_size,
            adv_temp     = args.adv_temp,
            lambda_contrast = args.lambda_contrast,
            use_soft_label  = args.use_soft_label)
        scheduler.step()

        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | "
              f"c: {model.c.item():.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

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

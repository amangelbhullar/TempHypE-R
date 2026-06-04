"""
CA-TempHypE-R-Pure-C: Fully Hyperbolic with ALL V4 enhancements
Adds to V5-Pure:
  - Frequency-aware embeddings (V2)
  - 2-layer HypMP subgraph encoder (V3)
  - Per-relation learnable curvature (V4)
  - Contrastive loss in hyperbolic space (V4)
  - Temporal smoothness regularization (V4)
"""
import argparse, math, os
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import geoopt
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import (set_seed, load_temporal_kg,
    build_snapshot_graphs, evaluate)
from rhgnn_v3 import build_history_vocab, build_history_mask

EPS = 1e-6

# ── Hyperbolic primitives ──────────────────────────────────────────
def artanh(x):
    x = x.clamp(-1+EPS, 1-EPS)
    return 0.5*(torch.log(1+x) - torch.log(1-x))

def expmap0(u, c=1.0):
    norm = u.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.tanh(math.sqrt(c)*norm) * u / (math.sqrt(c)*norm)

def logmap0(p, c=1.0):
    norm = p.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return artanh(math.sqrt(c)*norm) * p / (math.sqrt(c)*norm)

def mobius_add(x, y, c=1.0):
    xy = (x*y).sum(dim=-1, keepdim=True)
    x2 = (x*x).sum(dim=-1, keepdim=True)
    y2 = (y*y).sum(dim=-1, keepdim=True)
    num = (1+2*c*xy+c*y2)*x + (1-c*x2)*y
    den = (1+2*c*xy+c**2*x2*y2).clamp(min=EPS)
    return num/den

def mobius_matvec(M, x, c=1.0):
    Mx = x @ M.T
    Mx_norm = Mx.norm(dim=-1, keepdim=True).clamp(min=EPS)
    x_norm  = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.tanh(Mx_norm/x_norm*artanh(math.sqrt(c)*x_norm)) \
           * Mx/(math.sqrt(c)*Mx_norm)

def hyp_distance(x, y, c=1.0):
    diff = mobius_add(-x, y, c)
    norm = diff.norm(dim=-1).clamp(min=EPS, max=1/math.sqrt(c)-EPS)
    return 2/math.sqrt(c)*artanh(math.sqrt(c)*norm)

# ── HypLinear ─────────────────────────────────────────────────────
class HypLinear(nn.Module):
    def __init__(self, in_dim, out_dim, c=1.0, bias=True):
        super().__init__()
        self.c = c
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        self.use_bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        out = mobius_matvec(self.W, x, self.c)
        if self.use_bias:
            out = mobius_add(out, expmap0(self.b, self.c), self.c)
        return out

# ── HypGRUCell with temporal decay ───────────────────────────────
class HypGRUCell(nn.Module):
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.c   = c
        self.dim = dim
        self.Wz  = nn.Linear(dim*2, dim)
        self.Wr  = nn.Linear(dim*2, dim)
        self.Wh  = HypLinear(dim, dim, c=c)
        self.Uh  = HypLinear(dim, dim, c=c)

    def forward(self, h_hyp, x_hyp, delta_t=1.0):
        c = self.c
        h_tan = logmap0(h_hyp, c)
        x_tan = logmap0(x_hyp, c)
        concat = torch.cat([h_tan, x_tan], dim=-1)
        z = torch.sigmoid(self.Wz(concat))
        r = torch.sigmoid(self.Wr(concat))
        # Temporal decay on update gate
        decay = torch.exp(-0.1*torch.tensor(
            delta_t, device=h_hyp.device, dtype=h_hyp.dtype))
        z = z * decay
        r_h   = mobius_add(expmap0(r*h_tan, c), torch.zeros_like(h_hyp), c)
        Wh_rh = self.Wh(r_h)
        Uh_x  = self.Uh(x_hyp)
        h_tilde = expmap0(torch.tanh(logmap0(
            mobius_add(Wh_rh, Uh_x, c), c)), c)
        # Geodesic interpolation
        h_new = mobius_add(
            expmap0((1-z)*h_tan, c),
            expmap0(z*logmap0(h_tilde, c), c), c)
        return h_new

# ── 2-layer HypMP (subgraph encoder) ─────────────────────────────
class HypMP2Layer(nn.Module):
    """2-layer hyperbolic message passing — V3 enhancement."""
    def __init__(self, dim, num_relations, c=1.0):
        super().__init__()
        self.c = c
        # Layer 1 — relation-specific
        self.rel_lin1 = nn.ModuleList([
            HypLinear(dim, dim, c=c, bias=False)
            for _ in range(num_relations)])
        self.norm1 = nn.LayerNorm(dim)
        # Layer 2 — attention gate
        self.agg_lin  = HypLinear(dim, dim, c=c)
        self.attn_gate = nn.Linear(dim*2, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, nbr_hyp, rel_ids, self_hyp):
        c = self.c
        if len(nbr_hyp) == 0:
            return self_hyp
        # Layer 1: relation transform
        msgs1 = []
        for i, (n, r) in enumerate(zip(nbr_hyp, rel_ids)):
            m = self.rel_lin1[r](n.unsqueeze(0)).squeeze(0)
            msgs1.append(m)
        msgs1 = torch.stack(msgs1)
        # Layer 2: attention gate
        self_tan = logmap0(self_hyp, c)
        attn = torch.sigmoid(self.attn_gate(
            torch.cat([logmap0(msgs1,c),
                       self_tan.unsqueeze(0).expand_as(logmap0(msgs1,c))],
                      dim=-1)))
        weighted = (attn * logmap0(msgs1, c)).mean(0)
        agg = expmap0(torch.tanh(self.norm2(weighted)), c)
        return self.agg_lin(agg)

# ── HypODE ────────────────────────────────────────────────────────
class HypODE(nn.Module):
    def __init__(self, dim, c=1.0, dropout=0.1):
        super().__init__()
        self.c   = c
        self.f1  = HypLinear(dim, dim*2, c=c)
        self.f2  = HypLinear(dim*2, dim, c=c)
        self.drop= nn.Dropout(dropout)

    def forward(self, h_hyp, delta_t, steps=5):
        if delta_t <= 0: return h_hyp
        dt = delta_t / max(steps, 1)
        h  = h_hyp
        for _ in range(steps):
            v  = self.f1(h)
            v  = expmap0(self.drop(torch.relu(logmap0(v, self.c))), self.c)
            v  = self.f2(v)
            dh = logmap0(v, self.c)
            h  = expmap0(logmap0(h, self.c) + dt*dh, self.c)
        return h

# ── Main Model: CA-TempHypE-R-Pure-C ──────────────────────────────────
class TempHypERv5PureEnhanced(nn.Module):
    """
    CA-TempHypE-R-Pure-C: Fully hyperbolic + all V4 enhancements:
      V2: frequency-aware embeddings
      V3: 2-layer HypMP subgraph
      V4: per-relation curvature + contrastive + soft labels + smoothness
    """
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5,
                 smooth_label=0.1, lambda_contrast=0.1,
                 lambda_smooth=0.01, contrast_temp=0.07):
        super().__init__()
        self.num_entities    = num_entities
        self.num_relations   = num_relations
        self.dim             = dim
        self.ode_steps       = ode_steps
        self.smooth_label    = smooth_label
        self.lambda_contrast = lambda_contrast
        self.lambda_smooth   = lambda_smooth
        self.contrast_temp   = contrast_temp

        # ── Per-relation curvature (V4 enhancement) ──
        self.log_c_global = nn.Parameter(torch.tensor(math.log(init_curvature)))
        self.delta_c_r    = nn.Embedding(num_relations, 1)
        nn.init.zeros_(self.delta_c_r.weight)

        @property
        def c(self): return self.log_c_global.exp().item()

        # ── Frequency-aware embeddings (V2 enhancement) ──
        self.entity_emb  = nn.Parameter(torch.zeros(num_entities, dim))
        self.entity_freq = nn.Embedding(num_entities, dim//4)
        self.freq_gate   = nn.Linear(dim + dim//4, dim)
        nn.init.normal_(self.entity_emb,      std=1e-3)
        nn.init.normal_(self.entity_freq.weight, std=0.01)

        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.normal_(self.rel_emb.weight, std=0.001)

        # ── 2-layer HypMP subgraph encoder (V3 enhancement) ──
        # (built with global c — updated during forward)
        self._c_val = init_curvature
        self.hgru   = HypGRUCell(dim, c=init_curvature)
        self.ode    = HypODE(dim, c=init_curvature, dropout=dropout)
        self.hyp_mp = HypMP2Layer(dim, num_relations, c=init_curvature)

        # Per-relation rotation
        self.rel_rot = nn.Embedding(num_relations, dim*dim)
        nn.init.normal_(self.rel_rot.weight, std=0.001)

        # Copy mechanism (V3)
        self.copy_w  = nn.Parameter(torch.tensor(2.0))
        self.copy_g  = nn.Linear(dim, 1)
        self.bias    = nn.Parameter(torch.zeros(num_entities))
        self.history_vocab = None

        # Prev emb for smoothness
        self.register_buffer('prev_emb',
            torch.zeros(num_entities, dim))

        self.dropout = nn.Dropout(dropout)

    def get_c(self, r=None):
        """Get curvature — global or per-relation."""
        c_global = F.softplus(self.log_c_global).clamp(0.01, 10.0)
        if r is None:
            return c_global.item()
        dc = self.delta_c_r(r).squeeze(-1)
        return F.softplus(c_global + dc).clamp(0.01, 10.0)

    def _get_entity_hyp(self, ids, c=None):
        """Get entity embeddings on Poincaré ball."""
        if c is None: c = self.get_c()
        raw   = self.entity_emb[ids]
        freq  = self.entity_freq(ids)
        gate  = torch.sigmoid(self.freq_gate(
            torch.cat([raw, freq], dim=-1)))
        fused = gate * raw
        return expmap0(fused, c if isinstance(c,float)
                       else c.mean().item())

    def build_nbr_index(self, snapshots, max_nbrs=32):
        self._nbr_index = {}
        for tau, snap in snapshots.items():
            for ent, nbrs in snap.items():
                nbr_ids = [n for n,_ in nbrs[:max_nbrs]]
                rel_ids = [r for _,r in nbrs[:max_nbrs]]
                self._nbr_index[(tau,ent)] = (nbr_ids, rel_ids)
        print(f"Neighbour index: {len(self._nbr_index):,} entries")

    def refresh_msg_table(self, device):
        """Fast tangent-space precomputation — Mobius ops in forward only."""
        c = self.get_c()
        with torch.no_grad():
            all_raw  = self.entity_emb.data.to(device)
            all_freq = self.entity_freq.weight.data.to(device)
            gate_w   = self.freq_gate.weight.data.to(device)
            gate_b   = self.freq_gate.bias.data.to(device)
            fused    = torch.sigmoid(
                torch.cat([all_raw, all_freq], dim=-1) @ gate_w.T + gate_b
            ) * all_raw
            all_tan = fused  # tangent space approx

            self._msg_table = {}
            for (tau,ent),(nbr_ids,rel_ids) in self._nbr_index.items():
                if not nbr_ids:
                    self._msg_table[(tau,ent)] = torch.zeros(
                        self.dim, device=device)
                    continue
                nbr_t   = torch.tensor(nbr_ids, device=device)
                rel_t   = torch.tensor(rel_ids,  device=device)
                nbr_tan = all_tan[nbr_t]
                # Layer 1: relation transform in tangent space
                rel_W   = self.hyp_mp.rel_lin1
                msgs = []
                for i,(n_tan, r_id) in enumerate(zip(nbr_tan, rel_ids)):
                    W = rel_W[r_id].W.data.to(device)
                    msgs.append(n_tan @ W.T)
                msgs = torch.stack(msgs)
                # Layer 2: attention gate
                self_tan = all_tan[ent]
                attn = torch.sigmoid(self.hyp_mp.attn_gate(
                    torch.cat([msgs,
                        self_tan.unsqueeze(0).expand_as(msgs)], dim=-1)))
                agg = (attn * msgs).mean(0)
                self._msg_table[(tau,ent)] = torch.tanh(agg).detach()

    def _get_msg(self, ent_ids, tau_ids, device):
        msgs = []
        for eid, tid in zip(ent_ids.tolist(), tau_ids.tolist()):
            msgs.append(self._msg_table.get(
                (tid, eid), torch.zeros(self.dim, device=device)))
        # Return as hyperbolic points via expmap
        tan = torch.stack(msgs).to(device)
        return expmap0(tan, self.get_c())

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        device = self.entity_emb.device
        c_r    = self.get_c(r)
        c_g    = self.get_c()

        h_hyp  = self._get_entity_hyp(h, c_g)
        msg    = self._get_msg(h, tau, device)

        # H-GRU with temporal decay
        h_jump = self.hgru(h_hyp, msg, delta_t)

        # Neural ODE
        h_flow = self.ode(h_jump, delta_t, self.ode_steps)

        # Per-relation rotation + curvature
        R      = self.rel_rot(r).view(-1, self.dim, self.dim)
        h_tan  = logmap0(h_flow, c_g)
        h_rot_tan = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        # Use per-relation curvature for final embedding
        c_r_val = c_r.mean().item() if hasattr(c_r,'mean') else c_r
        h_rot  = expmap0(h_rot_tan, c_r_val)
        return h_rot, c_r_val

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0):
        device = h.device
        c_g    = self.get_c()
        h_rot, c_r = self.forward_entity(h, r, tau, snapshots, delta_t)

        # All entity embeddings
        all_ids = torch.arange(self.num_entities, device=device)
        all_hyp = self._get_entity_hyp(all_ids, c_g)

        # Hyperbolic distance scoring
        gen_score = torch.zeros(h.size(0), self.num_entities, device=device)
        chunk = 512
        for i in range(0, self.num_entities, chunk):
            tails = all_hyp[i:i+chunk]
            for j, hj in enumerate(h_rot):
                d = hyp_distance(
                    hj.unsqueeze(0).expand(len(tails),-1),
                    tails, c_r)
                gen_score[j,i:i+chunk] = -d
        gen_score = gen_score + self.bias.unsqueeze(0)

        # Copy mechanism (V3)
        if self.history_vocab is not None:
            hist_mask = build_history_mask(
                h, r, self.history_vocab,
                self.num_entities, device)
            copy_score = hist_mask * self.copy_w.abs()
            gate = torch.sigmoid(
                self.copy_g(logmap0(h_rot, c_r))).squeeze(-1).unsqueeze(-1)
            return gate*gen_score + (1-gate)*(gen_score+copy_score)
        return gen_score

    def compute_contrastive(self, h, r, t, tau, snapshots,
                            delta_t=1.0, n_neg=16):
        """Contrastive loss in hyperbolic space (V4 enhancement)."""
        device = h.device
        c_g    = self.get_c()
        h_rot, c_r = self.forward_entity(h, r, tau, snapshots, delta_t)
        anchor = F.normalize(logmap0(h_rot, c_r), dim=-1)
        pos    = F.normalize(logmap0(
            self._get_entity_hyp(t, c_g), c_g), dim=-1)
        neg_ids = torch.randint(0, self.num_entities,
                                (h.size(0)*n_neg,), device=device)
        neg = F.normalize(logmap0(
            self._get_entity_hyp(neg_ids, c_g), c_g), dim=-1)
        neg = neg.view(h.size(0), n_neg, self.dim)
        pos_sim = (anchor * pos).sum(-1) / self.contrast_temp
        neg_sim = torch.bmm(
            neg, anchor.unsqueeze(-1)).squeeze(-1) / self.contrast_temp
        logits  = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels  = torch.zeros(h.size(0), dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def temporal_smooth_loss(self):
        """Temporal smoothness regularization (V4 enhancement)."""
        return self.lambda_smooth * F.mse_loss(
            self.entity_emb, self.prev_emb.detach())

    def update_prev_emb(self):
        self.prev_emb.data.copy_(self.entity_emb.data)

def soft_label_loss(scores, true_tails, smooth=0.1):
    B, E = scores.size()
    target = torch.zeros(B, E, device=scores.device)
    target.scatter_(1, true_tails.unsqueeze(1), 1.0)
    target = (1-smooth)*target + smooth/E
    return F.kl_div(F.log_softmax(scores,-1), target, reduction='batchmean')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         type=str,   default="data/ICEWS14")
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--eval_batch",       type=int,   default=256)
    parser.add_argument("--dim",              type=int,   default=200)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lr_decay",         type=float, default=0.8)
    parser.add_argument("--lr_decay_every",   type=int,   default=50)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--curvature",        type=float, default=1.0)
    parser.add_argument("--ode_steps",        type=int,   default=5)
    parser.add_argument("--patience",         type=int,   default=60)
    parser.add_argument("--eval_every",       type=int,   default=10)
    parser.add_argument("--neg_ratio",        type=int,   default=10)
    parser.add_argument("--smooth_label",     type=float, default=0.1)
    parser.add_argument("--lambda_contrast",  type=float, default=0.1)
    parser.add_argument("--lambda_smooth",    type=float, default=0.01)
    parser.add_argument("--contrast_temp",    type=float, default=0.07)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save_path",        type=str,   default="checkpoints/rhgnn_v5_pure_enhanced.pt")
    parser.add_argument("--gpu",              type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    kg        = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(kg.train)
    num_e, num_r = kg.num_entities, kg.num_relations

    print(f"Dataset: {args.data_dir}")
    print(f"Entities={num_e} | Relations={num_r}")
    print(f"Train={len(kg.train)} | Valid={len(kg.valid)} | Test={len(kg.test)}")
    print(f"Device: {device} | CA-TempHypE-R-Pure-C (fully hyperbolic + all V4)")

    model = TempHypE-Rv5PureEnhanced(
        num_e, num_r,
        dim=args.dim,
        init_curvature=args.curvature,
        dropout=args.dropout,
        ode_steps=args.ode_steps,
        smooth_label=args.smooth_label,
        lambda_contrast=args.lambda_contrast,
        lambda_smooth=args.lambda_smooth,
        contrast_temp=args.contrast_temp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # History vocab (V3)
    vocab = build_history_vocab(kg.train)
    model.set_history_vocab(vocab) if hasattr(model,'set_history_vocab') \
        else setattr(model, 'history_vocab', vocab)

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr)

    best_mrr, no_improve = 0.0, 0
    sorted_ts = sorted(set(q[3] for q in kg.train))

    for epoch in range(1, args.epochs+1):
        model.train()
        model.build_nbr_index(snapshots)
        model.refresh_msg_table(device)

        total_loss, total_n = 0.0, 0
        ts_quads = defaultdict(list)
        for q in kg.train: ts_quads[q[3]].append(q)

        prev_ts = None
        for ts in sorted_ts:
            delta_t = (ts - prev_ts) if prev_ts is not None else 1.0
            prev_ts = ts
            quads = ts_quads[ts]

            for start in range(0, len(quads), args.batch_size):
                mini  = quads[start:start+args.batch_size]
                batch = torch.tensor(mini, dtype=torch.long, device=device)
                h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]

                scores = model.score_all_tails(h,r,tau,snapshots,delta_t)
                main_loss = soft_label_loss(scores, t, model.smooth_label)
                contrast  = model.compute_contrastive(
                    h,r,t,tau,snapshots,delta_t)
                smooth    = model.temporal_smooth_loss()
                loss = main_loss + args.lambda_contrast*contrast + smooth

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()*len(mini)
                total_n    += len(mini)

        model.update_prev_emb()
        avg_loss = total_loss/max(total_n,1)
        c_val    = model.get_c()

        if epoch % args.eval_every == 0:
            model.eval()
            model.refresh_msg_table(device)
            m = evaluate(model, kg.valid, kg.all_true,
                        snapshots, kg.time2id, device,
                        batch_size=args.eval_batch)
            mrr = m["MRR"]
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.4f} | c={c_val:.4f} "
                  f"| Valid MRR={mrr:.4f} H@1={m['Hits@1']:.4f} "
                  f"H@10={m['Hits@10']:.4f}")
            if mrr > best_mrr:
                best_mrr = mrr
                no_improve = 0
                torch.save({'model_state': model.state_dict(),
                           'epoch': epoch, 'valid_mrr': best_mrr,
                           'args': args}, args.save_path)
                print(f"  [saved] -> {args.save_path}")
            else:
                no_improve += args.eval_every
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:04d} | Loss={avg_loss:.4f} | c={c_val:.4f}")

        if epoch % args.lr_decay_every == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.lr_decay

    # Final test
    print(f"\nBest valid MRR: {best_mrr:.4f}")
    ckpt  = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    m = evaluate(model, kg.test, kg.all_true,
                snapshots, kg.time2id, device,
                batch_size=args.eval_batch)
    print(f"Test | MRR={m['MRR']:.4f} | H@1={m['Hits@1']:.4f} "
          f"| H@3={m['Hits@3']:.4f} | H@10={m['Hits@10']:.4f}")

if __name__ == '__main__':
    main()

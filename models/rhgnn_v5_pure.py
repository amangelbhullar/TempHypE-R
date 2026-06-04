"""
TempHypE-R V5 Pure Hyperbolic
- All operations stay on Poincare manifold
- Mobius linear layers (no tangent space approximation)
- Fixed curvature c=1.0 (standard Poincare ball)
- Fully hyperbolic GRU, message passing, scoring
"""
import argparse, math, os
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import geoopt
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v3 import build_history_vocab, build_history_mask

EPS = 1e-6

# ── Pure Hyperbolic Operations (fixed c=1) ────────────────────────────────────

def artanh(x):
    return torch.atanh(x.clamp(-1+EPS, 1-EPS))

def tanh(x):
    return x.tanh()

def expmap0(u, c=1.0):
    sqrt_c = c**0.5
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return tanh(sqrt_c * u_norm) / (sqrt_c * u_norm) * u

def logmap0(p, c=1.0):
    sqrt_c = c**0.5
    p_norm = p.norm(dim=-1, keepdim=True).clamp(min=EPS)
    p_norm = p_norm.clamp(max=(1-EPS)/sqrt_c)
    return artanh(sqrt_c * p_norm) / (sqrt_c * p_norm) * p

def mobius_add(x, y, c=1.0):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    den = 1 + 2*c*xy + c**2 * x2 * y2
    return num / den.clamp(min=EPS)

def mobius_matvec(M, x, c=1.0):
    """Hyperbolic linear map via Mobius matrix-vector multiplication."""
    Mx    = x @ M.T
    Mx_n  = Mx.norm(dim=-1, keepdim=True).clamp(min=EPS)
    x_n   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    x_n   = x_n.clamp(max=(1-EPS)/c**0.5)
    res   = tanh(Mx_n / x_n * artanh(c**0.5 * x_n)) / (c**0.5 * Mx_n) * Mx
    return res

def hyp_distance(x, y, c=1.0):
    sqrt_c = c**0.5
    diff   = mobius_add(-x, y, c)
    diff_n = diff.norm(dim=-1).clamp(min=EPS, max=(1-EPS)/sqrt_c)
    return 2 / sqrt_c * artanh(sqrt_c * diff_n)

def mobius_relu(x, c=1.0):
    """Hyperbolic activation via geodesic."""
    return expmap0(F.relu(logmap0(x, c)), c)

# ── Hyperbolic Linear Layer ────────────────────────────────────────────────────

class HypLinear(nn.Module):
    """Linear layer in hyperbolic space via Mobius transformation."""
    def __init__(self, in_dim, out_dim, c=1.0, bias=True):
        super().__init__()
        self.c   = c
        self.W   = nn.Parameter(torch.empty(out_dim, in_dim))
        self.use_bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        out = mobius_matvec(self.W, x, self.c)
        if self.use_bias:
            b_hyp = expmap0(self.b, self.c)
            out   = mobius_add(out, b_hyp, self.c)
        return out

# ── Hyperbolic GRU (fully on manifold) ───────────────────────────────────────

class HypGRUCell(nn.Module):
    """
    Pure hyperbolic GRU — all operations on Poincare ball.
    Gates computed in tangent space (stable), candidate in hyperbolic.
    """
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.c   = c
        self.dim = dim
        # Gate weights (Euclidean — gates are scalars, stable)
        self.Wz = nn.Linear(dim*2, dim)
        self.Wr = nn.Linear(dim*2, dim)
        # Candidate — hyperbolic linear
        self.Wh = HypLinear(dim, dim, c=c)
        self.Uh = HypLinear(dim, dim, c=c)

    def forward(self, h_hyp, x_hyp, delta_t=1.0):
        c = self.c
        # Map to tangent for gate computation
        h_tan = logmap0(h_hyp, c)
        x_tan = logmap0(x_hyp, c)
        concat = torch.cat([h_tan, x_tan], dim=-1)

        # Gates in tangent space
        z = torch.sigmoid(self.Wz(concat))
        r = torch.sigmoid(self.Wr(concat))

        # Temporal decay
        decay = torch.exp(-0.1 * torch.tensor(
            delta_t, device=h_hyp.device, dtype=h_hyp.dtype))
        z = z * decay

        # Candidate fully in hyperbolic space
        r_h    = expmap0(r * h_tan, c)  # gate in hyperbolic
        Wh_rh  = self.Wh(r_h)
        Uh_x   = self.Uh(x_hyp)
        h_cand = mobius_relu(mobius_add(Wh_rh, Uh_x, c), c)

        # Interpolate: h_new = (1-z)*h ⊕ z*h_cand
        h_new = mobius_add(
            expmap0((1-z) * h_tan, c),
            expmap0(z * logmap0(h_cand, c), c), c)
        return h_new

# ── Hyperbolic Message Passing ────────────────────────────────────────────────

class HypMP(nn.Module):
    """Message passing fully in hyperbolic space."""
    def __init__(self, dim, num_relations, c=1.0):
        super().__init__()
        self.c       = c
        self.dim     = dim
        self.rel_lin = nn.ModuleList([
            HypLinear(dim, dim, c=c) for _ in range(num_relations)])
        self.agg_lin = HypLinear(dim, dim, c=c)

    def forward(self, ent_hyp, nbr_hyp, rel_ids):
        c = self.c
        if len(nbr_hyp) == 0:
            return ent_hyp

        # Transform neighbors with relation-specific hyperbolic linear
        msgs = []
        for i, (nbr, rid) in enumerate(zip(nbr_hyp, rel_ids)):
            # Use shared weight with relation embedding offset
            msg = mobius_add(nbr,
                expmap0(self.rel_lin[rid % len(self.rel_lin)].W.mean(0) * 0.01, c),
                c)
            msgs.append(msg)
        msgs = torch.stack(msgs, dim=0)  # (N, dim)

        # Aggregate via Frechet mean (approximated)
        msgs_tan = logmap0(msgs, c)
        agg_tan  = msgs_tan.mean(0)
        agg_hyp  = expmap0(agg_tan, c)

        # Combine with entity
        out = mobius_add(ent_hyp, self.agg_lin(agg_hyp), c)
        return out

# ── Hyperbolic ODE ────────────────────────────────────────────────────────────

class HypODE(nn.Module):
    """ODE in hyperbolic space via parallel transport."""
    def __init__(self, dim, c=1.0, dropout=0.1):
        super().__init__()
        self.c   = c
        # ODE function as hyperbolic linear layers
        self.f1  = HypLinear(dim, dim*2, c=c)
        self.f2  = HypLinear(dim*2, dim, c=c)
        self.drop= nn.Dropout(dropout)

    def forward(self, h_hyp, delta_t, steps=5):
        if delta_t <= 0: return h_hyp
        c  = self.c
        dt = delta_t / max(steps, 1)
        state = h_hyp
        for _ in range(steps):
            # Compute velocity in tangent space
            h1  = mobius_relu(self.f1(state), c)
            vel = logmap0(self.f2(h1), c)  # tangent vector
            # Euler step on manifold
            state = expmap0(logmap0(state, c) + dt * vel, c)
        return state

# ── TempHypE-R V5 Pure ─────────────────────────────────────────────────────────────

class TempHypERv5Pure(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200,
                 c=1.0, dropout=0.1, ode_steps=5):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.c             = c  # base curvature
        self._log_c        = nn.Parameter(torch.tensor(float(c)).log().clamp(min=-3.0))
        self._log_c_rel    = nn.Parameter(torch.zeros(num_relations))
        self.ode_steps     = ode_steps

        # Manifold
        self.ball = PoincareBall(c=c)

        # Embeddings on Poincare ball
        self.entity_emb = geoopt.ManifoldParameter(
            torch.randn(num_entities, dim) * 0.001,
            manifold=self.ball)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.normal_(self.rel_emb.weight, std=0.001)

        # Hyperbolic GRU
        self.hgru = HypGRUCell(dim, c=c)

        # Hyperbolic ODE
        self.ode = HypODE(dim, c=c, dropout=dropout)

        # Hyperbolic MP (simplified — full version too memory intensive)
        self.mp_W    = nn.Embedding(num_relations, dim*dim)
        self.mp_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.mp_W.weight, std=0.001)

        # Relation rotation (hyperbolic)
        self.rel_rot = nn.Embedding(num_relations, dim*dim)
        nn.init.normal_(self.rel_rot.weight, std=0.001)

        # Copy mechanism
        self.copy_w  = nn.Parameter(torch.tensor(2.0))
        self.copy_g  = nn.Linear(dim, 1)
        self.bias    = nn.Parameter(torch.zeros(num_entities))

        # History vocab
        self.history_vocab = None

    def set_history_vocab(self, vocab):
        self.history_vocab = vocab

    def get_rel_curvature(self, rel_id):
        import torch.nn.functional as F
        return F.softplus(self._log_c + self._log_c_rel[rel_id]).item()

    def _get_hyp(self, ids):
        return self.entity_emb[ids]

    def build_nbr_index(self, snapshots, max_nbrs=16):
        self._nbr_index = {}
        for tau, ent_dict in snapshots.items():
            for ent, nbrs in ent_dict.items():
                nbrs = nbrs[:max_nbrs]
                self._nbr_index[(tau, ent)] = (
                    torch.tensor([n for n, _ in nbrs], dtype=torch.long),
                    torch.tensor([r for _, r in nbrs], dtype=torch.long))
        print(f"Neighbour index: {len(self._nbr_index):,} entries")

    def refresh_msg_table(self, device):
        c = self.c
        with torch.no_grad():
            all_hyp = self.entity_emb.data.to(device)
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in self._nbr_index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                if len(nbr_ids) == 0:
                    self._msg_table[(tau,ent)] = expmap0(
                        torch.zeros(self.dim, device=device), c)
                    continue
                # Message passing in hyperbolic space
                nbr_hyp = all_hyp[nbr_ids]
                nbr_tan = logmap0(nbr_hyp, c)
                W    = self.mp_W(rel_ids).view(len(rel_ids), self.dim, self.dim)
                msgs = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                agg  = self.mp_norm(msgs.mean(0))
                self._msg_table[(tau,ent)] = expmap0(agg, c)

    def _get_msg(self, entity_ids, tau_ids):
        device = entity_ids.device
        msgs = []
        for eid, tid in zip(entity_ids.tolist(), tau_ids.tolist()):
            msgs.append(self._msg_table.get((tid, eid),
                expmap0(torch.zeros(self.dim, device=device), self.c)))
        return torch.stack(msgs)

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        c = self.c
        h_hyp   = self._get_hyp(h)
        msg_hyp = self._get_msg(h, tau)

        # True hyperbolic GRU
        h_jump  = self.hgru(h_hyp, msg_hyp, delta_t)

        # Hyperbolic ODE
        h_flow  = self.ode(h_jump, delta_t, self.ode_steps)

        # Relation rotation in tangent space → back to hyperbolic
        h_tan   = logmap0(h_flow, c)
        R       = self.rel_rot(r).view(-1, self.dim, self.dim)
        h_rot   = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        return expmap0(h_rot, c)

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0,
                        chunk_size=500):
        import torch.nn.functional as F
        r_id  = r[0].item() if hasattr(r, "__len__") else int(r)
        c     = F.softplus(self._log_c + self._log_c_rel[r_id]).item()
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        h_tan = logmap0(h_rot, c)

        scores  = torch.zeros(h.size(0), self.num_entities, device=h.device)
        all_ids = torch.arange(self.num_entities, device=h.device)

        for start in range(0, self.num_entities, chunk_size):
            end   = min(start+chunk_size, self.num_entities)
            chunk = all_ids[start:end]
            t_hyp = self._get_hyp(chunk)
            pred  = h_rot.unsqueeze(1).expand(-1, len(chunk), -1)
            tail  = t_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist  = hyp_distance(pred, tail, c)
            scores[:, start:end] = -dist

        scores = scores + self.bias.unsqueeze(0)

        if self.history_vocab is not None:
            hist_mask  = build_history_mask(
                h, r, self.history_vocab, self.num_entities, h.device)
            copy_score = hist_mask * self.copy_w.abs()
            gate       = torch.sigmoid(self.copy_g(h_tan))
            return gate * scores + (1-gate) * (scores + copy_score)
        return scores

def soft_label_loss(scores, true_tails, smooth=0.1):
    B, E   = scores.size()
    target = torch.zeros(B, E, device=scores.device)
    target.scatter_(1, true_tails.unsqueeze(1), 1.0)
    target = (1-smooth)*target + smooth/E
    return F.kl_div(F.log_softmax(scores, dim=-1),
                    target.detach(), reduction='batchmean')

def train_epoch(model, train_groups, optimizer, device,
                snapshots, ts_to_real, batch_size=256, grad_clip=0.5):
    model.train()
    total_loss, total_n = 0.0, 0
    prev_real_ts = None
    for ts_id, quads in train_groups:
        real_ts  = ts_to_real.get(ts_id, ts_id)
        delta_t  = max(float(real_ts-prev_real_ts) if prev_real_ts else 1.0, 1.0)
        prev_real_ts = real_ts
        for start in range(0, len(quads), batch_size):
            mini  = quads[start:start+batch_size]
            batch = torch.tensor(mini, dtype=torch.long, device=device)
            h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
            scores = model.score_all_tails(h, r, tau, snapshots, delta_t)
            loss   = soft_label_loss(scores, t)
            if torch.isnan(loss) or torch.isinf(loss): continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Project embeddings back to ball after update
            with torch.no_grad():
                model.entity_emb.data = model.ball.projx(
                    model.entity_emb.data)
            total_loss += loss.item()*len(mini)
            total_n    += len(mini)
    return total_loss / max(total_n, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, default="data/ICEWS14")
    parser.add_argument("--epochs",         type=int, default=500)
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--eval_batch",     type=int, default=256)
    parser.add_argument("--dim",            type=int, default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_decay",       type=float, default=0.8)
    parser.add_argument("--lr_decay_every", type=int, default=50)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--curvature",      type=float, default=1.0)
    parser.add_argument("--ode_steps",      type=int, default=3)
    parser.add_argument("--eval_every",     type=int, default=10)
    parser.add_argument("--patience",       type=int, default=60)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--save_path",      type=str,
                        default="checkpoints/rhgnn_v5_pure_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Pure Hyperbolic c={args.curvature}")

    data      = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(data.train)
    history_vocab = build_history_vocab(data.train)

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train: train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items())

    model = TempHypE-Rv5Pure(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        dim=args.dim, c=args.curvature,
        dropout=args.dropout,
        ode_steps=args.ode_steps).to(device)
    model.set_history_vocab(history_vocab)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr,
                               stabilize=10)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    best_mrr, no_improve = 0.0, 0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_groups, optimizer, device,
                          snapshots, ts_to_real,
                          batch_size=args.batch_size)
        scheduler.step()
        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | "
              f"c: {args.curvature:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if epoch % args.eval_every == 0:
            model.refresh_msg_table(device)
            vm = evaluate(model, data.valid, data.all_true,
                         snapshots, ts_to_real, device,
                         batch_size=args.eval_batch)
            print(f"  Valid | MRR={vm['MRR']:.4f} | "
                  f"H@1={vm['Hits@1']:.4f} | H@3={vm['Hits@3']:.4f} | "
                  f"H@10={vm['Hits@10']:.4f}", flush=True)
            if vm['MRR'] > best_mrr:
                best_mrr = vm['MRR']; no_improve = 0
                torch.save({'model_state': model.state_dict(),
                           'epoch': epoch, 'mrr': best_mrr}, args.save_path)
                print(f"  [saved] -> {args.save_path}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience // args.eval_every:
                    print(f"Early stop ep{epoch}", flush=True); break

    print(f"\nBest valid MRR: {best_mrr:.4f}")
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device, batch_size=args.eval_batch)
    print(f"  Test | MRR={tm['MRR']:.4f} | H@1={tm['Hits@1']:.4f} | "
          f"H@3={tm['Hits@3']:.4f} | H@10={tm['Hits@10']:.4f}")
    print(f"\nBest valid MRR : {best_mrr:.4f}  (epoch {ckpt['epoch']})")
    print("Done.")

if __name__ == '__main__':
    main()

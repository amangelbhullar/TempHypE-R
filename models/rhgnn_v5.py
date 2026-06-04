"""
TempHypE-R V5 — True Hyperbolic using geoopt PoincareBall
Numerically stable hyperbolic operations via geoopt manifold
"""
import argparse, math, os
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import geoopt
from geoopt.optim import RiemannianAdam
import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v3 import build_history_vocab, build_history_mask

EPS = 1e-6

class TempHypERv5(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps

        # geoopt PoincareBall manifold — handles all numerical stability
        self.ball = geoopt.PoincareBall(c=init_curvature, learnable=True)

        # Embeddings on manifold
        self.entity_emb   = geoopt.ManifoldParameter(
            torch.zeros(num_entities, dim), manifold=self.ball)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # GRU gates (in tangent space — stable)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim)
        self.Uh = nn.Linear(dim, dim, bias=False)

        # ODE
        self.ode_func = nn.Sequential(
            nn.Linear(dim, dim*2), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim*2, dim))

        # Message passing
        self.rel_W   = nn.Embedding(num_relations, dim * dim)
        self.mp_gate = nn.Linear(dim * 2, dim)
        self.mp_norm = nn.LayerNorm(dim)

        # Scoring
        self.rel_rot     = nn.Embedding(num_relations, dim * dim)
        self.copy_weight = nn.Parameter(torch.tensor(2.0))
        self.copy_gate   = nn.Linear(dim, 1)
        self.bias        = nn.Parameter(torch.zeros(num_entities))
        self.dropout     = nn.Dropout(dropout)
        self.history_vocab = None

        # Init embeddings near origin
        nn.init.normal_(self.entity_emb, std=1e-3)
        nn.init.normal_(self.relation_emb.weight, std=0.01)
        nn.init.normal_(self.rel_W.weight, std=0.01)
        nn.init.normal_(self.rel_rot.weight, std=0.01)

    def set_history_vocab(self, vocab):
        self.history_vocab = vocab

    def _to_tangent(self, hyp_emb):
        """Map hyperbolic embeddings to tangent space at origin."""
        return self.ball.logmap0(hyp_emb)

    def _to_hyp(self, tan_emb):
        """Map tangent space to hyperbolic."""
        return self.ball.expmap0(tan_emb)

    def _get_entity_hyp(self, ids):
        """Get entity embeddings (already on manifold)."""
        return self.entity_emb[ids]

    def build_nbr_index(self, snapshots, max_nbrs=32):
        self._nbr_index = {}
        for tau, ent_dict in snapshots.items():
            for ent, nbrs in ent_dict.items():
                nbrs = nbrs[:max_nbrs]
                self._nbr_index[(tau, ent)] = (
                    torch.tensor([n for n, _ in nbrs], dtype=torch.long),
                    torch.tensor([r for _, r in nbrs], dtype=torch.long))
        print(f"Neighbour index: {len(self._nbr_index):,} entries")

    def refresh_msg_table(self, device):
        with torch.no_grad():
            # Get all embeddings in tangent space
            all_tan = self._to_tangent(self.entity_emb.to(device))
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in self._nbr_index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                if len(nbr_ids) == 0:
                    self._msg_table[(tau,ent)] = torch.zeros(self.dim,device=device)
                    continue
                nbr_tan = all_tan[nbr_ids]
                h_self  = all_tan[ent].unsqueeze(0).expand(len(nbr_ids),-1)
                W    = self.rel_W(rel_ids).view(len(nbr_ids), self.dim, self.dim)
                msgs = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                attn = torch.sigmoid(self.mp_gate(torch.cat([msgs, h_self], dim=-1)))
                agg  = (attn * msgs).mean(dim=0)
                self._msg_table[(tau,ent)] = self.mp_norm(torch.tanh(agg))

    def _get_msg(self, entity_ids, tau_ids):
        device = entity_ids.device
        msgs = []
        for eid, tid in zip(entity_ids.tolist(), tau_ids.tolist()):
            key = (tid, eid)
            msgs.append(self._msg_table.get(key,
                torch.zeros(self.dim, device=device)))
        return torch.stack(msgs, dim=0)

    def _hgru(self, h_hyp, msg_tan, delta_t=1.0):
        """H-GRU in tangent space — numerically stable."""
        h_tan  = self._to_tangent(h_hyp)
        r      = torch.sigmoid(self.Wr(h_tan) + self.Ur(msg_tan))
        z      = torch.sigmoid(self.Wz(h_tan) + self.Uz(msg_tan))
        h_cand = torch.tanh(self.Wh(h_tan) + self.Uh(r * msg_tan))
        decay  = torch.exp(-0.1 * torch.tensor(
            delta_t, device=h_tan.device, dtype=h_tan.dtype))
        v_new  = (z * decay) * h_cand + (1.0 - z * decay) * h_tan
        return self._to_hyp(v_new)

    def _ode_flow(self, h_hyp, delta_t):
        if delta_t <= 0.0: return h_hyp
        state = self._to_tangent(h_hyp)
        dt    = delta_t / max(self.ode_steps, 1)
        for _ in range(self.ode_steps):
            state = state + dt * self.ode_func(state)
        return self._to_hyp(state)

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        h_hyp   = self._get_entity_hyp(h)
        msg_tan = self._get_msg(h, tau)
        h_jump  = self._hgru(h_hyp, msg_tan, delta_t)
        h_flow  = self._ode_flow(h_jump, delta_t)
        # Relation rotation in tangent space
        h_tan   = self._to_tangent(h_flow)
        R       = self.rel_rot(r).view(-1, self.dim, self.dim)
        h_rot   = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        return self._to_hyp(h_rot)

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        t_hyp = self._get_entity_hyp(t)
        dist  = self.ball.dist(h_rot, t_hyp)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0, chunk_size=1000):
        h_rot   = self.forward_entity(h, r, tau, snapshots, delta_t)
        h_tan   = self._to_tangent(h_rot)
        all_ids = torch.arange(self.num_entities, device=h.device)
        scores  = torch.zeros(h.size(0), self.num_entities, device=h.device)

        for start in range(0, self.num_entities, chunk_size):
            end   = min(start + chunk_size, self.num_entities)
            chunk = all_ids[start:end]
            t_hyp = self._get_entity_hyp(chunk)
            pred  = h_rot.unsqueeze(1).expand(-1, len(chunk), -1)
            tail  = t_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist  = self.ball.dist(pred, tail)
            scores[:, start:end] = -dist

        scores = scores + self.bias.unsqueeze(0)

        if self.history_vocab is not None:
            hist_mask  = build_history_mask(
                h, r, self.history_vocab, self.num_entities, h.device)
            copy_score = hist_mask * self.copy_weight.abs()
            gate       = torch.sigmoid(self.copy_gate(h_tan))
            return gate * scores + (1.0 - gate) * (scores + copy_score)

        return scores

def soft_label_loss(scores, true_tails, smooth=0.1):
    B, E  = scores.size()
    target = torch.zeros(B, E, device=scores.device)
    target.scatter_(1, true_tails.unsqueeze(1), 1.0)
    target = (1 - smooth) * target + smooth / E
    return F.kl_div(F.log_softmax(scores, dim=-1),
                    target.detach(), reduction='batchmean')

def train_epoch(model, train_groups, optimizer, device,
                num_entities, snapshots, ts_to_real,
                batch_size=512, grad_clip=1.0):
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
            if torch.isnan(loss): continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(mini)
            total_n    += len(mini)
    return total_loss / max(total_n, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, default="data/ICEWS14")
    parser.add_argument("--epochs",         type=int, default=500)
    parser.add_argument("--batch_size",     type=int, default=512)
    parser.add_argument("--eval_batch",     type=int, default=512)
    parser.add_argument("--dim",            type=int, default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_decay",       type=float, default=0.8)
    parser.add_argument("--lr_decay_every", type=int, default=50)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--curvature",      type=float, default=1.0)
    parser.add_argument("--ode_steps",      type=int, default=5)
    parser.add_argument("--eval_every",     type=int, default=10)
    parser.add_argument("--patience",       type=int, default=60)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--save_path",      type=str,
                        default="checkpoints/rhgnn_v5_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    model = TempHypE-Rv5(
        num_entities=data.num_entities, num_relations=data.num_relations,
        dim=args.dim, init_curvature=args.curvature,
        dropout=args.dropout, ode_steps=args.ode_steps).to(device)
    model.set_history_vocab(history_vocab)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    best_mrr, no_improve = 0.0, 0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_groups, optimizer, device,
                          data.num_entities, snapshots, ts_to_real,
                          batch_size=args.batch_size)
        scheduler.step()
        c_val = model.ball.c.item() if hasattr(model.ball, 'c') else args.curvature
        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | c: {c_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if epoch % args.eval_every == 0:
            model.refresh_msg_table(device)
            vm = evaluate(model, data.valid, data.all_true,
                         snapshots, ts_to_real, device, batch_size=args.eval_batch)
            print(f"  Valid | MRR={vm['MRR']:.4f} | H@1={vm['Hits@1']:.4f} | "
                  f"H@3={vm['Hits@3']:.4f} | H@10={vm['Hits@10']:.4f}", flush=True)
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
    model.build_nbr_index(snapshots); model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device, batch_size=args.eval_batch)
    print(f"  Test | MRR={tm['MRR']:.4f} | H@1={tm['Hits@1']:.4f} | "
          f"H@3={tm['Hits@3']:.4f} | H@10={tm['Hits@10']:.4f}")
    print(f"\nBest valid MRR : {best_mrr:.4f}  (epoch {ckpt['epoch']})")
    print("Done.")

if __name__ == '__main__':
    main()

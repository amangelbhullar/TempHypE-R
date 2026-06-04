"""
TempHypE-R V1b — H-GRU + Neural ODE + Hyperbolic (no Freq Emb, no Adv Train)
Intermediate between V1 (H-LSTM) and V2 (H-GRU + Freq + Adv)
Shows contribution of H-GRU alone
"""
import argparse, math, os
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from geoopt.optim import RiemannianAdam
import sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v2 import expmap0, logmap0, hyp_distance, ODEFunc, adversarial_loss

EPS = 1e-6

class TempHypERv1b(nn.Module):
    """
    TempHypE-R with H-GRU (no frequency embeddings, no adversarial training)
    Clean ablation: V1 + H-GRU only
    """
    def __init__(self, num_entities, num_relations, dim=200,
                 init_curvature=1.0, dropout=0.1, ode_steps=5):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps

        # Learnable curvature
        self._log_c = nn.Parameter(
            torch.tensor(math.log(math.exp(init_curvature) - 1.0)))

        # Simple entity + relation embeddings (no freq)
        self.entity_emb   = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Relation-aware message passing
        self.rel_W   = nn.Embedding(num_relations, dim * dim)
        self.mp_gate = nn.Linear(dim * 2, dim)
        self.mp_norm = nn.LayerNorm(dim)

        # H-GRU (replaces H-LSTM from V1)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim)
        self.Uh = nn.Linear(dim, dim, bias=False)

        # Neural ODE
        self.ode_func = ODEFunc(dim, dropout)

        # Relation rotation + scoring
        self.rel_rot = nn.Embedding(num_relations, dim * dim)
        self.bias    = nn.Parameter(torch.zeros(num_entities))
        self.dropout = nn.Dropout(dropout)

        # Init
        for emb in [self.entity_emb, self.relation_emb,
                    self.rel_W, self.rel_rot]:
            nn.init.normal_(emb.weight, std=0.01)

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(5e-2, 5.0)

    def _to_hyp(self, ids):
        return expmap0(self.entity_emb(ids), self.c)

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
        c = self.c.detach()
        with torch.no_grad():
            all_tan = logmap0(
                expmap0(self.entity_emb.weight.to(device), c), c)
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in self._nbr_index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                if len(nbr_ids) == 0:
                    self._msg_table[(tau,ent)] = torch.zeros(
                        self.dim, device=device)
                    continue
                nbr_tan = all_tan[nbr_ids]
                h_self  = all_tan[ent].unsqueeze(0).expand(len(nbr_ids), -1)
                W    = self.rel_W(rel_ids).view(
                    len(nbr_ids), self.dim, self.dim)
                msgs = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                attn = torch.sigmoid(
                    self.mp_gate(torch.cat([msgs, h_self], dim=-1)))
                agg  = (attn * msgs).mean(dim=0)
                self._msg_table[(tau,ent)] = self.mp_norm(torch.tanh(agg))

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        c     = self.c
        h_hyp = self._to_hyp(entity_ids)
        h_tan = logmap0(h_hyp, c)
        r_mean= self.relation_emb.weight.mean(dim=0)
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
        if delta_t <= 0.0: return h_hyp
        c     = self.c
        state = logmap0(h_hyp, c)
        dt    = delta_t / max(self.ode_steps, 1)
        for _ in range(self.ode_steps):
            state = state + dt * self.ode_func(state)
        return expmap0(state, c)

    def forward_entity(self, h, r, tau, snapshots, delta_t=1.0):
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._hgru_jump(h_hyp, msg_tan, delta_t)
        h_flow  = self._ode_flow(h_jump, delta_t)
        h_tan   = logmap0(h_flow, self.c)
        R       = self.rel_rot(r).view(-1, self.dim, self.dim)
        h_rot   = torch.bmm(R, h_tan.unsqueeze(-1)).squeeze(-1)
        return expmap0(h_rot, self.c)

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        dist  = hyp_distance(h_rot, self._to_hyp(t), self.c)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0,
                        chunk_size=1000):
        c     = self.c
        h_rot = self.forward_entity(h, r, tau, snapshots, delta_t)
        all_ids   = torch.arange(self.num_entities, device=h.device)
        gen_score = torch.zeros(h.size(0), self.num_entities, device=h.device)
        for start in range(0, self.num_entities, chunk_size):
            end   = min(start+chunk_size, self.num_entities)
            chunk = all_ids[start:end]
            t_hyp = self._to_hyp(chunk)
            pred  = h_rot.unsqueeze(1).expand(-1, len(chunk), -1)
            tail  = t_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
            dist  = hyp_distance(pred, tail, c)
            gen_score[:, start:end] = -dist
        return gen_score + self.bias.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, default="data/ICEWS14")
    parser.add_argument("--epochs",         type=int, default=500)
    parser.add_argument("--batch_size",     type=int, default=1024)
    parser.add_argument("--eval_batch",     type=int, default=512)
    parser.add_argument("--dim",            type=int, default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_decay",       type=float, default=0.8)
    parser.add_argument("--lr_decay_every", type=int, default=50)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--curvature",      type=float, default=1.0)
    parser.add_argument("--ode_steps",      type=int, default=5)
    parser.add_argument("--neg_ratio",      type=int, default=10)
    parser.add_argument("--eval_every",     type=int, default=10)
    parser.add_argument("--patience",       type=int, default=60)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--save_path",      type=str,
                        default="checkpoints/rhgnn_v1b_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | TempHypE-R-V1b (H-GRU only, no Freq, no Adv)")

    data      = load_temporal_kg(args.data_dir)
    snapshots = build_snapshot_graphs(data.train)
    print(f"Entities: {data.num_entities} | Relations: {data.num_relations}")
    print(f"Train: {len(data.train)} | Valid: {len(data.valid)} | "
          f"Test: {len(data.test)}")

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train: train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items())

    model = TempHypE-Rv1b(
        num_entities  = data.num_entities,
        num_relations = data.num_relations,
        dim           = args.dim,
        init_curvature= args.curvature,
        dropout       = args.dropout,
        ode_steps     = args.ode_steps).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    best_mrr, no_improve = 0.0, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_n = 0.0, 0
        prev_real_ts = None
        for ts_id, quads in train_groups:
            real_ts  = ts_to_real.get(ts_id, ts_id)
            delta_t  = max(float(real_ts-prev_real_ts)
                          if prev_real_ts else 1.0, 1.0)
            prev_real_ts = real_ts
            for start in range(0, len(quads), args.batch_size):
                mini  = quads[start:start+args.batch_size]
                batch = torch.tensor(mini, dtype=torch.long, device=device)
                h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
                pos = model.score(h,r,t,tau,snapshots,delta_t)
                negs= torch.stack([model.score(h,r,
                         torch.randint(0,data.num_entities,(h.size(0),),
                                      device=device),
                         tau,snapshots,delta_t)
                         for _ in range(args.neg_ratio)], dim=-1)
                loss= adversarial_loss(pos, negs)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()*len(mini)
                total_n    += len(mini)
        scheduler.step()
        print(f"Epoch {epoch:04d} | Loss: {total_loss/total_n:.4f} | "
              f"c: {model.c.item():.4f} | "
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

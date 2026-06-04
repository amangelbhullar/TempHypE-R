"""
V4 WIKI Ablation — Parallel version
Each variant runs on specified GPU
Usage: python run_v4_ablation_wiki_parallel.py --variant 0 --gpu 0
"""
import torch, sys, math, argparse
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v2 import adversarial_loss, expmap0, logmap0
from rhgnn_v3 import build_history_vocab
from rhgnn_v4 import TempHypE-Rv4, soft_label_loss
from geoopt.optim import RiemannianAdam

EPOCHS     = 50
PATIENCE   = 20
EVAL_EVERY = 10
DATA_DIR   = 'data/WIKI-clean'

VARIANTS = [
    ('Full TempHypE-R-C',      {}, {}),
    ('w/o Contrastive',   {}, {'no_contrast': True}),
    ('w/o Soft Labels',   {}, {'no_soft': True}),
    ('w/o Temp Smooth',   {'no_smooth': True}, {'no_smooth': True}),
    ('w/o History Vocab', {'no_history': True}, {}),
    ('w/o Subgraph',      {'no_sgcn': True}, {}),
    ('w/o H-GRU',         {'no_hgru': True}, {}),
    ('w/o ODE',           {'no_ode': True}, {}),
    ('w/o Hyperbolic',    {'no_hyperbolic': True}, {}),
]

def make_model(data, device, no_history=False, no_hyperbolic=False,
               no_ode=False, no_hgru=False, no_sgcn=False, no_smooth=False):
    model = TempHypERCA(
        num_entities=data.num_entities, num_relations=data.num_relations,
        dim=200, dropout=0.1, ode_steps=5 if not no_ode else 0,
        num_sgcn_layers=0 if no_sgcn else 2,
        smooth_label=0.1, lambda_smooth=0.0 if no_smooth else 0.01,
    ).to(device)
    model.set_history_vocab(
        build_history_vocab(data.train) if not no_history else {})
    if no_hyperbolic:
        def _to_hyp_eucl(ids): return model.entity_emb(ids)
        def _score_all_eucl(h, r, tau, snapshots, delta_t=1.0):
            msg  = model._aggregate_messages(h, tau, snapshots)
            v    = model.entity_emb(h)
            rz   = torch.sigmoid(model.Wr(v) + model.Ur(msg))
            z    = torch.sigmoid(model.Wz(v) + model.Uz(msg))
            hc   = torch.tanh(model.Wh(v) + model.Uh(rz * msg))
            h_j  = z * hc + (1-z) * v
            pred = h_j + model.relation_emb(r)
            dist = (pred.unsqueeze(1) - model.entity_emb.weight.unsqueeze(0)).norm(dim=-1)
            return -dist + model.bias.unsqueeze(0)
        model._to_hyp = _to_hyp_eucl
        model.score_all_tails = _score_all_eucl
    if no_hgru:
        model._hgru_jump = lambda h_prev, msg, delta_t=1.0: h_prev
    if no_ode:
        model._ode_flow = lambda h, dt: h
    return model

def train_eval(model, train_groups, data, snapshots, ts_to_real, device,
               no_contrast=False, no_soft=False, no_smooth=False):
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    best_mrr, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, total_n = 0.0, 0
        prev_real_ts = None
        for ts_id, quads in train_groups:
            real_ts  = ts_to_real.get(ts_id, ts_id)
            delta_t  = max(float(real_ts - prev_real_ts) if prev_real_ts else 1.0, 1.0)
            prev_real_ts = real_ts
            for start in range(0, len(quads), 64):
                mini  = quads[start:start+64]
                batch = torch.tensor(mini, dtype=torch.long, device=device)
                h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
                if no_soft:
                    pos  = model.score(h,r,t,tau,snapshots,delta_t)
                    negs = torch.stack([model.score(h,r,
                               torch.randint(0,data.num_entities,(h.size(0),),device=device),
                               tau,snapshots,delta_t) for _ in range(10)], dim=-1)
                    main_loss = adversarial_loss(pos, negs)
                else:
                    scores    = model.score_all_tails(h,r,tau,snapshots,delta_t)
                    main_loss = soft_label_loss(scores, t, smooth=0.1)
                loss = main_loss
                if not no_contrast:
                    loss = loss + 0.1 * model.compute_contrastive(h,r,t,tau,snapshots,delta_t)
                if not no_smooth:
                    loss = loss + model.temporal_smooth_loss()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(mini)
                total_n    += len(mini)
        scheduler.step()
        model.update_prev_emb()

        if epoch % EVAL_EVERY == 0:
            model.refresh_msg_table(device)
            vm  = evaluate(model, data.valid, data.all_true,
                          snapshots, ts_to_real, device, batch_size=256)
            mrr = vm['MRR']
            print(f"  ep{epoch:04d} loss={total_loss/total_n:.4f} valid={mrr:.4f}", flush=True)
            if mrr > best_mrr:
                best_mrr   = mrr
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // EVAL_EVERY:
                    print(f"  Early stop ep{epoch}", flush=True)
                    break

    model.load_state_dict(best_state)
    model.to(device)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device, batch_size=256)
    return best_mrr, tm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=int, required=True,
                        help='Variant index 0-8')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:0')  # CUDA_VISIBLE_DEVICES handles mapping
    set_seed(42)

    name, model_kw, train_kw = VARIANTS[args.variant]
    print(f"Running variant {args.variant}: {name}", flush=True)

    data      = load_temporal_kg(DATA_DIR)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts
    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items())

    model = make_model(data, device, **model_kw)
    best_valid, tm = train_eval(model, train_groups, data, snapshots,
                                ts_to_real, device, **train_kw)

    print(f"\nRESULT | {name} | MRR={tm['MRR']:.4f} | "
          f"H@1={tm['Hits@1']:.4f} | H@3={tm['Hits@3']:.4f} | "
          f"H@10={tm['Hits@10']:.4f} | MAR={tm['MAR']:.1f}")

if __name__ == '__main__':
    main()

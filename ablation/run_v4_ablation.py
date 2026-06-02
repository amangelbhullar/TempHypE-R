"""
RHGNN V4 Ablation Study on ICEWS14
Shows contribution of each component added in V4:
Full V4 → remove one component at a time
"""
import torch, sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import (
    set_seed, load_temporal_kg, build_snapshot_graphs, evaluate)
from rhgnn_v2 import adversarial_loss
from rhgnn_v3 import build_history_vocab, build_history_mask
from rhgnn_v4 import RHGNNv4, soft_label_loss
from geoopt.optim import RiemannianAdam

DEVICE   = torch.device('cuda:0')
DATA_DIR = 'data/ICEWS14'
EPOCHS   = 200
PATIENCE = 40
EVAL_EVERY = 20

def train_eval(model, train_groups, data, snapshots, ts_to_real,
               optimizer, scheduler,
               no_contrast=False, no_soft=False, no_smooth=False):

    model.build_nbr_index(snapshots)
    model.refresh_msg_table(DEVICE)

    best_mrr, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, total_n = 0.0, 0
        prev_real_ts = None

        for ts_id, quads in train_groups:
            real_ts  = ts_to_real.get(ts_id, ts_id)
            delta_t  = float(real_ts - prev_real_ts) \
                       if prev_real_ts is not None else 1.0
            delta_t  = max(delta_t, 1.0)
            prev_real_ts = real_ts

            for start in range(0, len(quads), 1024):
                mini  = quads[start:start+1024]
                batch = torch.tensor(mini, dtype=torch.long, device=DEVICE)
                h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]

                # Main loss
                if no_soft:
                    pos  = model.score(h,r,t,tau,snapshots,delta_t)
                    negs = torch.stack([
                        model.score(h, r,
                            torch.randint(0, data.num_entities,
                                         (h.size(0),), device=DEVICE),
                            tau, snapshots, delta_t)
                        for _ in range(10)], dim=-1)
                    main_loss = adversarial_loss(pos, negs)
                else:
                    scores    = model.score_all_tails(
                        h, r, tau, snapshots, delta_t)
                    main_loss = soft_label_loss(scores, t, smooth=0.1)

                # Contrastive loss
                if no_contrast:
                    loss = main_loss
                else:
                    closs = model.compute_contrastive(
                        h, r, t, tau, snapshots, delta_t)
                    loss  = main_loss + 0.1 * closs

                # Smoothness
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
            model.refresh_msg_table(DEVICE)
            vm  = evaluate(model, data.valid, data.all_true,
                          snapshots, ts_to_real, DEVICE, batch_size=512)
            mrr = vm['MRR']
            print(f"  ep{epoch:04d} loss={total_loss/total_n:.4f} "
                  f"valid={mrr:.4f}", flush=True)
            if mrr > best_mrr:
                best_mrr   = mrr
                best_state = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // EVAL_EVERY:
                    print(f"  Early stop ep{epoch}", flush=True)
                    break

    # Test with best checkpoint
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(DEVICE)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, DEVICE, batch_size=512)
    return best_mrr, tm

def make_model(data, no_history=False, no_hyperbolic=False,
               no_ode=False, no_hgru=False, no_sgcn=False,
               no_smooth=False):
    model = RHGNNv4(
        num_entities    = data.num_entities,
        num_relations   = data.num_relations,
        dim             = 200,
        dropout         = 0.1,
        ode_steps       = 5 if not no_ode else 0,
        num_sgcn_layers = 0 if no_sgcn else 2,
        smooth_label    = 0.1,
        lambda_smooth   = 0.0 if no_smooth else 0.01,
    ).to(DEVICE)

    # Set history vocab
    if not no_history:
        model.set_history_vocab(build_history_vocab(data.train))
    else:
        model.set_history_vocab({})

    # Patch hyperbolic → Euclidean
    if no_hyperbolic:
        from rhgnn_v2 import expmap0, logmap0
        import torch.nn.functional as F

        def _to_hyp_eucl(ids):
            return model.entity_emb(ids)
        def _score_all_eucl(h, r, tau, snapshots, delta_t=1.0):
            msg = model._aggregate_messages(h, tau, snapshots)
            h_e = model.entity_emb(h)
            v   = logmap0(expmap0(h_e, model.c), model.c)
            rz  = torch.sigmoid(model.Wr(v) + model.Ur(msg))
            z   = torch.sigmoid(model.Wz(v) + model.Uz(msg))
            hc  = torch.tanh(model.Wh(v) + model.Uh(rz * msg))
            h_j = z * hc + (1-z) * v
            r_e = model.relation_emb(r)
            pred = h_j + r_e
            all_e = model.entity_emb.weight
            dist  = (pred.unsqueeze(1) - all_e.unsqueeze(0)).norm(dim=-1)
            return -dist + model.bias.unsqueeze(0)

        model._to_hyp = _to_hyp_eucl
        model.score_all_tails = _score_all_eucl

    # Patch H-GRU → identity
    if no_hgru:
        def _hgru_identity(h_prev_hyp, msg_tan, delta_t=1.0):
            return h_prev_hyp
        model._hgru_jump = _hgru_identity

    # Patch ODE → identity
    if no_ode:
        def _ode_identity(h_hyp, delta_t):
            return h_hyp
        model._ode_flow = _ode_identity

    return model

def run_variant(name, data, train_groups, snapshots, ts_to_real,
                model_kwargs={}, train_kwargs={}, seed=42):
    set_seed(seed)
    print(f"\n{'='*58}\nVariant: {name}", flush=True)

    model     = make_model(data, **model_kwargs)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8)

    best_valid, tm = train_eval(
        model, train_groups, data, snapshots, ts_to_real,
        optimizer, scheduler, **train_kwargs)
    return best_valid, tm

def main():
    print("Loading ICEWS14...", flush=True)
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

    # Define variants
    variants = [
        # name, model_kwargs, train_kwargs
        ('Full RHGNN-C',
         {}, {}),

        ('w/o Contrastive',
         {}, {'no_contrast': True}),

        ('w/o Soft Labels',
         {}, {'no_soft': True}),

        ('w/o Temporal Smooth',
         {'no_smooth': True}, {'no_smooth': True}),

        ('w/o History Vocab',
         {'no_history': True}, {}),

        ('w/o Subgraph Encoder',
         {'no_sgcn': True}, {}),

        ('w/o H-GRU',
         {'no_hgru': True}, {}),

        ('w/o ODE',
         {'no_ode': True}, {}),

        ('w/o Hyperbolic',
         {'no_hyperbolic': True}, {}),
    ]

    HDR = (f"{'Variant':<24} | {'MRR':>6} | {'H@1':>6} | "
           f"{'H@3':>6} | {'H@10':>6} | {'MAR':>7} | {'vs Full':>8}")
    SEP = '-' * 72
    results = []

    print(f"\nV4 Ablation Study — ICEWS14 ({EPOCHS} epochs)\n")
    print(f"{HDR}\n{SEP}")

    for name, model_kw, train_kw in variants:
        best_valid, tm = run_variant(
            name, data, train_groups, snapshots, ts_to_real,
            model_kwargs=model_kw, train_kwargs=train_kw)
        results.append((name, best_valid, tm))

        # Print table after each variant
        print(f"\n── Intermediate Table ──")
        print(f"{HDR}\n{SEP}")
        full_mrr = results[0][2]['MRR']
        for (n, bv, m) in results:
            diff = m['MRR'] - full_mrr
            sign = '+' if diff >= 0 else ''
            print(f"{n:<24} | {m['MRR']:>6.4f} | {m['Hits@1']:>6.4f} | "
                  f"{m['Hits@3']:>6.4f} | {m['Hits@10']:>6.4f} | "
                  f"{m['MAR']:>7.1f} | {sign}{diff:.4f}")

    # Final table
    print(f"\n{'='*72}")
    print(f"FINAL V4 ABLATION — ICEWS14 ({EPOCHS} epochs, patience={PATIENCE})")
    print(f"{HDR}\n{SEP}")
    full_mrr = results[0][2]['MRR']
    for (name, bv, tm) in results:
        diff = tm['MRR'] - full_mrr
        sign = '+' if diff >= 0 else ''
        pct  = diff / full_mrr * 100
        print(f"{name:<24} | {tm['MRR']:>6.4f} | {tm['Hits@1']:>6.4f} | "
              f"{tm['Hits@3']:>6.4f} | {tm['Hits@10']:>6.4f} | "
              f"{tm['MAR']:>7.1f} | {sign}{diff:.4f} ({sign}{pct:.1f}%)")

if __name__ == '__main__':
    main()

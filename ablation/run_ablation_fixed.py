"""
Ablation study using monkey-patching to disable components
without modifying rhgnn_end_to_end.py
"""
import torch, math, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import (
    load_temporal_kg, build_snapshot_graphs, RHGNN,
    train_one_epoch, evaluate
)
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────────────

def make_model(data, dim=200):
    return RHGNN(data.num_entities, data.num_relations, dim=dim).cuda()

def run_variant(name, data, snapshots, ts_to_real, device,
                patch_fn=None, epochs=200, patience=40,
                batch_size=1024, neg_ratio=5, lr=1e-3):

    model = make_model(data)
    if patch_fn:
        patch_fn(model)

    from geoopt.optim import RiemannianAdam
    optimizer = RiemannianAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8)

    # build training groups (same as main script)
    from collections import defaultdict
    train_groups = defaultdict(list)
    for q in data.train:
        train_groups[q[3]].append(q)
    train_groups = sorted(train_groups.items())

    best_mrr, best_state, no_improve = 0.0, None, 0

    # build index once before training
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_groups, optimizer, device,
            data.num_entities, snapshots, ts_to_real,
            neg_ratio=neg_ratio, batch_size=batch_size)
        scheduler.step()

        if epoch % 20 == 0:
            vm = evaluate(model, data.valid, data.all_true,
                         snapshots, ts_to_real, device, batch_size=512)
            mrr = vm['MRR']
            print(f"  [{name}] ep{epoch:04d} loss={loss:.4f} valid_MRR={mrr:.4f}",
                  flush=True)
            if mrr > best_mrr:
                best_mrr = mrr
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 20:
                print(f"  [{name}] early stop ep{epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    tm = evaluate(model, data.test, data.all_true,
                 snapshots, ts_to_real, device, batch_size=512)
    return best_mrr, tm

# ── ablation patches ─────────────────────────────────────────────────────────

def patch_no_ode(model):
    """Replace ODE flow with identity — no continuous-time evolution."""
    def _ode_identity(h_hyp, delta_t):
        return h_hyp
    model._ode_flow = _ode_identity
    print("  [patch] ODE disabled — identity flow")

def patch_no_hgru(model):
    """Replace H-GRU jump with identity — no gated memory update."""
    def _hgru_identity(h_prev_hyp, msg_tan):
        return h_prev_hyp
    model._hgru_jump = _hgru_identity
    print("  [patch] H-GRU disabled — identity jump")

def patch_no_hyperbolic(model):
    """Replace expmap0/logmap0 with identity — Euclidean geometry."""
    from rhgnn_end_to_end import expmap0, logmap0, mobius_add, hyp_distance
    import torch.nn.functional as F

    def _to_hyp_euclidean(ids):
        return model.entity_emb(ids)   # no expmap

    def _rel_hyp_euclidean(ids):
        return model.relation_emb(ids) # no expmap

    def score_all_tails_euclidean(h, r, tau, snapshots, delta_t=1.0):
        msg_tan = model._aggregate_messages(h, tau, snapshots)
        h_emb   = model.entity_emb(h)
        # GRU in Euclidean
        v_prev  = h_emb
        z       = torch.sigmoid(model.Wz(v_prev) + model.Uz(msg_tan))
        h_jump  = z * msg_tan + (1.0 - z) * v_prev
        # ODE in Euclidean tangent
        state   = h_jump
        dt      = delta_t / max(model.ode_steps, 1)
        for _ in range(model.ode_steps):
            state = state + dt * model.ode_func(state)
        # Euclidean translation instead of Mobius
        pred    = state + model.relation_emb(r)
        all_emb = model.entity_emb.weight          # (E, d)
        # L2 distance
        diff    = pred.unsqueeze(1) - all_emb.unsqueeze(0)
        dist    = diff.norm(dim=-1)
        return -dist + model.bias.unsqueeze(0)

    model._to_hyp   = _to_hyp_euclidean
    model._rel_hyp  = _rel_hyp_euclidean
    model.score_all_tails = score_all_tails_euclidean
    print("  [patch] Hyperbolic disabled — Euclidean geometry")

def patch_no_mp(model):
    """Replace message passing with zero vector — no GNN aggregation."""
    def _aggregate_zero(entity_ids, tau_ids, snapshots):
        return torch.zeros(entity_ids.size(0), model.dim,
                          device=entity_ids.device)
    model._aggregate_messages = _aggregate_zero
    print("  [patch] Message passing disabled — zero aggregation")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device   = torch.device('cuda:0')
    data_dir = 'data/ICEWS14'

    print(f"Loading {data_dir}...", flush=True)
    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    variants = [
        ('Full RHGNN',      None),
        ('w/o ODE',         patch_no_ode),
        ('w/o H-GRU',       patch_no_hgru),
        ('w/o Hyperbolic',  patch_no_hyperbolic),
        ('w/o Msg Passing', patch_no_mp),
    ]

    HDR = (f"{'Variant':<22} | {'MRR':>6} | {'H@1':>6} | "
           f"{'H@3':>6} | {'H@10':>6} | {'MAR':>7}")
    SEP = '-' * 62

    print(f"\n{HDR}\n{SEP}", flush=True)
    results = []

    for name, patch_fn in variants:
        print(f"\n{'='*50}\nTraining: {name}", flush=True)
        best_valid, tm = run_variant(
            name, data, snapshots, ts_to_real, device,
            patch_fn=patch_fn, epochs=200, patience=40)

        row = (f"{name:<22} | {tm['MRR']:>6.4f} | {tm['Hits@1']:>6.4f} | "
               f"{tm['Hits@3']:>6.4f} | {tm['Hits@10']:>6.4f} | {tm['MAR']:>7.1f}")
        results.append((name, best_valid, tm))

        print(f"\n── Intermediate Table ──")
        print(f"{HDR}\n{SEP}")
        for (n, bv, m) in results:
            print(f"{n:<22} | {m['MRR']:>6.4f} | {m['Hits@1']:>6.4f} | "
                  f"{m['Hits@3']:>6.4f} | {m['Hits@10']:>6.4f} | {m['MAR']:>7.1f}")

    print(f"\n{'='*62}")
    print(f"FINAL ABLATION TABLE — ICEWS14 (200 epochs, patience=40)")
    print(f"{HDR}\n{SEP}")
    full_mrr = results[0][2]['MRR']
    for (name, bv, tm) in results:
        drop = tm['MRR'] - full_mrr
        sign = '+' if drop >= 0 else ''
        print(f"{name:<22} | {tm['MRR']:>6.4f} | {tm['Hits@1']:>6.4f} | "
              f"{tm['Hits@3']:>6.4f} | {tm['Hits@10']:>6.4f} | "
              f"{tm['MAR']:>7.1f}  ({sign}{drop:.4f})")

if __name__ == '__main__':
    main()

"""
Extra metrics for paper:
1. Temporal MRR Decay
2. Relation-Stratified MRR
3. Parameter Count Table
4. Loss Smoothness (ODE stability)
"""
import torch, sys, re, os, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, TempHypE-R
from rhgnn_v4 import TempHypE-Rv4
from rhgnn_v3 import build_history_vocab

DEVICE = torch.device('cuda:0')

def load_v1(data, ckpt_path):
    model = TempHypE-R(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model

def load_v4(data, ckpt_path):
    model = TempHypERCA(data.num_entities, data.num_relations,
                    dim=200, num_sgcn_layers=2).to(DEVICE)
    model.set_history_vocab(build_history_vocab(data.train))
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model

@torch.no_grad()
def get_ranks(model, quads, all_true, snapshots, ts_to_real, batch_size=512):
    model.eval()
    true_tails_map = defaultdict(list)
    for (hh,rr,tt,ta) in all_true:
        true_tails_map[(hh,rr,ta)].append(tt)

    ts_groups = defaultdict(list)
    for q in quads:
        ts_groups[q[3]].append(q)

    ranks_out, ts_out, rel_out = [], [], []
    prev_real_ts = None

    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = float(real_ts - prev_real_ts) \
                  if prev_real_ts is not None else 1.0
        delta_t = max(delta_t, 1.0)
        prev_real_ts = real_ts
        quads_t = ts_groups[ts_id]

        for start in range(0, len(quads_t), batch_size):
            batch_q = quads_t[start:start+batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=DEVICE)
            h,r,true_t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
            scores  = model.score_all_tails(h, r, tau, snapshots, delta_t)
            for i,(hh,rr,tt,ta) in enumerate(batch_q):
                others = [c for c in true_tails_map[(hh,rr,ta)] if c!=tt]
                if others:
                    scores[i, torch.tensor(others,device=DEVICE)] = float('-inf')
            true_scores = scores[torch.arange(scores.size(0),device=DEVICE), true_t]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
            ranks_out.extend(rank.cpu().tolist())
            ts_out.extend([ts_id]*len(batch_q))
            rel_out.extend([q[1] for q in batch_q])

    return (np.array(ranks_out),
            np.array(ts_out),
            np.array(rel_out))

def mrr(r):    return float(np.mean(1.0/r)) if len(r)>0 else 0.0
def hits(r,k): return float(np.mean(r<=k))  if len(r)>0 else 0.0

# ── 1. Temporal MRR Decay ─────────────────────────────────────────────────────
def temporal_mrr_decay(ranks, ts_ids, name):
    print(f"\n── Temporal MRR Decay [{name}] ──")
    unique_ts = sorted(np.unique(ts_ids))
    n_bins    = 5
    bin_size  = max(1, len(unique_ts) // n_bins)
    bins      = [unique_ts[i:i+bin_size]
                 for i in range(0, len(unique_ts), bin_size)][:n_bins]
    labels    = ['Very Early', 'Early', 'Middle', 'Late', 'Very Late']

    print(f"  {'Period':<12} | {'N':>6} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6}")
    print(f"  {'-'*50}")
    mrrs = []
    for bin_ts, label in zip(bins, labels):
        mask = np.isin(ts_ids, bin_ts)
        r    = ranks[mask]
        if len(r) == 0: continue
        m = mrr(r)
        mrrs.append(m)
        print(f"  {label:<12} | {len(r):>6} | {m:>6.4f} | "
              f"{hits(r,1):>6.4f} | {hits(r,10):>6.4f}")
    if len(mrrs) >= 2:
        retention = mrrs[-1] / mrrs[0] if mrrs[0] > 0 else 0
        decay_pct = (mrrs[0] - mrrs[-1]) / mrrs[0] * 100
        print(f"  Retention ratio: {retention:.3f} | "
              f"Decay: {decay_pct:.1f}% | "
              f"{'✓ stable' if retention >= 0.85 else '⚠ degrading'}")

# ── 2. Relation-Stratified MRR ────────────────────────────────────────────────
def relation_stratified(ranks, rel_ids, data, name):
    print(f"\n── Relation-Stratified MRR [{name}] ──")
    rel_freq = defaultdict(int)
    for _,r,_,_ in data.train:
        rel_freq[r] += 1

    freq_vals = np.array([rel_freq.get(r,0) for r in rel_ids])
    bins = [
        ('Rare    (<10)',     freq_vals < 10),
        ('Medium (10-100)',  (freq_vals>=10) & (freq_vals<100)),
        ('Frequent(>=100)',  freq_vals >= 100),
    ]
    print(f"  {'Relation Type':<22} | {'N':>6} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6}")
    print(f"  {'-'*58}")
    for label, mask in bins:
        r = ranks[mask]
        if len(r) == 0: continue
        print(f"  {label:<22} | {len(r):>6} | {mrr(r):>6.4f} | "
              f"{hits(r,1):>6.4f} | {hits(r,10):>6.4f}")

# ── 3. Parameter Count ────────────────────────────────────────────────────────
def param_count_table():
    print(f"\n── Parameter Count Comparison ──")
    print(f"  {'Model':<20} | {'Dataset':<12} | {'Params':>12}")
    print(f"  {'-'*50}")

    for ds_name, data_dir in [('ICEWS14','data/ICEWS14'),
                               ('WIKI',   'data/WIKI-clean'),
                               ('YAGO',   'data/YAGO-clean')]:
        data = load_temporal_kg(data_dir)
        m1   = TempHypE-R(data.num_entities, data.num_relations, dim=200)
        m4   = TempHypERCA(data.num_entities, data.num_relations,
                       dim=200, num_sgcn_layers=2)
        p1   = sum(p.numel() for p in m1.parameters() if p.requires_grad)
        p4   = sum(p.numel() for p in m4.parameters() if p.requires_grad)
        print(f"  {'TempHypE-R-Base':<20} | {ds_name:<12} | {p1:>12,}")
        print(f"  {'TempHypE-R-C':<20} | {ds_name:<12} | {p4:>12,}")
        print(f"  Overhead: +{p4-p1:,} (+{(p4-p1)/p1*100:.1f}%)")
        print()

# ── 4. Loss Smoothness ────────────────────────────────────────────────────────
def loss_smoothness():
    print(f"\n── ODE Training Stability (Loss Smoothness) ──")
    print(f"  {'Model':<25} | {'Dataset':<12} | {'Mean Loss':>10} | {'Std Loss':>9} | {'Smoothness':>10}")
    print(f"  {'-'*72}")

    log_pairs = [
        ('TempHypE-R-Base', 'ICEWS14', 'logs/icews14_v4.log'),
        ('TempHypE-R-C',    'ICEWS14', 'logs/rhgnn_v4_icews14.log'),
        ('TempHypE-R-Base', 'WIKI',    'logs/wiki_v1.log'),
        ('TempHypE-R-C',    'WIKI',    'logs/rhgnn_v4_wiki.log'),
        ('TempHypE-R-Base', 'YAGO',    'logs/yago_v2.log'),
        ('TempHypE-R-C',    'YAGO',    'logs/rhgnn_v4_yago.log'),
    ]

    for model_name, ds_name, log_path in log_pairs:
        if not os.path.exists(log_path):
            print(f"  {model_name:<25} | {ds_name:<12} | log not found")
            continue
        losses = []
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Loss:\s*([\d.]+)', line)
                if m:
                    losses.append(float(m.group(1)))
        if not losses:
            continue
        # Use second half for stability measurement
        half    = losses[len(losses)//2:]
        mean_l  = np.mean(half)
        std_l   = np.std(half)
        smooth  = 1.0 / (std_l + 1e-8)
        print(f"  {model_name:<25} | {ds_name:<12} | {mean_l:>10.4f} | "
              f"{std_l:>9.6f} | {smooth:>10.2f}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_dataset(ds_name, data_dir, v1_ckpt, v4_ckpt):
    print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    for model_name, ckpt_path, loader in [
        ('TempHypE-R-Base', v1_ckpt, load_v1),
        ('TempHypE-R-C',    v4_ckpt, load_v4),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"  {model_name}: checkpoint not found — skip")
            continue
        try:
            print(f"\n--- {model_name} ---", flush=True)
            model = loader(data, ckpt_path)
            model.build_nbr_index(snapshots)
            model.refresh_msg_table(DEVICE)
            print(f"Scoring {len(data.test)} triples...", flush=True)
            ranks, ts_ids, rel_ids = get_ranks(
                model, data.test, data.all_true, snapshots, ts_to_real)
            print(f"MRR={mrr(ranks):.4f} | H@1={hits(ranks,1):.4f} | "
                  f"H@10={hits(ranks,10):.4f}")
            temporal_mrr_decay(ranks, ts_ids, f"{ds_name}-{model_name}")
            relation_stratified(ranks, rel_ids, data, f"{ds_name}-{model_name}")
        except Exception as e:
            print(f"  {model_name} FAILED: {e}", flush=True)

def main():
    param_count_table()
    loss_smoothness()

    datasets = [
        ('ICEWS14', 'data/ICEWS14',
         'checkpoints/rhgnn_icews14_v4.pt',
         'checkpoints/rhgnn_v4_icews14.pt'),
        ('WIKI', 'data/WIKI-clean',
         'checkpoints/rhgnn_wiki_v1.pt',
         'checkpoints/rhgnn_v4_wiki.pt'),
        ('YAGO', 'data/YAGO-clean',
         'checkpoints/rhgnn_yago_v2.pt',
         'checkpoints/rhgnn_v4_yago.pt'),
    ]

    for ds_name, data_dir, v1_ckpt, v4_ckpt in datasets:
        run_dataset(ds_name, data_dir, v1_ckpt, v4_ckpt)

    print(f"\n{'='*60}\nAll extra metrics complete.")

if __name__ == '__main__':
    main()

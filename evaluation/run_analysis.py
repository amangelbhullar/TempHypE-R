import torch, sys, re, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, RHGNN

DEVICE = torch.device('cuda:0')

def load_model(data, ckpt_path):
    model = RHGNN(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model

@torch.no_grad()
def score_all(model, quads, all_true, snapshots, ts_to_real, batch_size=512):
    model.eval()
    true_tails_map = defaultdict(list)
    for (hh, rr, tt, ta) in all_true:
        true_tails_map[(hh, rr, ta)].append(tt)
    head_degree = defaultdict(int)
    for (hh, rr, tt, ta) in all_true:
        head_degree[hh] += 1
    ts_groups = defaultdict(list)
    for q in quads:
        ts_groups[q[3]].append(q)
    ranks_out, deltas_out, degrees_out = [], [], []
    prev_real_ts = None
    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = float(real_ts - prev_real_ts) if prev_real_ts is not None else 1.0
        delta_t = max(delta_t, 1.0)
        prev_real_ts = real_ts
        quads_t = ts_groups[ts_id]
        for start in range(0, len(quads_t), batch_size):
            batch_q = quads_t[start:start+batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=DEVICE)
            h, r, true_t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
            scores  = model.score_all_tails(h, r, tau, snapshots, delta_t)
            for i, (hh, rr, tt, ta) in enumerate(batch_q):
                others = [c for c in true_tails_map[(hh, rr, ta)] if c != tt]
                if others:
                    scores[i, torch.tensor(others, device=DEVICE)] = float('-inf')
            true_scores = scores[torch.arange(scores.size(0), device=DEVICE), true_t]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
            ranks_out.extend(rank.cpu().tolist())
            deltas_out.extend([delta_t] * len(batch_q))
            degrees_out.extend([head_degree[q[0]] for q in batch_q])
    return np.array(ranks_out), np.array(deltas_out), np.array(degrees_out)

def mrr(r):   return float(np.mean(1.0/r)) if len(r)>0 else 0.0
def hits(r,k): return float(np.mean(r<=k))  if len(r)>0 else 0.0

def print_tps(ranks, deltas, name):
    print(f"\n── TPS: Temporal Persistence Score [{name}] ──")
    bins = [('Small  (Δt=1)',   deltas==1),
            ('Medium (Δt 2-5)', (deltas>1)&(deltas<=5)),
            ('Large  (Δt >5)',  deltas>5)]
    print(f"  {'Bin':<22} | {'N':>5} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6}")
    print(f"  {'-'*58}")
    mrrs = []
    for label, mask in bins:
        r = ranks[mask]
        if len(r)==0:
            print(f"  {label:<22} | {'0':>5} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}")
            mrrs.append(None)
            continue
        m = mrr(r)
        mrrs.append(m)
        print(f"  {label:<22} | {len(r):>5} | {m:>6.4f} | {hits(r,1):>6.4f} | {hits(r,10):>6.4f}")
    valid = [m for m in mrrs if m is not None]
    if len(valid) >= 2 and valid[0] > 0:
        tps = valid[-1] / valid[0]
        print(f"  TPS ratio (large/small): {tps:.3f}  {'✓ stable' if tps>=0.8 else '⚠ degrades over time'}")

def print_tgs(ranks, deltas, name):
    print(f"\n── TGS: Time Gap Sensitivity [{name}] ──")
    bins = [(1,1,'Δt=1'), (2,5,'Δt 2-5'), (6,10,'Δt 6-10'), (11,9999,'Δt >10')]
    print(f"  {'Gap':>10} | {'N':>5} | {'MRR':>6} | {'H@10':>6}")
    print(f"  {'-'*38}")
    mrrs = []
    for lo,hi,lab in bins:
        mask = (deltas>=lo)&(deltas<=hi)
        r = ranks[mask]
        if len(r)==0: continue
        m = mrr(r)
        mrrs.append(m)
        print(f"  {lab:>10} | {len(r):>5} | {m:>6.4f} | {hits(r,10):>6.4f}")
    if len(mrrs)>1:
        tgs = float(np.std(mrrs))
        print(f"  TGS std: {tgs:.4f}  {'✓ stable' if tgs<0.05 else '⚠ sensitive to gaps'}")

def print_deg(ranks, degrees, name):
    print(f"\n── DEG: Degree-Stratified MRR [{name}] ──")
    bins = [('Low   (1-10)',  (degrees>=1)&(degrees<=10)),
            ('Medium (11-50)',(degrees>=11)&(degrees<=50)),
            ('High   (>50)',  degrees>50)]
    print(f"  {'Degree Bin':<22} | {'N':>5} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6}")
    print(f"  {'-'*58}")
    for label, mask in bins:
        r = ranks[mask]
        if len(r)==0: continue
        print(f"  {label:<22} | {len(r):>5} | {mrr(r):>6.4f} | {hits(r,1):>6.4f} | {hits(r,10):>6.4f}")

def print_cui():
    print(f"\n── CUI: Curvature Utilization Index ──")
    print(f"  {'Dataset':<15} | {'Final c':>7} | {'CUI':>6} | Interpretation")
    print(f"  {'-'*60}")
    logs = [
        ('ICEWS14',    'logs/icews14_v4.log'),
        ('ICEWS18',    'logs/icews18_v3.log'),
        ('WIKI',       'logs/wiki_v1.log'),
        ('YAGO',       'logs/yago_v2.log'),
        ('GDELT',      'logs/gdelt_v2.log'),
        ('ICEWS05-15', 'logs/icews0515_v2.log'),
    ]
    for name, path in logs:
        try:
            cs = [float(m.group(1))
                  for line in open(path)
                  for m in [re.search(r'\| c:\s*([\d.]+)', line)] if m]
            if not cs:
                print(f"  {name:<15} | no data"); continue
            fc  = cs[-1]
            cui = 1.0 - fc
            if   fc < 0.05:  interp = "Very high curvature — strong hierarchy"
            elif fc < 0.15:  interp = "High curvature — moderate hierarchy"
            elif fc < 0.35:  interp = "Moderate curvature"
            else:            interp = "Low curvature — weak hierarchy"
            print(f"  {name:<15} | {fc:>7.4f} | {cui:>6.4f} | {interp}")
        except FileNotFoundError:
            print(f"  {name:<15} | log not found")

def run_dataset(name, data_dir, ckpt):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts
    model = load_model(data, ckpt)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(DEVICE)
    print(f"Scoring {len(data.test)} test triples...", flush=True)
    ranks, deltas, degrees = score_all(
        model, data.test, data.all_true, snapshots, ts_to_real)
    print(f"Overall MRR={mrr(ranks):.4f} | H@1={hits(ranks,1):.4f} | H@10={hits(ranks,10):.4f}")
    print_tps(ranks, deltas, name)
    print_tgs(ranks, deltas, name)
    print_deg(ranks, degrees, name)

def main():
    print_cui()
    datasets = [
        ('ICEWS14', 'data/ICEWS14',    'checkpoints/rhgnn_icews14_v4.pt'),
        ('WIKI',    'data/WIKI-clean',  'checkpoints/rhgnn_wiki_v1.pt'),
        ('YAGO',    'data/YAGO-clean',  'checkpoints/rhgnn_yago_v2.pt'),
    ]
    for name, data_dir, ckpt in datasets:
        run_dataset(name, data_dir, ckpt)
    print(f"\n{'='*60}\nAll analyses complete.")

if __name__ == '__main__':
    main()

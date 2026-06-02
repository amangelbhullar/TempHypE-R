import torch, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, RHGNN
from rhgnn_v4 import RHGNNv4
from rhgnn_v3 import build_history_vocab

DEVICE = torch.device('cuda:0')

def load_v1(data, ckpt):
    m = RHGNN(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

def load_v4(data, ckpt):
    m = RHGNNv4(data.num_entities, data.num_relations, dim=200, num_sgcn_layers=2).to(DEVICE)
    m.set_history_vocab(build_history_vocab(data.train))
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

@torch.no_grad()
def get_ranks(model, quads, all_true, snapshots, ts_to_real, batch_size=512):
    model.eval()
    true_tails = defaultdict(list)
    for (hh,rr,tt,ta) in all_true:
        true_tails[(hh,rr,ta)].append(tt)
    ts_groups = defaultdict(list)
    for q in quads:
        ts_groups[q[3]].append(q)
    ranks_out, rel_out = [], []
    prev_real_ts = None
    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = max(float(real_ts - prev_real_ts) if prev_real_ts else 1.0, 1.0)
        prev_real_ts = real_ts
        for start in range(0, len(ts_groups[ts_id]), batch_size):
            batch_q = ts_groups[ts_id][start:start+batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=DEVICE)
            h,r,true_t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
            scores  = model.score_all_tails(h, r, tau, snapshots, delta_t)
            for i,(hh,rr,tt,ta) in enumerate(batch_q):
                others = [c for c in true_tails[(hh,rr,ta)] if c!=tt]
                if others:
                    scores[i, torch.tensor(others,device=DEVICE)] = float('-inf')
            true_scores = scores[torch.arange(scores.size(0),device=DEVICE), true_t]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
            ranks_out.extend(rank.cpu().tolist())
            rel_out.extend([q[1] for q in batch_q])
    return np.array(ranks_out), np.array(rel_out)

def mrr(r):    return float(np.mean(1.0/r)) if len(r)>0 else 0.0
def hits(r,k): return float(np.mean(r<=k))  if len(r)>0 else 0.0

def relation_stratified(ranks, rel_ids, data, name):
    print(f"\n── Relation-Stratified MRR [{name}] ──")
    rel_freq = defaultdict(int)
    for _,r,_,_ in data.train: rel_freq[r] += 1
    freq_vals = np.array([rel_freq.get(r,0) for r in rel_ids])
    bins = [('Rare    (<10)',    freq_vals<10),
            ('Medium (10-100)', (freq_vals>=10)&(freq_vals<100)),
            ('Frequent(>=100)', freq_vals>=100)]
    print(f"  {'Type':<22} | {'N':>6} | {'MRR':>6} | {'H@1':>6} | {'H@10':>6}")
    print(f"  {'-'*55}")
    for label, mask in bins:
        r = ranks[mask]
        if len(r)==0: continue
        print(f"  {label:<22} | {len(r):>6} | {mrr(r):>6.4f} | "
              f"{hits(r,1):>6.4f} | {hits(r,10):>6.4f}")

def run(ds_name, data_dir, v1_ckpt, v4_ckpt):
    print(f"\n{'='*55}\nDataset: {ds_name}\n{'='*55}")
    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    for name, ckpt, loader in [('RHGNN-Base', v1_ckpt, load_v1),
                                ('RHGNN-C',   v4_ckpt, load_v4)]:
        try:
            print(f"\n--- {name} ---", flush=True)
            model = loader(data, ckpt)
            model.build_nbr_index(snapshots)
            model.refresh_msg_table(DEVICE)
            ranks, rel_ids = get_ranks(model, data.test, data.all_true,
                                       snapshots, ts_to_real)
            print(f"MRR={mrr(ranks):.4f} | H@1={hits(ranks,1):.4f} | "
                  f"H@10={hits(ranks,10):.4f}")
            relation_stratified(ranks, rel_ids, data, f"{ds_name}-{name}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")

run('ICEWS18',   'data/ICEWS18',
    'checkpoints/rhgnn_icews18_v3.pt',
    'checkpoints/rhgnn_v4_icews18.pt')
run('ICEWS05-15','data/ICEWS05-15',
    'checkpoints/rhgnn_icews0515_v2.pt',
    'checkpoints/rhgnn_v4_icews0515_seed2.pt')
print("\nDone.")

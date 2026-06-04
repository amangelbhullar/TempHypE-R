import torch, sys, numpy as np, os
from collections import defaultdict
from scipy import stats
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, TempHypE-R
from rhgnn_v2 import TempHypE-Rv2, expmap0, logmap0
from rhgnn_v3 import TempHypE-Rv3, build_history_vocab
from rhgnn_v4 import TempHypE-Rv4

DEVICE = torch.device('cuda:0')

def load_v1(data, ckpt):
    m = TempHypE-R(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

def load_v2(data, ckpt):
    m = TempHypERFA(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

def load_v3(data, ckpt):
    m = TempHypERMA(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    m.set_history_vocab(build_history_vocab(data.train))
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

def load_v4(data, ckpt):
    m = TempHypERCA(data.num_entities, data.num_relations, dim=200, num_sgcn_layers=2).to(DEVICE)
    m.set_history_vocab(build_history_vocab(data.train))
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['model_state'])
    return m

def get_embs(model):
    with torch.no_grad():
        model.eval()
        ids  = torch.arange(model.num_entities, device=DEVICE)
        embs = expmap0(model.entity_emb(ids), model.c)
        return embs.cpu().numpy()

def norm_hierarchy_corr(model, data):
    norms  = np.linalg.norm(get_embs(model), axis=-1)
    degree = defaultdict(int)
    for h,r,t,_ in data.train:
        degree[h]+=1; degree[t]+=1
    degrees = np.array([degree.get(i,0) for i in range(data.num_entities)], dtype=float)
    corr, _ = stats.pearsonr(norms, degrees)
    return corr

def delta_hyp(model, data, n=300):
    embs  = torch.tensor(get_embs(model))
    n_ent = min(300, data.num_entities)
    embs  = embs[:n_ent]
    deltas = []
    for _ in range(n):
        idx = np.random.choice(n_ent, 4, replace=False)
        w,x,y,z = [embs[i] for i in idx]
        def d(a,b): return (a-b).norm().item()
        gxy = 0.5*(d(x,z)+d(y,z)-d(x,y))
        gxz = 0.5*(d(x,y)+d(y,z)-d(x,z))
        gyz = 0.5*(d(x,y)+d(x,z)-d(y,z))
        ps  = sorted([gxy,gxz,gyz], reverse=True)
        deltas.append(ps[0]-ps[1])
    return np.mean(deltas)

def emb_quality(model, data):
    norms = np.linalg.norm(get_embs(model), axis=-1)
    return norms.mean(), norms.std(), norms.std()*norms.mean()

def hmrr(model, data, snapshots, ts_to_real):
    model.eval()
    degree = defaultdict(int)
    for h,r,t,_ in data.train:
        degree[h]+=1; degree[t]+=1
    max_deg = max(degree.values()) if degree else 1
    weights = np.array([degree.get(i,0)/max_deg for i in range(data.num_entities)])
    true_map = defaultdict(list)
    for (hh,rr,tt,ta) in data.all_true:
        true_map[(hh,rr,ta)].append(tt)
    ts_groups = defaultdict(list)
    for q in data.test: ts_groups[q[3]].append(q)
    ranks, tw = [], []
    prev = None
    with torch.no_grad():
        for ts_id in sorted(ts_groups.keys()):
            rt  = ts_to_real.get(ts_id, ts_id)
            dt  = max(float(rt-prev) if prev else 1.0, 1.0)
            prev= rt
            for s in range(0, len(ts_groups[ts_id]), 256):
                bq  = ts_groups[ts_id][s:s+256]
                bat = torch.tensor(bq, dtype=torch.long, device=DEVICE)
                h,r,t,tau = bat[:,0],bat[:,1],bat[:,2],bat[:,3]
                sc  = model.score_all_tails(h,r,tau,snapshots,dt)
                for i,(hh,rr,tt,ta) in enumerate(bq):
                    oth=[c for c in true_map[(hh,rr,ta)] if c!=tt]
                    if oth:
                        sc[i,torch.tensor(oth,device=DEVICE)]=float('-inf')
                ts2 = sc[torch.arange(sc.size(0),device=DEVICE),t]
                rk  = (sc>ts2.unsqueeze(1)).sum(dim=1).float()+1
                ranks.extend(rk.cpu().tolist())
                tw.extend(weights[t.cpu().numpy()].tolist())
    ranks = np.array(ranks)
    tw    = np.array(tw)
    std_mrr = float(np.mean(1.0/ranks))
    h_mrr   = float(np.sum((tw/tw.sum())/ranks))
    hi = ranks[tw >= np.percentile(tw,66)]
    lo = ranks[tw < np.percentile(tw,33)]
    return (std_mrr, h_mrr,
            float(np.mean(1.0/lo)) if len(lo) else 0,
            float(np.mean(1.0/hi)) if len(hi) else 0)

def run(ds_name, data_dir, models):
    print(f"\n{'='*90}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*90}")
    print(f"{'Model':<22} | {'δ-Hyp':>6} | {'NHC':>6} | {'‖·‖':>6} | "
          f"{'‖·‖σ':>6} | {'MRR':>6} | {'H-MRR':>7} | {'Ratio':>6} | "
          f"{'Lo-MRR':>7} | {'Hi-MRR':>7}")
    print(f"{'-'*95}")

    data = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    for name, ckpt, loader in models:
        if not os.path.exists(ckpt):
            print(f"{name:<22} | ckpt not found")
            continue
        try:
            model = loader(data, ckpt)
            model.build_nbr_index(snapshots)
            model.refresh_msg_table(DEVICE)
            dh         = delta_hyp(model, data)
            nhc        = norm_hierarchy_corr(model, data)
            mn,ns,pq   = emb_quality(model, data)
            std,hm,lo,hi = hmrr(model, data, snapshots, ts_to_real)
            ratio      = hm/std if std>0 else 0
            print(f"{name:<22} | {dh:>6.3f} | {nhc:>+6.3f} | {mn:>6.3f} | "
                  f"{ns:>6.3f} | {std:>6.4f} | {hm:>7.4f} | {ratio:>6.2f}x | "
                  f"{lo:>7.4f} | {hi:>7.4f}")
        except Exception as e:
            print(f"{name:<22} | ERROR: {e}")

def main():
    datasets = [
        ('ICEWS14', 'data/ICEWS14', [
            # V1 skipped - different architecture
            ('V2-TempHypE-R-Adv','checkpoints/rhgnn_v2_icews14.pt',   load_v2),
            ('V3-TempHypE-R-Hist','checkpoints/rhgnn_v3_icews14.pt',  load_v3),
            ('V4-TempHypE-R-C',  'checkpoints/rhgnn_v4_icews14.pt',   load_v4),
        ]),
        ('WIKI', 'data/WIKI-clean', [
            # V1 skipped - different architecture
            ('V2-TempHypE-R-Adv','checkpoints/rhgnn_v2_wiki.pt',      load_v2),
            ('V3-TempHypE-R-Hist','checkpoints/rhgnn_v3_wiki.pt',     load_v3),
            ('V4-TempHypE-R-C',  'checkpoints/rhgnn_v4_wiki.pt',      load_v4),
        ]),
        ('YAGO', 'data/YAGO-clean', [
            # V1 skipped - different architecture
            ('V2-TempHypE-R-Adv','checkpoints/rhgnn_v2_yago.pt',      load_v2),
            ('V3-TempHypE-R-Hist','checkpoints/rhgnn_v3_yago.pt',     load_v3),
            ('V4-TempHypE-R-C',  'checkpoints/rhgnn_v4_yago.pt',      load_v4),
        ]),
    ]
    for ds_name, data_dir, models in datasets:
        run(ds_name, data_dir, models)
    print(f"\n{'='*90}\nAll done.")

if __name__ == '__main__':
    main()

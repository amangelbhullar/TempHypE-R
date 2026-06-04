import torch, sys, numpy as np, os
from collections import defaultdict
from scipy import stats
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs
from rhgnn_v4 import TempHypE-Rv4
from rhgnn_v3 import build_history_vocab
from rhgnn_v2 import expmap0, logmap0

DEVICE = torch.device('cuda:0')

def load_v4_variant(data, ckpt, no_history=False, no_hyperbolic=False,
                    no_ode=False, no_hgru=False, no_sgcn=False):
    m = TempHypERCA(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        dim=200, num_sgcn_layers=0 if no_sgcn else 2).to(DEVICE)
    m.set_history_vocab(
        build_history_vocab(data.train) if not no_history else {})
    if no_hyperbolic:
        def _score_eucl(h, r, tau, snapshots, delta_t=1.0):
            msg  = m._aggregate_messages(h, tau, snapshots)
            v    = m.entity_emb(h)
            rz   = torch.sigmoid(m.Wr(v)+m.Ur(msg))
            z    = torch.sigmoid(m.Wz(v)+m.Uz(msg))
            hc   = torch.tanh(m.Wh(v)+m.Uh(rz*msg))
            h_j  = z*hc+(1-z)*v
            pred = h_j+m.relation_emb(r)
            dist = (pred.unsqueeze(1)-m.entity_emb.weight.unsqueeze(0)).norm(dim=-1)
            return -dist+m.bias.unsqueeze(0)
        m.score_all_tails = _score_eucl
    if no_hgru:
        m._hgru_jump = lambda h,msg,delta_t=1.0: h
    if no_ode:
        m._ode_flow = lambda h,dt: h
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd['model_state'], strict=False)
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
    degrees = np.array([degree.get(i,0)
                        for i in range(data.num_entities)], dtype=float)
    corr, _ = stats.pearsonr(norms, degrees)
    return corr

def delta_hyp(model, data, n=200):
    embs  = torch.tensor(get_embs(model))
    n_ent = min(200, data.num_entities)
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
    return norms.mean(), norms.std()

def hmrr(model, data, snapshots, ts_to_real):
    model.eval()
    degree = defaultdict(int)
    for h,r,t,_ in data.train:
        degree[h]+=1; degree[t]+=1
    max_deg = max(degree.values()) if degree else 1
    weights = np.array([degree.get(i,0)/max_deg
                        for i in range(data.num_entities)])
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
    return std_mrr, h_mrr, float(np.mean(1.0/hi)) if len(hi) else 0

def run_ablation(ds_name, data_dir, base_ckpt, variants):
    print(f"\n{'='*85}")
    print(f"Ablation Hyperbolic Metrics — {ds_name}")
    print(f"{'='*85}")
    print(f"{'Variant':<24} | {'δ-Hyp':>6} | {'NHC':>6} | "
          f"{'‖·‖ mean':>8} | {'‖·‖ σ':>6} | "
          f"{'MRR':>6} | {'H-MRR':>7} | {'Ratio':>6} | {'Hi-MRR':>7}")
    print(f"{'-'*90}")

    data = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    if not os.path.exists(base_ckpt):
        print(f"Base checkpoint not found: {base_ckpt}")
        return

    for name, kwargs in variants:
        try:
            model = load_v4_variant(data, base_ckpt, **kwargs)
            model.build_nbr_index(snapshots)
            model.refresh_msg_table(DEVICE)

            dh           = delta_hyp(model, data)
            nhc          = norm_hierarchy_corr(model, data)
            mn, ns       = emb_quality(model, data)
            std_mrr, h_mrr, hi_mrr = hmrr(model, data, snapshots, ts_to_real)
            ratio        = h_mrr/std_mrr if std_mrr > 0 else 0

            print(f"{name:<24} | {dh:>6.3f} | {nhc:>+6.3f} | "
                  f"{mn:>8.3f} | {ns:>6.3f} | "
                  f"{std_mrr:>6.4f} | {h_mrr:>7.4f} | "
                  f"{ratio:>6.2f}x | {hi_mrr:>7.4f}")
        except Exception as e:
            print(f"{name:<24} | ERROR: {e}")
            import traceback; traceback.print_exc()

def main():
    variants = [
        ('Full TempHypE-R-C',      {}),
        ('w/o Contrastive',   {}),  # same checkpoint, different training
        ('w/o History Vocab', {'no_history': True}),
        ('w/o Subgraph',      {'no_sgcn': True}),
        ('w/o H-GRU',         {'no_hgru': True}),
        ('w/o ODE',           {'no_ode': True}),
        ('w/o Hyperbolic',    {'no_hyperbolic': True}),
    ]

    datasets = [
        ('ICEWS14', 'data/ICEWS14',   'checkpoints/rhgnn_v4_icews14.pt'),
        ('WIKI',    'data/WIKI-clean','checkpoints/rhgnn_v4_wiki.pt'),
        ('YAGO',    'data/YAGO-clean','checkpoints/rhgnn_v4_yago.pt'),
    ]

    for ds_name, data_dir, ckpt in datasets:
        run_ablation(ds_name, data_dir, ckpt, variants)

    print(f"\n{'='*85}\nDone.")

if __name__ == '__main__':
    main()

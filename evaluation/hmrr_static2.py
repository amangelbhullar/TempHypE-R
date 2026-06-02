import numpy as np, torch, json, os
from collections import defaultdict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

DEVICE = 'cuda:0'

def compute_hmrr(model_name, ds_name, data_dir):
    def load_arr(split):
        return np.array([(p[0],p[1],p[2]) for l in open(f'{data_dir}/{split}.txt')
                        if l.strip() for p in [l.strip().split()] if len(p)>=3])

    train_arr = load_arr('train')
    valid_arr = load_arr('valid')
    test_arr  = load_arr('test')

    tf_train = TriplesFactory.from_labeled_triples(train_arr)
    tf_valid = TriplesFactory.from_labeled_triples(valid_arr,
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id)
    tf_test  = TriplesFactory.from_labeled_triples(test_arr,
        entity_to_id=tf_train.entity_to_id,
        relation_to_id=tf_train.relation_to_id)

    # Retrain (fast — 200 epochs)
    result = pipeline(
        training=tf_train, validation=tf_valid, testing=tf_test,
        model=model_name,
        model_kwargs=dict(embedding_dim=200),
        training_kwargs=dict(num_epochs=200, batch_size=1024),
        optimizer_kwargs=dict(lr=1e-3),
        device=DEVICE, random_seed=42,
        evaluation_kwargs=dict(batch_size=512),
    )
    m = result.model
    e2id = tf_train.entity_to_id  # PyKEEN's mapping
    r2id = tf_train.relation_to_id
    num_e = tf_train.num_entities

    ent_embs = m.entity_representations[0]().detach().cpu()
    rel_embs = m.relation_representations[0]().detach().cpu()

    mrr_std = result.metric_results.get_metric(
        "both.realistic.inverse_harmonic_mean_rank")

    # Build all_true filter using PyKEEN's IDs
    all_true = defaultdict(list)
    for arr in [train_arr, valid_arr, test_arr]:
        for h,r,t in arr:
            if h in e2id and r in r2id and t in e2id:
                all_true[(e2id[h], r2id[r])].append(e2id[t])

    # Degree weights
    degree = defaultdict(int)
    for h,r,t in train_arr:
        if h in e2id and t in e2id:
            degree[e2id[h]] += 1; degree[e2id[t]] += 1
    max_deg = max(degree.values()) if degree else 1
    weights = np.array([degree.get(i,0)/max_deg for i in range(num_e)])

    # Score test set
    ranks, tw = [], []
    for h,r,t in test_arr:
        if h not in e2id or r not in r2id or t not in e2id: continue
        hi, ri, ti = e2id[h], r2id[r], e2id[t]
        h_e = ent_embs[hi]
        r_e = rel_embs[ri]

        if model_name == 'TransE':
            scores = -(h_e + r_e - ent_embs).norm(dim=-1)
        else:  # DistMult
            scores = (h_e * r_e * ent_embs).sum(dim=-1)

        # Filter
        others = [c for c in all_true[(hi,ri)] if c != ti]
        if others: scores[torch.tensor(others)] = float('-inf')

        rank = (scores > scores[ti]).sum().item() + 1
        ranks.append(rank)
        tw.append(weights[ti])

    ranks = np.array(ranks, dtype=float)
    tw    = np.array(tw)
    w_norm = tw/tw.sum() if tw.sum()>0 else np.ones_like(tw)/len(tw)
    h_mrr  = float(np.sum(w_norm/ranks))
    hi_r   = ranks[tw >= np.percentile(tw,66)]
    lo_r   = ranks[tw <  np.percentile(tw,33)]

    # δ-hyperbolicity
    embs_np = ent_embs.numpy()
    norms   = np.linalg.norm(embs_np, axis=-1)
    from scipy import stats
    degrees_arr = np.array([degree.get(i,0) for i in range(num_e)], dtype=float)
    nhc, _ = stats.pearsonr(norms, degrees_arr)

    n_s = min(300, num_e)
    sample = torch.tensor(embs_np[:n_s])
    deltas = []
    for _ in range(300):
        idx = np.random.choice(n_s, 4, replace=False)
        w2,x,y,z2 = [sample[i] for i in idx]
        def d(a,b): return (a-b).norm().item()
        gxy=0.5*(d(x,z2)+d(y,z2)-d(x,y))
        gxz=0.5*(d(x,y)+d(y,z2)-d(x,z2))
        gyz=0.5*(d(x,y)+d(x,z2)-d(y,z2))
        ps=sorted([gxy,gxz,gyz],reverse=True)
        deltas.append(ps[0]-ps[1])

    ratio = h_mrr/mrr_std if mrr_std>0 else 0
    print(f'\n── {model_name} on {ds_name} ──')
    print(f'  MRR:             {mrr_std:.4f}')
    print(f'  H-MRR:           {h_mrr:.4f}')
    print(f'  H-MRR/MRR ratio: {ratio:.4f}x')
    print(f'  MRR low degree:  {np.mean(1.0/lo_r):.4f}')
    print(f'  MRR high degree: {np.mean(1.0/hi_r):.4f}')
    print(f'  δ-Hyp:           {np.mean(deltas):.4f}')
    print(f'  NHC:             {nhc:+.4f}')
    print(f'  norm std:        {norms.std():.4f}')
    return mrr_std, h_mrr, ratio, np.mean(deltas), nhc

results = {}
for model_name in ['DistMult', 'TransE']:
    for ds_name, data_dir in [('YAGO','data/YAGO-clean'),
                               ('ICEWS14','data/ICEWS14')]:
        key = f'{model_name}_{ds_name}'
        try:
            results[key] = compute_hmrr(model_name, ds_name, data_dir)
        except Exception as e:
            print(f'ERROR {key}: {e}')
            import traceback; traceback.print_exc()

print(f'\n{"="*70}')
print(f'{"Model":<22} | {"MRR":>6} | {"H-MRR":>7} | {"Ratio":>6} | {"δ-Hyp":>6} | {"NHC":>6}')
print('-'*70)
for k,v in results.items():
    print(f'{k:<22} | {v[0]:>6.4f} | {v[1]:>7.4f} | {v[2]:>6.2f}x | {v[3]:>6.3f} | {v[4]:>+6.3f}')

import torch, numpy as np, sys
from collections import defaultdict
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

DEVICE = 'cuda:0'

def hmrr_from_embs(embs, rel_embs, model_name, data_dir, ds_name):
    def load_str_triples(path):
        return [(p[0],p[1],p[2]) for l in open(path)
                if l.strip() for p in [l.strip().split()] if len(p)>=3]

    train_str = load_str_triples(f'{data_dir}/train.txt')
    test_str  = load_str_triples(f'{data_dir}/test.txt')
    valid_str = load_str_triples(f'{data_dir}/valid.txt')
    all_str   = train_str + valid_str + test_str

    # Rebuild entity mapping
    all_ents = sorted(set(h for h,r,t in all_str) | set(t for h,r,t in all_str))
    all_rels = sorted(set(r for h,r,t in all_str))
    e2id = {e:i for i,e in enumerate(all_ents)}
    r2id = {r:i for i,r in enumerate(all_rels)}
    num_e = len(all_ents)

    # Build filter
    all_true = defaultdict(list)
    for h,r,t in all_str:
        all_true[(e2id[h], r2id[r])].append(e2id[t])

    # Degree weights for H-MRR
    degree = defaultdict(int)
    for h,r,t in train_str:
        degree[e2id[h]] += 1; degree[e2id[t]] += 1
    max_deg = max(degree.values()) if degree else 1
    weights = np.array([degree.get(i,0)/max_deg for i in range(num_e)])

    embs_t    = torch.tensor(embs, dtype=torch.float32)
    rel_embs_t= torch.tensor(rel_embs, dtype=torch.float32)

    ranks, tw = [], []
    for h,r,t in test_str:
        if h not in e2id or r not in r2id or t not in e2id:
            continue
        hi, ri, ti = e2id[h], r2id[r], e2id[t]
        h_e = embs_t[hi]
        r_e = rel_embs_t[ri % len(rel_embs_t)]

        # Score all tails
        if model_name == 'TransE':
            scores = -(h_e + r_e - embs_t).norm(dim=-1)
        elif model_name == 'DistMult':
            scores = (h_e * r_e * embs_t).sum(dim=-1)
        else:
            scores = (h_e * r_e * embs_t).sum(dim=-1)

        # Filter
        others = [c for c in all_true[(hi,ri)] if c != ti]
        if others:
            scores[torch.tensor(others)] = float('-inf')

        rank = (scores > scores[ti]).sum().item() + 1
        ranks.append(rank)
        tw.append(weights[ti])

    ranks = np.array(ranks, dtype=float)
    tw    = np.array(tw)
    std_mrr = float(np.mean(1.0/ranks))
    w_norm  = tw/tw.sum() if tw.sum()>0 else np.ones_like(tw)/len(tw)
    h_mrr   = float(np.sum(w_norm/ranks))
    hi_mask = ranks[tw >= np.percentile(tw,66)]
    lo_mask = ranks[tw <  np.percentile(tw,33)]

    print(f'\n── H-MRR ({model_name} on {ds_name}) ──')
    print(f'  Standard MRR:     {std_mrr:.4f}')
    print(f'  H-MRR (weighted): {h_mrr:.4f}')
    print(f'  H-MRR/MRR ratio:  {h_mrr/std_mrr:.4f}x')
    print(f'  MRR low degree:   {np.mean(1.0/lo_mask):.4f}')
    print(f'  MRR high degree:  {np.mean(1.0/hi_mask):.4f}')
    return std_mrr, h_mrr

# Load saved embeddings and rerun scoring
datasets = [
    ('YAGO',    'data/YAGO-clean'),
    ('ICEWS14', 'data/ICEWS14'),
]
models = ['DistMult', 'TransE']

for model_name in models:
    for ds_name, data_dir in datasets:
        emb_path = f'checkpoints/{model_name.lower()}_{ds_name.lower()}_embs.npy'
        rel_path = f'checkpoints/{model_name.lower()}_{ds_name.lower()}_rel_embs.npy'
        if not __import__('os').path.exists(emb_path):
            print(f'Missing: {emb_path}')
            continue
        embs     = np.load(emb_path)
        rel_embs = np.load(rel_path) if __import__('os').path.exists(rel_path) else np.zeros((1,200))
        hmrr_from_embs(embs, rel_embs, model_name, data_dir, ds_name)

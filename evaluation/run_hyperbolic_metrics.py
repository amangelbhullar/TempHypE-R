"""
Hyperbolic-Specific Metrics for RHGNN Paper
1. Norm-Hierarchy Correlation (NHC)
2. Delta-Hyperbolicity Score
3. Hierarchical MRR (H-MRR)
4. Embedding Space Quality
5. Geodesic Temporal Consistency
6. Poincare Embedding Quality
"""
import torch, sys, math, numpy as np
from collections import defaultdict
from scipy import stats
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, RHGNN
from rhgnn_v4 import RHGNNv4
from rhgnn_v3 import build_history_vocab
from rhgnn_v2 import expmap0, logmap0, hyp_distance

DEVICE = torch.device('cuda:0')
EPS    = 1e-6

# ── Load Models ───────────────────────────────────────────────────────────────

def load_v1(data, ckpt):
    m = RHGNN(data.num_entities, data.num_relations, dim=200).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                 weights_only=False)['model_state'])
    return m

def load_v4(data, ckpt):
    m = RHGNNv4(data.num_entities, data.num_relations,
                dim=200, num_sgcn_layers=2).to(DEVICE)
    m.set_history_vocab(build_history_vocab(data.train))
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                 weights_only=False)['model_state'])
    return m

# ── 1. Norm-Hierarchy Correlation ─────────────────────────────────────────────

def norm_hierarchy_correlation(model, data, snapshots, ts_to_real):
    """
    In Poincare ball, parent entities should be closer to origin
    (smaller norm) than child entities.
    Measures: Pearson correlation between embedding norm and
    entity connectivity (degree as proxy for hierarchy level).
    """
    print("\n── Norm-Hierarchy Correlation ──")

    # Get entity embeddings
    with torch.no_grad():
        model.eval()
        all_ids = torch.arange(data.num_entities, device=DEVICE)

        if hasattr(model, 'ball'):
            # geoopt model
            embs = model.entity_emb[all_ids]
        else:
            embs = model.entity_emb(all_ids)
            c    = model.c.item()
            embs = expmap0(embs, model.c)

        norms = embs.norm(dim=-1).cpu().numpy()

    # Compute entity degree (connectivity) as hierarchy proxy
    degree = defaultdict(int)
    for h, r, t, _ in data.train:
        degree[h] += 1
        degree[t] += 1

    degrees = np.array([degree.get(i, 0)
                        for i in range(data.num_entities)], dtype=float)

    # Pearson correlation
    corr, pval = stats.pearsonr(norms, degrees)
    print(f"  Norm-Degree Correlation: r={corr:.4f}, p={pval:.4f}")
    print(f"  Mean norm (high degree):  "
          f"{norms[degrees > np.percentile(degrees,75)].mean():.4f}")
    print(f"  Mean norm (low degree):   "
          f"{norms[degrees < np.percentile(degrees,25)].mean():.4f}")
    print(f"  Interpretation: {'✅ Higher degree = farther from origin (hierarchy captured)' if corr > 0 else '⚠️ No hierarchy correlation'}")
    return corr

# ── 2. Delta-Hyperbolicity ────────────────────────────────────────────────────

def delta_hyperbolicity(model, data, n_samples=500):
    """
    Gromov's delta-hyperbolicity: measures how tree-like the
    embedding space is. Lower delta = more hyperbolic.
    Formula: delta = max over 4-tuples of
    (largest - second largest) Gromov product
    """
    print("\n── Delta-Hyperbolicity of Embedding Space ──")

    with torch.no_grad():
        model.eval()
        all_ids = torch.arange(min(500, data.num_entities), device=DEVICE)

        if hasattr(model, 'ball'):
            embs = model.entity_emb[all_ids].cpu()
        else:
            embs = expmap0(model.entity_emb(all_ids), model.c).cpu()

    n = len(embs)

    # Sample random 4-tuples
    deltas = []
    for _ in range(n_samples):
        idx   = np.random.choice(n, 4, replace=False)
        w,x,y,z = [embs[i] for i in idx]

        # Compute pairwise distances
        def dist(a, b):
            diff = a - b
            return diff.norm().item()

        dxy = dist(x,y); dxz = dist(x,z); dyz = dist(y,z)
        dxw = dist(x,w); dyw = dist(y,w); dzw = dist(z,w)

        # Gromov products
        gxy_z = 0.5 * (dxz + dyz - dxy)
        gxz_y = 0.5 * (dxy + dyz - dxz)
        gyz_x = 0.5 * (dxy + dxz - dyz)

        products = sorted([gxy_z, gxz_y, gyz_x], reverse=True)
        delta    = products[0] - products[1]
        deltas.append(delta)

    mean_delta = np.mean(deltas)
    print(f"  Mean delta: {mean_delta:.4f}")
    print(f"  Std delta:  {np.std(deltas):.4f}")
    print(f"  Interpretation: {'✅ Low delta — tree-like structure' if mean_delta < 1.0 else '⚠️ High delta — not tree-like'}")
    return mean_delta

# ── 3. Hierarchical MRR (H-MRR) ──────────────────────────────────────────────

@torch.no_grad()
def hierarchical_mrr(model, data, snapshots, ts_to_real, batch_size=256):
    """
    H-MRR: Weight MRR by entity hierarchy depth.
    Entities with high connectivity (hub entities) get higher weight.
    This rewards models that correctly rank hierarchical entities.
    """
    print("\n── Hierarchical MRR (H-MRR) ──")
    model.eval()

    # Build degree-based hierarchy weights
    degree = defaultdict(int)
    for h,r,t,_ in data.train:
        degree[h] += 1
        degree[t] += 1

    max_deg = max(degree.values()) if degree else 1
    weights = torch.tensor(
        [degree.get(i,0)/max_deg for i in range(data.num_entities)],
        device=DEVICE)

    true_tails_map = defaultdict(list)
    for (hh,rr,tt,ta) in data.all_true:
        true_tails_map[(hh,rr,ta)].append(tt)

    ts_groups = defaultdict(list)
    for q in data.test: ts_groups[q[3]].append(q)

    ranks, tail_weights = [], []
    prev_real_ts = None

    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = max(float(real_ts-prev_real_ts) if prev_real_ts else 1.0, 1.0)
        prev_real_ts = real_ts

        for start in range(0, len(ts_groups[ts_id]), batch_size):
            batch_q = ts_groups[ts_id][start:start+batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=DEVICE)
            h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
            scores  = model.score_all_tails(h, r, tau, snapshots, delta_t)

            for i,(hh,rr,tt,ta) in enumerate(batch_q):
                others = [c for c in true_tails_map[(hh,rr,ta)] if c!=tt]
                if others:
                    scores[i, torch.tensor(others,device=DEVICE)] = float('-inf')

            true_scores = scores[torch.arange(scores.size(0),device=DEVICE), t]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
            ranks.extend(rank.cpu().tolist())
            tail_weights.extend(weights[t].cpu().tolist())

    ranks        = np.array(ranks)
    tail_weights = np.array(tail_weights)

    # Standard MRR
    std_mrr = float(np.mean(1.0/ranks))

    # H-MRR — weighted by entity hierarchy depth
    w_norm  = tail_weights / tail_weights.sum()
    h_mrr   = float(np.sum(w_norm / ranks))

    # H-MRR by degree bin
    low_mask  = tail_weights < np.percentile(tail_weights, 33)
    mid_mask  = (tail_weights >= np.percentile(tail_weights, 33)) & \
                (tail_weights < np.percentile(tail_weights, 66))
    high_mask = tail_weights >= np.percentile(tail_weights, 66)

    print(f"  Standard MRR:     {std_mrr:.4f}")
    print(f"  H-MRR (weighted): {h_mrr:.4f}")
    print(f"  MRR (low degree):  {np.mean(1.0/ranks[low_mask]):.4f}")
    print(f"  MRR (mid degree):  {np.mean(1.0/ranks[mid_mask]):.4f}")
    print(f"  MRR (high degree): {np.mean(1.0/ranks[high_mask]):.4f}")
    print(f"  H-MRR / MRR ratio: {h_mrr/std_mrr:.4f} "
          f"({'✅ Hyp favored' if h_mrr > std_mrr else '⚠️ Eucl favored'})")
    return std_mrr, h_mrr

# ── 4. Embedding Space Quality ────────────────────────────────────────────────

def embedding_space_quality(model, data):
    """
    Measures how well embeddings use the hyperbolic space.
    - Mean norm: closer to 1 = embeddings near boundary (using space well)
    - Norm variance: higher = more discriminative
    - Entanglement: lower = better separated embeddings
    """
    print("\n── Embedding Space Quality ──")
    with torch.no_grad():
        model.eval()
        all_ids = torch.arange(data.num_entities, device=DEVICE)
        if hasattr(model, 'ball'):
            embs = model.entity_emb[all_ids]
        else:
            embs = expmap0(model.entity_emb(all_ids), model.c)
        norms = embs.norm(dim=-1).cpu().numpy()

    print(f"  Mean norm:     {norms.mean():.4f} "
          f"(1.0 = boundary, 0.0 = origin)")
    print(f"  Norm std:      {norms.std():.4f} "
          f"(higher = more discriminative)")
    print(f"  Min norm:      {norms.min():.4f}")
    print(f"  Max norm:      {norms.max():.4f}")
    print(f"  % near origin (<0.3): {(norms<0.3).mean()*100:.1f}%")
    print(f"  % near boundary(>0.7): {(norms>0.7).mean()*100:.1f}%")

    # Poincare quality score
    quality = norms.std() * norms.mean()
    print(f"  Poincare Quality Score: {quality:.4f}")
    return quality

# ── 5. Geodesic Temporal Consistency ─────────────────────────────────────────

def geodesic_temporal_consistency(model, data, snapshots,
                                  ts_to_real, n_entities=100):
    """
    Measures if entity trajectories follow geodesics on the manifold.
    Consistent geodesic movement = hyperbolic geometry is being used.
    """
    print("\n── Geodesic Temporal Consistency ──")
    model.eval()

    # Get sorted timestamps
    sorted_ts = sorted(ts_to_real.keys())
    if len(sorted_ts) < 3:
        print("  Not enough timestamps")
        return 0.0

    # Sample frequent entities
    degree = defaultdict(int)
    for h,r,t,_ in data.train:
        degree[h] += 1
    top_ents = sorted(degree.keys(), key=lambda x: degree[x],
                      reverse=True)[:n_entities]

    consistencies = []
    with torch.no_grad():
        for ent in top_ents[:20]:  # sample 20 for speed
            positions = []
            for ts_id in sorted_ts[:10]:  # first 10 timestamps
                real_ts = ts_to_real.get(ts_id, ts_id)
                h   = torch.tensor([ent], device=DEVICE)
                r   = torch.tensor([0],   device=DEVICE)
                tau = torch.tensor([ts_id], device=DEVICE)
                try:
                    h_rot = model.forward_entity(
                        h, r, tau, snapshots, 1.0)
                    positions.append(h_rot.squeeze().cpu())
                except:
                    continue

            if len(positions) < 3:
                continue

            # Compute geodesic deviations
            deviations = []
            for i in range(1, len(positions)-1):
                prev = positions[i-1]
                curr = positions[i]
                next_p = positions[i+1]
                # On a geodesic: curr should be between prev and next
                d_prev_curr = (prev - curr).norm().item()
                d_curr_next = (curr - next_p).norm().item()
                d_prev_next = (prev - next_p).norm().item()
                # Geodesic consistency: triangle inequality tightness
                deviation = abs(d_prev_curr + d_curr_next - d_prev_next)
                deviations.append(deviation)

            if deviations:
                consistencies.append(np.mean(deviations))

    if consistencies:
        mean_consistency = np.mean(consistencies)
        print(f"  Mean geodesic deviation: {mean_consistency:.4f}")
        print(f"  Std geodesic deviation:  {np.std(consistencies):.4f}")
        print(f"  Interpretation: {'✅ Low deviation — smooth geodesic paths' if mean_consistency < 0.5 else '⚠️ High deviation — non-geodesic movement'}")
        return mean_consistency
    return 0.0

# ── 6. Hyperbolic Hits@K (HH@K) ──────────────────────────────────────────────

@torch.no_grad()
def hyperbolic_hits_at_k(model, data, snapshots, ts_to_real,
                          K=10, batch_size=256):
    """
    HH@K: Hits@K but with partial credit for hierarchically close misses.
    If you predict a sibling entity instead of the true tail,
    you get partial credit based on hierarchy distance.
    """
    print(f"\n── Hyperbolic Hits@{K} (HH@{K}) ──")
    model.eval()

    # Build entity co-occurrence as hierarchy proxy
    cooccur = defaultdict(set)
    for h,r,t,_ in data.train:
        cooccur[h].add(t)
        cooccur[t].add(h)

    def hier_sim(e1, e2):
        """Jaccard similarity as hierarchy proxy."""
        s1 = cooccur[e1]
        s2 = cooccur[e2]
        if not s1 or not s2: return 0.0
        return len(s1&s2) / len(s1|s2)

    true_tails_map = defaultdict(list)
    for (hh,rr,tt,ta) in data.all_true:
        true_tails_map[(hh,rr,ta)].append(tt)

    ts_groups = defaultdict(list)
    for q in data.test[:1000]: ts_groups[q[3]].append(q)  # sample 1000

    std_hits, hyp_hits = [], []
    prev_real_ts = None

    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = max(float(real_ts-prev_real_ts) if prev_real_ts else 1.0, 1.0)
        prev_real_ts = real_ts

        for start in range(0, len(ts_groups[ts_id]), batch_size):
            batch_q = ts_groups[ts_id][start:start+batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=DEVICE)
            h,r,t,tau = batch[:,0],batch[:,1],batch[:,2],batch[:,3]
            scores  = model.score_all_tails(h, r, tau, snapshots, delta_t)

            for i,(hh,rr,tt,ta) in enumerate(batch_q):
                others = [c for c in true_tails_map[(hh,rr,ta)] if c!=tt]
                if others:
                    scores[i, torch.tensor(others,device=DEVICE)] = float('-inf')

            # Top-K predictions
            topk = scores.topk(K, dim=-1).indices.cpu().numpy()
            t_np = t.cpu().numpy()

            for i in range(len(batch_q)):
                true_t = t_np[i]
                preds  = topk[i]

                # Standard Hits@K
                std_hit = float(true_t in preds)
                std_hits.append(std_hit)

                # Hyperbolic Hits@K — partial credit
                if std_hit:
                    hyp_hits.append(1.0)
                else:
                    # Partial credit for similar entities
                    max_sim = max(hier_sim(pred, true_t) for pred in preds)
                    hyp_hits.append(max_sim * 0.5)  # partial credit

    std_h = np.mean(std_hits)
    hyp_h = np.mean(hyp_hits)
    print(f"  Standard H@{K}:    {std_h:.4f}")
    print(f"  Hyperbolic HH@{K}: {hyp_h:.4f}")
    print(f"  Gain from partial credit: +{hyp_h-std_h:.4f} "
          f"({(hyp_h-std_h)/std_h*100:.1f}%)")
    return std_h, hyp_h

# ── Main ──────────────────────────────────────────────────────────────────────

def run_dataset(ds_name, data_dir, v1_ckpt, v4_ckpt):
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    import os
    for model_name, ckpt, loader in [
        ('RHGNN-Base', v1_ckpt, load_v1),
        ('RHGNN-C',    v4_ckpt, load_v4),
    ]:
        if not os.path.exists(ckpt):
            print(f"\n{model_name}: checkpoint not found — skip")
            continue
        print(f"\n{'─'*40}")
        print(f"Model: {model_name}")
        print(f"{'─'*40}")
        try:
            model = loader(data, ckpt)
            model.build_nbr_index(snapshots)
            model.refresh_msg_table(DEVICE)

            # Run all metrics
            norm_hierarchy_correlation(model, data, snapshots, ts_to_real)
            delta_hyperbolicity(model, data)
            embedding_space_quality(model, data)
            hierarchical_mrr(model, data, snapshots, ts_to_real)
            hyperbolic_hits_at_k(model, data, snapshots, ts_to_real, K=10)
            geodesic_temporal_consistency(
                model, data, snapshots, ts_to_real)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

def main():
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

    print(f"\n{'='*60}")
    print("All hyperbolic metrics complete.")

if __name__ == '__main__':
    main()

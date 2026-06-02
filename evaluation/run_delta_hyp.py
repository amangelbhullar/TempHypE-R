"""
Compute Gromov delta-hyperbolicity for all datasets
and generate scatter plot data: delta vs MRR
"""
import sys, random, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg

def compute_delta(edges, num_nodes, sample=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # build adjacency
    adj = defaultdict(set)
    for (h, r, t, _) in edges:
        adj[h].add(t)
        adj[t].add(h)

    nodes = list(range(num_nodes))
    if len(nodes) > sample:
        nodes = random.sample(nodes, sample)

    # BFS shortest paths
    def bfs(src):
        dist = {src: 0}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    print(f"  Computing BFS for {len(nodes)} nodes...", flush=True)
    dists = {}
    for i, n in enumerate(nodes):
        dists[n] = bfs(n)
        if i % 100 == 0:
            print(f"  BFS {i}/{len(nodes)}", flush=True)

    # Gromov 4-point condition
    def d(u, v):
        if u in dists and v in dists[u]:
            return dists[u][v]
        if v in dists and u in dists[v]:
            return dists[v][u]
        return 999

    deltas = []
    sample_quads = random.sample(nodes, min(50, len(nodes)))
    for i, a in enumerate(sample_quads):
        for b in sample_quads[i+1:]:
            for c in sample_quads:
                for dd in sample_quads:
                    s1 = d(a,b) + d(c,dd)
                    s2 = d(a,c) + d(b,dd)
                    s3 = d(a,dd) + d(b,c)
                    vals = sorted([s1, s2, s3], reverse=True)
                    deltas.append((vals[0] - vals[1]) / 2.0)

    return float(np.mean(deltas)) if deltas else 0.0

datasets = [
    ('WIKI',       'data/WIKI-clean',  0.4932),
    ('YAGO',       'data/YAGO-clean',  0.3726),
    ('ICEWS05-15', 'data/ICEWS05-15',  0.2372),
    ('ICEWS14',    'data/ICEWS14',     0.2793),
    ('ICEWS18',    'data/ICEWS18',     0.1289),
    ('GDELT',      'data/GDELT',       0.1118),
]

print("Dataset          | delta  | MRR    | Interpretation")
print("-" * 60)

results = []
for name, data_dir, mrr in datasets:
    print(f"\nProcessing {name}...", flush=True)
    data  = load_temporal_kg(data_dir)
    edges = data.train
    delta = compute_delta(edges, data.num_entities, sample=300)
    results.append((name, delta, mrr))
    interp = "high hierarchy" if delta < 0.3 else "moderate" if delta < 0.6 else "low hierarchy"
    print(f"{name:<16} | {delta:.4f} | {mrr:.4f} | {interp}")

print("\n\nFinal Table (for scatter plot):")
print("Dataset          | delta  | MRR    ")
print("-" * 40)
for name, delta, mrr in sorted(results, key=lambda x: x[1]):
    print(f"{name:<16} | {delta:.4f} | {mrr:.4f}")

# correlation
deltas = np.array([r[1] for r in results])
mrrs   = np.array([r[2] for r in results])
corr   = np.corrcoef(deltas, mrrs)[0,1]
print(f"\nPearson correlation (delta vs MRR): {corr:.4f}")
print("Negative correlation = lower delta (more hierarchical) → higher MRR")
print("Done.")

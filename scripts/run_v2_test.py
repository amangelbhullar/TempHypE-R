import torch, sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v2 import TempHypE-Rv2

device = torch.device('cuda:0')

datasets = [
    ('ICEWS14-v2', 'data/ICEWS14', 'checkpoints/rhgnn_v2_icews14.pt'),
]

for name, data_dir, ckpt_path in datasets:
    print(f"\nEvaluating {name}...", flush=True)
    data      = load_temporal_kg(data_dir)
    snapshots = build_snapshot_graphs(data.train)
    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts
    model = TempHypERFA(data.num_entities, data.num_relations, dim=200).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    m = evaluate(model, data.test, data.all_true,
                snapshots, ts_to_real, device, batch_size=512)
    print(f"{name} | MRR={m['MRR']:.4f} | H@1={m['Hits@1']:.4f} | "
          f"H@3={m['Hits@3']:.4f} | H@10={m['Hits@10']:.4f} | MAR={m['MAR']:.1f}")
print("\nDone.")

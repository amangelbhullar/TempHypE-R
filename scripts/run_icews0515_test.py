import torch, sys
sys.path.insert(0, '.')
from rhgnn_end_to_end import load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v4 import TempHypE-Rv4
from rhgnn_v3 import build_history_vocab

device = torch.device('cuda:0')
data      = load_temporal_kg('data/ICEWS05-15')
snapshots = build_snapshot_graphs(data.train)
ts_to_real = {}
for raw_ts in data.sorted_timestamps:
    tid = data.time2id.get(str(raw_ts), raw_ts)
    ts_to_real[tid] = raw_ts
model = TempHypERCA(data.num_entities, data.num_relations,
                dim=200, num_sgcn_layers=2).to(device)
model.set_history_vocab(build_history_vocab(data.train))
ckpt = torch.load('checkpoints/rhgnn_v4_icews0515.pt',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.build_nbr_index(snapshots)
model.refresh_msg_table(device)
m = evaluate(model, data.test, data.all_true,
            snapshots, ts_to_real, device, batch_size=512)
print(f"ICEWS05-15 s42 | MRR={m['MRR']:.4f} | H@1={m['Hits@1']:.4f} | "
      f"H@3={m['Hits@3']:.4f} | H@10={m['Hits@10']:.4f} | MAR={m['MAR']:.1f}")

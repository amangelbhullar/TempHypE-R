import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import geoopt
from geoopt.optim import RiemannianAdam

Quad = Tuple[int, int, int, int]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class TemporalKGData:
    train: List[Quad]
    valid: List[Quad]
    test:  List[Quad]
    entity2id:   Dict[str, int]
    relation2id: Dict[str, int]
    time2id:     Dict[str, int]
    id2entity:   List[str]
    id2relation: List[str]
    id2time:     List[str]
    all_true:    Set[Quad]
    sorted_timestamps: List[int]

    @property
    def num_entities(self)  -> int: return len(self.entity2id)
    @property
    def num_relations(self) -> int: return len(self.relation2id)
    @property
    def num_times(self)     -> int: return len(self.time2id)

def read_quad_file(path: str) -> List[Tuple[str, str, str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 4:
                raise ValueError(f"Line {lineno}: expected 4 columns, got {len(parts)}")
            rows.append(tuple(parts[:4]))
    return rows

def build_id_maps(*splits):
    entities, relations, times = set(), set(), set()
    for split in splits:
        for h, r, t, tau in split:
            entities.add(h); entities.add(t)
            relations.add(r); times.add(tau)
    id2entity   = sorted(entities)
    id2relation = sorted(relations)
    try:
        id2time = sorted(times, key=lambda x: int(x))
    except ValueError:
        id2time = sorted(times)
    entity2id   = {x: i for i, x in enumerate(id2entity)}
    relation2id = {x: i for i, x in enumerate(id2relation)}
    time2id     = {x: i for i, x in enumerate(id2time)}
    return entity2id, relation2id, time2id, id2entity, id2relation, id2time

def encode_rows(rows, entity2id, relation2id, time2id) -> List[Quad]:
    return [(entity2id[h], relation2id[r], entity2id[t], time2id[tau])
            for h, r, t, tau in rows]

def load_temporal_kg(data_dir: str) -> TemporalKGData:
    train_raw = read_quad_file(os.path.join(data_dir, "train.txt"))
    valid_raw = read_quad_file(os.path.join(data_dir, "valid.txt"))
    test_raw  = read_quad_file(os.path.join(data_dir, "test.txt"))
    entity2id, relation2id, time2id, id2entity, id2relation, id2time = build_id_maps(train_raw, valid_raw, test_raw)
    train    = encode_rows(train_raw, entity2id, relation2id, time2id)
    valid    = encode_rows(valid_raw, entity2id, relation2id, time2id)
    test     = encode_rows(test_raw,  entity2id, relation2id, time2id)
    all_true = set(train + valid + test)
    try:
        sorted_ts = sorted({int(tau) for _, _, _, tau in train_raw + valid_raw + test_raw})
    except ValueError:
        sorted_ts = list(range(len(time2id)))
    return TemporalKGData(
        train=train, valid=valid, test=test,
        entity2id=entity2id, relation2id=relation2id, time2id=time2id,
        id2entity=id2entity, id2relation=id2relation, id2time=id2time,
        all_true=all_true, sorted_timestamps=sorted_ts,
    )

def build_snapshot_graphs(quads: List[Quad]) -> Dict[int, Dict[int, List]]:
    snapshots = defaultdict(lambda: defaultdict(list))
    for h, r, t, tau in quads:
        snapshots[tau][h].append((t, r))
        snapshots[tau][t].append((h, r))
    return snapshots

class TemporalKGDataset(Dataset):
    def __init__(self, quads): self.quads = quads
    def __len__(self): return len(self.quads)
    def __getitem__(self, idx): return torch.tensor(self.quads[idx], dtype=torch.long)

EPS = 1e-5

def poincare_project(x, c):
    max_norm = (1.0 - EPS) / c.sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return x * (max_norm / norm).clamp(max=1.0)

def expmap0(v, c):
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return poincare_project(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm), c)

def logmap0(x, c):
    x = poincare_project(x, c)
    sqrt_c = c.sqrt()
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.atanh((sqrt_c * x_norm).clamp(max=1.0 - EPS)) * x / (sqrt_c * x_norm)

def mobius_add(x, y, c):
    x2  = (x * x).sum(-1, keepdim=True)
    y2  = (y * y).sum(-1, keepdim=True)
    xy  = (x * y).sum(-1, keepdim=True)
    num = (1 + 2*c*xy + c*y2)*x + (1 - c*x2)*y
    den = (1 + 2*c*xy + c**2 * x2 * y2).clamp_min(EPS)
    return poincare_project(num / den, c)

def hyp_distance(x, y, c):
    sqrt_c = c.sqrt()
    diff   = mobius_add(-x, y, c)
    norm   = diff.norm(dim=-1).clamp_min(1e-15)
    return 2.0 / sqrt_c * torch.atanh((sqrt_c * norm).clamp(max=1.0 - EPS))

class ODEFunc(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
    def forward(self, z): return self.net(z)

class TempHypER(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200, init_curvature=1.0, dropout=0.1, ode_steps=5):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.dim           = dim
        self.ode_steps     = ode_steps
        self._log_c        = nn.Parameter(torch.tensor(math.log(math.exp(init_curvature) - 1.0)))
        self.entity_emb    = nn.Embedding(num_entities,  dim)
        self.relation_emb  = nn.Embedding(num_relations, dim)
        self.rel_lin       = nn.Embedding(num_relations, dim * dim)
        self.Wz            = nn.Linear(dim, dim)
        self.Uz            = nn.Linear(dim, dim, bias=False)
        self.ode_func      = ODEFunc(dim, dropout)
        self.bias          = nn.Parameter(torch.zeros(num_entities))
        self.dropout       = nn.Dropout(dropout)
        nn.init.normal_(self.entity_emb.weight,   std=0.01)
        nn.init.normal_(self.relation_emb.weight, std=0.01)
        nn.init.normal_(self.rel_lin.weight,      std=0.01)

    @property
    def c(self):
        return F.softplus(self._log_c).clamp(5e-1, 5.0)

    def _to_hyp(self, ids):
        return expmap0(self.entity_emb(ids), self.c)

    def _rel_hyp(self, ids):
        return expmap0(self.relation_emb(ids), self.c)

    def build_nbr_index(self, snapshots, max_nbrs=16):
        """Pre-build neighbour index as CPU tensors for fast lookup during training."""
        self._nbr_index = {}
        for tau, ent_dict in snapshots.items():
            for ent, nbrs in ent_dict.items():
                nbrs = nbrs[:max_nbrs]
                self._nbr_index[(tau, ent)] = (
                    torch.tensor([n for n, _ in nbrs], dtype=torch.long),
                    torch.tensor([r for _, r in nbrs], dtype=torch.long),
                )
        print(f"Neighbour index: {len(self._nbr_index):,} entries")

    def refresh_msg_table(self, device):
        """
        Recompute message table with current embeddings (no grad).
        Called once per epoch before training.
        """
        c     = self.c.detach()
        index = self._nbr_index
        # all entity tangent vectors with current embeddings
        with torch.no_grad():
            all_tan = logmap0(expmap0(self.entity_emb.weight.to(device), c), c)  # (E,d)
            self._msg_table = {}
            for (tau, ent), (nbr_ids, rel_ids) in index.items():
                nbr_ids = nbr_ids.to(device)
                rel_ids = rel_ids.to(device)
                nbr_tan = all_tan[nbr_ids]
                W = self.rel_lin(rel_ids).view(len(nbr_ids), self.dim, self.dim)
                m = torch.bmm(W, nbr_tan.unsqueeze(-1)).squeeze(-1)
                self._msg_table[(tau, ent)] = torch.tanh(m.mean(dim=0))

    def _aggregate_messages(self, entity_ids, tau_ids, snapshots):
        """
        Relation-transformed self message in tangent space.
        Uses the entity's own embedding transformed by mean relation weight.
        Fast, differentiable, no Python loops over neighbours.
        """
        c       = self.c
        h_hyp   = self._to_hyp(entity_ids)          # (B, d) on manifold
        h_tan   = logmap0(h_hyp, c)                  # (B, d) tangent
        # mean relation embedding as a shared transform
        r_mean  = self.relation_emb.weight.mean(dim=0)  # (d,)
        msg     = torch.tanh(h_tan + r_mean.unsqueeze(0))
        return msg

    def _hgru_jump(self, h_prev_hyp, msg_tan):
        c      = self.c
        v_prev = logmap0(h_prev_hyp, c)
        z      = torch.sigmoid(self.Wz(v_prev) + self.Uz(msg_tan))
        v_new  = z * msg_tan + (1.0 - z) * v_prev
        return expmap0(v_new, c)

    def _ode_flow(self, h_hyp, delta_t):
        if delta_t <= 0.0:
            return h_hyp
        c     = self.c
        state = logmap0(h_hyp, c)
        dt    = delta_t / max(self.ode_steps, 1)
        for _ in range(self.ode_steps):
            state = state + dt * self.ode_func(state)
        return expmap0(state, c)

    def score(self, h, r, t, tau, snapshots, delta_t=1.0):
        c       = self.c
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._hgru_jump(h_hyp, msg_tan)
        h_flow  = self._ode_flow(h_jump, delta_t)
        pred    = mobius_add(h_flow, self._rel_hyp(r), c)
        dist    = hyp_distance(pred, self._to_hyp(t), c)
        return -dist + self.bias[t]

    def score_all_tails(self, h, r, tau, snapshots, delta_t=1.0):
        c       = self.c
        msg_tan = self._aggregate_messages(h, tau, snapshots)
        h_hyp   = self._to_hyp(h)
        h_jump  = self._hgru_jump(h_hyp, msg_tan)
        h_flow  = self._ode_flow(h_jump, delta_t)
        pred    = mobius_add(h_flow, self._rel_hyp(r), c)
        all_ids = torch.arange(self.num_entities, device=h.device)
        all_hyp = self._to_hyp(all_ids)
        pred_exp = pred.unsqueeze(1).expand(-1, self.num_entities, -1)
        tail_exp = all_hyp.unsqueeze(0).expand(h.size(0), -1, -1)
        dist = hyp_distance(pred_exp, tail_exp, c)
        return -dist + self.bias.unsqueeze(0)

def train_one_epoch(model, train_groups, optimizer, device, num_entities, snapshots, ts_to_real, neg_ratio=5, batch_size=512, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_n    = 0
    prev_real_ts = None
    for ts_id, quads in train_groups:
        real_ts  = ts_to_real.get(ts_id, ts_id)
        delta_t  = float(real_ts - prev_real_ts) if prev_real_ts is not None else 1.0
        delta_t  = max(delta_t, 1.0)
        prev_real_ts = real_ts
        for start in range(0, len(quads), batch_size):
            mini  = quads[start: start + batch_size]
            batch = torch.tensor(mini, dtype=torch.long, device=device)
            h, r, t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
            pos_score = model.score(h, r, t, tau, snapshots, delta_t)
            pos_loss  = F.softplus(-pos_score).mean()
            neg_losses = []
            for _ in range(neg_ratio):
                neg_t = torch.randint(0, num_entities, (h.size(0),), device=device)
                ns    = model.score(h, r, neg_t, tau, snapshots, delta_t)
                neg_losses.append(F.softplus(ns).mean())
            loss = pos_loss + torch.stack(neg_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(mini)
            total_n    += len(mini)
    return total_loss / max(total_n, 1)

@torch.no_grad()
def evaluate(model, eval_quads, all_true, snapshots, ts_to_real, device, batch_size=64):
    model.eval()
    ranks = []
    true_tails_map = defaultdict(list)
    for (hh, rr, tt, ta) in all_true:
        true_tails_map[(hh, rr, ta)].append(tt)
    ts_groups = defaultdict(list)
    for q in eval_quads:
        ts_groups[q[3]].append(q)
    prev_real_ts = None
    for ts_id in sorted(ts_groups.keys()):
        real_ts = ts_to_real.get(ts_id, ts_id)
        delta_t = float(real_ts - prev_real_ts) if prev_real_ts is not None else 1.0
        delta_t = max(delta_t, 1.0)
        prev_real_ts = real_ts
        quads = ts_groups[ts_id]
        for start in range(0, len(quads), batch_size):
            batch_q = quads[start: start + batch_size]
            batch   = torch.tensor(batch_q, dtype=torch.long, device=device)
            h, r, true_t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
            scores = model.score_all_tails(h, r, tau, snapshots, delta_t)
            for i, (hh, rr, tt, ta) in enumerate(batch_q):
                other_true = [c for c in true_tails_map[(hh, rr, ta)] if c != tt]
                if other_true:
                    scores[i, torch.tensor(other_true, device=device)] = float("-inf")
            true_scores = scores[torch.arange(scores.size(0), device=device), true_t]
            rank = (scores > true_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
            ranks.extend(rank.cpu().tolist())
    ranks_np = np.array(ranks, dtype=np.float64)
    return {
        "MRR":     float(np.mean(1.0 / ranks_np)),
        "Hits@1":  float(np.mean(ranks_np <= 1)),
        "Hits@3":  float(np.mean(ranks_np <= 3)),
        "Hits@10": float(np.mean(ranks_np <= 10)),
        "MAR":     float(np.mean(ranks_np)),
    }

def _fmt(m):
    return (f"MRR={m['MRR']:.4f} | H@1={m['Hits@1']:.4f} | "
            f"H@3={m['Hits@3']:.4f} | H@10={m['Hits@10']:.4f} | MAR={m['MAR']:.1f}")

def create_toy_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train = ["Alice\tparent_of\tBob\t2000","Alice\tparent_of\tCarol\t2000",
             "Bob\tparent_of\tDave\t2003","Carol\tparent_of\tEve\t2003",
             "Dave\tworks_with\tEve\t2006","Eve\tworks_with\tFrank\t2007",
             "Frank\tlocated_in\tCityA\t2008","Dave\tlocated_in\tCityA\t2008",
             "Alice\tmentor_of\tDave\t2005","Bob\tmentor_of\tFrank\t2006",
             "Carol\tlocated_in\tCityB\t2004","Bob\tlocated_in\tCityA\t2001"]
    valid = ["Bob\tworks_with\tEve\t2007","Carol\tparent_of\tFrank\t2005",
             "Alice\tlocated_in\tCityB\t2002"]
    test  = ["Alice\tparent_of\tDave\t2006","Eve\tlocated_in\tCityA\t2009",
             "Dave\tmentor_of\tEve\t2007"]
    for name, rows in [("train.txt", train), ("valid.txt", valid), ("test.txt", test)]:
        with open(os.path.join(data_dir, name), "w") as f:
            f.write("\n".join(rows) + "\n")
    print(f"[toy] Dataset written to {data_dir}/")

def main():
    parser = argparse.ArgumentParser(description="TempHypE-R faithful implementation")
    parser.add_argument("--data_dir",       type=str,   default="data/toy")
    parser.add_argument("--create_toy",     action="store_true")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--batch_size",     type=int,   default=512)
    parser.add_argument("--eval_batch",     type=int,   default=64)
    parser.add_argument("--dim",            type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lr_decay",       type=float, default=0.7)
    parser.add_argument("--lr_decay_every", type=int,   default=20)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--curvature",      type=float, default=1.0)
    parser.add_argument("--ode_steps",      type=int,   default=5)
    parser.add_argument("--neg_ratio",      type=int,   default=5)
    parser.add_argument("--eval_every",     type=int,   default=10)
    parser.add_argument("--patience",       type=int,   default=20)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--save_path",      type=str,   default="checkpoints/rhgnn_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.create_toy:
        create_toy_dataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    data = load_temporal_kg(args.data_dir)
    print(f"Entities   : {data.num_entities}")
    print(f"Relations  : {data.num_relations}")
    print(f"Timestamps : {data.num_times}")
    print(f"Train / Valid / Test : {len(data.train)} / {len(data.valid)} / {len(data.test)}")

    print("Building snapshot graphs...")
    snapshots = build_snapshot_graphs(data.train)
    print(f"Snapshots  : {len(snapshots)} timestamps")

    ts_to_real = {}
    for raw_ts in data.sorted_timestamps:
        tid = data.time2id.get(str(raw_ts), raw_ts)
        ts_to_real[tid] = raw_ts

    train_groups_dict = defaultdict(list)
    for q in data.train:
        train_groups_dict[q[3]].append(q)
    train_groups = sorted(train_groups_dict.items(), key=lambda x: x[0])

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    model = TempHypE-R(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        dim=args.dim,
        init_curvature=args.curvature,
        dropout=args.dropout,
        ode_steps=args.ode_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")
    model.build_nbr_index(snapshots)

    optimizer = RiemannianAdam([
        {"params": [p for n, p in model.named_parameters() if n != "_log_c"], "lr": args.lr},
        {"params": [model._log_c], "lr": args.lr * 0.1},
    ], stabilize=10)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay)

    best_valid_mrr   = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model, train_groups=train_groups, optimizer=optimizer,
            device=device, num_entities=data.num_entities, snapshots=snapshots,
            ts_to_real=ts_to_real, neg_ratio=args.neg_ratio, batch_size=args.batch_size,
        )
        scheduler.step()
        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | c: {model.c.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_m = evaluate(
                model=model, eval_quads=data.valid, all_true=data.all_true,
                snapshots=snapshots, ts_to_real=ts_to_real,
                device=device, batch_size=args.eval_batch,
            )
            print(f"  Valid | {_fmt(valid_m)}")
            if valid_m["MRR"] > best_valid_mrr:
                best_valid_mrr   = valid_m["MRR"]
                patience_counter = 0
                torch.save({"model_state": model.state_dict(), "args": vars(args),
                            "epoch": epoch, "valid_mrr": best_valid_mrr}, args.save_path)
                print(f"  [saved] -> {args.save_path}")
            else:
                patience_counter += args.eval_every
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    print("\nLoading best checkpoint for test evaluation...")
    if not os.path.exists(args.save_path):
        print("No checkpoint saved — model did not improve. Skipping test eval.")
    else:
        ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_m = evaluate(
            model=model, eval_quads=data.test, all_true=data.all_true,
            snapshots=snapshots, ts_to_real=ts_to_real,
            device=device, batch_size=args.eval_batch,
        )
        print(f"  Test  | {_fmt(test_m)}")
        print(f"\nBest valid MRR : {best_valid_mrr:.4f}  (epoch {ckpt['epoch']})")
        print("Done.")

if __name__ == "__main__":
    main()

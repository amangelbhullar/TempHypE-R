# RHGNN: Curvature-Aware Recurrent Hyperbolic GNNs for TKG Reasoning

> **Anonymous submission to CIKM 2026**  
> Paper: *Curvature-Aware Recurrent Hyperbolic GNNs for TKG Reasoning*

---

## Overview

This repository contains the official implementation of the **RHGNN model family** — a systematic suite of six models that progressively combine hyperbolic geometry, gated temporal memory, and continuous dynamics for temporal knowledge graph (TKG) extrapolation.

Each model adds exactly one component over the previous, enabling clean isolation of each design choice:

| Model | Key Addition | ICEWS14 MRR |
|-------|-------------|-------------|
| RHGNN-HLSTM (V1a) | H-LSTM + Neural ODE + Poincaré ball | 0.279 |
| RHGNN-HGRU (V1b) | H-GRU + temporal decay | 0.294 |
| RHGNN-FA (V2) | Frequency-aware embeddings | 0.311 |
| RHGNN-MA (V3) | 2-layer RGCN + historical vocabulary | 0.338 |
| **RHGNN-CA (V4)** | Per-relation curvature + contrastive loss | **0.344** |
| RHGNN-FH (V5) | Fully hyperbolic Möbius ops (geometry ablation) | 0.277 |

**Key contributions:**
- The RHGNN model family systematically isolating each design component
- Novel evaluation metrics: **H-MRR** and **δ-hyperbolicity** revealing standard MRR underestimates hierarchical reasoning by **1.57–1.90×**
- Finding: hybrid tangent-space approach outperforms fully Möbius operations while converging **3× faster**

---

## Repository Structure

```
RHGNN-Family/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── rhgnn_v1a.py               # RHGNN-HLSTM: H-LSTM + ODE baseline
│   ├── rhgnn_v1b.py               # RHGNN-HGRU: H-GRU + temporal decay
│   ├── rhgnn_v2.py                # RHGNN-FA: frequency-aware embeddings
│   ├── rhgnn_v3.py                # RHGNN-MA: subgraph + history vocab
│   ├── rhgnn_v4.py                # RHGNN-CA: full model (best)
│   ├── rhgnn_v5_pure.py           # RHGNN-FH: fully hyperbolic ablation
│   │
│   └── components/                # Shared architectural components
│       ├── __init__.py
│       ├── hyperbolic.py          # Poincaré ball ops (exp, log, Möbius)
│       ├── hgru.py                # Hyperbolic GRU cell
│       ├── neural_ode.py          # Neural ODE with Euler solver
│       ├── rgcn.py                # 2-layer RGCN message passing
│       ├── curvature.py           # Per-relation curvature module
│       └── scoring.py             # Geodesic scoring + copy mechanism
│
├── evaluation/                     # Evaluation metrics
│   ├── __init__.py
│   ├── standard_mrr.py            # Filtered MRR, Hits@K
│   ├── h_mrr.py                   # H-MRR: degree-weighted hierarchical MRR
│   └── delta_hyperbolicity.py     # Gromov δ-hyperbolicity of embeddings
│
├── data/                           # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py                 # TKG dataset loader
│   ├── preprocess.py              # Preprocessing scripts
│   └── README.md                  # Dataset download instructions
│
├── utils/                          # Training utilities
│   ├── __init__.py
│   ├── trainer.py                 # Training loop with RiemannianAdam
│   ├── losses.py                  # KL + InfoNCE + smoothness losses
│   ├── riemannian_adam.py         # RiemannianAdam optimizer
│   └── neighbor_cache.py          # Cached neighbor index builder
│
├── configs/                        # Hyperparameter configurations
│   ├── icews14.yaml
│   ├── icews18.yaml
│   ├── icews05_15.yaml
│   ├── wiki.yaml
│   ├── yago.yaml
│   └── gdelt.yaml
│
├── scripts/                        # Run scripts
│   ├── train.py                   # Main training entry point
│   ├── evaluate.py                # Evaluation on test set
│   ├── run_all_datasets.sh        # Reproduce all results
│   └── run_ablation.sh            # Reproduce ablation study
│
└── experiments/                    # Experiment logs and checkpoints
    └── README.md
```

---

## Installation

```bash
git clone https://github.com/clearsubmission/RHGNN-Family.git
cd RHGNN-Family
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA ≥ 11.3 (recommended)
- torchdiffeq
- geoopt (for RiemannianAdam)

```bash
pip install torch torchvision
pip install torchdiffeq geoopt
```

---

## Datasets

We evaluate on six TKG benchmarks:

| Dataset | Entities | Relations | Train | Valid | Test |
|---------|----------|-----------|-------|-------|------|
| ICEWS14 | 7,128 | 230 | 63,685 | 13,823 | 13,222 |
| ICEWS18 | 23,033 | 256 | 373,018 | 45,995 | 49,545 |
| ICEWS05-15 | 10,488 | 251 | 368,962 | 46,275 | 46,092 |
| WIKI | 12,554 | 24 | 539,286 | 67,538 | 63,110 |
| YAGO | 10,623 | 10 | 161,540 | 19,523 | 20,026 |
| GDELT | 7,691 | 240 | 2,735,685 | 341,961 | 341,961 |

Download and place datasets under `data/`:

```bash
# Datasets will be released upon paper acceptance
# Preprocessing:
python data/preprocess.py --dataset ICEWS14 --data_dir data/ICEWS14
```

---

## Training

**Train RHGNN-CA (V4, full model) on ICEWS14:**

```bash
python scripts/train.py \
    --model rhgnn_v4 \
    --dataset ICEWS14 \
    --data_dir data/ICEWS14 \
    --dim 200 \
    --epochs 500 \
    --batch_size 1024 \
    --lr 1e-3 \
    --lr_decay 0.8 \
    --lr_decay_every 50 \
    --patience 60 \
    --smooth_label 0.1 \
    --lambda_contrast 0.1 \
    --contrast_temp 0.07 \
    --lambda_smooth 0.01 \
    --ode_steps 5 \
    --seed 42 \
    --gpu 0
```

**Train all versions on all datasets:**

```bash
bash scripts/run_all_datasets.sh
```

**Using config files:**

```bash
python scripts/train.py --config configs/icews14.yaml --model rhgnn_v4
```

---

## Evaluation

**Evaluate on test set (standard MRR + H-MRR + δ-hyperbolicity):**

```bash
python scripts/evaluate.py \
    --model rhgnn_v4 \
    --checkpoint experiments/rhgnn_v4_icews14.pt \
    --dataset ICEWS14 \
    --data_dir data/ICEWS14 \
    --metrics mrr hits h_mrr delta_hyp nhc
```

**Reproduce ablation study:**

```bash
bash scripts/run_ablation.sh --dataset ICEWS14
bash scripts/run_ablation.sh --dataset YAGO
```

---

## Main Results

### Event-based datasets

| Model | ICEWS14 MRR | ICEWS18 MRR | ICEWS05-15 MRR |
|-------|------------|------------|----------------|
| RE-Net | .457 | .429 | .421 |
| CyGNet | .486 | .467 | .426 |
| CENET | .534 | .511 | .498 |
| **RHGNN-CA (Ours)** | **.344** | **.244** | **.422** |

### Hierarchical datasets

| Model | WIKI MRR | YAGO MRR |
|-------|---------|---------|
| HyTE | .254 | .144 |
| RE-Net | .520 | .652 |
| CENET | .684 | .841 |
| **RHGNN-CA (Ours)** | **.518** | **.562** |

### H-MRR reveals systematic underestimation

Standard MRR underestimates RHGNN-CA's hierarchical reasoning:

| Dataset | Standard MRR | H-MRR | Ratio |
|---------|-------------|-------|-------|
| ICEWS14 | 0.342 | 0.629 | **1.84×** |
| WIKI | 0.521 | 0.993 | **1.90×** |
| YAGO | 0.558 | 0.873 | **1.57×** |

---

## Novel Evaluation Metrics

### H-MRR (Hierarchical MRR)

Standard MRR treats all entities equally. H-MRR weights rankings by entity connectivity (degree as hierarchy proxy):

```
H-MRR = Σᵢ (wᵢ / ‖w‖₁) · (1/rankᵢ)
wᵢ = deg(tᵢ) / maxⱼ deg(j)
```

Hub entities (high degree, near Poincaré origin) receive higher weight, revealing that RHGNN-CA is **1.57–1.90× better** than standard MRR suggests.

**Compute H-MRR:**
```python
from evaluation.h_mrr import compute_h_mrr

h_mrr = compute_h_mrr(
    rankings=test_rankings,
    entity_degrees=dataset.entity_degrees
)
```

### δ-Hyperbolicity

Gromov's δ measures how tree-like the learned embedding space is (lower = more hyperbolic):

```python
from evaluation.delta_hyperbolicity import compute_delta_hyp

delta = compute_delta_hyp(embeddings, sample_size=500)
```

---

## Key Findings

**1. Hyperbolic geometry is the most critical component**

Removing it drops MRR by 70% on ICEWS14 and 93% on YAGO. The H-MRR/MRR ratio explodes from 1.84× to 3.62×, revealing that standard MRR severely underestimates its importance.

**2. Hybrid tangent-space beats fully Möbius (V5)**

RHGNN-FH (V5) replaces all operations with Möbius computations. Despite identical components, it converges 3× slower and achieves lower MRR on flat datasets (ICEWS14: 0.277 vs 0.344). Geometric benefits come from **manifold-constrained representations and geodesic scoring**, not curved computation.

**3. Per-relation curvature discovers geometry automatically**

- ICEWS14 (flat events): cᵣ → 0.05 (near-Euclidean)
- YAGO (taxonomic): cᵣ ≈ 0.7 (high curvature)

No manual tuning required.

---

## Hyperparameter Summary

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| `dim` | 200 | Embedding dimension |
| `lr` | 1e-3 | Learning rate (RiemannianAdam) |
| `lr_decay` | 0.8 | LR decay factor |
| `lr_decay_every` | 50 | Decay every N epochs |
| `patience` | 60 | Early stopping patience |
| `ode_steps` | 5 | Euler steps for Neural ODE |
| `smooth_label` | 0.1 | Label smoothing ε |
| `lambda_contrast` | 0.1 | InfoNCE loss weight |
| `contrast_temp` | 0.07 | Contrastive temperature τ_c |
| `lambda_smooth` | 0.01 | Smoothness regularizer λ_s |
| `batch_size` | 1024 | Training batch size |

---

## Citation

```bibtex
@inproceedings{anonymous2026rhgnn,
  title     = {Curvature-Aware Recurrent Hyperbolic {GNN}s for {TKG} Reasoning},
  author    = {Anonymous},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Knowledge and Information Management (CIKM 2026)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This code is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

We thank the authors of [geoopt](https://github.com/geoopt/geoopt) for the RiemannianAdam implementation and [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for the ODE solver. Experiments were conducted on NVIDIA H100 GPUs.

# Multi-Class Inductive Matrix Completion by Low-Rank Tensor Decomposition

Implementation of **multi-class Inductive Matrix Completion (IMC)** via Canonical Polyadic Decomposition (CPD).

This repository extends **Inductive Matrix Completion (IMC)** from regression / binary classification to **multi-class classification** by modeling the three-way interaction among *head entities*, *tail entities*, and *relation categories* as a latent tensor, factorized with **Canonical Polyadic Decomposition (CPD)**. The result is a feature-based (inductive) probabilistic classifier that predicts a full distribution over relation types for any entity pair — including entities never seen during training.

---

## Overview

Traditional IMC learns a low-rank map from feature space to a real-valued association matrix, and is limited to continuous or binary outputs. Many real-world relation-prediction tasks are categorical with a large label space (tags, knowledge-graph relation types, drug side effects). This work:

- Reformulates IMC as a **multi-class softmax** model via a CPD of a latent tensor `head × tail × relation` (when `#classes = 1` it reduces to classical IMC),
- Derives the full **gradient and Hessian** of the multi-class cross-entropy objective and solves it with an **alternating trust-region Newton** optimizer,
- Constructs **BFS-based inductive subgraph splits** of FB15k-237 with strictly *entity-disjoint* train/test sets,
- Encodes entities with a **prompt-based PLM pipeline** (RoBERTa-large, Qwen2.5-3B) using a `PoolEncoder` aggregation,
- Evaluates against classical classifiers, KGE models (DistMult / ComplEx / RotatE), and the GNN-based TyleR, plus a **conformal-prediction** analysis for calibrated uncertainty.

The method (**IMC**) achieves the best MRR and Hits@K on **every** setting tested, and produces the smallest valid conformal prediction sets.

### The core model

For an entity pair `(i, j)` with features `x_i, y_j`, the probability of relation `r` is

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{P}(A_{ij}=r|x_i,y_j)=\frac{\exp\!\left\{\sum_{l=1}^{L}(x_i^{\top}w_l)(y_j^{\top}h_l)c_{lr}\right\}}{\sum_{s=1}^{R}\exp\!\left\{\sum_{l=1}^{L}(x_i^{\top}w_l)(y_j^{\top}h_l)c_{ls}\right\}}" alt="softmax-CPD probability" />
</p>

The factor matrices `W`, `H`, `C` are optimized to minimize the multi-class cross-entropy plus L2 regularization, alternating over the three blocks with a trust-region Newton method (conjugate-gradient inner solver).

---

## Key results

Mean (± std) across 10 random seed splits. IMC outperforms all baselines on all eight settings; `LR-C` (logistic regression on concatenated features) is the strongest baseline.

| Dataset | IMC MRR | IMC Hits@1 | IMC Hits@3 | IMC Hits@10 |
|---|---|---|---|---|
| FB15k-237 v1 · Qwen | **.8138** ± .0197 | **.7048** ± .0227 | **.9105** ± .0233 | **.9748** ± .0128 |
| FB15k-237 v1 · RoBERTa | **.8040** ± .0261 | **.6849** ± .0369 | **.9110** ± .0194 | **.9772** ± .0131 |
| FB15k-237 v2 · Qwen | **.8034** ± .0105 | **.6792** ± .0136 | **.9180** ± .0106 | **.9765** ± .0045 |
| FB15k-237 v2 · RoBERTa | **.7922** ± .0092 | **.6671** ± .0128 | **.9047** ± .0126 | **.9717** ± .0081 |
| FB15k-237 v3 · Qwen | **.8280** ± .0097 | **.7249** ± .0121 | **.9263** ± .0083 | **.9826** ± .0045 |
| FB15k-237 v3 · RoBERTa | **.8152** ± .0090 | **.7100** ± .0117 | **.9074** ± .0114 | **.9745** ± .0046 |
| MovieLens 10M | **.4835** ± .0056 | **.3867** ± .0050 | **.5377** ± .0074 | **.6643** ± .0091 |
| TwoSides | **.3180** ± .0059 | **.2150** ± .0031 | **.3520** ± .0070 | **.5680** ± .0068 |

---

## Repository structure

```
EnhancedIMC/
├── IMC/                              # Core code (the IMC method + baselines)
│   ├── main-fb237.py                 # IMC training entry point: alternating TR-Newton + conformal prediction
│   ├── CategorialLoss.py             # Multi-class cross-entropy loss, gradient, and Hessian-vector products
│   ├── SparseRelationMatrix.py       # Sparse relation-matrix construction (CuPy)
│   ├── generate_plm_embeddings.py    # Prompt-based PLM feature extraction (PoolEncoder pipeline)
│   ├── baseline-fb237.py             # Classical baselines: FT, Logistic Regression, Random Forest, LightGBM
│   ├── kge-baseline-fb237.py         # KGE baselines: DistMult, ComplEx, RotatE (PLM-input)
│   ├── tyler_fb237.py                # TyleR (RGCN + PLM) baseline driver
│   ├── tyler_model.py                # TyleR classifier + trainer (RGCN, subgraph dataset, training loop)
│   ├── tyler_subgraph.py             # TyleR subgraph extraction (BFS + DRNL labeling → DGL)
│   ├── tyler/                        # TyleR RGCN model (layers, aggregators)
│   ├── tune-fb237-cross-seed.py      # Cross-seed hyperparameter grid search (selects by avg valid-MRR)
│   ├── tune-fb237.py                 # Single-seed tuning variant
│   ├── model_downloader.py           # Download PLMs locally via ModelScope
│   ├── compute_dataset_statistics.py # Dataset statistics
│   ├── data/                         # FB15k-237 BFS inductive splits (base + 10 seeds per version)
│   └── results/                      # Unified + summary result CSVs, tuning logs
│
├── new_seeds.py                      # One-click: regenerate splits + update seed lists + copy labels + embeddings
├── run_all.py                        # Run all methods across 10 seeds, compute mean ± std
├── run_one.py                        # Run all methods on a single version
└── reshuffle_splits.py               # BFS + shuffle + clean pipeline → multi-seed splits (the canonical splitter)
```

The FB15k-237 experiments live in `IMC/`. The MovieLens 10M and TwoSides experiments (which use the BLP/`bert-base-cased` and PubMedBERT pipelines) are implemented in a companion codebase (`OldIMC/`).

> `IMC/embeddings/` (generated PLM embeddings) and `IMC/models/` (downloaded PLM weights) are produced locally by the scripts below and excluded from this repo via `.gitignore`.

> **External references (not vendored in this repo).** The BFS split methodology follows [GraIL](https://arxiv.org/abs/1911.06962) (Teru et al., ICML 2020) and the prompt-based PLM encoding follows TyleR (De Bellis et al., EMNLP 2025). The pieces actually used at runtime are reimplemented here — `reshuffle_splits.py` (BFS + shuffle + clean), `IMC/tyler/` + `IMC/tyler_fb237.py` (TyleR model), and `generate_plm_embeddings.py` (prompt pipeline) — so the results reproduce without the upstream repos. The only missing piece is the raw FB15k-237 graph, which is needed solely to *regenerate* the BFS splits from scratch (see below).

---

## Environment & installation

The core IMC solver is written with **CuPy** for GPU-accelerated tensor operations; feature extraction and KGE baselines use **PyTorch**. The reference hardware is a 0.2× NVIDIA H20 (20 GB VRAM) with 16 GB RAM; peak memory stays under 6 GB VRAM.

| Requirement | Purpose |
|---|---|
| Python 3.12 | base environment |
| [CuPy](https://cupy.dev/) (`cupy-cuda12x`) | IMC loss/gradient/Hessian on GPU |
| [PyTorch](https://pytorch.org/) + torchvision | PLM feature extraction, KGE baselines |
| [transformers](https://huggingface.co/docs/transformers) (HuggingFace) | RoBERTa / Qwen model loading |
| scikit-learn, scipy, pandas, numpy | data handling, metrics, classifiers |
| lightgbm | LightGBM baseline |
| tqdm | progress bars |
| modelscope | local model download |
| DGL + PyTorch 2.3 (Python 3.8) | **TyleR baseline only** (separate `tyler` conda env) |
| matplotlib | figure generation |

Suggested setup (adjust the CUDA builds to your driver):

```bash
conda create -n imc python=3.12
conda activate imc

# CUDA-coupled packages first (match your driver/CUDA toolkit)
pip install torch==2.11 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install cupy-cuda12x==13.0.0

# Everything else
pip install -r requirements.txt
```

> The exact CuPy / PyTorch builds depend on your CUDA version — see the header of `requirements.txt` for alternatives.

The **TyleR** baseline requires DGL, which conflicts with some base-env libraries on Windows. It is run from a separate `tyler` environment:

```bash
conda create -n tyler python=3.8
conda activate tyler
conda install -c dglteam/label/th23_cu121 dgl
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

`run_all.py` automatically locates this environment (a `python.exe` under `envs/tyler`) and falls back to the active interpreter otherwise.

Download the PLMs locally (or let `generate_plm_embeddings.py` fetch them from HuggingFace):

```bash
cd IMC
python model_downloader.py --model qwen      # Qwen/Qwen2.5-3B
python model_downloader.py --model roberta   # FacebookAI/roberta-large
```

---

## Datasets

### FB15k-237 (three BFS-based inductive versions)

Starting from the full FB15k-237 graph (14,541 entities, 237 relations, 310,116 triples), we randomly select root nodes and run breadth-first search (≤ 3 hops) to extract a **training subgraph**; the **test subgraph** is extracted from the remaining masked graph, guaranteeing **zero entity overlap**. Edge subsampling controls density. For each of three increasing scales (`v1`, `v2`, `v3`) we generate **10 random seed splits**, and within each split the test triples are split 1:1 into validation and test (out-of-entity-pair ratio = 1.0).

| Dataset | #Relations | #Entities | #Train | #Valid | #Test | Out-of-entity-pair |
|---|---|---|---|---|---|---|
| FB15k-237 v1 | 142 | ≈1,093 | ≈1,993 | ≈206 | ≈205 | 1.0 |
| FB15k-237 v2 | 172 | ≈1,660 | ≈4,145 | ≈474 | ≈473 | 1.0 |
| FB15k-237 v3 | 183 | ≈2,501 | ≈7,406 | ≈866 | ≈865 | 1.0 |
| MovieLens 10M | 2,243 | — | 35,628 | 6,435 | 6,402 | 1.0 |
| TwoSides | 157 | — | 30,826 | 3,938 | 4,006 | 1.0 |

Each dataset directory under `IMC/data/` contains `train.txt`, `valid.txt`, `test.txt` (tab-separated `head \t relation \t tail`) plus metadata (`label.tsv`, `type.tsv`, `ontology.tsv`).

### MovieLens 10M and TwoSides

- **MovieLens 10M**: user–movie tag prediction (single-label, most frequent tag per pair); user features from aggregated tag text, movie features from titles/genres.
- **TwoSides**: drug–drug interaction / side-effect prediction from FAERS; features from RxNorm drug descriptions encoded with PubMedBERT.

These use the BLP framework (`bert-base-cased`, 64-dim embeddings) and are implemented in the companion `OldIMC/` codebase.

---

## Reproducing the experiments

### 1. (Optional) Generate the seed splits

The repository ships with the precomputed splits for seeds `7001–7010`. To regenerate them (or generate a fresh seed range), run from `EnhancedIMC/`:

```bash
# Regenerate splits for seeds 7001–7010 across all three versions
python new_seeds.py --base-seed 7001 --seeds 10
```

`new_seeds.py` performs the full pipeline in one shot: it generates BFS-based splits via `reshuffle_splits.py`, updates the seed lists in all scripts, copies label/type/ontology files, and generates PLM embeddings for the first seed (then copies them across the remaining seeds).

> **Note on split regeneration.** The precomputed BFS subgraphs are already shipped in `IMC/data/`, so the results reproduce out-of-the-box via `new_seeds.py` / `run_all.py` — no raw graph needed. Re-sampling the subgraph *from the raw full FB15k-237 graph* is outside the scope of this repo; the raw graph is available from the [GraIL repository](https://arxiv.org/abs/1911.06962) if needed.

### 2. Generate PLM entity embeddings

For a specific version/model (done automatically by `new_seeds.py` if needed):

```bash
cd IMC
python generate_plm_embeddings.py --version fb237_v1_ind --model qwen --aggregation sum
python generate_plm_embeddings.py --version fb237_v1_ind --model roberta --aggregation sum
```

Six prompt templates (e.g. *"[ENTITY] is a type of [MASK]."*) are fed through the PLM; the masked/final-token hidden states pass through a `PromptEncoder` (LayerNorm → Linear(·→128) → Dropout), are aggregated by the `PoolEncoder` (sum → ReLU → Dropout → Linear → sigmoid), and produce a **128-dimensional** entity embedding. `--aggregation` supports `sum`, `mean`, `concat`, `attn` (`sum` is used throughout).

### 3. Cross-seed hyperparameter tuning

For each `(version, model)` pair, run a grid search over `k` (latent rank), `lambda` (L2), and `bias` (bias dimensions) on 3 tuning seeds, selecting the configuration with the highest **average validation MRR**:

```bash
cd IMC
python tune-fb237-cross-seed.py --version fb237_v1_ind --model qwen
python tune-fb237-cross-seed.py --version fb237_v1_ind --model roberta
python tune-fb237-cross-seed.py --version fb237_v2_ind --model qwen
python tune-fb237-cross-seed.py --version fb237_v2_ind --model roberta
python tune-fb237-cross-seed.py --version fb237_v3_ind --model qwen
python tune-fb237-cross-seed.py --version fb237_v3_ind --model roberta
```

Default grid: `k ∈ {50,75,100,125,150,175,200}`, `lambda ∈ {1,5,10,50,100,500,1000}`, `bias ∈ {16,32,48,64,80,96}`, tuning seeds `{7001,7005,7009}`. Override with `--k_values`, `--lambda_values`, `--bias_values`, `--tune_seeds`. Results are written to `results/tune_cross_*_detail.csv` and `*_summary.csv`.

### 4. Run all methods on all 10 seeds

Using the selected hyperparameters, run the classical baselines, KGE models, IMC, and TyleR for a given `(version, model)`:

```bash
# from EnhancedIMC/
python run_all.py --version fb237_v1_ind --model qwen     --aggregation sum --k 150 --lambda 100 --bias 48
python run_all.py --version fb237_v1_ind --model roberta  --aggregation sum --k 200 --lambda 100 --bias 64
python run_all.py --version fb237_v2_ind --model qwen     --aggregation sum --k 200 --lambda 100 --bias 80
python run_all.py --version fb237_v2_ind --model roberta  --aggregation sum --k 150 --lambda 100 --bias 96
python run_all.py --version fb237_v3_ind --model qwen     --aggregation sum --k 200 --lambda 100 --bias 96
python run_all.py --version fb237_v3_ind --model roberta  --aggregation sum --k 200 --lambda 100 --bias 64
```

Each sub-script appends per-seed rows to `IMC/results/fb237_unified_results.csv`; after all seeds finish, `run_all.py` computes **mean ± std** into `IMC/results/fb237_summary_results.csv`. Already-completed experiments are skipped automatically (use `--no-skip` to force re-runs, `--summary-only` to only re-aggregate, `--dry-run` to preview the command list).

### 5. Selected hyperparameters

Chosen by cross-seed tuning (best average validation MRR on 3 seeds); `maxiter = 50` outer alternations, 5 trust-region Newton iterations per block, 10 CG iterations, initial radius `Δ₀ = 1.0`.

| Version | PLM | k (rank) | λ | bias | Max alt. iters |
|---|---|---|---|---|---|
| v1 | Qwen | 150 | 100 | 48 | 50 |
| v1 | RoBERTa | 200 | 100 | 64 | 50 |
| v2 | Qwen | 200 | 100 | 80 | 50 |
| v2 | RoBERTa | 150 | 100 | 96 | 50 |
| v3 | Qwen | 200 | 100 | 96 | 50 |
| v3 | RoBERTa | 200 | 100 | 64 | 50 |

For MovieLens 10M: `k = 400, λ = 100, bias = 32, maxiter = 20`. For TwoSides: `k = 50, λ = 100, bias = 2, maxiter = 50`.

---

## Method details

### Alternating trust-region Newton

The joint non-convex problem is split into three convex subproblems over `W`, `H`, and `C`. Each block is minimized by a **trust-region Newton method** whose quadratic subproblem is solved with **conjugate gradients** (Hessian-vector products are computed exactly and batched). This gives reliable monotonic convergence across all datasets.

### Bias-augmented features

Optionally, `bias` all-ones columns are appended to the feature matrix (`X → [X, 1_n^{(b)}]`), so the model learns head/tail **main effects** in addition to feature–feature interactions, while remaining entity-agnostic (hence inductive). See `main-fb237.py` for the exact logit expansion.

### Conformal prediction

Given a trained probabilistic model, split-conformal prediction sets are constructed from a calibration set (the validation split) with nonconformity score `s = 1 − P(r | h, t)`, giving finite-sample marginal coverage guarantees. IMC produces the smallest valid sets across all settings — often 2–10× smaller than the next-best baseline.

---

## Baselines

| Family | Methods | Notes |
|---|---|---|
| Feature translation | FT | TransE-style mean translation vectors (FB15k-237 only) |
| Classifiers | Logistic Regression, Random Forest, LightGBM × {concat, subtract} | head/tail feature concatenation or subtraction |
| KGE (adapted) | DistMult, ComplEx, RotatE | frozen PLM features + learnable linear projection into 512-dim KGE space; trained with cross-entropy |
| GNN-based | TyleR | RGCN encoder over PLM node features (FB15k-237 only, `tyler` env) |

---

## Evaluation metrics

- **Point prediction** — MRR, Hits@1, Hits@3, Hits@10 (rank of the true relation among all candidates).
- **Conformal prediction** — marginal coverage and average set size at significance levels `α ∈ {0.01, 0.05, 0.10}`.

---

## Acknowledgements

This work builds on and adapts code from:

- **GraIL** — *Inductive Relation Prediction by Subgraph Reasoning* (Teru et al., ICML 2020): BFS subgraph-sampling methodology.
- **TyleR** — *Type-Less yet Type-Aware Inductive Link Prediction* (De Bellis et al., EMNLP 2025): prompt-based PLM encoding pipeline and `PoolEncoder`.
- **BLP** — *BERT for Link Prediction* (Daza et al., 2021): feature pipeline for MovieLens/TwoSides.

---

## License

This repository is released for research and reproducibility purposes.

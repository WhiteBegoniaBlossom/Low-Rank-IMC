"""
KGE Baselines for IMC: DistMult, ComplEx, RotatE.

All three use PLM features as fixed input via a learnable linear projection layer
(analogous to IMC's W/H matrices), trained with cross-entropy loss for direct
comparability with IMC.

Usage:
  python kge-baseline-fb237.py --version fb237_v1_ind --model DistMult
  python kge-baseline-fb237.py --version fb237_v1_ind --model ComplEx
  python kge-baseline-fb237.py --version fb237_v1_ind --model RotatE
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from SparseRelationMatrix import create_sparse_relation_matrices

# ============================================================
# Data loading (same as main-fb237.py)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

V1_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V2_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V3_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
VERSIONS = [
    f"fb237_v1_ind_seed{s}" for s in V1_SEEDS
] + [
    f"fb237_v2_ind_seed{s}" for s in V2_SEEDS
] + [
    f"fb237_v3_ind_seed{s}" for s in V3_SEEDS
]


def load_fb237_data(version, model="roberta", aggregation="sum"):
    import pandas as pd

    version_dir = os.path.join(DATA_DIR, version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"Dataset directory not found: {version_dir}")

    train_triples = pd.read_csv(
        os.path.join(version_dir, "train.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    valid_triples = pd.read_csv(
        os.path.join(version_dir, "valid.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    test_triples = pd.read_csv(
        os.path.join(version_dir, "test.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )

    emb_path = os.path.join(EMBED_DIR, f"{version}_{model}_{aggregation}_embeddings.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. "
            f"Run generate_plm_embeddings.py first."
        )
    with open(emb_path, "rb") as f:
        node2emb = pickle.load(f)

    print(f"[{version}] Train: {len(train_triples)}, Valid: {len(valid_triples)}, "
          f"Test: {len(test_triples)}, Embeddings: {len(node2emb)}")
    return train_triples, valid_triples, test_triples, node2emb


def prepare_features_fb237(train_triples, valid_triples, test_triples, node2emb, random_seed=28):
    np.random.seed(random_seed)

    all_entities = set()
    for df in [train_triples, valid_triples, test_triples]:
        all_entities.update(df["head"].unique())
        all_entities.update(df["tail"].unique())

    print(f"Number of entities: {len(all_entities)}")

    missing = [e for e in all_entities if e not in node2emb]
    if missing:
        print(f"{len(missing)} entities missing in embeddings, filling with random vectors")
        emb_dim = next(iter(node2emb.values())).shape[0]
        for e in missing:
            node2emb[e] = (np.random.randn(emb_dim).astype(np.float32) * 0.1)

    entity_list = sorted(all_entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_list)}

    emb_dim = len(next(iter(node2emb.values())))
    X_features = np.zeros((len(entity_list), emb_dim), dtype=np.float32)
    for e, idx in entity_to_idx.items():
        X_features[idx] = node2emb[e]

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    print(f"Feature matrix shape: {X_features.shape}")
    return entity_to_idx, X_features


# ============================================================
# Triple Dataset for training
# ============================================================

class TripleDataset(Dataset):
    def __init__(self, triples, entity_to_idx, relation_encoder):
        self.heads = []
        self.relations = []
        self.tails = []

        for _, row in triples.iterrows():
            if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
                self.heads.append(entity_to_idx[row['head']])
                self.tails.append(entity_to_idx[row['tail']])
                self.relations.append(relation_encoder.transform([row['relation']])[0])

        self.heads = torch.tensor(self.heads, dtype=torch.long)
        self.tails = torch.tensor(self.tails, dtype=torch.long)
        self.relations = torch.tensor(self.relations, dtype=torch.long)
        self.num_entities = len(entity_to_idx)

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx], self.relations[idx], self.tails[idx]


# ============================================================
# PLM-Input KGE Models (PLM features as fixed input, not learned embeddings)
# ============================================================

class PLMInputBaseKGE(nn.Module):
    """Base class for KGE models with a learnable linear projection from PLM
    feature space to KGE embedding space.

    This is the fair analogue to IMC's W and H matrices: IMC learns
    W (d×k) and H (k×d) to project entity features into a latent interaction
    space; the KGE baselines learn a single shared Linear(d→kge_dim) to
    project entity features into the KGE scoring space.  Without this
    projection, raw 4096-dim PLM features cannot effectively participate in
    bilinear scoring functions designed for low-dimensional learned embeddings.
    """
    def __init__(self, plm_features, kge_dim, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('plm_features', plm_features)
        self.plm_dim = plm_features.shape[1]
        self.kge_dim = kge_dim
        self.entity_proj = nn.Linear(self.plm_dim, kge_dim)
        nn.init.xavier_uniform_(self.entity_proj.weight)

    def _get_features(self, head, tail):
        h_raw = self.dropout(self.plm_features[head])
        t_raw = self.dropout(self.plm_features[tail])
        h = self.entity_proj(h_raw)
        t = self.entity_proj(t_raw)
        return h, t

    def forward(self, head, tail):
        raise NotImplementedError


class PLMInputDistMult(PLMInputBaseKGE):
    def __init__(self, plm_features, num_relations, dim=None, dropout=0.3):
        kge_dim = dim if dim is not None else plm_features.shape[1]
        super().__init__(plm_features, kge_dim, dropout=dropout)
        self.rel_emb = nn.Embedding(num_relations, self.kge_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, head, tail):
        h, t = self._get_features(head, tail)
        return (h * t) @ self.rel_emb.weight.T   # (batch, num_relations)


class PLMInputComplEx(PLMInputBaseKGE):
    def __init__(self, plm_features, num_relations, dim=None, dropout=0.3):
        kge_dim = dim if dim is not None else plm_features.shape[1]
        super().__init__(plm_features, kge_dim, dropout=dropout)
        self.half = self.kge_dim // 2
        self.rel_emb_re = nn.Embedding(num_relations, self.half)
        self.rel_emb_im = nn.Embedding(num_relations, self.half)
        nn.init.xavier_uniform_(self.rel_emb_re.weight)
        nn.init.xavier_uniform_(self.rel_emb_im.weight)

    def forward(self, head, tail):
        h, t = self._get_features(head, tail)
        h_re, h_im = h[:, :self.half], h[:, self.half:2*self.half]
        t_re, t_im = t[:, :self.half], t[:, self.half:2*self.half]
        R_re = self.rel_emb_re.weight      # (num_relations, half)
        R_im = self.rel_emb_im.weight
        scores = (h_re * t_re) @ R_re.T \
               - (h_im * t_im) @ R_re.T \
               + (h_re * t_im) @ R_im.T \
               + (h_im * t_re) @ R_im.T
        return scores                      # (batch, num_relations)


class PLMInputRotatE(PLMInputBaseKGE):
    def __init__(self, plm_features, num_relations, dim=None, dropout=0.3):
        kge_dim = dim if dim is not None else plm_features.shape[1]
        super().__init__(plm_features, kge_dim, dropout=dropout)
        self.half = self.kge_dim // 2
        self.rel_phase = nn.Embedding(num_relations, self.half)
        nn.init.uniform_(self.rel_phase.weight, a=-np.pi, b=np.pi)

    def forward(self, head, tail):
        h, t = self._get_features(head, tail)
        h_re, h_im = h[:, :self.half], h[:, self.half:2*self.half]
        t_re, t_im = t[:, :self.half], t[:, self.half:2*self.half]
        phase = self.rel_phase.weight       # (num_relations, half)
        R_re = torch.cos(phase)
        R_im = torch.sin(phase)
        scores = (h_re * t_re) @ R_re.T \
               - (h_im * t_im) @ R_re.T \
               + (h_re * t_im) @ R_im.T \
               + (h_im * t_re) @ R_im.T
        return scores


MODEL_CLASSES = {
    "DistMult": PLMInputDistMult,
    "ComplEx": PLMInputComplEx,
    "RotatE": PLMInputRotatE,
}


# ============================================================
# Training
# ============================================================

def train_kge(model, train_triples, entity_to_idx, relation_encoder,
              valid_triples=None, test_triples=None,
              epochs=100, batch_size=1024, lr=1e-3, weight_decay=0.0,
              device="cuda", eval_every=10, version="", save_best=True,
              lr_patience=15, lr_factor=0.5):
    """Train a PLM-input KGE model with cross-entropy loss (same loss as IMC)."""
    model = model.to(device)

    dataset = TripleDataset(train_triples, entity_to_idx, relation_encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=lr_patience,
        min_lr=1e-6, verbose=True)

    best_valid_mrr = -1.0
    best_state = None
    best_epoch = 0
    history = {"epoch": [], "loss": [], "valid_mrr": [], "valid_hits1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_heads, batch_rels, batch_tails in dataloader:
            batch_heads = batch_heads.to(device)
            batch_rels = batch_rels.to(device)
            batch_tails = batch_tails.to(device)

            scores = model(batch_heads, batch_tails)  # (batch, num_relations)
            loss = F.cross_entropy(scores, batch_rels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)

        print(f"[{version}] Epoch {epoch:3d}/{epochs} | loss={avg_loss:.6f}", end="")

        if valid_triples is not None and (epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            valid_mrr, valid_hits, _ = evaluate_relation_prediction(
                model, valid_triples, entity_to_idx, relation_encoder, device, verbose=False)
            history["valid_mrr"].append(valid_mrr)
            history["valid_hits1"].append(valid_hits["Hits@1"])
            print(f" | valid MRR={valid_mrr:.4f} Hits@1={valid_hits['Hits@1']:.4f}", end="")

            scheduler.step(valid_mrr)

            if save_best and valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
        else:
            history["valid_mrr"].append(0)
            history["valid_hits1"].append(0)

        print()

    if best_state is not None:
        print(f"[{version}] Loading best model from epoch {best_epoch} (valid MRR={best_valid_mrr:.4f})")
        model.load_state_dict(best_state)

    return model, history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_relation_prediction(model, triples, entity_to_idx, relation_encoder,
                                 device, verbose=True, name="Test"):
    model.eval()
    heads = []
    tails = []
    true_rels = []

    for _, row in triples.iterrows():
        if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
            heads.append(entity_to_idx[row['head']])
            tails.append(entity_to_idx[row['tail']])
            true_rels.append(relation_encoder.transform([row['relation']])[0])

    if not heads:
        return 0.0, {"Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    heads = torch.tensor(heads, dtype=torch.long, device=device)
    tails = torch.tensor(tails, dtype=torch.long, device=device)
    true_rels = torch.tensor(true_rels, dtype=torch.long)
    num_relations = len(relation_encoder.classes_)

    all_ranks = []
    all_preds = []
    eval_batch = 256

    for i in range(0, len(heads), eval_batch):
        h_batch = heads[i:i + eval_batch]
        t_batch = tails[i:i + eval_batch]
        true_batch = true_rels[i:i + eval_batch]

        scores = model(h_batch, t_batch)  # (batch, num_relations)

        _, sorted_idx = torch.sort(scores, descending=True, dim=1)
        preds = sorted_idx[:, 0].cpu()

        for j in range(len(h_batch)):
            rank = (sorted_idx[j] == true_batch[j].to(device)).nonzero(as_tuple=True)[0].item() + 1
            all_ranks.append(rank)
        all_preds.extend(preds.tolist())

    mrr = float(np.mean([1.0 / r for r in all_ranks]))
    hits = {}
    for k in [1, 3, 10]:
        hits[f"Hits@{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in all_ranks]))

    acc = accuracy_score(true_rels.tolist(), all_preds)

    if verbose:
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} MRR: {mrr:.4f}")
        for k, v in hits.items():
            print(f"{name} {k}: {v:.4f}")

    return mrr, hits, acc


# ============================================================
# Conformal prediction
# ============================================================

@torch.no_grad()
def kge_conformal_prediction(model, calib_triples, test_triples,
                             entity_to_idx, relation_encoder,
                             alpha_values, device):
    model.eval()
    num_relations = len(relation_encoder.classes_)

    def prepare_data(triples):
        heads, tails, labels = [], [], []
        for _, row in triples.iterrows():
            if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
                heads.append(entity_to_idx[row['head']])
                tails.append(entity_to_idx[row['tail']])
                labels.append(relation_encoder.transform([row['relation']])[0])
        return (torch.tensor(heads, dtype=torch.long, device=device),
                torch.tensor(tails, dtype=torch.long, device=device),
                np.array(labels))

    calib_h, calib_t, calib_labels = prepare_data(calib_triples)
    test_h, test_t, test_labels = prepare_data(test_triples)

    if len(calib_h) == 0 or len(test_h) == 0:
        return {}

    calib_scores = model(calib_h, calib_t)
    calib_probs = F.softmax(calib_scores, dim=1).cpu().numpy()

    test_scores = model(test_h, test_t)
    test_probs = F.softmax(test_scores, dim=1).cpu().numpy()

    results = {}
    for alpha in alpha_values:
        calib_nonconformity = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
        n_calib = len(calib_nonconformity)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_hat = np.quantile(calib_nonconformity, q_level, method='higher')

        pred_sets = [set(np.where(probs >= 1.0 - q_hat)[0]) for probs in test_probs]
        coverage = np.mean([lbl in ps for lbl, ps in zip(test_labels, pred_sets)])
        avg_size = np.mean([len(ps) for ps in pred_sets])

        print(f"  Alpha={alpha}: Coverage={coverage:.4f}, Avg Set Size={avg_size:.2f}")
        results[alpha] = {"coverage": float(coverage), "avg_set_size": float(avg_size)}

    return results


# ============================================================
# Run a single experiment
# ============================================================

def run_kge_experiment(version, model_name, model_args, aggregation,
                       random_seed, device, epochs, batch_size, lr, weight_decay, dim,
                       dropout, lr_patience=15, lr_factor=0.5):
    """Run a KGE experiment with PLM features projected via a learnable linear
    layer into KGE embedding space, then scored with the KGE scoring function.
    Only the projection layer and relation embeddings are learned — entity
    features themselves are frozen.  This is the fair analogue to IMC's
    learned W/H projection matrices.
    """
    tag = model_name
    print("\n" + "=" * 70)
    print(f"  {tag} on {version} ({model_args}/{aggregation})")
    print("=" * 70)

    train_triples, valid_triples, test_triples, node2emb = load_fb237_data(
        version, model=model_args, aggregation=aggregation)

    entity_to_idx, X_features = prepare_features_fb237(
        train_triples, valid_triples, test_triples, node2emb, random_seed)

    all_relations = sorted(set(train_triples['relation'].unique())
                          | set(valid_triples['relation'].unique())
                          | set(test_triples['relation'].unique()))
    R_train, relation_encoder, num_relations = create_sparse_relation_matrices(
        train_triples, entity_to_idx, all_relations=all_relations)

    num_entities = X_features.shape[0]
    print(f"Entities={num_entities}, Feature dim={X_features.shape[1]}, Relations={num_relations}")

    plm_features_tensor = torch.from_numpy(X_features.astype(np.float32))

    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(plm_features_tensor, num_relations, dim, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

    t0 = time.time()
    model, history = train_kge(
        model, train_triples, entity_to_idx, relation_encoder,
        valid_triples=valid_triples, test_triples=test_triples,
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        device=device, eval_every=5, version=f"{version}/{tag}",
        save_best=True, lr_patience=lr_patience, lr_factor=lr_factor
    )
    train_time = time.time() - t0

    model.eval()
    train_mrr, train_hits, train_acc = evaluate_relation_prediction(
        model, train_triples, entity_to_idx, relation_encoder, device, name="Train")
    valid_mrr, valid_hits, valid_acc = evaluate_relation_prediction(
        model, valid_triples, entity_to_idx, relation_encoder, device, name="Valid")
    test_mrr, test_hits, test_acc = evaluate_relation_prediction(
        model, test_triples, entity_to_idx, relation_encoder, device, name="Test")

    alpha_values = [0.01, 0.05, 0.1]
    conformal = kge_conformal_prediction(
        model, valid_triples, test_triples,
        entity_to_idx, relation_encoder, alpha_values, device)

    print(f"Training time: {train_time:.1f}s")

    return {
        "version": version,
        "model_name": tag,
        "plm_model": model_args,
        "aggregation": aggregation,
        "dim": dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "train_time_s": train_time,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "train_mrr": train_mrr,
        "train_hits": train_hits,
        "valid_mrr": valid_mrr,
        "valid_hits": valid_hits,
        "test_mrr": test_mrr,
        "test_hits": test_hits,
        "conformal": conformal,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="KGE Baselines for IMC: DistMult, ComplEx, RotatE (PLM-input only)")
    parser.add_argument("--version", type=str, default=None,
                        help="Specific version (e.g. fb237_v1)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 3 inductive versions")
    parser.add_argument("--model", type=str, default="DistMult",
                        choices=["DistMult", "ComplEx", "RotatE"],
                        help="KGE model")
    parser.add_argument("--plm_model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"],
                        help="PLM model for embeddings")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"])
    parser.add_argument("--dim", type=int, default=512,
                        help="Embedding dimension (total real params per entity)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization strength")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate on entity embeddings")
    parser.add_argument("--lr_patience", type=int, default=20,
                        help="LR scheduler patience (epochs without improvement)")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="LR scheduler reduction factor")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    args = parser.parse_args()

    if args.all:
        versions = VERSIONS
    elif args.version:
        versions = [args.version]
    else:
        print("Specify --version <name> or --all")
        return

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = []
    for version in versions:
        result = run_kge_experiment(
            version=version,
            model_name=args.model,
            model_args=args.plm_model,
            aggregation=args.aggregation,
            random_seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dim=args.dim,
            dropout=args.dropout,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
        )
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("KGE BASELINE SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"\n{r['version']} / {r['model_name']} ({r['plm_model']}/{r['aggregation']}):")
        print(f"  dim={r['dim']}, epochs={r['epochs']}, time={r['train_time_s']:.1f}s")
        print(f"  Test MRR={r['test_mrr']:.4f}, "
              f"Hits@1={r['test_hits']['Hits@1']:.4f}, "
              f"Hits@3={r['test_hits']['Hits@3']:.4f}, "
              f"Hits@10={r['test_hits']['Hits@10']:.4f}")
        print(f"  Valid MRR={r['valid_mrr']:.4f}, Train MRR={r['train_mrr']:.4f}")

    # Save results to unified CSV (shared with IMC and baseline)
    csv_path = os.path.join(os.path.dirname(__file__), "results", "fb237_unified_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            "method": r["model_name"],
            "model": r["plm_model"],
            "aggregation": r["aggregation"],
            "version": r["version"],
            "train_time_s": round(r["train_time_s"], 1),
            "train_acc": round(r["train_acc"], 6),
            "valid_acc": round(r["valid_acc"], 6),
            "mrr": round(r["test_mrr"], 6),
            "hits_1": round(r["test_hits"]["Hits@1"], 6),
            "hits_3": round(r["test_hits"]["Hits@3"], 6),
            "hits_10": round(r["test_hits"]["Hits@10"], 6),
        }
        for alpha in [0.01, 0.05, 0.1]:
            if alpha in r.get("conformal", {}):
                cr = r["conformal"][alpha]
                row[f"conformal_cov_a{alpha}"] = round(cr["coverage"], 6)
                row[f"conformal_size_a{alpha}"] = round(cr["avg_set_size"], 4)
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        keys = df[["method", "model", "aggregation", "version"]].apply(tuple, axis=1).tolist()
        mask = existing[["method", "model", "aggregation", "version"]].apply(tuple, axis=1).isin(keys)
        existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    try:
        main()
        print("All KGE baseline runs complete.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

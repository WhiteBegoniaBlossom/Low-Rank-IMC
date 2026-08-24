"""
TyleR relation prediction on FB237 using IMC pre-computed PLM embeddings.

Ported from tyler-main/train.py + tyler-main/test_ranking_rel.py.
Uses IMC's saved embedding format ({version}_{model}_{aggregation}_embeddings.pkl).

Usage:
    python tyler_fb237.py --version fb237_v1_ind_seed1006 --model roberta --aggregation sum
    python tyler_fb237.py --version fb237_v1_ind --model roberta --aggregation sum
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import pandas as pd

# Ensure IMC/ is on the path so we can import tyler_model
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# All imports are relative to IMC/ directory (cwd)
from tyler_model import (
    process_files,
    load_imc_embeddings,
    TyleRClassifier,
    SubgraphDataset,
    Trainer,
    evaluate_relation_prediction,
)

# ============================================================
# Paths
# ============================================================

DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
EMBED_DIR = os.path.join(_SCRIPT_DIR, "embeddings")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

UNIFIED_CSV = os.path.join(RESULTS_DIR, "fb237_unified_results.csv")


# ============================================================
# Conformal Prediction for TyleR
# ============================================================

def tyler_conformal_prediction(model, valid_dataset, test_dataset,
                                alpha_values, device, batch_size=16):
    """
    Conformal prediction for TyleR relation prediction.

    Uses the validation set as calibration data to construct prediction sets
    for test triples. Nonconformity score = 1 - softmax_prob(true_relation).

    Args:
        model: trained TyleRClassifier
        valid_dataset: SubgraphDataset for validation triples (calibration)
        test_dataset: SubgraphDataset for test triples
        alpha_values: list of alpha values (e.g. [0.01, 0.05, 0.1])
        device: torch device
        batch_size: batch size for inference

    Returns:
        dict mapping alpha -> {'coverage': float, 'avg_set_size': float}
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tyler_model import collate_dgl_rel, move_batch_to_device_dgl_rel

    model.eval()

    def collect_probs_and_labels(dataset):
        """Collect softmax probabilities and true labels for a dataset."""
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_dgl_rel,
        )
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                data, rel_labels = move_batch_to_device_dgl_rel(batch, device)
                scores = model(data)  # [batch_size, num_rels]
                probs = F.softmax(scores, dim=1)  # softmax over relation dimension
                all_probs.append(probs.cpu().numpy())
                all_labels.append(rel_labels.cpu().numpy())
        return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

    print("\n--- Conformal Prediction ---")
    print("Collecting calibration probabilities (validation set)...")
    calib_probs, calib_labels = collect_probs_and_labels(valid_dataset)
    num_rels = calib_probs.shape[1]
    print(f"  Calibration set: {len(calib_labels)} triples, "
          f"{num_rels} relation classes")

    print("Collecting test probabilities...")
    test_probs, test_labels = collect_probs_and_labels(test_dataset)
    print(f"  Test set: {len(test_labels)} triples")

    results = {}
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}:")

        # Nonconformity scores: 1 - P(true_class)
        calib_scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
        n_calib = len(calib_scores)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_hat = np.quantile(calib_scores, q_level, method='higher')
        print(f"  Calibration set size: {n_calib}, Quantile: {q_hat:.4f}")

        # Build prediction sets
        prediction_sets = []
        for probs in test_probs:
            pred_set = set(np.where(probs >= 1.0 - q_hat)[0])
            prediction_sets.append(pred_set)

        # Evaluate
        coverage = np.mean([true_label in pred_set
                           for true_label, pred_set in zip(test_labels, prediction_sets)])
        avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])

        results[alpha] = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
        }
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Avg. Set Size: {avg_set_size:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TyleR relation prediction with IMC embeddings"
    )

    # Dataset
    parser.add_argument("--version", type=str, required=True,
                        help="Dataset version (e.g. fb237_v1_ind_seed1006)")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"],
                        help="PLM model used for embeddings")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"],
                        help="Embedding aggregation method")

    # Model architecture
    parser.add_argument("--hop", type=int, default=3,
                        help="Hops for subgraph extraction")
    parser.add_argument("--num_gcn_layers", type=int, default=3,
                        help="Number of RGCN layers")
    parser.add_argument("--emb_dim", type=int, default=32,
                        help="GNN embedding dimension")
    parser.add_argument("--attn_rel_emb_dim", type=int, default=32,
                        help="Attention relation embedding dimension")
    parser.add_argument("--num_bases", type=int, default=4,
                        help="Number of basis functions for RGCN weights")
    parser.add_argument("--gnn_agg_type", type=str, default="sum",
                        choices=["sum", "mlp", "gru"],
                        help="Aggregator type")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate in GNN")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Edge dropout rate")
    parser.add_argument("--has_attn", type=lambda x: x.lower() == 'true',
                        default=True,
                        help="Whether to use attention in RGCN")
    parser.add_argument("--add_ht_emb", type=lambda x: x.lower() == 'true',
                        default=True,
                        help="Concatenate head/tail embeddings with graph repr")
    parser.add_argument("--is_comp", type=str, default="sub",
                        choices=["mult", "sub"],
                        help="Composition operator")
    parser.add_argument("--add_traspose_rels", type=lambda x: x.lower() == 'true',
                        default=False,
                        help="Add symmetric inverse relations")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="L2 weight decay")
    parser.add_argument("--margin", type=float, default=10,
                        help="MarginRankingLoss margin")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed")

    # Misc
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Use CPU only")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.disable_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    print(f"Device: {device}")
    print(f"Version: {args.version}")
    print(f"Model: {args.model}, Aggregation: {args.aggregation}")
    print(f"Seed: {args.seed}")

    # ============================================================
    # Load data
    # ============================================================

    version_dir = os.path.join(DATA_DIR, args.version)
    if not os.path.isdir(version_dir):
        print(f"ERROR: Dataset directory not found: {version_dir}")
        return

    train_path = os.path.join(version_dir, "train.txt")
    valid_path = os.path.join(version_dir, "valid.txt")
    test_path = os.path.join(version_dir, "test.txt")

    for p in [train_path, valid_path, test_path]:
        if not os.path.exists(p):
            print(f"ERROR: Missing file: {p}")
            return

    print("\n--- Building graph ---")
    data_info = process_files(
        train_path, valid_path, test_path,
        add_traspose_rels=args.add_traspose_rels,
    )
    print(f"  Entities: {data_info['num_ents']}")
    print(f"  Relations: {data_info['num_rels']}")
    print(f"  Augmented relations: {data_info['aug_num_rels']}")
    print(f"  Train triples: {len(data_info['train_triples'])}")

    # ============================================================
    # Load IMC embeddings
    # ============================================================

    emb_path = os.path.join(
        EMBED_DIR, f"{args.version}_{args.model}_{args.aggregation}_embeddings.pkl"
    )
    print(f"\n--- Loading embeddings ---")
    print(f"  Path: {emb_path}")

    if not os.path.exists(emb_path):
        print(f"ERROR: Embedding file not found: {emb_path}")
        print(f"  Run generate_plm_embeddings.py first.")
        return

    entity_embedding_matrix = load_imc_embeddings(
        emb_path, data_info['entity2id']
    )
    sem_dim = entity_embedding_matrix.shape[1]
    print(f"  Embedding dimension: {sem_dim}")

    # ============================================================
    # Load valid/test triples
    # ============================================================

    def load_triples(path, entity2id, relation2id):
        """Load triples from file into integer ID array."""
        data = []
        skipped = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    if h in entity2id and t in entity2id and r in relation2id:
                        data.append([entity2id[h], entity2id[t], relation2id[r]])
                    else:
                        skipped += 1
        if skipped > 0:
            print(f"  Skipped {skipped} triples with unknown entities/relations")
        return np.array(data)

    train_triples = data_info['train_triples']
    valid_triples = load_triples(
        valid_path, data_info['entity2id'], data_info['relation2id']
    )
    test_triples = load_triples(
        test_path, data_info['entity2id'], data_info['relation2id']
    )
    print(f"  Valid triples: {len(valid_triples)}")
    print(f"  Test triples: {len(test_triples)}")

    # ============================================================
    # Build subgraph datasets
    # ============================================================

    print(f"\n--- Building train subgraphs ---")
    train_dataset = SubgraphDataset(
        data_info, train_triples,
        hop=args.hop, enclosing_sub_graph=True,
        max_nodes_per_hop=None, device=device,
    )

    print(f"\n--- Building valid subgraphs ---")
    valid_dataset = SubgraphDataset(
        data_info, valid_triples,
        hop=args.hop, enclosing_sub_graph=True,
        max_nodes_per_hop=None, device=device,
    )

    print(f"\n--- Building test subgraphs ---")
    test_dataset = SubgraphDataset(
        data_info, test_triples,
        hop=args.hop, enclosing_sub_graph=True,
        max_nodes_per_hop=None, device=device,
    )

    # ============================================================
    # Build model params
    # ============================================================

    n_feat_dim = train_dataset.n_feat_dim  # DRNL one-hot dimension only
    inp_dim = n_feat_dim + sem_dim  # DRNL + IMC embeddings (added by model)

    # Create params namespace (matching tyler's conventions)
    class Params:
        pass

    params = Params()
    params.inp_dim = inp_dim
    params.emb_dim = args.emb_dim
    params.attn_rel_emb_dim = args.attn_rel_emb_dim
    params.num_rels = data_info['num_rels']
    params.aug_num_rels = data_info['aug_num_rels']
    params.num_bases = args.num_bases
    params.num_gcn_layers = args.num_gcn_layers
    params.dropout = args.dropout
    params.edge_dropout = args.edge_dropout
    params.has_attn = args.has_attn
    params.is_comp = args.is_comp
    params.gnn_agg_type = args.gnn_agg_type
    params.add_ht_emb = args.add_ht_emb
    params.max_label_value = train_dataset.max_n_label
    params.device = device

    # Training params
    params.lr = args.lr
    params.l2 = args.l2
    params.margin = args.margin
    params.batch_size = args.batch_size
    params.num_workers = 0

    print(f"\n--- Model params ---")
    print(f"  inp_dim: {inp_dim} (DRNL + sem_dim={sem_dim})")
    print(f"  emb_dim: {args.emb_dim}")
    print(f"  num_gcn_layers: {args.num_gcn_layers}")
    print(f"  num_rels: {data_info['num_rels']}")
    print(f"  max_label_value: {train_dataset.max_n_label}")

    # ============================================================
    # Create model
    # ============================================================

    print(f"\n--- Creating model ---")
    model = TyleRClassifier(params, data_info['relation2id']).to(device)
    model.load_entity_embeddings(entity_embedding_matrix)
    print(model)

    # ============================================================
    # Train
    # ============================================================

    print(f"\n--- Training ---")
    exp_dir = os.path.join(_SCRIPT_DIR, "experiments",
                           f"tyler_{args.version}_{args.model}_{args.aggregation}")
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(params, model, train_dataset, valid_dataset)

    t_start = time.time()
    best_mrr = trainer.train(
        num_epochs=args.num_epochs,
        early_stop=args.early_stop,
        save_every=args.save_every,
        exp_dir=exp_dir,
    )
    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.1f}s, best valid MRR: {best_mrr:.4f}")

    # ============================================================
    # Conformal Prediction
    # ============================================================

    alpha_values = [0.01, 0.05, 0.1]
    conformal_results = tyler_conformal_prediction(
        model, valid_dataset, test_dataset,
        alpha_values, device, batch_size=args.batch_size,
    )

    # ============================================================
    # Evaluate on test set
    # ============================================================

    print(f"\n--- Test evaluation ---")
    test_metrics = evaluate_relation_prediction(
        model, data_info, test_triples,
        batch_size=args.batch_size,
        hop=args.hop,
        enclosing_sub_graph=True,
    )

    # Compute train accuracy
    model.eval()
    train_preds = []
    train_labels = []
    from torch.utils.data import DataLoader
    from tyler_model import collate_dgl_rel, move_batch_to_device_dgl_rel

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_dgl_rel,
    )
    with torch.no_grad():
        for batch in train_loader:
            data, rel_labels = move_batch_to_device_dgl_rel(batch, device)
            scores = model(data)
            train_preds += scores.argmax(dim=1).cpu().tolist()
            train_labels += rel_labels.cpu().tolist()
    train_acc = (np.array(train_preds) == np.array(train_labels)).mean()

    # Compute valid accuracy
    valid_acc = None
    result = trainer.validate()
    if result is not None:
        valid_acc = result['acc']

    # ============================================================
    # Save results
    # ============================================================

    print(f"\n--- Saving results ---")

    row = {
        'method': 'TyleR',
        'model': args.model,
        'aggregation': args.aggregation,
        'version': args.version,
        'train_time_s': round(train_time, 1),
        'train_acc': round(train_acc, 6),
        'valid_acc': round(valid_acc, 6) if valid_acc is not None else None,
        'mrr': round(test_metrics['mrr'], 6),
        'hits_1': round(test_metrics['hits_1'], 6),
        'hits_3': round(test_metrics['hits_3'], 6),
        'hits_10': round(test_metrics['hits_10'], 6),
    }
    for alpha in [0.01, 0.05, 0.1]:
        if alpha in conformal_results:
            cr = conformal_results[alpha]
            row[f'conformal_cov_a{alpha}'] = round(cr['coverage'], 6)
            row[f'conformal_size_a{alpha}'] = round(cr['avg_set_size'], 4)

    df = pd.DataFrame([row])

    if os.path.exists(UNIFIED_CSV):
        existing = pd.read_csv(UNIFIED_CSV)
        # Remove duplicate entry if exists
        keys = df[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).tolist()
        mask = existing[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).isin(keys)
        existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(UNIFIED_CSV, index=False)
    print(f"Results saved to {UNIFIED_CSV}")
    print(f"\nFinal test metrics: MRR={test_metrics['mrr']:.4f}, "
          f"Hits@1={test_metrics['hits_1']:.4f}, "
          f"Hits@3={test_metrics['hits_3']:.4f}, "
          f"Hits@10={test_metrics['hits_10']:.4f}")


if __name__ == "__main__":
    main()

"""
Grid search over k, lambda, bias for IMC on fb237 datasets.

Reuses data loading and IMC training from main-fb237.py. For each parameter
combination, trains IMC and records validation accuracy + test metrics.

Usage:
    # Small search on smallest dataset
    python tune-fb237.py --version fb237_v1

    # Custom search ranges
    python tune-fb237.py --version fb237_v1 --k_values 50,70,100 --lambda_values 500,1000,2000

    # Run on all 4 inductive versions
    python tune-fb237.py --all
"""
import os
import sys
import itertools
import argparse
import gc
import importlib.util
import cupy as cp
import numpy as np
import pandas as pd

# Import from main-fb237.py (hyphen in filename prevents direct import)
_main_path = os.path.join(os.path.dirname(__file__), "main-fb237.py")
_spec = importlib.util.spec_from_file_location("main_fb237", _main_path)
_m237 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m237)

load_fb237_data = _m237.load_fb237_data
prepare_features_fb237 = _m237.prepare_features_fb237
IMC = _m237.IMC
reconstruct_accuracy = _m237.reconstruct_accuracy
evaluate_on_subset = _m237.evaluate_on_subset
compute_mrr = _m237.compute_mrr
compute_hits_at_k = _m237.compute_hits_at_k
setup_gpu_environment = _m237.setup_gpu_environment
VERSIONS = _m237.VERSIONS

from SparseRelationMatrix import create_sparse_relation_matrices

# Default grid (kept small — user can override via CLI)
DEFAULT_K_VALUES = [50, 75, 100, 125, 150, 175, 200]
DEFAULT_LAMBDA_VALUES = [1, 5, 10, 50, 100, 500, 1000]
DEFAULT_BIAS_VALUES = [16, 32, 48, 64, 80, 96]


def train_one_config(version, model, aggregation, k, lambda_cat, bias,
                     random_seed=28, maxiter=50):
    """Train IMC with a single (k, lambda, bias) configuration.
    Returns dict of metrics, or None on failure."""
    print(f"\n{'='*60}")
    print(f"Config: k={k}, lambda={lambda_cat}, bias={bias}")
    print(f"{'='*60}")

    try:
        train_triples, valid_triples, test_triples, node2emb = load_fb237_data(
            version, model=model, aggregation=aggregation)
        entity_to_idx_base, X_features_base = prepare_features_fb237(
            train_triples, valid_triples, test_triples, node2emb, random_seed
        )

        all_relations = sorted(set(train_triples['relation'].unique())
                              | set(valid_triples['relation'].unique())
                              | set(test_triples['relation'].unique()))
        R_train, relation_encoder, num_relations = create_sparse_relation_matrices(
            train_triples, entity_to_idx_base, all_relations=all_relations
        )

        # Extend with bias dimensions
        if bias == 0:
            X_features = X_features_base
            entity_to_idx = entity_to_idx_base
        else:
            original_dim = X_features_base.shape[1]
            total_dim = original_dim + bias
            X_features_extended = cp.zeros((X_features_base.shape[0], total_dim), dtype=cp.float32)
            X_features_extended[:, :original_dim] = X_features_base
            X_features_extended[:, -bias:] = 1.0
            X_features = X_features_extended
            entity_to_idx = entity_to_idx_base

        W, H, C_tensor, predict_proba, train_time, msg, _, _, _ = IMC(
            R_train, X_features, X_features, k, lambda_cat, maxiter,
            C=num_relations,
            valid_triples=valid_triples, entity_to_idx=entity_to_idx,
            relation_encoder=relation_encoder, eval_interval=1,
            random_seed=random_seed
        )

        if cp.isnan(W).any().get() or cp.isnan(H).any().get() or cp.isnan(C_tensor).any().get():
            print(f"  FAILED: NaN detected")
            return {
                'k': k, 'lambda': lambda_cat, 'bias': bias,
                'train_time': train_time, 'convergence': msg,
                'valid_acc': None, 'test_acc': None,
                'valid_mrr': None, 'test_mrr': None,
                'hits_1': None, 'hits_3': None, 'hits_10': None,
                'train_acc': None,
            }

        valid_acc = evaluate_on_subset(
            valid_triples, W, H, C_tensor, entity_to_idx, relation_encoder, X_features, "Valid "
        )
        test_acc = evaluate_on_subset(
            test_triples, W, H, C_tensor, entity_to_idx, relation_encoder, X_features, "Test "
        )
        train_acc = reconstruct_accuracy(R_train, X_features, W, H, C_tensor)
        valid_mrr = compute_mrr(valid_triples, W, H, C_tensor, entity_to_idx, relation_encoder, X_features)
        test_mrr = compute_mrr(test_triples, W, H, C_tensor, entity_to_idx, relation_encoder, X_features)
        hits = compute_hits_at_k(test_triples, W, H, C_tensor, entity_to_idx, relation_encoder, X_features)

        result = {
            'k': k, 'lambda': lambda_cat, 'bias': bias,
            'train_time': round(train_time, 1),
            'convergence': msg,
            'train_acc': round(train_acc, 4),
            'valid_acc': round(valid_acc, 4),
            'test_acc': round(test_acc, 4),
            'valid_mrr': round(valid_mrr, 4),
            'test_mrr': round(test_mrr, 4),
            'hits_1': round(hits['Hits@1'], 4),
            'hits_3': round(hits['Hits@3'], 4),
            'hits_10': round(hits['Hits@10'], 4),
        }
        print(f"  valid_MRR={valid_mrr:.4f}, test_MRR={test_mrr:.4f}, valid_acc={valid_acc:.4f}")
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'k': k, 'lambda': lambda_cat, 'bias': bias,
            'train_time': 0, 'convergence': str(e),
            'valid_acc': None, 'test_acc': None,
            'valid_mrr': None, 'test_mrr': None,
            'hits_1': None, 'hits_3': None, 'hits_10': None,
            'train_acc': None,
        }

    finally:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


def grid_search(version, model, aggregation, k_values, lambda_values, bias_values,
                random_seed=28, maxiter=50):
    """Run grid search over all parameter combinations."""
    combinations = list(itertools.product(k_values, lambda_values, bias_values))
    total = len(combinations)
    print(f"\nGrid search on {version}: {total} combinations "
          f"(k={k_values}, lambda={lambda_values}, bias={bias_values})")

    results = []
    csv_path = os.path.join(os.path.dirname(__file__), "results",
                            f"tune_{version}_{model}_{aggregation}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    for idx, (k_val, lam_val, b_val) in enumerate(combinations):
        print(f"\n--- [{idx+1}/{total}] ---")
        result = train_one_config(
            version, model, aggregation,
            k=k_val, lambda_cat=lam_val, bias=b_val,
            random_seed=random_seed, maxiter=maxiter
        )
        result['model'] = model
        result['aggregation'] = aggregation
        result['version'] = version
        results.append(result)

        # Save incrementally
        df = pd.DataFrame(results)
        cols = ['model', 'aggregation', 'version', 'k', 'lambda', 'bias',
                'valid_acc', 'test_acc', 'train_acc', 'valid_mrr', 'test_mrr',
                'hits_1', 'hits_3', 'hits_10',
                'train_time', 'convergence']
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(csv_path, index=False)

        # Print best so far (by valid_MRR)
        valid_results = [r for r in results if r['valid_mrr'] is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: r['valid_mrr'])
            print(f"  Best so far: k={best['k']}, lambda={best['lambda']}, "
                  f"bias={best['bias']}, valid_MRR={best['valid_mrr']:.4f}, test_MRR={best['test_mrr']:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE: {version} ({model}/{aggregation})")
    print(f"{'='*60}")
    valid_results = [r for r in results if r['valid_mrr'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda r: r['valid_mrr'])
        print(f"Best config (by valid_MRR): k={best['k']}, lambda={best['lambda']}, bias={best['bias']}")
        print(f"  valid_MRR={best['valid_mrr']:.4f}, test_MRR={best['test_mrr']:.4f}, valid_acc={best['valid_acc']:.4f}")
    else:
        print("All runs failed.")

    print(f"\nSaved to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search k, lambda, bias for IMC on fb237"
    )
    parser.add_argument("--version", type=str, default="fb237_v1_ind",
                        help="Dataset version (default: fb237_v1_ind)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 3 inductive versions")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"])
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"])
    parser.add_argument("--k_values", type=str, default="50, 75, 100, 125, 150, 175, 200",
                        help="Comma-separated k values")
    parser.add_argument("--lambda_values", type=str, default="1, 5, 10, 50, 100, 500, 1000",
                        help="Comma-separated lambda values")
    parser.add_argument("--bias_values", type=str, default="16, 32, 48, 64, 80, 96",
                        help="Comma-separated bias values")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Max IMC outer iterations")
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    lambda_values = [float(x.strip()) for x in args.lambda_values.split(",")]
    bias_values = [int(x.strip()) for x in args.bias_values.split(",")]

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    pool = setup_gpu_environment()

    versions = VERSIONS if args.all else [args.version]

    try:
        for v in versions:
            grid_search(
                v, args.model, args.aggregation,
                k_values, lambda_values, bias_values,
                random_seed=args.seed, maxiter=args.maxiter
            )
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    finally:
        pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

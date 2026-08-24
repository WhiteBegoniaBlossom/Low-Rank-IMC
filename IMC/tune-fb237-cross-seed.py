"""
Cross-seed hyperparameter tuning for IMC on fb237 datasets.

Trains each (k, lambda, bias) configuration on MULTIPLE tuning seeds,
selects the best config by AVERAGE MRR across seeds. This produces
more robust parameters than single-seed tuning.

Usage:
    # Default: tune on 3 seeds
    python tune-fb237-cross-seed.py --version fb237_v1_ind --model qwen

    # Custom tuning seeds & search ranges
    python tune-fb237-cross-seed.py --version fb237_v1_ind --model roberta \\
        --tune_seeds 7001,7005,7008 \\
        --k_values 150,175,200,225,250 \\
        --lambda_values 30,50,75,100,150

    # Run on all 3 versions with both models
    python tune-fb237-cross-seed.py --all --model qwen
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

# Default grid (same as tune-fb237.py)
DEFAULT_K_VALUES = [50, 75, 100, 125, 150, 175, 200]
DEFAULT_LAMBDA_VALUES = [1, 5, 10, 50, 100, 500, 1000]
DEFAULT_BIAS_VALUES = [16, 32, 48, 64, 80, 96]

# Seeds held out for final evaluation — these are NEVER used during tuning
HELDOUT_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]

# Default tuning seeds — a small subset used for hyperparameter selection
DEFAULT_TUNE_SEEDS = [7001, 7005, 7008]


def train_one_config(version, model, aggregation, k, lambda_cat, bias,
                     random_seed=28, maxiter=50):
    """Train IMC with a single (k, lambda, bias) configuration on a single seed.
    Returns dict of metrics, or None on failure."""
    print(f"\n{'='*60}")
    print(f"Config: k={k}, lambda={lambda_cat}, bias={bias}, seed={random_seed}")
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
                'k': k, 'lambda': lambda_cat, 'bias': bias, 'seed': random_seed,
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
            'k': k, 'lambda': lambda_cat, 'bias': bias, 'seed': random_seed,
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
            'k': k, 'lambda': lambda_cat, 'bias': bias, 'seed': random_seed,
            'train_time': 0, 'convergence': str(e),
            'valid_acc': None, 'test_acc': None,
            'valid_mrr': None, 'test_mrr': None,
            'hits_1': None, 'hits_3': None, 'hits_10': None,
            'train_acc': None,
        }

    finally:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


def cross_seed_grid_search(version, model, aggregation,
                           k_values, lambda_values, bias_values,
                           tune_seeds, maxiter=50):
    """Run grid search over all param combinations, evaluate on multiple seeds.
    Selects best config by AVERAGE MRR across tuning seeds."""
    combinations = list(itertools.product(k_values, lambda_values, bias_values))
    total_combos = len(combinations)
    total_runs = total_combos * len(tune_seeds)
    print(f"\n{'='*70}")
    print(f"Cross-seed grid search on {version} ({model}/{aggregation})")
    print(f"  {total_combos} param combos × {len(tune_seeds)} seeds = {total_runs} runs")
    print(f"  Tuning seeds: {tune_seeds}")
    print(f"  k={k_values}")
    print(f"  lambda={lambda_values}")
    print(f"  bias={bias_values}")
    print(f"{'='*70}")

    # Per-seed detailed results
    detail_csv = os.path.join(os.path.dirname(__file__), "results",
                              f"tune_cross_{version}_{model}_{aggregation}_detail.csv")
    # Averaged summary (one row per config)
    summary_csv = os.path.join(os.path.dirname(__file__), "results",
                               f"tune_cross_{version}_{model}_{aggregation}_summary.csv")
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)

    all_details = []  # every single run

    for combo_idx, (k_val, lam_val, b_val) in enumerate(combinations):
        combo_results = []
        for seed in tune_seeds:
            run_idx = combo_idx * len(tune_seeds) + tune_seeds.index(seed) + 1
            seed_version = f"{version}_seed{seed}"
            print(f"\n--- [{run_idx}/{total_runs}] combo={combo_idx+1}/{total_combos}, "
                  f"k={k_val}, λ={lam_val}, bias={b_val}, seed={seed} ---")

            result = train_one_config(
                seed_version, model, aggregation,
                k=k_val, lambda_cat=lam_val, bias=b_val,
                random_seed=seed, maxiter=maxiter
            )
            result['model'] = model
            result['aggregation'] = aggregation
            result['version'] = seed_version
            all_details.append(result)
            combo_results.append(result)

            # Save details incrementally
            df_detail = pd.DataFrame(all_details)
            cols = ['model', 'aggregation', 'version', 'k', 'lambda', 'bias', 'seed',
                    'valid_acc', 'test_acc', 'train_acc', 'valid_mrr', 'test_mrr',
                    'hits_1', 'hits_3', 'hits_10',
                    'train_time', 'convergence']
            df_detail = df_detail[[c for c in cols if c in df_detail.columns]]
            df_detail.to_csv(detail_csv, index=False)

        # Compute and print average for this combo across seeds
        valid_results = [r for r in combo_results if r['valid_mrr'] is not None]
        if valid_results:
            avg_valid_mrr = np.mean([r['valid_mrr'] for r in valid_results])
            avg_test_mrr = np.mean([r['test_mrr'] for r in valid_results])
            avg_valid = np.mean([r['valid_acc'] for r in valid_results])
            print(f"  >> COMBO AVG across {len(valid_results)}/{len(tune_seeds)} seeds: "
                  f"valid_MRR={avg_valid_mrr:.4f}, test_MRR={avg_test_mrr:.4f}, valid_acc={avg_valid:.4f}")

        # Save averaged summary incrementally
        summary_rows = _build_summary(all_details, model, aggregation, version)
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            summary_cols = ['model', 'aggregation', 'version', 'k', 'lambda', 'bias',
                            'n_seeds', 'valid_mrr_mean', 'valid_mrr_std',
                            'test_mrr_mean', 'test_mrr_std',
                            'valid_acc_mean', 'valid_acc_std',
                            'test_acc_mean', 'test_acc_std',
                            'hits_1_mean', 'hits_1_std',
                            'hits_3_mean', 'hits_3_std',
                            'hits_10_mean', 'hits_10_std',
                            'train_acc_mean', 'train_time_mean']
            df_summary = df_summary[[c for c in summary_cols if c in df_summary.columns]]
            df_summary = df_summary.sort_values('valid_mrr_mean', ascending=False)
            df_summary.to_csv(summary_csv, index=False)

    # Final summary
    print(f"\n{'='*70}")
    print(f"CROSS-SEED GRID SEARCH COMPLETE: {version} ({model}/{aggregation})")
    print(f"{'='*70}")

    summary_rows = _build_summary(all_details, model, aggregation, version)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        best = max(summary_rows, key=lambda r: r['valid_mrr_mean'] if r['valid_mrr_mean'] is not None else -1)
        print(f"\nBest config (by avg valid_MRR across {len(tune_seeds)} seeds):")
        print(f"  k={best['k']}, lambda={best['lambda']}, bias={best['bias']}")
        print(f"  avg valid_MRR={best['valid_mrr_mean']:.4f} +/- {best['valid_mrr_std']:.4f}")
        print(f"  avg test_MRR={best['test_mrr_mean']:.4f} +/- {best['test_mrr_std']:.4f}")
        print(f"  avg valid_acc={best['valid_acc_mean']:.4f} +/- {best['valid_acc_std']:.4f}")
        print(f"  avg hits@1={best['hits_1_mean']:.4f} +/- {best['hits_1_std']:.4f}")

        # Print top-5 configs
        print(f"\nTop-5 configurations:")
        sorted_rows = sorted(summary_rows,
                            key=lambda r: r['valid_mrr_mean'] if r['valid_mrr_mean'] is not None else -1,
                            reverse=True)
        for i, row in enumerate(sorted_rows[:5]):
            print(f"  {i+1}. k={row['k']}, λ={row['lambda']}, bias={row['bias']}  "
                  f"valid_MRR={row['valid_mrr_mean']:.4f}+/-{row['valid_mrr_std']:.4f}  "
                  f"valid_acc={row['valid_acc_mean']:.4f}")

    print(f"\nDetail CSV: {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    return all_details


def _build_summary(all_details, model, aggregation, version):
    """Aggregate per-seed results into per-config summary rows."""
    df = pd.DataFrame(all_details)
    if df.empty:
        return []

    # Filter to only the target version/model/aggregation
    # version in df is like "fb237_v1_ind_seed7001" — match by prefix
    df = df[(df['model'] == model) & (df['aggregation'] == aggregation) & (df['version'].str.startswith(version + '_seed'))]

    group_cols = ['k', 'lambda', 'bias']
    metric_cols = ['valid_mrr', 'test_mrr', 'valid_acc', 'test_acc',
                   'hits_1', 'hits_3', 'hits_10',
                   'train_acc', 'train_time']

    rows = []
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['model'] = model
        row['aggregation'] = aggregation
        row['version'] = version
        row['n_seeds'] = group['seed'].nunique()

        for metric in metric_cols:
            if metric in group.columns:
                vals = group[metric].dropna()
                if len(vals) > 0:
                    row[f'{metric}_mean'] = round(vals.mean(), 6)
                    row[f'{metric}_std'] = round(vals.std(ddof=1), 6) if len(vals) > 1 else 0.0
                else:
                    row[f'{metric}_mean'] = None
                    row[f'{metric}_std'] = None

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Cross-seed grid search for IMC — select params by avg MRR across multiple seeds"
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
    parser.add_argument("--tune_seeds", type=str, default="7001, 7005, 7009",
                        help="Comma-separated seeds used for tuning (default: 7001,7005,7009)")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Max IMC outer iterations")
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    lambda_values = [float(x.strip()) for x in args.lambda_values.split(",")]
    bias_values = [int(x.strip()) for x in args.bias_values.split(",")]
    tune_seeds = [int(x.strip()) for x in args.tune_seeds.split(",")]

    # Validate tuning seeds
    for s in tune_seeds:
        if s not in HELDOUT_SEEDS:
            print(f"WARNING: seed {s} is not in the standard heldout set {HELDOUT_SEEDS}")

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    pool = setup_gpu_environment()

    # Use base versions (without seed suffix) — cross_seed_grid_search
    # internally appends _seed{seed} for each tuning seed.
    BASE_VERSIONS = ["fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind"]
    versions = BASE_VERSIONS if args.all else [args.version]

    try:
        for v in versions:
            cross_seed_grid_search(
                v, args.model, args.aggregation,
                k_values, lambda_values, bias_values,
                tune_seeds=tune_seeds, maxiter=args.maxiter
            )
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    finally:
        pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

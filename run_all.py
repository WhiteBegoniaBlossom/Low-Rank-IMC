"""
Run all methods on all 10 seeds for a given dataset version + PLM model,
then compute mean +/- std across seeds.

Usage:
  python run_all.py --version fb237_v1_ind --model qwen --aggregation sum
  python run_all.py --version fb237_v1_ind --model roberta --aggregation sum --k 70 --lambda 1000.0 --bias 32
  python run_all.py --all --model qwen --aggregation sum
  python run_all.py --summary-only   # just recompute summary from existing CSV

Output:
  - IMC/results/fb237_unified_results.csv   (per-run, each sub-script appends)
  - IMC/results/fb237_summary_results.csv    (mean +/- std across seeds)
"""

import subprocess
import sys
import os
import re
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMC_DIR = os.path.join(BASE_DIR, "IMC")

BASELINE_SCRIPT = os.path.join(IMC_DIR, "baseline-fb237.py")
KGE_SCRIPT = os.path.join(IMC_DIR, "kge-baseline-fb237.py")
IMC_SCRIPT = os.path.join(IMC_DIR, "main-fb237.py")
TYLER_SCRIPT = os.path.join(IMC_DIR, "tyler_fb237.py")

# TyleR needs DGL which only works in the 'tyler' conda env (base env has
# graphbolt C++ library import issue on Windows). Other methods use base env.
TYLER_PYTHON = os.path.join(os.path.dirname(sys.executable), "envs", "tyler", "python.exe")
if not os.path.exists(TYLER_PYTHON):
    # Fallback: try to find it from the conda installation
    _conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    _candidate = os.path.join(_conda_prefix, "envs", "tyler", "python.exe")
    if os.path.exists(_candidate):
        TYLER_PYTHON = _candidate

RESULTS_DIR = os.path.join(IMC_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

UNIFIED_CSV = os.path.join(RESULTS_DIR, "fb237_unified_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "fb237_summary_results.csv")

BASE_TO_SEEDS = {
    "fb237_v1_ind": [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010],
    "fb237_v2_ind": [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010],
    "fb237_v3_ind": [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010],
}
ALL_BASE_VERSIONS = ["fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind"]

# KGE model names as they appear in the CSV "method" column
KGE_METHODS = ["DistMult", "ComplEx", "RotatE"]


def get_completed_experiments():
    """Read existing CSV and return set of (method, model, aggregation, version) tuples."""
    if not os.path.exists(UNIFIED_CSV):
        return set()
    df = pd.read_csv(UNIFIED_CSV)
    keys = df[['method', 'model', 'aggregation', 'version']].drop_duplicates()
    return set(tuple(r) for r in keys.values)


def build_commands(version, model, aggregation, k, lambda_cat, bias, skip_existing=None):
    """
    Build list of (label, cmd_list) for a single version.
    cmd_list format is [python, script, arg1, arg2, ...] for direct subprocess call.
    """
    python = sys.executable
    all_cmds = []

    # 1. Classical baselines — all in one run (baseline-fb237.py writes 7 method rows)
    BASELINE_METHOD_NAMES = [
        "feature_translation",
        "lr_concat", "lr_subtract",
        "rf_concat", "rf_subtract",
        "lgbm_concat", "lgbm_subtract",
    ]
    if not skip_existing or not all(
        (m, model, aggregation, version) in skip_existing for m in BASELINE_METHOD_NAMES
    ):
        all_cmds.append((
            f"{version} | baselines",
            [python, BASELINE_SCRIPT,
             "--version", version,
             "--model", model,
             "--aggregation", aggregation]
        ))

    # 2. KGE baselines (DistMult, ComplEx, RotatE)
    for kge_model in KGE_METHODS:
        if skip_existing and (kge_model, model, aggregation, version) in skip_existing:
            continue
        all_cmds.append((
            f"{version} | {kge_model}",
            [python, KGE_SCRIPT,
             "--version", version,
             "--model", kge_model,
             "--plm_model", model,
             "--aggregation", aggregation]
        ))

    # 3. IMC
    if not skip_existing or ("IMC", model, aggregation, version) not in skip_existing:
        all_cmds.append((
            f"{version} | IMC",
            [python, IMC_SCRIPT,
             "--version", version,
             "--model", model,
             "--aggregation", aggregation,
             "--k", str(k),
             "--lambda", str(lambda_cat),
             "--bias", str(bias)]
        ))

    # 4. TyleR (RGCN + PLM embeddings) — uses tyler conda env for DGL
    if not skip_existing or ("TyleR", model, aggregation, version) not in skip_existing:
        all_cmds.append((
            f"{version} | TyleR",
            [TYLER_PYTHON, TYLER_SCRIPT,
             "--version", version,
             "--model", model,
             "--aggregation", aggregation]
        ))

    return all_cmds


def run_one(label, cmd_list):
    """Run a single experiment. Returns True on success."""
    print(f"  {label} ... ", end="", flush=True)
    try:
        result = subprocess.run(cmd_list, cwd=IMC_DIR,
                               capture_output=True, text=True, timeout=10800)
        if result.returncode == 0:
            print("OK")
            return True
        else:
            # Print last line of stderr for debugging
            err_lines = result.stderr.strip().split('\n')
            last_err = err_lines[-1] if err_lines else "unknown error"
            print(f"FAILED ({last_err[:120]})")
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except Exception as e:
        print(f"ERROR ({e})")
        return False


def compute_summary():
    """Read fb237_unified_results.csv and compute mean +/- std."""
    if not os.path.exists(UNIFIED_CSV):
        print(f"ERROR: {UNIFIED_CSV} not found")
        return None

    df = pd.read_csv(UNIFIED_CSV)

    # Map version -> base_version
    _seed_pattern = re.compile(r'^(.+_ind)_seed\d+$')

    def get_base_version(v):
        m = _seed_pattern.match(str(v))
        if m:
            base = m.group(1)
            if base in ALL_BASE_VERSIONS:
                return base
        if str(v) in ALL_BASE_VERSIONS:
            return str(v)
        return str(v)

    df['base_version'] = df['version'].apply(get_base_version)

    # Only include known base versions in summary
    df = df[df['base_version'].isin(ALL_BASE_VERSIONS)]

    # Dynamically discover metric columns: numeric columns that aren't housekeeping
    HOUSEKEEPING = {'method', 'model', 'aggregation', 'version', 'base_version', 'train_time_s'}
    all_metric_cols = [c for c in df.columns if c not in HOUSEKEEPING
                       and pd.api.types.is_numeric_dtype(df[c])]

    group_cols = ['method', 'model', 'aggregation', 'base_version']
    rows = []

    for keys, group in df.groupby(group_cols):
        n_seeds = group['version'].nunique()
        row = dict(zip(group_cols, keys))
        row['n_seeds'] = n_seeds

        for metric in all_metric_cols:
            if metric in group.columns:
                vals = group[metric].dropna()
                if len(vals) > 0:
                    row[f'{metric}_mean'] = round(vals.mean(), 6)
                    row[f'{metric}_std'] = round(vals.std(ddof=1), 6) if len(vals) > 1 else 0.0
                else:
                    row[f'{metric}_mean'] = None
                    row[f'{metric}_std'] = None

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No results to summarize.")
        return None

    method_order = {
        'feature_translation': 0,
        'lr_concat': 1, 'lr_subtract': 2,
        'rf_concat': 3, 'rf_subtract': 4,
        'lgbm_concat': 5, 'lgbm_subtract': 6,
        'DistMult': 7, 'ComplEx': 8, 'RotatE': 9,
        'IMC': 10, 'TyleR': 11,
    }
    summary['_sort'] = summary['method'].map(method_order).fillna(99)
    summary = summary.sort_values(['base_version', 'model', '_sort']).drop(columns=['_sort'])

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary saved to {SUMMARY_CSV}")
    return summary


def print_summary(summary):
    """Print a compact readable summary table."""
    if summary is None or summary.empty:
        return

    print(f"\n{'='*130}")
    print("SUMMARY: mean +/- std across seeds")
    print(f"{'='*130}")

    for base_ver in ALL_BASE_VERSIONS:
        sub = summary[summary['base_version'] == base_ver]
        if sub.empty:
            continue
        print(f"\n--- {base_ver} ---")
        header = (f"{'Method':<18} {'PLM':<8}"
                  f" {'MRR':>22} {'Hits@1':>22} {'Hits@10':>22}"
                  f" {'Cov(a=.1)':>14} {'Size(a=.1)':>14} {'n':>5}")
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            def fmt(col):
                m = r.get(f'{col}_mean')
                s = r.get(f'{col}_std')
                if pd.isna(m):
                    return "N/A".center(22)
                return f"{m:.4f}+/-{s:.4f}".center(22)

            def cfmt(col, width=14):
                m = r.get(f'{col}_mean')
                s = r.get(f'{col}_std')
                if pd.isna(m):
                    return "N/A".center(width)
                return f"{m:.4f}+/-{s:.4f}".center(width)

            n = int(r['n_seeds'])
            print(f"{r['method']:<18} {r['model']:<8} "
                  f"{fmt('mrr')} {fmt('hits_1')} {fmt('hits_10')} "
                  f"{cfmt('conformal_cov_a0.1')} {cfmt('conformal_size_a0.1')} {n:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all methods on all seeds, compute mean+/-std")
    parser.add_argument("--version", type=str, default=None,
                        help="Base version (e.g. fb237_v1_ind)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 3 base versions")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"])
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"])
    parser.add_argument("--k", type=int, default=70)
    parser.add_argument("--lambda", type=float, default=1000.0, dest="lambda_cat")
    parser.add_argument("--bias", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run experiments even if already in CSV")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only compute summary from existing CSV")
    args = parser.parse_args()

    if args.summary_only:
        summary = compute_summary()
        print_summary(summary)
        return

    # Determine base versions to process
    if args.all:
        base_versions = ALL_BASE_VERSIONS
    elif args.version:
        v = args.version
        if v in BASE_TO_SEEDS:
            base_versions = [v]
        else:
            # Try fuzzy match
            matched = [bv for bv in ALL_BASE_VERSIONS if bv.startswith(v) or v.startswith(bv.split('_ind')[0])]
            if matched:
                base_versions = [matched[0]]
            else:
                print(f"Unknown version: {v}")
                print(f"Known: {ALL_BASE_VERSIONS}")
                return
    else:
        print("Specify --version <name> or --all")
        return

    skip_existing = None if args.no_skip else get_completed_experiments()

    # Build flat list of all commands
    all_labels = []
    all_cmds = []

    for bv in base_versions:
        seeds = BASE_TO_SEEDS.get(bv, [])
        versions = [bv] if not seeds else [f"{bv}_seed{s}" for s in seeds]
        for ver in versions:
            cmds = build_commands(ver, args.model, args.aggregation,
                                  args.k, args.lambda_cat, args.bias,
                                  skip_existing=skip_existing)
            for label, cmd_list in cmds:
                all_labels.append(label)
                all_cmds.append(cmd_list)

    total = len(all_cmds)
    print(f"Experiments to run: {total}")
    print(f"PLM: {args.model}, Agg: {args.aggregation}")
    print(f"IMC: k={args.k}, lambda={args.lambda_cat}, bias={args.bias}")
    print(f"Skip existing: {not args.no_skip}")
    print()

    if args.dry_run:
        for label, cmd_list in zip(all_labels, all_cmds):
            print(f"  {label}: {' '.join(cmd_list)}")
        print()
        return

    if total == 0:
        print("All experiments already completed. Use --summary-only to see results.")
        summary = compute_summary()
        print_summary(summary)
        return

    # Run
    count = 0
    failed = []
    for i, (label, cmd_list) in enumerate(zip(all_labels, all_cmds)):
        count += 1
        print(f"[{count}/{total}] ", end="")
        if not run_one(label, cmd_list):
            failed.append(label)

    print(f"\n{'='*60}")
    print(f"Done: {count - len(failed)}/{total} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    # Compute summary
    print(f"\nComputing summary across seeds...")
    summary = compute_summary()
    print_summary(summary)


if __name__ == "__main__":
    main()

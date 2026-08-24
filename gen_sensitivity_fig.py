"""
Generate parameter sensitivity plots for FB15k-237 v3, MovieLens, and TwoSides.

FB15k-237 v3 data from EnhancedIMC cross-seed tuning (all (k, lambda, bias)
combinations averaged across tuning seeds).
MovieLens and TwoSides data from OldIMC per-parameter tuning CSVs.

Produces 3 separate figure files (one per dataset), each with 3 rows
(k, lambda, bias) — matching the subfigure layout used in the paper.

Usage:
    python gen_sensitivity_fig.py --fb237-model qwen
    python gen_sensitivity_fig.py --fb237-model roberta --output-dir ../Essay/fig
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMC_RESULTS = os.path.join(SCRIPT_DIR, "IMC", "results")
OLDIMC_RESULTS = os.path.join(os.path.dirname(SCRIPT_DIR), "OldIMC", "results")

# ---- Optimal hyperparameters for FB237 v3 ----
FB237_V3_OPTIMAL = {
    "qwen":   {"k": 200, "lambda": 100, "bias": 96},
    "roberta": {"k": 200, "lambda": 100, "bias": 64},
}

# ---- Dataset display info ----
DATASET_NAMES = ["fb237_v3", "movielens", "twosides"]
DATASET_LABELS = {
    "fb237_v3":  "FB15k-237 v3",
    "movielens": "MovieLens",
    "twosides":  "TwoSides",
}
DATASET_COLORS = {
    "fb237_v3":  "#1f77b4",
    "movielens": "#ff7f0e",
    "twosides":  "#2ca02c",
}
DATASET_MARKERS = {
    "fb237_v3":  "o",
    "movielens": "s",
    "twosides":  "^",
}

PARAM_NAMES = ["k", "lambda", "bias"]
PARAM_LABELS = {
    "k":      r"Latent dimension $L$",
    "lambda": r"Regularization coefficient $\lambda$",
    "bias":   r"Bias dimension $b$",
}


def load_fb237_v3_sensitivity(model):
    """Extract per-parameter sensitivity from EnhancedIMC cross-seed tuning detail CSV."""
    detail_csv = os.path.join(IMC_RESULTS, f"tune_cross_fb237_v3_ind_{model}_sum_detail.csv")

    if not os.path.exists(detail_csv):
        print(f"ERROR: {detail_csv} not found.")
        print(f"Run: python tune-fb237-cross-seed.py --version fb237_v3_ind --model {model}")
        sys.exit(1)

    df = pd.read_csv(detail_csv)
    optimal = FB237_V3_OPTIMAL[model]

    # Average metrics across tuning seeds for each (k, lambda, bias) combination
    group_cols = ["k", "lambda", "bias"]
    agg = df.groupby(group_cols).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    ).reset_index()

    def extract_slice(param_name):
        """Rows where the other two params are fixed at optimal values."""
        if param_name == "k":
            mask = (agg["lambda"] == optimal["lambda"]) & (agg["bias"] == optimal["bias"])
        elif param_name == "lambda":
            mask = (agg["k"] == optimal["k"]) & (agg["bias"] == optimal["bias"])
        elif param_name == "bias":
            mask = (agg["k"] == optimal["k"]) & (agg["lambda"] == optimal["lambda"])
        else:
            raise ValueError(f"Unknown param: {param_name}")

        sub = agg[mask].sort_values(param_name)
        if len(sub) == 0:
            print(f"WARNING: No data for {param_name} slice in FB237 v3 "
                  f"(k={optimal['k']}, lambda={optimal['lambda']}, bias={optimal['bias']}). "
                  f"Check that the grid covers these fixed values.")
        return {
            "x": sub[param_name].values,
            "y": sub["test_acc_mean"].values,
            "y_err": sub["test_acc_std"].values,
        }

    return {p: extract_slice(p) for p in PARAM_NAMES}


def load_oldimc_sensitivity(dataset):
    """Load per-parameter sensitivity from OldIMC tuning CSVs."""
    param_files = {
        "k":      os.path.join(OLDIMC_RESULTS, f"{dataset}_tuning_k.csv"),
        "lambda": os.path.join(OLDIMC_RESULTS, f"{dataset}_tuning_lambda.csv"),
        "bias":   os.path.join(OLDIMC_RESULTS, f"{dataset}_tuning_bias.csv"),
    }

    result = {}
    for param_name, filepath in param_files.items():
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping {param_name} for {dataset}")
            continue
        df = pd.read_csv(filepath).sort_values(param_name)
        result[param_name] = {
            "x": df[param_name].values,
            "y": df["test_accuracy"].values,
            "y_err": None,
        }
    return result


def plot_param_row(ax, data, param_name, color):
    """Plot one parameter's sensitivity curve on a given axis."""
    if data is None or len(data["x"]) == 0:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
        return

    x_vals = data["x"]
    y_vals = data["y"]
    y_err = data.get("y_err")

    if param_name == "bias":
        # Equal spacing with integer labels
        x_pos = np.arange(1, len(x_vals) + 1)
        ax.plot(x_pos, y_vals, color=color, marker="o", markersize=4,
                linewidth=1.5, linestyle="-")
        if y_err is not None and np.any(y_err > 0):
            ax.fill_between(x_pos, y_vals - y_err, y_vals + y_err,
                            color=color, alpha=0.15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(v)}" for v in x_vals], fontsize=9)
    elif param_name == "lambda":
        ax.plot(x_vals, y_vals, color=color, marker="s", markersize=4,
                linewidth=1.5, linestyle="-")
        if y_err is not None and np.any(y_err > 0):
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err,
                            color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v}" for v in x_vals], fontsize=8, rotation=30)
        ax.minorticks_off()
    else:  # k
        ax.plot(x_vals, y_vals, color=color, marker="^", markersize=4,
                linewidth=1.5, linestyle="-")
        if y_err is not None and np.any(y_err > 0):
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err,
                            color=color, alpha=0.15)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{int(v)}" for v in x_vals], fontsize=9)

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)


def make_dataset_figure(dataset, data, output_dir):
    """Create a single-dataset sensitivity figure (3 rows × 1 col)."""
    color = DATASET_COLORS[dataset]

    fig, axes = plt.subplots(3, 1, figsize=(5, 5.5), dpi=600)
    plt.subplots_adjust(hspace=0.65)

    for i, param_name in enumerate(PARAM_NAMES):
        ax = axes[i]
        param_data = data.get(param_name)
        plot_param_row(ax, param_data, param_name, color)

        # Y-label on middle row only
        if i == 1:
            ax.set_ylabel("Test accuracy", fontsize=10)

        # X-label = parameter name
        ax.set_xlabel(PARAM_LABELS[param_name], fontsize=10)

        # Auto-tighten y-limits
        if param_data is not None and len(param_data["y"]) > 0:
            y_min, y_max = np.min(param_data["y"]), np.max(param_data["y"])
            margin = 0.08 * (y_max - y_min) if y_max > y_min else 0.01
            ax.set_ylim(y_min - margin, y_max + margin)

    # Save
    base_name = f"sensitivity_{dataset}"
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    png_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  [{DATASET_LABELS[dataset]}] -> {base_name}.pdf, {base_name}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sensitivity plots (3 datasets)")
    parser.add_argument("--fb237-model", type=str, default="qwen",
                        choices=["qwen", "roberta"],
                        help="PLM model for FB237 v3 sensitivity")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "Essay", "fig"),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading sensitivity data...")
    print(f"  FB15k-237 v3: EnhancedIMC ({args.fb237_model})")
    print("  MovieLens:    OldIMC")
    print("  TwoSides:     OldIMC")
    print()

    fb237_data = load_fb237_v3_sensitivity(args.fb237_model)
    movielens_data = load_oldimc_sensitivity("movielens")
    twosides_data = load_oldimc_sensitivity("twosides")

    data_map = {
        "fb237_v3": fb237_data,
        "movielens": movielens_data,
        "twosides": twosides_data,
    }

    print("Generating figures...")
    for dataset in DATASET_NAMES:
        make_dataset_figure(dataset, data_map[dataset], args.output_dir)

    print("\nDone. All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()

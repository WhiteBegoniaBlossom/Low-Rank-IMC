"""
Generate convergence curves for FB15k-237 v1, v2, v3 from EnhancedIMC experiments.

Runs IMC on one seed per version with optimal hyperparameters (from cross-seed tuning),
captures the outer loss trajectory, and produces a 3-subfigure convergence plot.

Usage:
    python gen_convergence_fig.py --model roberta
    python gen_convergence_fig.py --model qwen --seed 7001
"""

import os
import sys
import argparse
import gc
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

# Add IMC directory to path
IMC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMC")
sys.path.insert(0, IMC_DIR)

# Import from main-fb237.py (hyphen in filename prevents direct import)
_main_path = os.path.join(IMC_DIR, "main-fb237.py")
_spec = importlib.util.spec_from_file_location("main_fb237", _main_path)
_m237 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m237)

train_fb237_version = _m237.train_fb237_version
setup_gpu_environment = _m237.setup_gpu_environment
VERSIONS = _m237.VERSIONS

# Optimal hyperparameters from cross-seed tuning (3-seed average valid-MRR)
OPTIMAL_PARAMS = {
    "roberta": {
        "fb237_v1_ind": {"k": 200, "lambda": 100, "bias": 64},
        "fb237_v2_ind": {"k": 175, "lambda": 100, "bias": 96},
        "fb237_v3_ind": {"k": 200, "lambda": 100, "bias": 64},
    },
    "qwen": {
        "fb237_v1_ind": {"k": 150, "lambda": 100, "bias": 48},
        "fb237_v2_ind": {"k": 200, "lambda": 100, "bias": 80},
        "fb237_v3_ind": {"k": 200, "lambda": 100, "bias": 96},
    },
}

VERSION_LABELS = {
    "fb237_v1_ind": r"FB15k-237 v1 ($\sim$2.4K triples)",
    "fb237_v2_ind": r"FB15k-237 v2 ($\sim$5.1K triples)",
    "fb237_v3_ind": r"FB15k-237 v3 ($\sim$9.1K triples)",
}

VERSION_ORDER = ["fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate convergence curves for FB15k-237 v1/v2/v3")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "qwen"])
    parser.add_argument("--seed", type=int, default=7001,
                        help="Seed to use for convergence curves (default: 7001)")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Maximum alternating iterations")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Essay", "fig"),
                        help="Output directory for figures")
    parser.add_argument("--save-loss", action="store_true", default=True,
                        help="Save loss trajectories as .txt files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing loss files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loss_dir = os.path.join(os.path.dirname(__file__), "IMC", "results", "loss_curves")
    os.makedirs(loss_dir, exist_ok=True)

    if args.model not in OPTIMAL_PARAMS:
        print(f"Unknown model: {args.model}. Options: {list(OPTIMAL_PARAMS.keys())}")
        sys.exit(1)

    params_dict = OPTIMAL_PARAMS[args.model]

    # Collect loss trajectories
    loss_data = {}

    if not args.skip_train:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        pool = setup_gpu_environment()

        try:
            for version in VERSION_ORDER:
                params = params_dict[version]
                version_seed = f"{version}_seed{args.seed}"

                print(f"\n{'='*60}")
                print(f"Training: {version_seed}")
                print(f"  k={params['k']}, lambda={params['lambda']}, bias={params['bias']}")
                print(f"  model={args.model}, aggregation=sum")
                print(f"{'='*60}")

                # Run IMC training and capture outer loss trajectory
                _, outer_losses, _ = train_fb237_version(
                    version_seed,
                    k=params["k"],
                    lambda_cat=params["lambda"],
                    bias=params["bias"],
                    model=args.model,
                    aggregation="sum",
                    random_seed=args.seed,
                    maxiter_cat=args.maxiter,
                )

                outer_losses_float = [float(l) for l in outer_losses]
                loss_data[version] = outer_losses_float

                # Save loss trajectory
                loss_file = os.path.join(loss_dir, f"loss_{version}_{args.model}_seed{args.seed}.txt")
                np.savetxt(loss_file, outer_losses_float)
                print(f"Loss trajectory saved to {loss_file}")

                # Clean up GPU memory between versions
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

        finally:
            pool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()

    else:
        # Load existing loss files
        for version in VERSION_ORDER:
            loss_file = os.path.join(loss_dir, f"loss_{version}_{args.model}_seed{args.seed}.txt")
            if os.path.exists(loss_file):
                loss_data[version] = np.loadtxt(loss_file).tolist()
                print(f"Loaded {loss_file}: {len(loss_data[version])} iterations")
            else:
                print(f"WARNING: {loss_file} not found. Run without --skip-train first.")
                sys.exit(1)

    # Generate 3-subfigure convergence plot
    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 600,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=600)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]

    for idx, version in enumerate(VERSION_ORDER):
        ax = axes[idx]
        losses = loss_data[version]

        ax.plot(losses, marker=markers[idx], markersize=5, linewidth=1.5,
                linestyle=linestyles[idx], color=colors[idx],
                markerfacecolor="none", markeredgewidth=1.0)

        ax.set_xlabel("# iterations")
        if idx == 0:
            ax.set_ylabel("Objective value")

        # Version label with triple count
        ax.set_title(VERSION_LABELS[version], fontsize=11)

        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_axisbelow(True)

        # Log scale for y-axis if loss range spans orders of magnitude
        loss_min, loss_max = min(losses), max(losses)
        if loss_max / max(loss_min, 1e-10) > 50:
            ax.set_yscale("log")

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "convergence_fb237.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    # Also save PNG for easy preview
    png_path = os.path.join(args.output_dir, "convergence_fb237.png")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    print(f"\nConvergence figure saved to:")
    print(f"  {output_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()

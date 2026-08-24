"""
Run all methods on a single dataset version.

Usage:
  python run_one.py --version fb237_v1_ind_seed2012 --model roberta --aggregation sum
  python run_one.py --version fb237_v1_ind --model qwen --aggregation sum
"""
import subprocess
import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMC_DIR = os.path.join(BASE_DIR, "IMC")

BASELINE_SCRIPT = os.path.join(IMC_DIR, "baseline-fb237.py")
KGE_SCRIPT = os.path.join(IMC_DIR, "kge-baseline-fb237.py")
IMC_SCRIPT = os.path.join(IMC_DIR, "main-fb237.py")
TYLER_SCRIPT = os.path.join(IMC_DIR, "tyler_fb237.py")

# TyleR needs the 'tyler' conda env for DGL; other methods use base env.
TYLER_PYTHON = os.path.join(os.path.dirname(sys.executable), "envs", "tyler", "python.exe")
if not os.path.exists(TYLER_PYTHON):
    TYLER_PYTHON = sys.executable  # fallback


def run(label, cmd, python_exe=None):
    """Run a single experiment. Returns True on success."""
    if python_exe is None:
        python_exe = sys.executable
    full_cmd = [python_exe] + cmd
    print(f"  {label} ... ", end="", flush=True)
    try:
        result = subprocess.run(full_cmd, cwd=IMC_DIR,
                                capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            print("OK")
            return True
        else:
            err = result.stderr.strip().split('\n')[-1] if result.stderr.strip() else "unknown"
            print(f"FAILED ({err[:120]})")
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all methods on one dataset version")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"])
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"])
    parser.add_argument("--k", type=int, default=70, help="IMC factorization rank")
    parser.add_argument("--lambda", type=float, default=1000.0, dest="lambda_cat",
                        help="IMC L2 regularization")
    parser.add_argument("--bias", type=int, default=32, help="IMC bias dimensions")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        choices=["baselines", "DistMult", "ComplEx", "RotatE", "IMC", "TyleR"],
                        help="Methods to skip")
    args = parser.parse_args()

    version = args.version
    model = args.model
    agg = args.aggregation

    print(f"Version: {version}, Model: {model}, Aggregation: {agg}")
    print(f"Skip: {args.skip if args.skip else 'none'}")
    print()

    # Build command list
    commands = []

    # 1. Classical baselines (Linear, RF, LightGBM, FT)
    if "baselines" not in args.skip:
        commands.append(("baselines", [BASELINE_SCRIPT, "--version", version,
                                        "--model", model, "--aggregation", agg]))

    # 2. KGE baselines
    for kge in ["DistMult", "ComplEx", "RotatE"]:
        if kge not in args.skip:
            commands.append((kge, [KGE_SCRIPT, "--version", version,
                                   "--model", kge, "--plm_model", model,
                                   "--aggregation", agg]))

    # 3. IMC
    if "IMC" not in args.skip:
        commands.append(("IMC", [IMC_SCRIPT, "--version", version,
                                  "--model", model, "--aggregation", agg,
                                  "--k", str(args.k),
                                  "--lambda", str(args.lambda_cat),
                                  "--bias", str(args.bias)]))

    # 4. TyleR (uses tyler conda env)
    if "TyleR" not in args.skip:
        commands.append(("TyleR", [TYLER_SCRIPT, "--version", version,
                                    "--model", model, "--aggregation", agg],
                         TYLER_PYTHON))

    # Run
    total = len(commands)
    ok = 0
    for i, cmd_info in enumerate(commands):
        label = cmd_info[0]
        cmd = cmd_info[1]
        py = cmd_info[2] if len(cmd_info) > 2 else None
        print(f"[{i+1}/{total}] ", end="")
        if run(label, cmd, py):
            ok += 1

    print(f"\nDone: {ok}/{total} succeeded")


if __name__ == "__main__":
    main()

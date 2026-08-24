"""
One-click script: generate new seed splits, update code, copy labels, generate embeddings.

Usage:
  python new_seeds.py --base-seed 8001 --seeds 10
  python new_seeds.py --base-seed 9001 --seeds 5 --version v1
  python new_seeds.py --base-seed 8001 --seeds 10 --model roberta
  python new_seeds.py --base-seed 8001 --seeds 10 --models qwen,roberta
"""
import os, sys, subprocess, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMC_DIR = os.path.join(BASE_DIR, "IMC")
DATA_DIR = os.path.join(IMC_DIR, "data")
EMB_DIR = os.path.join(IMC_DIR, "embeddings")

VERSIONS = ["v1", "v2", "v3"]

# ---- helpers ----

def run(cmd, cwd=None, timeout=600):
    cwd = cwd or BASE_DIR
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stdout, result.stderr


def update_seed_lists(seeds, versions):
    """Update all *_SEEDS lists in Python scripts."""
    files_to_update = {
        os.path.join(BASE_DIR, "run_all.py"): "BASE_TO_SEEDS",
        os.path.join(IMC_DIR, "main-fb237.py"): ("V1_SEEDS", "V2_SEEDS", "V3_SEEDS"),
        os.path.join(IMC_DIR, "baseline-fb237.py"): ("V1_SEEDS", "V2_SEEDS", "V3_SEEDS"),
        os.path.join(IMC_DIR, "kge-baseline-fb237.py"): ("V1_SEEDS", "V2_SEEDS", "V3_SEEDS"),
        os.path.join(IMC_DIR, "generate_plm_embeddings.py"): ("_V1_SEEDS", "_V2_SEEDS", "_V3_SEEDS"),
    }

    seed_str = str(seeds)
    for ver_idx, ver in enumerate(versions):
        old_seeds = list(range(7001, 7011))  # previous default
        old_str = str(old_seeds)

        for filepath, var_names in files_to_update.items():
            with open(filepath) as f:
                content = f.read()

            if isinstance(var_names, str):
                # run_all.py: single dict entry per version
                var_name = var_names
                # Find and replace the version's seed list in BASE_TO_SEEDS
                pattern = f'"{ver}_ind": {old_str}'
                replacement = f'"{ver}_ind": {seed_str}'
                if pattern in content:
                    content = content.replace(pattern, replacement)
                else:
                    # Try with current actual seeds
                    import re
                    match = re.search(rf'"fb237_{ver}_ind":\s*\[([^\]]+)\]', content)
                    if match:
                        content = content[:match.start()] + \
                                  f'"fb237_{ver}_ind": {seed_str}' + \
                                  content[match.end():]
            else:
                # IMC scripts: V1_SEEDS / V2_SEEDS / V3_SEEDS
                var_name = var_names[ver_idx]
                # Find the line like "V1_SEEDS = [7001, ...]"
                import re
                pattern = rf'{var_name}\s*=\s*\[[^\]]*\]'
                content = re.sub(pattern, f'{var_name} = {seed_str}', content)

            with open(filepath, "w") as f:
                f.write(content)

        print(f"  Updated {ver} seed lists in all scripts")


def copy_labels(seeds, versions):
    """Copy label.tsv and type.tsv from original _ind to each seed directory."""
    for ver in versions:
        base = f"fb237_{ver}_ind"
        src = os.path.join(DATA_DIR, base)
        for s in seeds:
            dst = os.path.join(DATA_DIR, f"{base}_seed{s}")
            for fname in ["label.tsv", "type.tsv", "ontology.tsv"]:
                src_file = os.path.join(src, fname)
                if os.path.exists(src_file):
                    shutil.copy(src_file, os.path.join(dst, fname))
        print(f"  {ver}: labels copied to {len(seeds)} seeds")


def generate_embeddings(seeds, versions, models):
    """Generate embeddings for one seed per version per model, then copy."""
    for model in models:
        for ver in versions:
            base = f"fb237_{ver}_ind"
            first_seed = seeds[0]
            version_name = f"{base}_seed{first_seed}"

            # Remove old embeddings to force regeneration
            emb_name = f"{version_name}_{model}_sum_embeddings.pkl"
            logits_name = f"{version_name}_{model}_id2logits.pt"
            for f in [emb_name, logits_name]:
                path = os.path.join(EMB_DIR, f)
                if os.path.exists(path):
                    os.remove(path)

            print(f"  Generating {model} embeddings for {version_name}...")
            ok, stdout, stderr = run([
                sys.executable, "generate_plm_embeddings.py",
                "--version", version_name,
                "--model", model,
                "--aggregation", "sum",
            ], cwd=IMC_DIR, timeout=1200)

            if not ok:
                print(f"    FAILED: {stderr[-200:]}")
                continue

            # Copy to other seeds
            for s in seeds[1:]:
                other = f"{base}_seed{s}"
                src_emb = os.path.join(EMB_DIR, emb_name)
                dst_emb = os.path.join(EMB_DIR, f"{other}_{model}_sum_embeddings.pkl")
                src_log = os.path.join(EMB_DIR, logits_name)
                dst_log = os.path.join(EMB_DIR, f"{other}_{model}_id2logits.pt")
                shutil.copy(src_emb, dst_emb)
                shutil.copy(src_log, dst_log)

            print(f"  {ver}/{model}: generated + copied to {len(seeds)} seeds")


# ---- main ----

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="One-click: generate new seed splits + update code + labels + embeddings")
    parser.add_argument("--base-seed", type=int, required=True,
                        help="Starting seed number (e.g. 8001)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds to generate")
    parser.add_argument("--version", type=str, default=None,
                        choices=VERSIONS, help="Single version (default: all 3)")
    parser.add_argument("--models", type=str, default="qwen,roberta",
                        help="Comma-separated model list (default: qwen,roberta)")
    parser.add_argument("--skip-splits", action="store_true",
                        help="Skip split generation (only update + labels + embeddings)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    versions = VERSIONS if args.version is None else [args.version]
    models = [m.strip() for m in args.models.split(",")]

    print(f"\n{'='*60}")
    print(f"NEW SEEDS PIPELINE")
    print(f"{'='*60}")
    print(f"  Seeds:     {seeds}")
    print(f"  Versions:  {versions}")
    print(f"  Models:    {models}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] Would execute the following steps:")
        print(f"  1. Generate splits via reshuffle_splits.py")
        print(f"  2. Update seed lists in 5 Python scripts")
        print(f"  3. Copy label.tsv/type.tsv to all seed dirs")
        print(f"  4. Generate embeddings for {models}")
        return

    # Step 1: Generate splits
    if not args.skip_splits:
        print("[1/4] Generating seed splits...")
        cmd = [
            sys.executable, "reshuffle_splits.py",
            "--base-seed", str(args.base_seed),
            "--seeds", str(args.seeds),
        ]
        if args.version:
            cmd += ["--version", args.version]
        ok, stdout, stderr = run(cmd)
        if not ok:
            print(f"ERROR: {stderr}")
            return
        print(stdout.strip().split('\n')[-5:])  # last few lines

    # Step 2: Update seed lists
    print("\n[2/4] Updating seed lists in code...")
    update_seed_lists(seeds, versions)

    # Step 3: Copy labels
    print("\n[3/4] Copying label/type files...")
    copy_labels(seeds, versions)

    # Step 4: Generate embeddings
    if not args.skip_embeddings:
        print(f"\n[4/4] Generating embeddings for {models}...")
        generate_embeddings(seeds, versions, models)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"\n  Seeds:  {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"  Models: {models}")
    print(f"\nReady to run experiments:")
    for ver in versions:
        print(f"  python run_all.py --version fb237_{ver}_ind --model {models[0]} --aggregation sum")


if __name__ == "__main__":
    main()

"""
Generate and select 10 consistent seed splits per version.
Strategy: generate many candidates, select the 10 with most similar entity counts.

Usage:
  python select_splits.py          # full generate + select
  python select_splits.py --analyze-only   # just analyze existing
  python select_splits.py --select-only    # select from existing candidates
"""

import subprocess
import sys
import os
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(BASE_DIR, "generate_splits.py")
DATA_DIR = os.path.join(BASE_DIR, "grail-master", "data")
IMC_DATA_DIR = os.path.join(BASE_DIR, "IMC", "data")

VERSION_CONFIGS = {
    "fb237_v1": {"max_nodes": 350, "target_entities": 1093},
    "fb237_v2": {"max_nodes": 550, "target_entities": 1660},
    "fb237_v3": {"max_nodes": 800, "target_entities": 2501},
}

# Generate 30 candidates per version, select 10
CANDIDATES_PER_VERSION = 30
SELECT_PER_VERSION = 10


def get_stats(directory):
    entities = set()
    relations = set()
    total = 0
    for f in ['train.txt', 'valid.txt', 'test.txt']:
        fpath = os.path.join(directory, f)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as fh:
            for line in fh:
                p = line.strip().split('\t')
                if len(p) == 3:
                    entities.add(p[0])
                    entities.add(p[2])
                    relations.add(p[1])
                    total += 1
    return {"entities": len(entities), "relations": len(relations), "triples": total}


def scan_existing(version_base):
    """Scan existing _ind_seed* directories and return stats dict."""
    existing = {}
    prefix = version_base + "_ind_seed"
    for d in os.listdir(DATA_DIR):
        if d.startswith(prefix) and os.path.isdir(os.path.join(DATA_DIR, d)):
            stats = get_stats(os.path.join(DATA_DIR, d))
            # parse seed from dirname
            seed_str = d[len(prefix):]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            existing[seed] = stats
    return existing


def generate_candidate(version_base, max_nodes, seed):
    """Run generate_splits.py for a single candidate. Returns stats or None."""
    temp_name = f"{version_base}_tmp_sel_{seed}"
    final_dir = os.path.join(DATA_DIR, f"{version_base}_ind_seed{seed}")

    # Skip if already exists
    if os.path.isdir(final_dir):
        return get_stats(final_dir)

    cmd = [
        sys.executable, SCRIPT,
        "--dataset", "FB15K237",
        "--new_dataset", temp_name,
        "--max_nodes_per_hop", str(max_nodes),
        "--max_nodes_per_hop_test", str(max_nodes),
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, cwd=BASE_DIR,
                          capture_output=True, text=True)
    if result.returncode != 0:
        # Check if it's a StopIteration / BFS exhaustion
        if "StopIteration" in result.stderr:
            print(f"    seed={seed}: BFS exhausted (graph too fragmented), skipping")
        else:
            print(f"    seed={seed}: FAILED - {result.stderr[-200:]}")
        # Cleanup temp
        for suffix in ["", "_ind"]:
            tmp = os.path.join(DATA_DIR, temp_name + suffix)
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)
        return None

    # Rename _ind to final name
    src_ind = os.path.join(DATA_DIR, temp_name + "_ind")
    if os.path.isdir(src_ind):
        os.rename(src_ind, final_dir)
        stats = get_stats(final_dir)
    else:
        stats = None

    # Remove transductive temp
    src_trans = os.path.join(DATA_DIR, temp_name)
    if os.path.isdir(src_trans):
        shutil.rmtree(src_trans)

    return stats


def select_best_10(candidates):
    """
    Given dict of {seed: {entities, triples, relations}},
    select the 10 seeds with the smallest entity count variance.
    Uses sliding window on sorted entity counts.
    """
    if len(candidates) < SELECT_PER_VERSION:
        return list(candidates.keys())

    # Sort by entity count
    sorted_items = sorted(candidates.items(), key=lambda x: x[1]["entities"])

    best_seeds = None
    best_range = float('inf')

    # Sliding window of size SELECT_PER_VERSION
    for i in range(len(sorted_items) - SELECT_PER_VERSION + 1):
        window = sorted_items[i:i + SELECT_PER_VERSION]
        ents = [v["entities"] for _, v in window]
        w_range = max(ents) - min(ents)
        if w_range < best_range:
            best_range = w_range
            best_seeds = [s for s, _ in window]

    return best_seeds


def clean_rejected(version_base, keep_seeds):
    """Remove _ind_seed* directories not in keep_seeds."""
    prefix = version_base + "_ind_seed"
    for d in os.listdir(DATA_DIR):
        if d.startswith(prefix) and os.path.isdir(os.path.join(DATA_DIR, d)):
            seed_str = d[len(prefix):]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            if seed not in keep_seeds:
                path = os.path.join(DATA_DIR, d)
                shutil.rmtree(path)
                # Also remove from IMC
                imc_path = os.path.join(IMC_DATA_DIR, d)
                if os.path.isdir(imc_path):
                    shutil.rmtree(imc_path)
                print(f"  Removed: {d}")


def sync_to_imc(version_base, seeds):
    """Copy selected _ind_seed dirs to IMC/data/."""
    os.makedirs(IMC_DATA_DIR, exist_ok=True)
    for seed in seeds:
        name = f"{version_base}_ind_seed{seed}"
        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(IMC_DATA_DIR, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--candidates", type=int, default=CANDIDATES_PER_VERSION)
    parser.add_argument("--select", type=int, default=SELECT_PER_VERSION)
    args = parser.parse_args()

    # Use a deterministic sequence of seeds for candidates
    base_seeds = list(range(1001, 1001 + args.candidates))

    all_selected = {}

    for ver_base, config in VERSION_CONFIGS.items():
        max_nodes = config["max_nodes"]
        print(f"\n{'='*60}")
        print(f"Processing {ver_base} (max_nodes={max_nodes})")
        print(f"{'='*60}")

        if args.select_only:
            # Only select from existing
            print("  Scanning existing...")
            existing = scan_existing(ver_base)
            print(f"  Found {len(existing)} existing candidates")
            candidates = existing
        else:
            # Generate candidates
            candidates = {}
            for i, seed in enumerate(base_seeds):
                print(f"  [{i+1}/{args.candidates}] seed={seed}...", end=" ", flush=True)
                stats = generate_candidate(ver_base, max_nodes, seed)
                if stats:
                    candidates[seed] = stats
                    print(f"{stats['entities']}e/{stats['triples']}t/{stats['relations']}r")
                else:
                    print("SKIPPED")

            print(f"\n  Generated {len(candidates)}/{args.candidates} valid candidates")

        if args.analyze_only:
            # Just print stats
            print(f"\n  All candidates for {ver_base}:")
            for s, st in sorted(candidates.items(), key=lambda x: x[1]["entities"]):
                print(f"    seed{s}: {st['entities']}e, {st['triples']}t, {st['relations']}r")
            continue

        if len(candidates) < args.select:
            print(f"  ERROR: Only {len(candidates)} candidates, need {args.select}")
            continue

        # Select best
        best_seeds = select_best_10(candidates)
        best_stats = [candidates[s] for s in best_seeds]
        ents = [s["entities"] for s in best_stats]
        trips = [s["triples"] for s in best_stats]
        rels = [s["relations"] for s in best_stats]

        print(f"\n  Selected {len(best_seeds)} seeds: {sorted(best_seeds)}")
        print(f"  Entity range: {min(ents)}-{max(ents)} (span={max(ents)-min(ents)})")
        print(f"  Triple range: {min(trips)}-{max(trips)}")
        print(f"  Relation range: {min(rels)}-{max(rels)}")

        # Show selected
        for s in sorted(best_seeds):
            st = candidates[s]
            print(f"    seed{s}: {st['entities']}e, {st['triples']}t, {st['relations']}r")

        # Cleanup rejected
        if not args.select_only:
            print(f"\n  Cleaning up rejected...")
            clean_rejected(ver_base, best_seeds)

        # Sync to IMC
        print(f"  Syncing to IMC/data/...")
        sync_to_imc(ver_base, best_seeds)

        all_selected[ver_base] = best_seeds

    if args.analyze_only:
        return

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SELECTION SUMMARY")
    print(f"{'='*60}")
    for ver_base, seeds in all_selected.items():
        print(f"\n{ver_base}: {sorted(seeds)}")
        for seed in sorted(seeds):
            name = f"{ver_base}_ind_seed{seed}"
            st = get_stats(os.path.join(IMC_DATA_DIR, name))
            print(f"  {name}: {st['entities']}e, {st['triples']}t, {st['relations']}r")

    # Also show base versions
    print(f"\n--- Base versions ---")
    for ver_base in VERSION_CONFIGS:
        name = f"{ver_base}_ind"
        st = get_stats(os.path.join(IMC_DATA_DIR, name))
        print(f"  {name}: {st['entities']}e, {st['triples']}t, {st['relations']}r")

    print(f"\nDone. Now run:")
    print(f"  cd IMC && python generate_plm_embeddings.py --all --model roberta --aggregation sum")


if __name__ == "__main__":
    main()

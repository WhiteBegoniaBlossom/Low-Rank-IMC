"""
Batch generate 10 different seeded splits for each of fb237 v1/v2/v3 _ind datasets.
Uses the same BFS subgraph sampling methodology as grail-master's prepare_meta_data.py.

Parameters calibrated to match original dataset entity counts:
  - v1_ind: ~1093 entities  ->  max_nodes=350
  - v2_ind: ~1660 entities  ->  max_nodes=550
  - v3_ind: ~2501 entities  ->  max_nodes=800

Output directories follow the naming convention fb237_v{1,2,3}_ind_seed{seed},
matching the format expected by IMC/main-fb237.py.
"""

import subprocess
import sys
import os
import shutil
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(BASE_DIR, "generate_splits.py")
DATA_DIR = os.path.join(BASE_DIR, "grail-master", "data")
IMC_DATA_DIR = os.path.join(BASE_DIR, "IMC", "data")

VERSIONS = {
    "fb237_v1": {"max_nodes": 350},
    "fb237_v2": {"max_nodes": 550},
    "fb237_v3": {"max_nodes": 800},
}

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def get_triple_counts(directory):
    """Count entities, relations, and triples in a dataset directory."""
    entities = set()
    relations = set()
    total = 0
    for f in ['train.txt', 'valid.txt', 'test.txt']:
        fpath = os.path.join(directory, f)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    entities.add(parts[0])
                    entities.add(parts[2])
                    relations.add(parts[1])
                    total += 1
    return len(entities), len(relations), total


def run_generation(version_base, max_nodes, seed):
    """Run generate_splits.py and rename output to correct convention."""
    # The script creates {new_dataset}/ and {new_dataset}_ind/
    # We want {version_base}_ind_seed{seed}/ as the final _ind directory name
    temp_name = f"{version_base}_tmp_seed{seed}"
    final_name = f"{version_base}_ind_seed{seed}"

    cmd = [
        sys.executable, SCRIPT,
        "--dataset", "FB15K237",
        "--new_dataset", temp_name,
        "--max_nodes_per_hop", str(max_nodes),
        "--max_nodes_per_hop_test", str(max_nodes),
        "--seed", str(seed),
    ]
    print(f"\n{'='*60}")
    print(f"Generating: {final_name}  (max_nodes={max_nodes}, seed={seed})")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Failed to generate {final_name}")
        return None

    # Rename {temp_name}_ind -> {final_name}
    src_ind = os.path.join(DATA_DIR, temp_name + "_ind")
    dst_ind = os.path.join(DATA_DIR, final_name)
    if os.path.exists(dst_ind):
        shutil.rmtree(dst_ind)
    os.rename(src_ind, dst_ind)

    # Remove the transductive directory (not needed for IMC)
    src_trans = os.path.join(DATA_DIR, temp_name)
    if os.path.exists(src_trans):
        shutil.rmtree(src_trans)

    # Collect stats
    entities, relations, triples = get_triple_counts(dst_ind)
    stats = {"entities": entities, "relations": relations, "triples": triples}
    print(f"  -> {final_name}: {entities} entities, {relations} relations, {triples} triples")
    return stats


def main():
    total = len(VERSIONS) * len(SEEDS)
    count = 0
    all_stats = {}

    print("=" * 60)
    print(f"Batch generating {total} datasets ({len(VERSIONS)} versions x {len(SEEDS)} seeds)")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)

    for version_base, config in VERSIONS.items():
        max_nodes = config["max_nodes"]
        for seed in SEEDS:
            count += 1
            print(f"\n[{count}/{total}] {version_base} seed={seed}")
            stats = run_generation(version_base, max_nodes, seed)
            if stats:
                key = f"{version_base}_ind_seed{seed}"
                all_stats[key] = stats

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE: {len(all_stats)}/{total} successful")
    print(f"{'='*60}")

    if len(all_stats) < total:
        expected = {f"{v}_ind_seed{s}" for v in VERSIONS for s in SEEDS}
        missing = expected - set(all_stats.keys())
        if missing:
            print(f"Missing: {sorted(missing)}")

    # Summary table
    print(f"\n{'Dataset':<35} {'Entities':>10} {'Relations':>10} {'Triples':>10}")
    print("-" * 65)
    for v in sorted(all_stats.keys()):
        s = all_stats[v]
        print(f"{v:<35} {s['entities']:>10} {s['relations']:>10} {s['triples']:>10}")

    # Compare with originals
    print(f"\n--- Original dataset stats for comparison ---")
    for v in ["fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind"]:
        d = os.path.join(DATA_DIR, v)
        if os.path.isdir(d):
            e, r, t = get_triple_counts(d)
            print(f"{v:<35} {e:>10} {r:>10} {t:>10}")

    # Copy to IMC/data/
    print(f"\n--- Copying _ind datasets to IMC/data/ ---")
    os.makedirs(IMC_DATA_DIR, exist_ok=True)
    for key in all_stats:
        src = os.path.join(DATA_DIR, key)
        dst = os.path.join(IMC_DATA_DIR, key)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copied: {key}")

    print(f"\nDone. {len(all_stats)} datasets ready in:")
    print(f"  {DATA_DIR}")
    print(f"  {IMC_DATA_DIR}")


if __name__ == "__main__":
    main()

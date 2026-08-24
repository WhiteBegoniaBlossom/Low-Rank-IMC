"""
Compute dataset statistics for all inductive splits and write two CSV files:
  - dataset_statistics.csv: per-split detail
  - dataset_statistics_summary.csv: mean ± std per version

Usage:
    python compute_dataset_statistics.py                          # auto-detect IMC/data/
    python compute_dataset_statistics.py --data_dir ./data         # specify data dir
    python compute_dataset_statistics.py --pattern "fb237_v*_ind*" # filter directories
"""
import os
import sys
import argparse
import numpy as np
from collections import defaultdict


def count_split(dir_path):
    """Count entities, relations, and triples in train/valid/test.txt of a split directory."""
    stats = {}
    for split_name in ("train", "valid", "test"):
        file_path = os.path.join(dir_path, f"{split_name}.txt")
        if not os.path.exists(file_path):
            print(f"  WARNING: {file_path} not found, skipping")
            continue

        entities = set()
        relations = set()
        n_triples = 0

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                head, rel, tail = parts[0], parts[1], parts[2]
                entities.add(head)
                entities.add(tail)
                relations.add(rel)
                n_triples += 1

        stats[split_name] = {
            "entities": len(entities),
            "relations": len(relations),
            "triples": n_triples,
        }

    return stats


def extract_version_seed(dir_name):
    """Parse version and seed from directory name like 'fb237_v1_ind_seed28' or 'fb237_v1_ind_nr10_seed28'."""
    # Try to match: fb237_v<N>_ind[_...]_seed<M>
    parts = dir_name.split("_")
    version = None
    seed = None
    for i, p in enumerate(parts):
        if p.startswith("fb237-v") or (p == "fb237" and i + 1 < len(parts) and parts[i + 1].startswith("v")):
            if p == "fb237":
                version = f"fb237_{parts[i + 1]}"
            else:
                version = p.replace("-", "_")
        if p.startswith("seed"):
            try:
                seed = int(p[4:])
            except ValueError:
                pass
    # Fallback: look for fb237_vN pattern
    if version is None:
        for p in parts:
            if p.startswith("fb237"):
                version = p
                break
    if version is None:
        # Try to reconstruct
        if "fb237" in parts:
            idx = parts.index("fb237")
            if idx + 1 < len(parts):
                version = f"fb237_{parts[idx + 1]}"
    return version, seed


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for inductive splits")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing split subdirectories (default: IMC/data/)")
    parser.add_argument("--pattern", type=str, default="fb237_v*_ind*",
                        help="Glob pattern to match split directories")
    parser.add_argument("--output_detail", type=str, default=None,
                        help="Path for detail CSV (default: data_dir/dataset_statistics.csv)")
    parser.add_argument("--output_summary", type=str, default=None,
                        help="Path for summary CSV (default: data_dir/dataset_statistics_summary.csv)")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), "data")

    # Find all matching directories
    import glob
    pattern = os.path.join(args.data_dir, args.pattern)
    all_dirs = sorted(glob.glob(pattern))

    # Filter: only directories that contain train.txt, and are NOT _meta directories
    split_dirs = []
    for d in all_dirs:
        if not os.path.isdir(d):
            continue
        if d.endswith("_meta"):
            continue
        if os.path.exists(os.path.join(d, "train.txt")):
            split_dirs.append(d)

    if not split_dirs:
        print(f"No split directories found matching: {pattern}")
        print("Make sure train.txt exists in each split directory.")
        sys.exit(1)

    # Collect per-split stats
    detail_rows = []
    version_data = defaultdict(list)

    for d in split_dirs:
        dir_name = os.path.basename(d)
        version, seed = extract_version_seed(dir_name)
        if version is None:
            print(f"  WARNING: Could not parse version from '{dir_name}', skipping")
            continue
        if seed is None:
            print(f"  WARNING: Could not parse seed from '{dir_name}', skipping")
            continue

        print(f"Processing {dir_name} ...")
        stats = count_split(d)
        if len(stats) != 3:
            print(f"  WARNING: Incomplete splits in {dir_name}, skipping")
            continue

        train = stats["train"]
        valid = stats["valid"]
        test = stats["test"]
        total_triples = train["triples"] + valid["triples"] + test["triples"]

        detail_rows.append({
            "version": version,
            "seed": seed,
            "train_entities": train["entities"],
            "train_relations": train["relations"],
            "train_triples": train["triples"],
            "valid_entities": valid["entities"],
            "valid_relations": valid["relations"],
            "valid_triples": valid["triples"],
            "test_entities": test["entities"],
            "test_relations": test["relations"],
            "test_triples": test["triples"],
            "total_triples": total_triples,
        })

        version_data[version].append({
            "train_entities": train["entities"],
            "train_relations": train["relations"],
            "train_triples": train["triples"],
            "valid_entities": valid["entities"],
            "valid_relations": valid["relations"],
            "valid_triples": valid["triples"],
            "test_entities": test["entities"],
            "test_relations": test["relations"],
            "test_triples": test["triples"],
            "total_triples": total_triples,
        })

    # Write detail CSV
    detail_path = args.output_detail or os.path.join(args.data_dir, "dataset_statistics.csv")
    fields = ["version", "seed", "train_entities", "train_relations", "train_triples",
              "valid_entities", "valid_relations", "valid_triples",
              "test_entities", "test_relations", "test_triples", "total_triples"]

    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for row in detail_rows:
            f.write(",".join(str(row[f]) for f in fields) + "\n")
    print(f"\nDetail CSV written to: {detail_path} ({len(detail_rows)} splits)")

    # Write summary CSV
    summary_path = args.output_summary or os.path.join(args.data_dir, "dataset_statistics_summary.csv")
    metric_fields = ["train_entities", "train_relations", "train_triples",
                     "valid_entities", "valid_relations", "valid_triples",
                     "test_entities", "test_relations", "test_triples", "total_triples"]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("version,seeds," + ",".join(metric_fields) + "\n")
        for version in sorted(version_data.keys()):
            data = version_data[version]
            n = len(data)
            row = [version, str(n)]
            for mf in metric_fields:
                vals = [d[mf] for d in data]
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)  # sample std
                row.append(f"{mean:.0f} +/- {std:.0f}")
            f.write(",".join(row) + "\n")
    print(f"Summary CSV written to: {summary_path} ({len(version_data)} versions)")


if __name__ == "__main__":
    main()

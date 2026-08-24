"""
Regenerate random seed splits that match the original _ind split characteristics.

Instead of BFS-based subgraph sampling (which produces overly dense subgraphs),
this uses RANDOM NODE SAMPLING from the full FB15K237 graph. Random sampling
naturally produces sparser subgraphs (edges/entity ≈ 1-2, matching original).

Usage:
  python regenerate_splits.py                    # generate + select (all 3 versions)
  python regenerate_splits.py --analyze-only     # analyze existing seed splits
  python regenerate_splits.py --version v1       # only v1
"""
import os
import sys
import math
import argparse
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "IMC", "data")

# Full FB15K237 location
FB15K_DIR = os.path.join(BASE_DIR, "grail-master", "data", "FB15K237")

# Target stats from the ORIGINAL _ind splits (these are what we want to match)
TARGET_STATS = {
    "fb237_v1_ind": {"entities": 1093, "triples": 1993, "relations": 142},
    "fb237_v2_ind": {"entities": 1660, "triples": 4145, "relations": 172},
    "fb237_v3_ind": {"entities": 2501, "triples": 7406, "relations": 183},
}

# Number of candidates and selections
CANDIDATES_PER_VERSION = 50
SELECT_PER_VERSION = 10


def load_full_fb237():
    """Load all triples from the full FB15K237 dataset."""
    triples = []
    entities = set()
    relations = set()

    train_path = os.path.join(FB15K_DIR, "train.txt")
    valid_path = os.path.join(FB15K_DIR, "valid.txt")
    test_path = os.path.join(FB15K_DIR, "test.txt")

    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    triples.append((h, r, t))
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)

    return triples, list(entities), relations


def generate_split(fb_triples, entity_list, target_stats, seed,
                   valid_ratio=0.1, test_ratio=0.1):
    """
    Generate a single split by randomly sampling entities from the full graph.

    Args:
        fb_triples: list of (head, rel, tail) from full FB15K237
        entity_list: list of all entities in FB15K237
        target_stats: dict with target 'entities' count
        seed: random seed
        valid_ratio, test_ratio: split ratios

    Returns:
        dict with train/valid/test triples and stats, or None if too few triples
    """
    rng = np.random.RandomState(seed)

    # Sample entities (target the original's entity count)
    target_n = target_stats["entities"]
    # Slightly oversample to account for isolated entities
    n_sample = int(target_n * 1.05)
    sampled_ents = set(rng.choice(entity_list, size=min(n_sample, len(entity_list)),
                                   replace=False))

    # Extract all triples whose head AND tail are in the sampled set
    extracted = []
    extracted_rels = set()
    extracted_ents = set()
    for h, r, t in fb_triples:
        if h in sampled_ents and t in sampled_ents:
            extracted.append((h, r, t))
            extracted_rels.add(r)
            extracted_ents.add(h)
            extracted_ents.add(t)

    if len(extracted) < 50:
        return None  # Too few triples

    # Shuffle and split
    indices = rng.permutation(len(extracted))
    n_test = max(1, int(math.ceil(len(extracted) * test_ratio)))
    n_valid = max(1, int(math.ceil(len(extracted) * valid_ratio)))
    n_train = len(extracted) - n_valid - n_test

    if n_train < 10:
        return None

    train_triples = [extracted[i] for i in indices[:n_train]]
    valid_triples = [extracted[i] for i in indices[n_train:n_train + n_valid]]
    test_triples = [extracted[i] for i in indices[n_train + n_valid:]]

    return {
        "train": train_triples,
        "valid": valid_triples,
        "test": test_triples,
        "stats": {
            "entities": len(extracted_ents),
            "triples": n_train,
            "relations": len(extracted_rels),
        },
    }


def save_split(output_dir, split_data):
    """Save train/valid/test.txt to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for name in ["train", "valid", "test"]:
        path = os.path.join(output_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in split_data[name]:
                f.write(f"{h}\t{r}\t{t}\n")


def score_candidate(stats, target_stats):
    """
    Score a candidate by how close it is to the target.
    Lower is better. Uses normalized Euclidean distance.
    """
    score = 0.0
    for key in ["entities", "triples", "relations"]:
        if key in stats and key in target_stats:
            # Normalize by target value
            diff = (stats[key] - target_stats[key]) / max(target_stats[key], 1)
            score += diff * diff
    return score


def select_best(candidates, n_select=10):
    """Select the n_select candidates with the best scores."""
    sorted_candidates = sorted(candidates, key=lambda x: x["score"])
    return sorted_candidates[:n_select]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate seed splits matching original _ind characteristics"
    )
    parser.add_argument("--version", type=str, default=None,
                        choices=["v1", "v2", "v3"],
                        help="Only process one version")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing splits")
    parser.add_argument("--candidates", type=int, default=CANDIDATES_PER_VERSION)
    parser.add_argument("--select", type=int, default=SELECT_PER_VERSION)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without saving")
    args = parser.parse_args()

    print("Loading full FB15K237...")
    fb_triples, entity_list, fb_relations = load_full_fb237()
    print(f"  {len(fb_triples)} triples, {len(entity_list)} entities, "
          f"{len(fb_relations)} relations")

    versions = ["v1", "v2", "v3"] if args.version is None else [args.version]
    all_selected = {}

    for ver in versions:
        base_name = f"fb237_{ver}_ind"
        if base_name not in TARGET_STATS:
            print(f"WARNING: No target stats for {base_name}, skipping")
            continue

        target = TARGET_STATS[base_name]
        print(f"\n{'='*60}")
        print(f"Processing {base_name}")
        print(f"  Target: {target['entities']}e / {target['triples']}t / "
              f"{target['relations']}r")
        print(f"{'='*60}")

        if args.analyze_only:
            # Analyze existing
            prefix = f"{base_name}_seed"
            existing = []
            for d in os.listdir(DATA_DIR):
                if d.startswith(prefix) and os.path.isdir(os.path.join(DATA_DIR, d)):
                    seed_str = d[len(prefix):]
                    try:
                        seed = int(seed_str)
                    except ValueError:
                        continue
                    ents = set()
                    rels = set()
                    n = 0
                    train_path = os.path.join(DATA_DIR, d, "train.txt")
                    if not os.path.exists(train_path):
                        continue
                    with open(train_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 3:
                                ents.add(parts[0])
                                ents.add(parts[2])
                                rels.add(parts[1])
                                n += 1
                    stats = {"entities": len(ents), "triples": n,
                             "relations": len(rels)}
                    sc = score_candidate(stats, target)
                    existing.append({"seed": seed, "stats": stats, "score": sc})
                    ratio = n / max(len(ents), 1)
                    print(f"  seed{seed}: {stats['entities']}e / {stats['triples']}t "
                          f"/ {stats['relations']}r | e/e={ratio:.1f} | score={sc:.4f}")
            print(f"\n  Found {len(existing)} existing seed splits")
            continue

        # Generate candidates
        print(f"  Generating {args.candidates} candidates...")
        candidates = []
        base_seeds = list(range(2001, 2001 + args.candidates))

        for i, seed in enumerate(base_seeds):
            split = generate_split(fb_triples, entity_list, target, seed)
            if split is None:
                print(f"    [{i+1}/{args.candidates}] seed={seed}: SKIPPED (too few triples)")
                continue

            stats = split["stats"]
            sc = score_candidate(stats, target)
            candidates.append({"seed": seed, "split": split, "stats": stats,
                               "score": sc})
            ratio = stats["triples"] / max(stats["entities"], 1)
            print(f"    [{i+1}/{args.candidates}] seed={seed}: "
                  f"{stats['entities']}e / {stats['triples']}t / "
                  f"{stats['relations']}r | e/e={ratio:.1f} | score={sc:.4f}")

        if len(candidates) < args.select:
            print(f"  ERROR: Only {len(candidates)} valid candidates, "
                  f"need {args.select}")
            continue

        # Select best
        best = select_best(candidates, args.select)
        print(f"\n  Selected {len(best)} seeds:")
        for entry in best:
            s = entry["stats"]
            ratio = s["triples"] / max(s["entities"], 1)
            print(f"    seed{entry['seed']}: {s['entities']}e / {s['triples']}t "
                  f"/ {s['relations']}r | e/e={ratio:.1f} | score={entry['score']:.4f}")

        if args.dry_run:
            print(f"\n  [DRY RUN] Would save {len(best)} splits to IMC/data/")
            all_selected[base_name] = [e["seed"] for e in best]
            continue

        # Remove old seed splits for this version
        prefix = f"{base_name}_seed"
        for d in os.listdir(DATA_DIR):
            if d.startswith(prefix) and os.path.isdir(os.path.join(DATA_DIR, d)):
                old_path = os.path.join(DATA_DIR, d)
                print(f"  Removing old: {d}")
                shutil.rmtree(old_path)

        # Save new splits
        for entry in best:
            seed = entry["seed"]
            dir_name = f"{base_name}_seed{seed}"
            output_dir = os.path.join(DATA_DIR, dir_name)
            save_split(output_dir, entry["split"])
            print(f"  Saved: {dir_name}/")

        all_selected[base_name] = [e["seed"] for e in best]

    if args.analyze_only or args.dry_run:
        return

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SELECTION SUMMARY")
    print(f"{'='*60}")
    for base_name, seeds in all_selected.items():
        print(f"\n{base_name}: {sorted(seeds)}")
        for seed in sorted(seeds):
            dir_name = f"{base_name}_seed{seed}"
            dd = os.path.join(DATA_DIR, dir_name)
            ents = set()
            rels = set()
            n = 0
            with open(os.path.join(dd, "train.txt")) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        ents.add(parts[0])
                        ents.add(parts[2])
                        rels.add(parts[1])
                        n += 1
            ratio = n / max(len(ents), 1)
            print(f"  {dir_name}: {len(ents)}e / {n}t / {len(rels)}r | e/e={ratio:.1f}")

    print(f"\nOriginal targets:")
    for name, t in TARGET_STATS.items():
        print(f"  {name}: {t['entities']}e / {t['triples']}t / {t['relations']}r")

    print(f"\nDone! Next steps:")
    print(f"  python run_all.py --version <version>_ind --model roberta --aggregation sum")
    print(f"  (Re-generate embeddings first if using a different PLM model)")


if __name__ == "__main__":
    main()

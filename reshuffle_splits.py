"""
Generate multi-seed train/valid/test splits from original fb237 _ind datasets.

Method:
1. Keep the BFS-sampled subgraph fixed (entities + triples unchanged)
2. Shuffle triples with different seeds, split into train/valid/test
3. Apply clean_data postprocessing: move valid/test triples with unknown
   entities/relations back to the train set (same as grail-master's clean_data.py)

This EXACTLY replicates the original two-stage pipeline:
  prepare_meta_data.py (BFS + shuffle + split)  →  clean_data.py (postprocess)

Usage:
  python reshuffle_splits.py                    # all 3 versions, 10 seeds each
  python reshuffle_splits.py --version v1        # v1 only
  python reshuffle_splits.py --version v1 --seeds 20  # 20 seeds
"""
import os, sys, math, argparse, shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMC_DIR = os.path.join(BASE_DIR, "IMC", "data")

VERSIONS = ["v1", "v2", "v3"]


def load_triples(directory):
    """Load all triples from train.txt, valid.txt, test.txt in a directory."""
    triples = []
    for fname in ["train.txt", "valid.txt", "test.txt"]:
        fpath = os.path.join(directory, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples


def save_split(output_dir, train, valid, test):
    """Save train/valid/test.txt to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train.txt", train), ("valid.txt", valid), ("test.txt", test)]:
        with open(os.path.join(output_dir, name), "w") as f:
            for h, r, t in data:
                f.write(f"{h}\t{r}\t{t}\n")


def reshuffle(triples, seed, valid_ratio=0.1, test_ratio=0.1):
    """
    Shuffle triples with np.random and split into train/valid/test.
    Exact same logic as grail-master's get_splits().
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(triples))
    shuffled = [triples[i] for i in indices]

    n_test = max(1, int(math.ceil(len(shuffled) * test_ratio)))
    n_valid = max(1, int(math.ceil(len(shuffled) * valid_ratio)))
    n_train = len(shuffled) - n_valid - n_test

    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]

    return train, valid, test


def clean_split(train, valid, test):
    """
    Postprocess: move valid/test triples with unknown entities or relations
    back to the train set. Exactly replicates grail-master's clean_data.py logic.

    This ensures that valid and test sets only contain triples whose head, tail,
    and relation ALL appear in the (possibly expanded) training set.
    """
    # Collect initial train entities and relations
    train_ent = set()
    train_rels = set()
    for h, r, t in train:
        train_ent.add(h)
        train_ent.add(t)
        train_rels.add(r)

    # Process valid: move unknown triples to train
    filtered_valid = []
    for h, r, t in valid:
        if h in train_ent and r in train_rels and t in train_ent:
            filtered_valid.append((h, r, t))
        else:
            train.append((h, r, t))
            train_ent.add(h)
            train_ent.add(t)
            train_rels.add(r)

    # Process test: move unknown triples to train
    filtered_test = []
    for h, r, t in test:
        if h in train_ent and r in train_rels and t in train_ent:
            filtered_test.append((h, r, t))
        else:
            train.append((h, r, t))
            train_ent.add(h)
            train_ent.add(t)
            train_rels.add(r)

    return train, filtered_valid, filtered_test


def reshuffle_with_clean(triples, seed, valid_ratio=0.1, test_ratio=0.1):
    """
    Full pipeline: reshuffle + split + clean postprocessing.
    Returns (train, valid, test) lists of triples.
    """
    train, valid, test = reshuffle(triples, seed, valid_ratio, test_ratio)
    train, valid, test = clean_split(train, valid, test)
    return train, valid, test


def main():
    parser = argparse.ArgumentParser(
        description="Reshuffle original _ind splits with different seeds")
    parser.add_argument("--version", type=str, default=None, choices=VERSIONS)
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seed splits per version")
    parser.add_argument("--base-seed", type=int, default=7001,
                        help="Starting seed for shuffling")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    versions = VERSIONS if args.version is None else [args.version]

    for ver in versions:
        base_name = f"fb237_{ver}_ind"

        # Load original triples (all three splits combined)
        src_dir = os.path.join(IMC_DIR, base_name)
        if not os.path.isdir(src_dir):
            print(f"  {base_name}: source not found at {src_dir}, skipping")
            continue

        triples = load_triples(src_dir)
        entities = set()
        relations = set()
        for h, r, t in triples:
            entities.add(h); entities.add(t); relations.add(r)

        print(f"\n{'='*60}")
        print(f"{base_name}: {len(entities)} entities, {len(relations)} relations, "
              f"{len(triples)} triples")
        print(f"{'='*60}")

        seeds = list(range(args.base_seed, args.base_seed + args.seeds))
        print(f"  Generating {len(seeds)} seeds: {seeds[0]}..{seeds[-1]}")

        if args.dry_run:
            print(f"  [DRY RUN] Would save to IMC/data/{base_name}_seedXXXX/")
            continue

        # Remove old seed splits for this version
        prefix = f"{base_name}_seed"
        for d in os.listdir(IMC_DIR):
            if d.startswith(prefix) and os.path.isdir(os.path.join(IMC_DIR, d)):
                shutil.rmtree(os.path.join(IMC_DIR, d))
                print(f"  Removed old: {d}")

        # Generate new seed splits
        saved = []
        for seed in seeds:
            train, valid, test = reshuffle_with_clean(triples, seed)
            dir_name = f"{base_name}_seed{seed}"
            output_dir = os.path.join(IMC_DIR, dir_name)
            save_split(output_dir, train, valid, test)
            saved.append(seed)

            # Compute stats
            train_ent = set(); train_rel = set()
            for h, r, t in train:
                train_ent.add(h); train_ent.add(t); train_rel.add(r)
            print(f"  Saved {dir_name}/ ({len(train)}t/{len(valid)}v/{len(test)}e "
                  f"— {len(train_ent)} train entities, {len(train_rel)} train rels)")

        print(f"  Done: {len(saved)} seeds for {base_name}")

    if not args.dry_run:
        print(f"\n{'='*60}")
        print("ALL DONE. Next steps:")
        print(f"{'='*60}")
        print(f"  1. Update seed lists if seed range changed")
        print(f"  2. Generate PLM embeddings for new seed splits")
        print(f"  3. Run experiments: python run_all.py --version fb237_v1_ind --model qwen --aggregation sum")


if __name__ == "__main__":
    main()

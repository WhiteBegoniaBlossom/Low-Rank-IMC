"""
Generate BFS-based seed splits that match the original fb237_v1_ind statistics.

Strategy:
1. Use BFS subgraph sampling (same as grail-master's prepare_meta_data.py)
2. Calibrate max_nodes_per_hop to roughly match target entity count
3. If the extracted subgraph has too many edges, subsample edges to match
   the target edges-per-entity ratio — this preserves BFS community structure
   while matching the original's sparsity.
4. Generate N candidates per version, select the best M.

Usage:
  python generate_bfs_splits_v2.py                    # generate all 3 versions
  python generate_bfs_splits_v2.py --version v1        # only v1
  python generate_bfs_splits_v2.py --version v1 --seeds 5  # 5 seeds for v1
  python generate_bfs_splits_v2.py --analyze-only      # analyze existing
"""

import os
import sys
import math
import random
import argparse
import shutil
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import csc_matrix

# ============================================================
# BFS functions (replicated from grail-master)
# ============================================================

def _sp_row_vec_from_idx_list(idx_list, dim):
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])
    return neighbors


def _bfs_relational(adj, roots, max_nodes_per_hop=None, rng=None):
    """
    BFS for graphs. Uses a local RandomState for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()

    visited = set()
    current_lvl = set(roots)

    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            # Use rng.choice for deterministic sampling
            next_lvl_list = sorted(next_lvl)
            indices = rng.choice(len(next_lvl_list), size=max_nodes_per_hop, replace=False)
            next_lvl = set(next_lvl_list[i] for i in indices)

        yield next_lvl
        current_lvl = set.union(next_lvl)


def incidence_matrix(adj_list):
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


# ============================================================
# Data loading
# ============================================================

def process_files(files):
    entity2id = {}
    relation2id = {}
    triplets = {}
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (triplets['train'][:, 0][idx].squeeze(1),
              triplets['train'][:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))
        ))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


# ============================================================
# Subgraph extraction
# ============================================================

def get_active_relations(adj_list):
    act_rels = []
    for r, adj in enumerate(adj_list):
        if len(adj.tocoo().row.tolist()) > 0:
            act_rels.append(r)
    return act_rels


def get_subgraph_nodes(adj_list, hops, max_nodes_per_hop, n_roots, rng):
    """BFS from random roots, return list of sampled node IDs."""
    A_incidence = incidence_matrix(adj_list)

    idx = rng.choice(range(len(A_incidence.tocoo().row)), size=n_roots, replace=False)
    roots = set(
        [A_incidence.tocoo().row[i] for i in idx] +
        [A_incidence.tocoo().col[i] for i in idx]
    )

    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop, rng)
    lvls = []
    for _ in range(hops):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            break

    nodes = list(roots) + list(set().union(*lvls))
    return nodes


def mask_nodes(adj_list, nodes):
    """Zero out all edges incident to the given nodes."""
    masked_adj_list = [adj.copy() for adj in adj_list]
    for node in nodes:
        for adj in masked_adj_list:
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsr()
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsc()
    for adj in masked_adj_list:
        adj.eliminate_zeros()
    return masked_adj_list


def extract_triplets(adj_list, nodes, valid_rels):
    """Extract all triplets between given nodes, restricted to valid_rels."""
    subgraph = [adj[nodes, :][:, nodes] for adj in adj_list]
    active_rels = get_active_relations(subgraph)
    common_rels = list(set(active_rels).intersection(set(valid_rels)))

    all_triplets = []
    for r in common_rels:
        for (i, j) in zip(subgraph[r].tocoo().row, subgraph[r].tocoo().col):
            all_triplets.append([nodes[i], nodes[j], r])
    all_triplets = np.array(all_triplets)

    # Remove self-connections
    ind = np.argwhere(all_triplets[:, 0] == all_triplets[:, 1])
    all_triplets = np.delete(all_triplets, ind, axis=0)

    return all_triplets, common_rels


# ============================================================
# Edge subsampling to match target density
# ============================================================

def subsample_edges(triplets, target_total, rng):
    """
    Randomly subsample edges to match target total count.
    Preserves the train/valid/test ratio.
    """
    if len(triplets) <= target_total:
        return triplets  # Already sparse enough, no subsampling needed

    indices = rng.choice(len(triplets), size=target_total, replace=False)
    return triplets[indices]


# ============================================================
# Main generation
# ============================================================

def generate_one_seed(adj_list, params, seed):
    """
    Generate a single _ind dataset using BFS methodology.

    Returns:
        dict with train/valid/test triples and stats, or None on failure.
    """
    rng = np.random.RandomState(seed)

    # Stage 1: sample train graph nodes
    meta_train_nodes = get_subgraph_nodes(
        adj_list, params['hops'], params['max_nodes_train'],
        params['n_roots'], rng
    )

    # Stage 2: mask and sample test graph nodes
    masked_adj_list = mask_nodes(adj_list, meta_train_nodes)
    meta_test_nodes = get_subgraph_nodes(
        masked_adj_list, params['hops_test'] + 1, params['max_nodes_test'],
        params['n_roots'], rng
    )

    # Check disjointness
    overlap = set(meta_train_nodes).intersection(set(meta_test_nodes))
    if overlap:
        return None  # Should not happen with proper masking

    # Extract triplets from test nodes using train-active relations
    # First get train active relations
    _, train_active_rels = extract_triplets(
        adj_list, meta_train_nodes, range(len(adj_list))
    )

    if len(train_active_rels) < 5:
        return None  # Too few relations

    # Extract triplets for the inductive graph
    all_triplets, active_rels = extract_triplets(
        adj_list, meta_test_nodes, train_active_rels
    )

    if len(all_triplets) < 50:
        return None

    # Edge subsampling to match target density (only if target is set)
    target_total = params.get('target_triples_total', None)
    if target_total is not None and len(all_triplets) > target_total:
        all_triplets = subsample_edges(all_triplets, target_total, rng)

    # Shuffle and split
    rng.shuffle(all_triplets)
    n_test = max(1, int(math.ceil(len(all_triplets) * params['test_ratio'])))
    n_valid = max(1, int(math.ceil(len(all_triplets) * params['valid_ratio'])))
    n_train = len(all_triplets) - n_valid - n_test

    if n_train < 10:
        return None

    train_triplets = all_triplets[:n_train]
    valid_triplets = all_triplets[n_train:n_train + n_valid]
    test_triplets = all_triplets[n_train + n_valid:]

    # Compute stats
    all_entities = set()
    all_relations = set()
    for t in train_triplets:
        all_entities.add(t[0]); all_entities.add(t[1])
        all_relations.add(t[2])  # Actually t[2] is the relation ID

    return {
        'train': train_triplets,
        'valid': valid_triplets,
        'test': test_triplets,
        'entities': len(set(
            [t[0] for t in train_triplets] + [t[1] for t in train_triplets]
        )),
        'relations': len(set(t[2] for t in train_triplets)),
        'train_count': n_train,
        'valid_count': n_valid,
        'test_count': n_test,
        'total': len(all_triplets),
    }


def calibrate_max_nodes(adj_list, params, seed, target, tolerance=0.3):
    """
    Search for max_nodes that gives closest stats to target.
    Applies edge subsampling during calibration so scores are accurate.
    """
    target_entities = target['entities']
    target_total = target['total_triples']

    best_mn = None
    best_score = float('inf')
    best_result = None

    # Try different n_roots values to capture more diverse graph regions
    best_overall_mn = None
    best_overall_score = float('inf')
    best_overall_result = None

    for n_roots in [1, 2, 3]:
        trial_params = dict(params)
        trial_params['n_roots'] = n_roots
        trial_params['target_triples_total'] = target_total

        # Search max_nodes for this n_roots
        search_range = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800]

        for mn in search_range:
            trial_params['max_nodes_train'] = mn
            trial_params['max_nodes_test'] = mn

            result = generate_one_seed(adj_list, trial_params, seed)
            if result is None:
                continue

            e_diff = abs(result['entities'] - target_entities) / target_entities
            t_diff = abs(result['total'] - target_total) / target_total
            r_diff = abs(result['relations'] - target['relations']) / max(target['relations'], 1)
            score = e_diff * 1.0 + t_diff * 0.5 + r_diff * 0.3

            if score < best_overall_score:
                best_overall_score = score
                best_overall_mn = mn
                best_overall_result = result
                # Store the best params
                params['best_n_roots'] = n_roots
                params['best_score'] = score

            if e_diff < tolerance and t_diff < tolerance:
                break

        if best_overall_score < tolerance:
            break

    return best_overall_mn, best_overall_result

    for mn in search_range:
        trial_params = dict(params)
        trial_params['max_nodes_train'] = mn
        trial_params['max_nodes_test'] = mn
        trial_params['target_triples_total'] = target_total  # enable edge subsampling

        result = generate_one_seed(adj_list, trial_params, seed)
        if result is None:
            continue

        # Weighted score
        e_diff = abs(result['entities'] - target_entities) / target_entities
        t_diff = abs(result['total'] - target_total) / target_total
        r_diff = abs(result['relations'] - target['relations']) / max(target['relations'], 1)
        score = e_diff * 1.0 + t_diff * 0.5 + r_diff * 0.3

        if score < best_score:
            best_score = score
            best_mn = mn
            best_result = result

        if e_diff < tolerance and t_diff < tolerance:
            break

    return best_mn, best_result


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for triple in triplets:
            s, o, r = int(triple[0]), int(triple[1]), int(triple[2])
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


# ============================================================
# Target stats (from original grail-master fb237_v1/v2/v3_ind)
# ============================================================

TARGET_CONFIGS = {
    "fb237_v1_ind": {
        "entities": 1093,
        "relations": 142,
        "train_triples": 1993,
        "total_triples": 2404,
        "max_nodes_range": [100, 400],
    },
    "fb237_v2_ind": {
        "entities": 1660,
        "relations": 172,
        "train_triples": 4145,
        "total_triples": 5092,
        "max_nodes_range": [200, 600],
    },
    "fb237_v3_ind": {
        "entities": 2501,
        "relations": 183,
        "train_triples": 7406,
        "total_triples": 9137,
        "max_nodes_range": [300, 900],
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "grail-master", "data")
IMC_DATA_DIR = os.path.join(BASE_DIR, "IMC", "data")
FB15K_DIR = os.path.join(DATA_DIR, "FB15K237")

# Default BFS parameters matching grail-master
DEFAULT_BFS_PARAMS = {
    'hops': 3,
    'hops_test': 3,
    'n_roots': 1,
    'max_nodes_train': 300,
    'max_nodes_test': 300,
    'valid_ratio': 0.1,
    'test_ratio': 0.1,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate BFS-based seed splits matching original fb237 _ind statistics"
    )
    parser.add_argument("--version", type=str, default=None,
                        choices=["v1", "v2", "v3"],
                        help="Only process one version")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seed splits to generate")
    parser.add_argument("--candidates", type=int, default=30,
                        help="Number of candidates to generate per version")
    parser.add_argument("--base-seed", type=int, default=3001,
                        help="Starting seed for candidates")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only find best max_nodes, don't generate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without saving")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyze existing seed splits")
    args = parser.parse_args()

    # Load full FB15K237
    print("Loading full FB15K237...")
    files = {
        'train': os.path.join(FB15K_DIR, 'train.txt'),
        'valid': os.path.join(FB15K_DIR, 'valid.txt'),
        'test': os.path.join(FB15K_DIR, 'test.txt'),
    }
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)
    print(f"  {len(entity2id)} entities, {len(relation2id)} relations, "
          f"{len(triplets['train'])} train triples")

    versions = ["v1", "v2", "v3"] if args.version is None else [args.version]
    all_selected = {}

    for ver in versions:
        base_name = f"fb237_{ver}_ind"
        if base_name not in TARGET_CONFIGS:
            continue

        target = TARGET_CONFIGS[base_name]
        print(f"\n{'='*60}")
        print(f"Processing {base_name}")
        print(f"  Target: {target['entities']}e / {target['train_triples']}t_train "
              f"/ {target['total_triples']}t_total / {target['relations']}r")
        print(f"{'='*60}")

        if args.analyze_only:
            prefix = f"{base_name}_seed"
            existing = []
            for d in os.listdir(IMC_DATA_DIR):
                if d.startswith(prefix) and os.path.isdir(os.path.join(IMC_DATA_DIR, d)):
                    ents, rels, tr = set(), set(), 0
                    train_path = os.path.join(IMC_DATA_DIR, d, "train.txt")
                    if not os.path.exists(train_path):
                        continue
                    with open(train_path) as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                ents.add(parts[0]); ents.add(parts[2])
                                rels.add(parts[1]); tr += 1
                    print(f"  {d}: {len(ents)}e {len(rels)}r {tr}t_train")
            continue

        # Phase 1: calibrate max_nodes
        print(f"\n  Phase 1: Calibrating max_nodes_per_hop...")

        # Try a few calibration seeds to find robust max_nodes
        calib_seeds = [args.base_seed, args.base_seed + 100, args.base_seed + 200]
        calib_results = []

        for calib_seed in calib_seeds:
            print(f"    Calibrating with seed={calib_seed}...")
            best_mn, best_stats = calibrate_max_nodes(
                adj_list, DEFAULT_BFS_PARAMS, calib_seed, target
            )
            if best_mn is not None:
                calib_results.append((calib_seed, best_mn, best_stats))
                print(f"      best max_nodes={best_mn}: {best_stats['entities']}e "
                      f"{best_stats['total']}t (target: {target['entities']}e {target['total_triples']}t)")

        if not calib_results:
            print(f"  ERROR: Calibration failed for {base_name}")
            continue

        # Use median max_nodes
        mns = sorted([r[1] for r in calib_results])
        median_mn = mns[len(mns) // 2]
        print(f"\n  Calibrated max_nodes_per_hop = {median_mn} (median of {mns})")

        if args.calibrate_only:
            continue

        # Phase 2: generate candidates
        print(f"\n  Phase 2: Generating {args.candidates} candidates...")

        params = dict(DEFAULT_BFS_PARAMS)
        params['max_nodes_train'] = median_mn
        params['max_nodes_test'] = median_mn
        params['target_triples_total'] = target['total_triples']

        candidates = []
        candidate_seeds = list(range(args.base_seed, args.base_seed + args.candidates))

        for i, seed in enumerate(candidate_seeds):
            result = generate_one_seed(adj_list, params, seed)
            if result is None:
                continue

            # Score: weighted distance from target
            e_diff = (result['entities'] - target['entities']) / target['entities']
            t_diff = (result['total'] - target['total_triples']) / target['total_triples']
            r_diff = (result['relations'] - target['relations']) / max(target['relations'], 1)
            score = abs(e_diff) * 1.0 + abs(t_diff) * 0.5 + abs(r_diff) * 0.3

            candidates.append({
                'seed': seed,
                'result': result,
                'score': score,
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{args.candidates}] seed={seed}: "
                      f"{result['entities']}e/{result['total']}t/{result['relations']}r "
                      f"score={score:.4f}")

        if len(candidates) < args.seeds:
            print(f"  ERROR: Only {len(candidates)} valid candidates, need {args.seeds}")
            continue

        # Phase 3: select best
        candidates.sort(key=lambda x: x['score'])
        selected = candidates[:args.seeds]

        print(f"\n  Phase 3: Selected {len(selected)} seeds:")
        for entry in selected:
            r = entry['result']
            e2e = r['total'] / max(r['entities'], 1)
            print(f"    seed={entry['seed']}: {r['entities']}e {r['total']}t "
                  f"{r['relations']}r e2e={e2e:.2f} score={entry['score']:.4f}")

        if args.dry_run:
            all_selected[base_name] = [s['seed'] for s in selected]
            continue

        # Phase 4: save to IMC/data
        print(f"\n  Phase 4: Saving to IMC/data/...")

        # Remove old seed splits for this version
        prefix = f"{base_name}_seed"
        for d in os.listdir(IMC_DATA_DIR):
            if d.startswith(prefix) and os.path.isdir(os.path.join(IMC_DATA_DIR, d)):
                old_path = os.path.join(IMC_DATA_DIR, d)
                print(f"    Removing old: {d}")
                shutil.rmtree(old_path)

        for entry in selected:
            seed = entry['seed']
            result = entry['result']
            dir_name = f"{base_name}_seed{seed}"
            output_dir = os.path.join(IMC_DATA_DIR, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            save_to_file(output_dir, 'train.txt', result['train'], id2entity, id2relation)
            save_to_file(output_dir, 'valid.txt', result['valid'], id2entity, id2relation)
            save_to_file(output_dir, 'test.txt', result['test'], id2entity, id2relation)
            print(f"    Saved: {dir_name}/")

        all_selected[base_name] = [s['seed'] for s in selected]

    if args.analyze_only or args.dry_run or args.calibrate_only:
        return

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nOriginal targets:")
    for name, t in TARGET_CONFIGS.items():
        print(f"  {name}: {t['entities']}e / {t['train_triples']}t / {t['relations']}r / {t['total_triples']}total")

    print(f"\nGenerated seed splits in IMC/data/:")
    for base_name, seeds in all_selected.items():
        print(f"\n  {base_name}:")
        for seed in sorted(seeds):
            d = os.path.join(IMC_DATA_DIR, f"{base_name}_seed{seed}")
            ents, rels, tr = set(), set(), 0
            with open(os.path.join(d, "train.txt")) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        ents.add(parts[0]); ents.add(parts[2])
                        rels.add(parts[1]); tr += 1
            e2e = tr / max(len(ents), 1)
            print(f"    seed{seed}: {len(ents)}e {len(rels)}r {tr}t e2e={e2e:.2f}")

    print(f"\nNext steps:")
    print(f"  1. Generate PLM embeddings:")
    for base_name in all_selected:
        for seed in sorted(all_selected[base_name]):
            print(f"     python IMC/generate_plm_embeddings.py --version {base_name}_seed{seed} --model roberta --aggregation sum")
    print(f"  2. Run experiments:")
    print(f"     python run_all.py --version fb237_v1_ind --model roberta --aggregation sum")


if __name__ == "__main__":
    main()

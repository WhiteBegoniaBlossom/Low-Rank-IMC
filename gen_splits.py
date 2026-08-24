"""
Generate BFS-based seed splits matching original _ind statistics.

Core methodology (matching grail-master):
1. BFS sample train subgraph from full graph
2. Mask train nodes, BFS sample test subgraph from remainder
3. Extract all edges between test nodes, restricted to train-active relations
4. Edge subsample to match target density
5. Split into train/valid/test

Usage:
  python gen_splits.py                    # generate v1/v2/v3, 10 seeds each
  python gen_splits.py --version v1       # v1 only
  python gen_splits.py --version v1 --n_roots 2 --max_nodes 400 --seeds 10
  python gen_splits.py --dry-run          # preview without saving
"""
import os, sys, math, argparse, shutil
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import csc_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "grail-master", "data")
IMC_DIR = os.path.join(BASE_DIR, "IMC", "data")
FB15K_DIR = os.path.join(DATA_DIR, "FB15K237")

# Target stats from original grail-master fb237_v1/v2/v3_ind
TARGETS = {
    "v1": {"entities": 1093, "relations": 142, "total": 2404, "train": 1993,
           "n_roots": 2, "max_nodes": 400},
    "v2": {"entities": 1660, "relations": 172, "total": 5092, "train": 4145,
           "n_roots": 2, "max_nodes": 550},
    "v3": {"entities": 2501, "relations": 183, "total": 9137, "train": 7406,
           "n_roots": 2, "max_nodes": 700},
}

# ---- sparse matrix helpers ----

def _sp_row_vec(idx_list, dim):
    data = np.ones(len(idx_list))
    return ssp.csr_matrix((data, (np.zeros(len(idx_list)), list(idx_list))), shape=(1, dim))

def _get_neighbors(adj, nodes):
    sp = _sp_row_vec(list(nodes), adj.shape[1])
    return set(ssp.find(sp.dot(adj))[1])

def _bfs_relational(adj, roots, max_per_hop, rng):
    visited = set()
    current = set(roots)
    while current:
        for v in current:
            visited.add(v)
        nxt = _get_neighbors(adj, current)
        nxt -= visited
        if max_per_hop and max_per_hop < len(nxt):
            lst = sorted(nxt)
            nxt = set(lst[i] for i in rng.choice(len(lst), size=max_per_hop, replace=False))
        yield nxt
        current = set.union(nxt)

def incidence_matrix(adj_list):
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        coo = adj.tocoo()
        rows += coo.row.tolist(); cols += coo.col.tolist(); dats += coo.data.tolist()
    return ssp.csc_matrix((np.array(dats), (np.array(rows), np.array(cols))), shape=dim)

# ---- data loading ----

def load_fb237():
    entity2id, relation2id = {}, {}
    ent, rel = 0, 0
    triplets = {}

    for split in ['train', 'valid', 'test']:
        data = []
        with open(os.path.join(FB15K_DIR, f'{split}.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3: continue
                h, r, t = parts
                for x in [h, t]:
                    if x not in entity2id:
                        entity2id[x] = ent; ent += 1
                if r not in relation2id:
                    relation2id[r] = rel; rel += 1
                data.append([entity2id[h], entity2id[t], relation2id[r]])
        triplets[split] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (triplets['train'][:, 0][idx].squeeze(1),
              triplets['train'][:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation

# ---- subgraph extraction ----

def get_subgraph_nodes(adj_list, hops, max_per_hop, n_roots, rng):
    A = incidence_matrix(adj_list)
    idx = rng.choice(len(A.tocoo().row), size=n_roots, replace=False)
    roots = set([A.tocoo().row[i] for i in idx] + [A.tocoo().col[i] for i in idx])
    gen = _bfs_relational(A, roots, max_per_hop, rng)
    lvls = []
    for _ in range(hops):
        try: lvls.append(next(gen))
        except StopIteration: break
    return list(roots) + list(set().union(*lvls))

def mask_nodes(adj_list, nodes):
    result = [adj.copy() for adj in adj_list]
    for node in nodes:
        for adj in result:
            adj.data[adj.indptr[node]:adj.indptr[node+1]] = 0
            adj = adj.tocsr()
            adj.data[adj.indptr[node]:adj.indptr[node+1]] = 0
            adj = adj.tocsc()
    for adj in result:
        adj.eliminate_zeros()
    return result

def extract_triplets(adj_list, nodes, valid_rels):
    sub = [adj[nodes, :][:, nodes] for adj in adj_list]
    active = [r for r, a in enumerate(sub) if len(a.tocoo().row.tolist()) > 0]
    common = list(set(active).intersection(set(valid_rels)))
    trips = []
    for r in common:
        for i, j in zip(sub[r].tocoo().row, sub[r].tocoo().col):
            trips.append([nodes[i], nodes[j], r])
    trips = np.array(trips)
    # remove self-loops
    mask_arr = trips[:, 0] != trips[:, 1]
    return trips[mask_arr], common

# ---- generation ----

def generate_one(adj_list, target, seed):
    """Generate a single _ind split. Returns dict with train/valid/test arrays and stats."""
    rng = np.random.RandomState(seed)
    n_roots = target.get('n_roots', 2)
    max_nodes = target.get('max_nodes', 400)
    target_total = target['total']

    # Stage 1: train subgraph nodes
    train_nodes = get_subgraph_nodes(adj_list, 3, max_nodes, n_roots, rng)

    # Get train-active relations
    _, train_rels = extract_triplets(adj_list, train_nodes, range(len(adj_list)))
    if len(train_rels) < 5:
        return None

    # Stage 2: mask and extract test subgraph nodes
    masked = mask_nodes(adj_list, train_nodes)
    test_nodes = get_subgraph_nodes(masked, 4, max_nodes, n_roots, rng)

    overlap = set(train_nodes).intersection(set(test_nodes))
    if overlap:
        return None

    # Stage 3: extract edges for test graph (from FULL adj_list, restricted to train_rels)
    all_trips, active_rels = extract_triplets(adj_list, test_nodes, train_rels)
    if len(all_trips) < 50:
        return None

    # Stage 4: edge subsampling to match target density
    if len(all_trips) > target_total:
        idx = rng.choice(len(all_trips), size=target_total, replace=False)
        all_trips = all_trips[idx]

    # Stage 5: shuffle and split
    rng.shuffle(all_trips)
    n_test = max(1, int(math.ceil(len(all_trips) * 0.1)))
    n_valid = max(1, int(math.ceil(len(all_trips) * 0.1)))
    n_train = len(all_trips) - n_valid - n_test
    if n_train < 10:
        return None

    train_t = all_trips[:n_train]
    valid_t = all_trips[n_train:n_train + n_valid]
    test_t = all_trips[n_train + n_valid:]

    train_ents = set([int(t[0]) for t in train_t] + [int(t[1]) for t in train_t])
    train_rels_set = set(int(t[2]) for t in train_t)

    return {
        'train': train_t, 'valid': valid_t, 'test': test_t,
        'entities': len(train_ents),
        'relations': len(train_rels_set),
        'train_count': n_train,
        'total': len(all_trips),
    }

def score_result(result, target):
    e_d = abs(result['entities'] - target['entities']) / target['entities']
    t_d = abs(result['total'] - target['total']) / target['total']
    r_d = abs(result['relations'] - target['relations']) / max(target['relations'], 1)
    return e_d * 1.0 + t_d * 0.5 + r_d * 0.3

def save_split(directory, triplets, id2entity, id2relation):
    os.makedirs(directory, exist_ok=True)
    for name, arr in [('train.txt', triplets['train']),
                       ('valid.txt', triplets['valid']),
                       ('test.txt', triplets['test'])]:
        with open(os.path.join(directory, name), 'w') as f:
            for t in arr:
                s, o, r = int(t[0]), int(t[1]), int(t[2])
                f.write(f"{id2entity[s]}\t{id2relation[r]}\t{id2entity[o]}\n")

# ---- main ----

def main():
    parser = argparse.ArgumentParser(description="Generate BFS-based seed splits")
    parser.add_argument("--version", type=str, default=None, choices=["v1", "v2", "v3"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=5001)
    parser.add_argument("--n_roots", type=int, default=None,
                        help="Override default n_roots")
    parser.add_argument("--max_nodes", type=int, default=None,
                        help="Override default max_nodes_per_hop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading FB15K237...")
    adj_list, _, entity2id, relation2id, id2entity, id2relation = load_fb237()
    print(f"  {len(entity2id)} entities, {len(relation2id)} relations")

    versions = ["v1", "v2", "v3"] if args.version is None else [args.version]
    all_selected = {}

    for ver in versions:
        target = dict(TARGETS[ver])
        if args.n_roots: target['n_roots'] = args.n_roots
        if args.max_nodes: target['max_nodes'] = args.max_nodes

        base_name = f"fb237_{ver}_ind"
        print(f"\n{'='*60}")
        print(f"{base_name}: target {target['entities']}e/{target['total']}t/{target['relations']}r")
        print(f"  n_roots={target['n_roots']}, max_nodes={target['max_nodes']}")
        print(f"{'='*60}")

        # Generate candidates
        candidates = []
        seed_start = args.base_seed
        for i in range(args.candidates):
            seed = seed_start + i
            result = generate_one(adj_list, target, seed)
            if result is None:
                continue
            sc = score_result(result, TARGETS[ver])
            e2e = result['total'] / max(result['entities'], 1)
            candidates.append({'seed': seed, 'result': result, 'score': sc})
            if (i+1) % 5 == 0 or i == 0:
                print(f"  [{len(candidates)}] seed={seed}: {result['entities']}e/{result['total']}t/"
                      f"{result['relations']}r e2e={e2e:.2f} score={sc:.4f}")

        if len(candidates) < args.seeds:
            print(f"  FAILED: only {len(candidates)} candidates, need {args.seeds}")
            continue

        # Select best
        candidates.sort(key=lambda x: x['score'])
        selected = candidates[:args.seeds]
        print(f"\n  Selected {len(selected)} seeds:")
        for entry in selected:
            r = entry['result']
            e2e = r['total'] / max(r['entities'], 1)
            print(f"    seed={entry['seed']}: {r['entities']}e/{r['total']}t/"
                  f"{r['relations']}r e2e={e2e:.2f} score={entry['score']:.4f}")

        if args.dry_run:
            all_selected[base_name] = [s['seed'] for s in selected]
            continue

        # Save
        prefix = f"{base_name}_seed"
        for d in os.listdir(IMC_DIR):
            if d.startswith(prefix) and os.path.isdir(os.path.join(IMC_DIR, d)):
                print(f"    Removing old: {d}")
                shutil.rmtree(os.path.join(IMC_DIR, d))

        for entry in selected:
            seed = entry['seed']
            dir_name = f"{base_name}_seed{seed}"
            save_split(os.path.join(IMC_DIR, dir_name), entry['result'], id2entity, id2relation)
            print(f"    Saved: {dir_name}/")

        all_selected[base_name] = [s['seed'] for s in selected]

    # Final summary
    print(f"\n{'='*60}")
    print("DONE. Generated seed splits:")
    print(f"{'='*60}")
    print(f"\n  Original targets:")
    for ver, t in TARGETS.items():
        print(f"    fb237_{ver}_ind: {t['entities']}e/{t['total']}t/{t['relations']}r e2e={t['total']/t['entities']:.2f}")
    print(f"\n  To use in experiments, update run_all.py BASE_TO_SEEDS with:")
    for base_name, seeds in all_selected.items():
        print(f"    {base_name}: {sorted(seeds)}")
    print(f"\n  Then generate PLM embeddings and run experiments.")

if __name__ == "__main__":
    main()

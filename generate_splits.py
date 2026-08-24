"""
Standalone script to generate fb237 inductive dataset splits.
Replicates the exact logic of grail-master/utils/prepare_meta_data.py
without requiring DGL (which fails to import on Windows).

Generates 10 different random seeds for each of v1/v2/v3 _ind datasets.
"""

import os
import math
import random
import argparse
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import csc_matrix


# ============================================================
# Functions replicated from dgl_utils.py (no DGL dependency)
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


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accommodate node sampling.
    """
    visited = set()
    current_lvl = set(roots)

    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(sorted(next_lvl), max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


# ============================================================
# Functions replicated from graph_utils.py
# ============================================================

def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


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


# ============================================================
# Functions replicated from data_utils.py
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

            if triplet[1] in relation2id:
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


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


# ============================================================
# Functions replicated from prepare_meta_data.py
# ============================================================

def get_active_relations(adj_list):
    act_rels = []
    for r, adj in enumerate(adj_list):
        if len(adj.tocoo().row.tolist()) > 0:
            act_rels.append(r)
    return act_rels


def get_avg_degree(adj_list):
    adj_mat = incidence_matrix(adj_list)
    degree = []
    for node in range(adj_list[0].shape[0]):
        degree.append(np.sum(adj_mat[node, :]))
    return np.mean(degree)


def get_splits(adj_list, nodes, valid_rels=None, valid_ratio=0.1, test_ratio=0.1):
    subgraph = [adj[nodes, :][:, nodes] for adj in adj_list]

    active_rels = get_active_relations(subgraph)
    common_rels = list(set(active_rels).intersection(set(valid_rels)))

    print('Average degree : ', get_avg_degree(subgraph))
    print('Nodes: ', len(nodes))
    print('Links: ', np.sum(get_edge_count(subgraph)))
    print('Active relations: ', len(common_rels))

    all_triplets = []
    for r in common_rels:
        for (i, j) in zip(subgraph[r].tocoo().row, subgraph[r].tocoo().col):
            all_triplets.append([nodes[i], nodes[j], r])
    all_triplets = np.array(all_triplets)

    ind = np.argwhere(all_triplets[:, 0] == all_triplets[:, 1])
    all_triplets = np.delete(all_triplets, ind, axis=0)
    print('Links after deleting self connections : %d' % len(all_triplets))

    np.random.shuffle(all_triplets)
    train_split = int(math.ceil(len(all_triplets) * (1 - valid_ratio - test_ratio)))
    valid_split = int(math.ceil(len(all_triplets) * (1 - test_ratio)))

    train_triplets = all_triplets[:train_split]
    valid_triplets = all_triplets[train_split:valid_split]
    test_triplets = all_triplets[valid_split:]

    return train_triplets, valid_triplets, test_triplets, common_rels


def get_subgraph(adj_list, hops, max_nodes_per_hop):
    A_incidence = incidence_matrix(adj_list)

    idx = np.random.choice(range(len(A_incidence.tocoo().row)), size=params.n_roots, replace=False)
    roots = set([A_incidence.tocoo().row[id] for id in idx] + [A_incidence.tocoo().col[id] for id in idx])

    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(hops):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            break

    nodes = list(roots) + list(set().union(*lvls))

    return nodes


def mask_nodes(adj_list, nodes):
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


# ============================================================
# Main
# ============================================================

def main(params):
    files = {
        'train': os.path.join(params.data_dir, params.dataset, 'train.txt'),
        'valid': os.path.join(params.data_dir, params.dataset, 'valid.txt'),
        'test': os.path.join(params.data_dir, params.dataset, 'test.txt'),
    }

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)

    meta_train_nodes = get_subgraph(adj_list, params.hops, params.max_nodes_per_hop)

    masked_adj_list = mask_nodes(adj_list, meta_train_nodes)

    meta_test_nodes = get_subgraph(masked_adj_list, params.hops_test + 1, params.max_nodes_per_hop_test)

    print('Common nodes among the two disjoint datasets (should ideally be zero): ',
          set(meta_train_nodes).intersection(set(meta_test_nodes)))
    tmp = [adj[meta_train_nodes, :][:, meta_train_nodes] for adj in masked_adj_list]
    print('Residual edges (should be zero) : ', np.sum(get_edge_count(tmp)))

    print("================")
    print("Train graph stats")
    print("================")
    train_triplets, valid_triplets, test_triplets, train_active_rels = get_splits(
        adj_list, meta_train_nodes, range(len(adj_list)))
    print("================")
    print("Meta-test graph stats")
    print("================")
    meta_train_triplets, meta_valid_triplets, meta_test_triplets, meta_active_rels = get_splits(
        adj_list, meta_test_nodes, train_active_rels)

    print("================")
    print('Extra rels (should be empty): ', set(meta_active_rels) - set(train_active_rels))

    # Save transductive dataset
    trans_dir = os.path.join(params.output_dir, params.new_dataset)
    if not os.path.exists(trans_dir):
        os.makedirs(trans_dir)
    save_to_file(trans_dir, 'train.txt', train_triplets, id2entity, id2relation)
    save_to_file(trans_dir, 'valid.txt', valid_triplets, id2entity, id2relation)
    save_to_file(trans_dir, 'test.txt', test_triplets, id2entity, id2relation)

    # Save inductive dataset (_ind)
    ind_dir = os.path.join(params.output_dir, params.new_dataset + '_ind')
    if not os.path.exists(ind_dir):
        os.makedirs(ind_dir)
    save_to_file(ind_dir, 'train.txt', meta_train_triplets, id2entity, id2relation)
    save_to_file(ind_dir, 'valid.txt', meta_valid_triplets, id2entity, id2relation)
    save_to_file(ind_dir, 'test.txt', meta_test_triplets, id2entity, id2relation)

    # Print summary
    ind_stats = {
        'train': len(meta_train_triplets),
        'valid': len(meta_valid_triplets),
        'test': len(meta_test_triplets),
        'total': len(meta_train_triplets) + len(meta_valid_triplets) + len(meta_test_triplets),
        'nodes': len(set(meta_test_nodes)),
        'relations': len(meta_active_rels),
    }
    print(f"\nInductive dataset ({params.new_dataset}_ind) summary:")
    for k, v in ind_stats.items():
        print(f"  {k}: {v}")

    return ind_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fb237 inductive splits')

    parser.add_argument("--dataset", "-d", type=str, default="FB15K237")
    parser.add_argument("--new_dataset", "-nd", type=str, default="fb237_test")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing the base dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output datasets")
    parser.add_argument("--n_roots", "-n", type=int, default=1)
    parser.add_argument("--hops", "-H", type=int, default=3)
    parser.add_argument("--max_nodes_per_hop", "-m", type=int, default=2500)
    parser.add_argument("--hops_test", "-HT", type=int, default=3)
    parser.add_argument("--max_nodes_per_hop_test", "-mt", type=int, default=2500)
    parser.add_argument("--seed", "-s", type=int, default=28)

    params = parser.parse_args()

    if params.data_dir is None:
        params.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'grail-master', 'data')
    if params.output_dir is None:
        params.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'grail-master', 'data')

    np.random.seed(params.seed)
    random.seed(params.seed)

    main(params)

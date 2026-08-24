"""
Subgraph extraction utilities for TyleR relation prediction.
Ported from tyler-main/test_ranking_rel.py and tyler-main/OEILP/utils/.
Uses scipy sparse matrices for BFS + DRNL labeling, networkx for DGL conversion.
"""
import random
import numpy as np
import scipy.sparse as ssp
import torch
import dgl
import networkx as nx


# ============================================================
# BFS-based subgraph extraction
# ============================================================

def incidence_matrix(adj_list):
    """Combine per-relation adjacency matrices into a single incidence matrix."""
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


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create a sparse row vector with ones at the given indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Get all neighbors of a set of nodes using sparse adjacency."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])
    return neighbors


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """BFS generator yielding node sets at each hop level."""
    visited = set()
    current_lvl = set(roots)
    next_lvl = set()
    while current_lvl:
        for v in current_lvl:
            visited.add(v)
        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited
        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(list(next_lvl), max_nodes_per_hop))
        yield next_lvl
        current_lvl = set.union(next_lvl)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    """Get all nodes within h hops of the root nodes."""
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def remove_nodes(A_incidence, nodes):
    """Remove specified nodes from an incidence matrix."""
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


# ============================================================
# Double-Radius Node Labeling (DRNL)
# ============================================================

def node_label(subgraph, max_distance=1):
    """
    Compute DRNL (double-radius node labeling) for a subgraph.
    Returns (labels, enclosing_subgraph_node_indices).
    labels[i] = [dist_to_head, dist_to_tail] for node i.
    """
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [
        np.clip(
            ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:],
            0, 1e7,
        )
        for r, sg in enumerate(sgs_single_root)
    ]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False,
                                  max_nodes_per_hop=None, max_node_label_value=None):
    """
    Extract enclosing subgraph around (head, tail) and compute DRNL labels.

    Args:
        ind: (head_id, tail_id) tuple
        rel: relation id (unused for extraction itself, kept for API compatibility)
        A_list: list of scipy CSC adjacency matrices per relation
        h: number of hops for subgraph extraction
        enclosing_sub_graph: if True, only keep nodes in intersection of head/tail neighborhoods
        max_nodes_per_hop: cap on nodes per BFS hop
        max_node_label_value: cap on DRNL label values

    Returns:
        (pruned_subgraph_nodes, pruned_labels)
    """
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([
            np.minimum(label, max_node_label_value).tolist() for label in pruned_labels
        ])

    return pruned_subgraph_nodes, pruned_labels


# ============================================================
# DGL graph construction
# ============================================================

def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Convert a list of scipy CSC adjacency matrices (one per relation)
    to a DGL heterogeneous graph via networkx.

    Args:
        graph: list of scipy CSC matrices, one per relation
        n_feats: optional node features

    Returns:
        DGL graph with edge attribute 'type' (relation id)
    """
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {"type": rel}))
        g_nx.add_edges_from(nx_triplets)
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=["type"])
    if n_feats is not None:
        g_dgl.ndata["feat"] = torch.tensor(n_feats)
    return g_dgl


# ============================================================
# Node feature preparation
# ============================================================

def prepare_features(subgraph, n_labels, r_label, max_n_label, n_feats=None):
    """
    Prepare node features for a subgraph:
    - One-hot encode DRNL labels
    - Set head/tail node IDs (1=head, 2=tail)
    - Optionally concatenate external features (e.g., PLM embeddings)

    Args:
        subgraph: DGL graph
        n_labels: DRNL labels [num_nodes, 2]
        r_label: relation label (for edge labeling)
        max_n_label: [max_dist_sub, max_dist_obj] for one-hot sizing
        n_feats: optional external features [num_nodes, feat_dim]

    Returns:
        subgraph with ndata['feat'], ndata['id'], ndata['r_label'], edata['label']
    """
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata["feat"] = torch.FloatTensor(n_feats).to(subgraph.device)
    subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long).to(subgraph.device)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1
    n_ids[tail_id] = 2
    subgraph.ndata["id"] = torch.FloatTensor(n_ids).to(subgraph.device)

    subgraph.ndata["r_label"] = torch.LongTensor(np.ones(n_nodes) * r_label).to(subgraph.device)
    return subgraph


# ============================================================
# Subgraph extraction for relation prediction (no target edge)
# ============================================================

def get_subgraph_for_rel_prediction(link, adj_list, dgl_adj_list, max_node_label_value,
                                     hop=3, enclosing_sub_graph=True):
    """
    Extract subgraph for relation prediction evaluation.
    NOTE: Does NOT add the target relation edge (that would leak the answer).

    Args:
        link: (head_id, tail_id, rel_id)
        adj_list: list of scipy adjacency matrices per relation
        dgl_adj_list: full DGL graph (for subgraph extraction)
        max_node_label_value: [max_dist_sub, max_dist_obj]
        hop: number of hops
        enclosing_sub_graph: whether to use enclosing subgraph

    Returns:
        DGL subgraph with prepared features
    """
    head, tail, rel = link[0], link[1], link[2]
    nodes, node_labels = subgraph_extraction_labeling(
        (head, tail), rel, adj_list, h=hop,
        enclosing_sub_graph=enclosing_sub_graph,
        max_node_label_value=max_node_label_value,
    )
    subgraph = dgl_adj_list.subgraph(nodes)

    # NOTE: do NOT add the target relation edge between head and tail
    # for relation prediction — that would leak the answer.

    n_feats = None
    subgraph = prepare_features(subgraph, node_labels, rel, max_node_label_value, n_feats)

    return subgraph

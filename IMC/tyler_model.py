"""
TyleR model adapted for IMC pre-computed PLM embeddings.
Ported from tyler-main/ITLP/tyler_simple.py and tyler-main/ITLP/trainer.py.

Key changes from original:
- No PoolEncoder/PromptEncoder — uses IMC's pre-aggregated embeddings directly
- No ontology features (init_onto_use always False)
- entity_embedding_matrix loaded from IMC .pkl files
- init_entity_features() always concatenates DRNL + IMC embeddings
"""
import os
import pickle
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl import mean_nodes

from tyler.rgcn_model import RGCN
from tyler_subgraph import (
    subgraph_extraction_labeling,
    prepare_features,
    ssp_multigraph_to_dgl,
    get_subgraph_for_rel_prediction,
    incidence_matrix,
)


# ============================================================
# Utility: build entity/relation mappings and adjacency matrices
# ============================================================

def process_files(train_path, valid_path=None, test_path=None,
                  add_traspose_rels=False):
    """
    Read triple files and build entity2id, relation2id, adjacency matrices.

    Args:
        train_path: path to train.txt
        valid_path: path to valid.txt (optional, for relation vocab)
        test_path: path to test.txt (optional, for relation vocab)
        add_traspose_rels: whether to add inverse relations

    Returns dict with:
        entity2id, id2entity, relation2id, id2relation,
        adj_list (list of scipy CSC per relation, from train),
        dgl_adj_list (full DGL graph),
        train_triples (np.array [n_train, 3] with integer IDs),
        num_rels, aug_num_rels
    """
    entity2id = {}
    relation2id = {}

    def read_triples(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    data.append((parts[0], parts[1], parts[2]))
        return data

    # Read train triples
    train_data = read_triples(train_path)

    # Collect all relations from train (+ valid/test if provided) for vocab
    all_relations = set()
    for _, r, _ in train_data:
        all_relations.add(r)
    if valid_path:
        for _, r, _ in read_triples(valid_path):
            all_relations.add(r)
    if test_path:
        for _, r, _ in read_triples(test_path):
            all_relations.add(r)

    # Build relation2id (sorted for determinism)
    for rel in sorted(all_relations):
        relation2id[rel] = len(relation2id)

    # Build entity2id from train triples
    for h, _, t in train_data:
        if h not in entity2id:
            entity2id[h] = len(entity2id)
        if t not in entity2id:
            entity2id[t] = len(entity2id)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    num_ents = len(entity2id)

    # Convert train triples to integer IDs
    train_triples_list = []
    for h, r, t in train_data:
        if r in relation2id:
            train_triples_list.append([
                entity2id[h], entity2id[t], relation2id[r]
            ])
    train_triples = np.array(train_triples_list)

    # Build adjacency matrices per relation (from train only)
    num_rels = len(relation2id)
    adj_list = []
    for i in range(num_rels):
        idx = np.argwhere(train_triples[:, 2] == i)
        if len(idx) > 0:
            adj = ssp.csc_matrix(
                (np.ones(len(idx), dtype=np.uint8),
                 (train_triples[:, 0][idx].squeeze(1),
                  train_triples[:, 1][idx].squeeze(1))),
                shape=(num_ents, num_ents),
            )
        else:
            adj = ssp.csc_matrix((num_ents, num_ents), dtype=np.uint8)
        adj_list.append(adj)

    # Optionally add transpose relations
    if add_traspose_rels:
        adj_list_aug = adj_list + [adj.T for adj in adj_list]
    else:
        adj_list_aug = adj_list

    aug_num_rels = len(adj_list_aug)
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return {
        'entity2id': entity2id,
        'id2entity': id2entity,
        'relation2id': relation2id,
        'id2relation': id2relation,
        'adj_list': adj_list,
        'adj_list_aug': adj_list_aug,
        'dgl_adj_list': dgl_adj_list,
        'train_triples': train_triples,
        'num_ents': num_ents,
        'num_rels': num_rels,
        'aug_num_rels': aug_num_rels,
    }


# ============================================================
# Load IMC embeddings
# ============================================================

def load_imc_embeddings(emb_path, entity2id, sem_dim=None):
    """
    Load IMC pre-computed embeddings and build a tensor indexed by entity2id.

    Args:
        emb_path: path to .pkl file
        entity2id: dict mapping entity name -> integer ID
        sem_dim: expected embedding dimension (inferred if None)

    Returns:
        torch.Tensor [num_entities, sem_dim]
    """
    with open(emb_path, 'rb') as f:
        node2emb = pickle.load(f)

    if sem_dim is None:
        # Infer from first embedding
        sem_dim = next(iter(node2emb.values())).shape[0]

    num_ents = len(entity2id)
    emb_matrix = np.zeros((num_ents, sem_dim), dtype=np.float32)

    missing_count = 0
    for name, idx in entity2id.items():
        if name in node2emb:
            emb_matrix[idx] = node2emb[name].astype(np.float32)
        else:
            # Fill missing with small random (same as IMC baselines)
            emb_matrix[idx] = np.random.randn(sem_dim).astype(np.float32) * 0.1
            missing_count += 1

    if missing_count > 0:
        print(f"  Warning: {missing_count}/{num_ents} entities missing from embeddings, "
              f"filled with random")

    return torch.from_numpy(emb_matrix)


# ============================================================
# TyleR Model (adapted from TextGraphClassifier)
# ============================================================

class TyleRClassifier(nn.Module):
    """
    Adapted TyleR model for relation prediction on knowledge graph subgraphs.

    Uses IMC pre-computed PLM embeddings (fixed, not trainable) concatenated
    with DRNL structural features as input to a 3-layer RGCN.

    Original: tyler-main/ITLP/tyler_simple.py TextGraphClassifier
    """

    def __init__(self, params, relation2id):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.num_rels = params.num_rels
        self.device = params.device

        # RGCN backbone
        self.gnn = RGCN(params)

        # Relation embedding for message passing
        self.rel_emb = nn.Embedding(
            params.num_rels + 1, params.inp_dim,
            sparse=False, padding_idx=params.num_rels
        )

        self.dropout = nn.Dropout(params.dropout)

        # Final relation scoring layer
        if params.add_ht_emb:
            self.fc_layer = nn.Linear(
                3 * params.num_gcn_layers * params.emb_dim,
                params.num_rels
            )
        else:
            self.fc_layer = nn.Linear(
                params.num_gcn_layers * params.emb_dim,
                params.num_rels
            )

        # Fixed entity embeddings (loaded from IMC .pkl, not trainable)
        self.register_buffer('entity_embedding_matrix', torch.empty(0))

    def init_entity_features(self, g):
        """
        Initialize entity features for RGCN input.

        g.ndata['feat'] contains only DRNL one-hot features.
        We look up IMC embeddings via _ID and concatenate them.
        This mirrors the original TextGraphClassifier's init_ent_emb_matrix.
        """
        ent_ids = g.ndata['_ID']
        ent_feats = self.entity_embedding_matrix[ent_ids]
        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_feats], dim=1)

    def load_entity_embeddings(self, emb_tensor):
        """
        Load IMC pre-aggregated embeddings into the fixed buffer.

        Args:
            emb_tensor: [num_entities, sem_dim] float tensor
        """
        self.entity_embedding_matrix = emb_tensor.to(self.device)

    def forward(self, data):
        g = data

        # Initialize features: DRNL one-hot + IMC embeddings
        self.init_entity_features(g)

        # Relation embeddings
        r = self.rel_emb.weight.clone()

        # RGCN forward pass
        g.ndata['h'], r_emb_out = self.gnn(g, r)

        # Mean-pool over all node representations
        out_dim = self.params.num_gcn_layers * self.params.emb_dim
        g_out = mean_nodes(g, 'repr').view(-1, out_dim)

        # Extract head and tail node embeddings
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat(
                [g_out, head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim)],
                dim=1,
            )
        else:
            g_rep = g_out

        output = self.fc_layer(g_rep)
        return output


# ============================================================
# Subgraph Dataset
# ============================================================

class SubgraphDataset(torch.utils.data.Dataset):
    """
    Dataset that extracts enclosing subgraphs for all triples and caches
    raw (nodes, labels, rel) tuples. DGL subgraphs are built on-the-fly
    in __getitem__ to avoid stale ndata from model modifications.

    Each item is a (DGL subgraph, relation_label) pair.
    """

    def __init__(self, data_info, triples,
                 hop=3, enclosing_sub_graph=True, max_nodes_per_hop=None,
                 device='cuda'):
        """
        Args:
            data_info: dict from process_files()
            triples: np.array [n, 3] of (head_id, tail_id, rel_id)
            hop: BFS hops for subgraph extraction
            enclosing_sub_graph: whether to use enclosing subgraph
            max_nodes_per_hop: cap on nodes per BFS hop
            device: 'cuda' or 'cpu'
        """
        self.data_info = data_info
        self.triples = triples
        self.hop = hop
        self.enclosing_sub_graph = enclosing_sub_graph
        self.max_nodes_per_hop = max_nodes_per_hop
        self.device = device

        self.adj_list = data_info['adj_list']
        self.dgl_adj_list = data_info['dgl_adj_list']
        self.num_rels = data_info['num_rels']

        print(f"  Extracting {len(triples)} subgraphs (hop={hop})...")

        # Extract all subgraph nodes and labels, compute global max_n_label
        self.items = []
        max_label_0, max_label_1 = 0, 0

        for i, (head, tail, rel) in enumerate(triples):
            nodes, node_labels = subgraph_extraction_labeling(
                (head, tail), rel, self.adj_list,
                h=hop, enclosing_sub_graph=enclosing_sub_graph,
                max_nodes_per_hop=max_nodes_per_hop, max_node_label_value=None,
            )
            self.items.append({
                'nodes': nodes,
                'labels': node_labels,
                'rel': rel,
            })
            if len(node_labels) > 0:
                max_label_0 = max(max_label_0, node_labels[:, 0].max())
                max_label_1 = max(max_label_1, node_labels[:, 1].max())

        self.max_n_label = np.array([max_label_0, max_label_1])
        print(f"  Max DRNL labels: {self.max_n_label}")

        # Feature dimension: just DRNL one-hot (IMC embeddings added by model)
        self.n_feat_dim = (self.max_n_label[0] + 1 + self.max_n_label[1] + 1)

        print(f"  DRNL feature dimension: {self.n_feat_dim}")
        print(f"  Subgraph dataset ready: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """Build DGL subgraph on-the-fly from cached nodes/labels.
        Only adds DRNL structural features; IMC embeddings are added
        by the model's init_entity_features() via _ID lookup."""
        entry = self.items[index]
        nodes = entry['nodes']
        node_labels = entry['labels']
        rel = entry['rel']

        # Extract DGL subgraph from the full graph (on CPU)
        subgraph = self.dgl_adj_list.subgraph(nodes)

        # Move graph to target device first, then set features
        subgraph = subgraph.to(self.device)

        # Prepare DRNL structural features only (no IMC embeddings)
        subgraph = prepare_features(
            subgraph, node_labels, rel, self.max_n_label, n_feats=None
        )

        return subgraph, rel


# ============================================================
# DGL batching utilities
# ============================================================

def collate_dgl_rel(samples):
    """Collate function for DataLoader: batch DGL graphs."""
    graphs_pos, r_labels_pos = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)
    return batched_graph_pos, r_labels_pos


def move_batch_to_device_dgl_rel(batch, device):
    """Move a batch of DGL graphs to the specified device."""
    g_dgl_pos, r_labels_pos = batch
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    g_dgl_pos = g_dgl_pos.to(device)
    return g_dgl_pos, r_labels_pos


# ============================================================
# Ranking metrics
# ============================================================

def compute_ranking_metrics(ranks):
    """Compute MRR and Hits@K from a list of ranks."""
    ranks = np.array(ranks)
    hits_1 = (ranks <= 1).mean() if len(ranks) > 0 else 0
    hits_3 = (ranks <= 3).mean() if len(ranks) > 0 else 0
    hits_10 = (ranks <= 10).mean() if len(ranks) > 0 else 0
    mrr = float(np.mean(1.0 / ranks)) if len(ranks) > 0 else 0
    return {
        'hits_1': hits_1,
        'hits_3': hits_3,
        'hits_10': hits_10,
        'mrr': mrr,
        'support': len(ranks),
    }


# ============================================================
# Training
# ============================================================

class Trainer:
    """Training loop adapted from tyler-main/ITLP/trainer.py."""

    def __init__(self, params, model, train_dataset, valid_dataset=None):
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_rels = params.num_rels
        self.updates_counter = 0

        model_params = list(self.model.parameters())
        trainable_params = filter(lambda p: p.requires_grad, model_params)
        print(f"Total parameters: {sum(p.numel() for p in model_params):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = torch.optim.Adam(
            model_params, lr=params.lr, weight_decay=params.l2
        )
        self.criterion = nn.MarginRankingLoss(params.margin, reduction='sum')

        self.best_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_dgl_rel,
        )

        for b_idx, batch in enumerate(dataloader):
            data, rel_labels = move_batch_to_device_dgl_rel(
                batch, self.params.device
            )
            batch_size = rel_labels.size(0)

            self.optimizer.zero_grad()
            scores = self.model(data)  # [batch_size, num_rels]

            # Positive scores
            pos_scores = scores[torch.arange(batch_size, device=self.params.device),
                                rel_labels]

            # Negative scores: random incorrect relation
            neg_rels = torch.randint(
                0, self.num_rels, (batch_size,),
                device=self.params.device
            )
            collision = (neg_rels == rel_labels)
            while collision.any():
                neg_rels[collision] = torch.randint(
                    0, self.num_rels,
                    (collision.sum().item(),),
                    device=self.params.device
                )
                collision = (neg_rels == rel_labels)

            neg_scores = scores[torch.arange(batch_size, device=self.params.device),
                                neg_rels]

            loss = self.criterion(
                pos_scores, neg_scores,
                torch.ones(batch_size, device=self.params.device)
            )

            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1
            torch.cuda.empty_cache()

            with torch.no_grad():
                all_preds += scores.argmax(dim=1).detach().cpu().tolist()
                all_labels += rel_labels.detach().cpu().tolist()
                total_loss += loss.item()

        acc = (np.array(all_preds) == np.array(all_labels)).mean() if all_labels else 0
        return total_loss, acc

    def validate(self):
        """Evaluate on validation set, return metrics dict."""
        if self.valid_dataset is None:
            return None

        self.model.eval()
        all_scores = []
        all_labels = []

        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_dgl_rel,
        )

        with torch.no_grad():
            for batch in dataloader:
                data, rel_labels = move_batch_to_device_dgl_rel(
                    batch, self.params.device
                )
                score_pos = self.model(data)
                all_scores.append(score_pos.detach().cpu())
                all_labels.append(rel_labels.cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute ranks
        ranks = []
        for i in range(all_labels.size(0)):
            scores = all_scores[i]
            true_label = all_labels[i].item()
            true_score = scores[true_label]
            rank = (scores >= true_score).sum().item()
            ranks.append(rank)

        ranks = np.array(ranks)
        mrr = float(np.mean(1.0 / ranks))
        hits_1 = float(np.mean(ranks <= 1))
        hits_3 = float(np.mean(ranks <= 3))
        hits_10 = float(np.mean(ranks <= 10))
        acc = float((all_scores.argmax(dim=1) == all_labels).float().mean())

        return {
            'mrr': mrr,
            'hits_1': hits_1,
            'hits_3': hits_3,
            'hits_10': hits_10,
            'acc': acc,
            'support': all_labels.size(0),
        }

    def train(self, num_epochs, early_stop=10, save_every=10, exp_dir=None):
        """Main training loop."""
        self.best_metric = 0
        self.not_improved_count = 0

        for epoch in range(1, num_epochs + 1):
            loss, acc = self.train_epoch()

            # Validate
            result = self.validate()
            if result is not None:
                valid_mrr = result['mrr']
                print(f"Epoch {epoch:3d}: loss={loss:.4f}, train_acc={acc:.4f}, "
                      f"valid_mrr={valid_mrr:.4f}, hits@1={result['hits_1']:.4f}, "
                      f"hits@10={result['hits_10']:.4f}")

                if valid_mrr >= self.best_metric:
                    self.best_metric = valid_mrr
                    self.not_improved_count = 0
                    if exp_dir:
                        self._save_checkpoint(exp_dir, 'best_graph_classifier.pth')
                        print(f"  -> Best model saved (MRR={valid_mrr:.4f})")
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count >= early_stop:
                        print(f"Early stopping after {epoch} epochs "
                              f"(no improvement for {early_stop} validations)")
                        break
            else:
                print(f"Epoch {epoch:3d}: loss={loss:.4f}, train_acc={acc:.4f}")

            if exp_dir and epoch % save_every == 0:
                self._save_checkpoint(exp_dir, f'graph_classifier_epoch{epoch}.pth')

        return self.best_metric

    def _save_checkpoint(self, exp_dir, filename):
        path = os.path.join(exp_dir, filename)
        torch.save(self.model.state_dict(), path)


# ============================================================
# Test-time relation prediction evaluation
# ============================================================

def evaluate_relation_prediction(model, data_info, test_triples,
                                  batch_size=16, hop=3,
                                  enclosing_sub_graph=True):
    """
    Evaluate relation prediction on test triples.
    For each (head, tail) pair, rank the true relation among all relations.

    Args:
        model: trained TyleRClassifier
        data_info: dict from process_files()
        test_triples: np.array [n, 3] of (head_id, tail_id, rel_id)
        batch_size: batch size for inference
        hop: subgraph extraction hops
        enclosing_sub_graph: whether to use enclosing subgraph

    Returns:
        dict with mrr, hits_1, hits_3, hits_10, support
    """
    model.eval()
    device = model.device

    adj_list = data_info['adj_list']
    dgl_adj_list = data_info['dgl_adj_list']
    max_n_label = model.gnn.max_label_value

    ranks = []
    subgraph_sizes = []

    print(f"Evaluating relation prediction on {len(test_triples)} test triples")
    print(f"Number of relations: {data_info['num_rels']}")

    for i, (head, tail, true_rel) in enumerate(test_triples):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(test_triples)}")

        link = [head, tail, true_rel]
        subgraph = get_subgraph_for_rel_prediction(
            link, adj_list, dgl_adj_list, max_n_label,
            hop=hop, enclosing_sub_graph=enclosing_sub_graph,
        )
        subgraph_sizes.append(subgraph.number_of_nodes())
        subgraph = subgraph.to(device)

        with torch.no_grad():
            scores = model(subgraph)
        if isinstance(scores, tuple):
            scores = scores[0]
        scores = scores.squeeze(0).detach().cpu().numpy()

        true_score = scores[true_rel]
        rank = int(np.sum(scores >= true_score))
        ranks.append(rank)

    metrics = compute_ranking_metrics(ranks)
    avg_size = np.mean(subgraph_sizes)
    print(f"Avg subgraph size: {avg_size:.1f} nodes")
    print(f"Results: {metrics}")
    return metrics

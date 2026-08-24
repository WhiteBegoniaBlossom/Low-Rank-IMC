# Operations about sparse relation matrix.
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as cusp


class SparseRelationMatrix:
    """
    Sparse correlation matrix class
    """

    def __init__(self, num_entities, num_relations):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.data = []  # Store (row, col, relation_id) triples
        self._coo_matrix = None
        self._csr_matrix = None
        self._csc_matrix = None

    def add_relation(self, head_idx, tail_idx, relation_idx):
        """Add a relation triple"""
        self.data.append((head_idx, tail_idx, relation_idx))

    def build_matrices(self):
        if not self.data:
            raise ValueError("No data available to construct matrix")

        rows, cols, relations = zip(*self.data)

        # Creating a COO-formatted sparse matrix using float32
        self._coo_matrix = sp.coo_matrix(
            (np.array(relations, dtype=np.float32), (rows, cols)),
            shape=(self.num_entities, self.num_entities),
            dtype=np.float32
        )

        # Convert to CSR and CSC formats for efficient computation
        self._csr_matrix = self._coo_matrix.tocsr()
        self._csc_matrix = self._coo_matrix.tocsc()

        return self

    @property
    def coo_matrix(self):
        if self._coo_matrix is None:
            self.build_matrices()
        return self._coo_matrix

    @property
    def csr_matrix(self):
        if self._csr_matrix is None:
            self.build_matrices()
        return self._csr_matrix

    @property
    def csc_matrix(self):
        if self._csc_matrix is None:
            self.build_matrices()
        return self._csc_matrix

    def get_nonzero_indices(self):
        """Get the index of the non-zero element"""
        return self.coo_matrix.row, self.coo_matrix.col

    def get_nonzero_values(self):
        """Get the value of non-zero elements"""
        return self.coo_matrix.data

    def get_density(self):
        """Calculate matrix density"""
        total_elements = self.num_entities * self.num_entities
        nonzero_elements = len(self.data)
        return nonzero_elements / total_elements

    def to_dense(self):
        """Convert to a dense matrix"""
        return self.coo_matrix.toarray()

    def to_cupy(self):
        """Convert to a CuPy sparse matrix"""
        return cp.asarray(self.to_dense(), dtype=cp.float32)


def create_sparse_relation_matrices(triples, entity_to_idx, relations_file=None,
                                    all_relations=None):
    """
    Creating a sparse correlation matrix
    """
    num_entities = len(entity_to_idx)
    relation_encoder = LabelEncoder()
    if relations_file is not None:
        # Load all relationships from file
        with open(relations_file, 'r') as f:
            all_relations = [line.strip() for line in f]
        relation_encoder.fit(all_relations)
    elif all_relations is not None:
        # Fit on an explicit list of all relation names (train+valid+test)
        relation_encoder.fit(all_relations)
    else:
        # Encoding relationship tags (train only — may miss rare relations)
        relation_labels = triples['relation'].values
        relation_encoder.fit(relation_labels)

    num_relations = len(relation_encoder.classes_)

    # Collect data
    rows, cols, data = [], [], []

    for _, row in triples.iterrows():
        head_idx = entity_to_idx[row['head']]
        tail_idx = entity_to_idx[row['tail']]
        relation_idx = relation_encoder.transform([row['relation']])[0] + 1

        rows.append(head_idx)
        cols.append(tail_idx)
        data.append(relation_idx)

    # Creating a CuPy sparse matrix
    rows_cp = cp.array(rows, dtype=cp.int32)
    cols_cp = cp.array(cols, dtype=cp.int32)
    data_cp = cp.array(data, dtype=cp.float32)

    sparse_R = cusp.coo_matrix((data_cp, (rows_cp, cols_cp)),
                               shape=(num_entities, num_entities))

    print(f"CuPy sparse matrix creation complete: {len(data)} non-zero elements")
    return sparse_R, relation_encoder, num_relations


def create_sparse_mask(sparse_R):
    """
    Creating a sparse mask from a sparse relation matrix
    """
    rows, cols = sparse_R.row.get(), sparse_R.col.get()
    data = np.ones_like(rows, dtype=np.float32)

    mask_coo = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(sparse_R.shape[0], sparse_R.shape[1]),
        dtype=np.float32
    )

    return mask_coo


def create_sparse_R_onehot(sparse_R, num_classes):
    """
    Creates a sparse one-hot encoded matrix.
    Return format: (row_indices, col_indices, onehot_vectors)
    """
    rows, cols = sparse_R.row.get(), sparse_R.col.get()
    relations = sparse_R.data.get()

    # Convert relation values back to integers for indexing
    relations_int = relations.astype(np.int32)

    # Each non-zero element corresponds to a one-hot vector
    onehot_vectors = np.zeros((len(relations_int), num_classes), dtype=np.float32)
    for i, rel in enumerate(relations_int):
        onehot_vectors[i, int(rel) - 1] = 1.0

    return rows, cols, onehot_vectors

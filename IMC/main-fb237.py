"""
Train IMC model on tyler-main's pre-split fb237 dataset versions using
PLM node embeddings (from generate_plm_embeddings.py).

The IMC model training process (CategorialLoss, SparseRelationMatrix, IMC function,
trust-region Newton optimization) is reused unchanged from the existing IMC codebase.

Usage:
    python main-fb237.py --version fb237_v1
    python main-fb237.py --all
"""
import os
import sys
import pickle
import time
import argparse
import gc
import numpy as np
import cupy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.special import softmax

from CategorialLoss import (
    optimized_categorical_loss_sparse,
    optimized_compute_full_gradient_sparse,
    compute_C_grad_and_loss, compute_H_hvp, compute_C_hvp, compute_W_hvp,
    compute_H_grad_and_loss, compute_W_grad_and_loss,
)
from SparseRelationMatrix import (
    create_sparse_mask, create_sparse_R_onehot, create_sparse_relation_matrices,
)

# ============================================================
# Data loading
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

V1_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V2_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V3_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
VERSIONS = [
    f"fb237_v1_ind_seed{s}" for s in V1_SEEDS
] + [
    f"fb237_v2_ind_seed{s}" for s in V2_SEEDS
] + [
    f"fb237_v3_ind_seed{s}" for s in V3_SEEDS
]


def load_fb237_data(version, model="roberta", aggregation="sum"):
    """Load a single fb237 dataset version's triples and PLM embeddings."""
    import pandas as pd

    version_dir = os.path.join(DATA_DIR, version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"Dataset directory not found: {version_dir}")

    train_triples = pd.read_csv(
        os.path.join(version_dir, "train.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    valid_triples = pd.read_csv(
        os.path.join(version_dir, "valid.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    test_triples = pd.read_csv(
        os.path.join(version_dir, "test.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )

    emb_path = os.path.join(EMBED_DIR, f"{version}_{model}_{aggregation}_embeddings.pkl")
    print(f"Embedding: {emb_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. "
            f"Run generate_plm_embeddings.py --version {version} --model {model} --aggregation {aggregation} first."
        )
    with open(emb_path, "rb") as f:
        node2emb = pickle.load(f)

    print(f"[{version}] Train: {len(train_triples)}, Valid: {len(valid_triples)}, "
          f"Test: {len(test_triples)}, Embeddings: {len(node2emb)}")
    return train_triples, valid_triples, test_triples, node2emb


def _pretrain_mlp_projector(X_scaled, head_indices, tail_indices, rel_labels,
                            num_relations, out_dim, hidden_dim, dropout,
                            epochs, lr, batch_size, random_seed):
    """Pre-train a shared MLP + bilinear scorer (DistMult-style) to learn
    task-specific feature projections.  Returns projected features X_proj
    with shape (n_entities, out_dim)."""
    cp.random.seed(random_seed)
    n_entities, in_dim = X_scaled.shape

    if hidden_dim is None:
        hidden_dim = max(in_dim, out_dim)

    # He init for MLP
    W1 = cp.random.randn(in_dim, hidden_dim, dtype=cp.float32) * cp.sqrt(2.0 / in_dim).item()
    b1 = cp.zeros(hidden_dim, dtype=cp.float32)
    W2 = cp.random.randn(hidden_dim, out_dim, dtype=cp.float32) * cp.sqrt(2.0 / hidden_dim).item()
    b2 = cp.zeros(out_dim, dtype=cp.float32)
    # Xavier init for relation embeddings
    R = cp.random.randn(num_relations, out_dim, dtype=cp.float32) * cp.sqrt(1.0 / out_dim).item()

    # Adam state
    params = [W1, b1, W2, b2, R]
    m = [cp.zeros_like(p) for p in params]
    v = [cp.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t = 0
    weight_decay = 1e-5

    n_triples = len(head_indices)
    n_batches = (n_triples + batch_size - 1) // batch_size

    def _mlp_forward(x):
        h_raw = x @ W1 + b1
        h_relu = cp.maximum(h_raw, 0)
        mask = None
        if dropout > 0:
            mask = (cp.random.rand(*h_relu.shape, dtype=cp.float32) > dropout).astype(cp.float32)
            mask /= 1.0 - dropout
            h_relu = h_relu * mask
        h_proj = h_relu @ W2 + b2
        return h_proj, h_raw, h_relu, mask

    def _mlp_backward(d_proj, relu, raw, mask, x_idx):
        """Backprop dL/d(proj) through the MLP. Returns (dW1, db1, dW2, db2)."""
        d_relu = d_proj @ W2.T
        if mask is not None:
            d_relu = d_relu * mask
        d_raw = d_relu * (raw > 0).astype(cp.float32)
        dW1 = X_scaled[x_idx].T @ d_raw
        db1 = cp.sum(d_raw, axis=0)
        dW2 = relu.T @ d_proj
        db2 = cp.sum(d_proj, axis=0)
        return dW1, db1, dW2, db2

    for epoch in range(epochs):
        perm = cp.random.permutation(n_triples)
        head_shuf = head_indices[perm]
        tail_shuf = tail_indices[perm]
        rel_shuf = rel_labels[perm]
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_triples)
            h_idx = head_shuf[start:end]
            t_idx = tail_shuf[start:end]
            r_lbl = rel_shuf[start:end]
            bs = end - start

            # Forward
            h_proj, h_raw, h_relu, h_mask = _mlp_forward(X_scaled[h_idx])
            t_proj, t_raw, t_relu, t_mask = _mlp_forward(X_scaled[t_idx])

            scores = (h_proj * t_proj) @ R.T      # (bs, num_relations)
            scores_max = cp.max(scores, axis=1, keepdims=True)
            exp_scores = cp.exp(scores - scores_max)
            probs = exp_scores / (cp.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

            log_probs = cp.log(probs + 1e-10)
            nll = -cp.mean(log_probs[cp.arange(bs), r_lbl])
            l2_reg = weight_decay * (cp.sum(W1 ** 2) + cp.sum(W2 ** 2) + cp.sum(R ** 2))
            loss = nll + l2_reg
            epoch_loss += float(loss)

            # Backward: dL/d(scores)
            delta = probs.copy()
            delta[cp.arange(bs), r_lbl] -= 1.0
            delta /= bs

            interactions = h_proj * t_proj          # (bs, out_dim)

            # dL/d(h_proj), dL/d(t_proj)
            delta_dot_R = delta @ R                  # (bs, out_dim)
            dh_proj = t_proj * delta_dot_R
            dt_proj = h_proj * delta_dot_R

            # dL/dR
            dR = delta.T @ interactions + 2.0 * weight_decay * R

            # Backprop through shared MLP (accumulate head + tail gradients)
            dW1_h, db1_h, dW2_h, db2_h = _mlp_backward(dh_proj, h_relu, h_raw, h_mask, h_idx)
            dW1_t, db1_t, dW2_t, db2_t = _mlp_backward(dt_proj, t_relu, t_raw, t_mask, t_idx)

            dW1 = dW1_h + dW1_t + 2.0 * weight_decay * W1
            db1 = db1_h + db1_t
            dW2 = dW2_h + dW2_t + 2.0 * weight_decay * W2
            db2 = db2_h + db2_t

            grads = [dW1, db1, dW2, db2, dR]
            t += 1
            for i in range(5):
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i]
                v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] ** 2
                m_hat = m[i] / (1.0 - beta1 ** t)
                v_hat = v[i] / (1.0 - beta2 ** t)
                params[i] -= lr * m_hat / (cp.sqrt(v_hat) + eps_adam)
            W1, b1, W2, b2, R = params

        print(f"  Pretrain epoch {epoch + 1}/{epochs}, loss={epoch_loss / max(n_batches, 1):.6f}")

    # Extract projected features for ALL entities
    X_proj, _, _, _ = _mlp_forward(X_scaled)
    return X_proj


def prepare_features_fb237(train_triples, valid_triples, test_triples, node2emb, random_seed=28,
                           pretrain=False, proj_dim=512, pretrain_epochs=20,
                           pretrain_lr=1e-3, pretrain_batch_size=4096, dropout=0.3):
    """Construct standardized feature matrix from PLM embeddings.
    When pretrain=True, pre-trains an MLP projector using the training triples
    before feeding features into IMC."""
    np.random.seed(random_seed)
    cp.random.seed(random_seed)

    all_entities = set()
    for df in [train_triples, valid_triples, test_triples]:
        all_entities.update(df["head"].unique())
        all_entities.update(df["tail"].unique())

    print(f"Number of entities: {len(all_entities)}")

    missing = [e for e in all_entities if e not in node2emb]
    if missing:
        print(f"{len(missing)} entities missing in embeddings, filling with random vectors")
        emb_dim = next(iter(node2emb.values())).shape[0]
        for e in missing:
            node2emb[e] = (np.random.randn(emb_dim).astype(np.float32) * 0.1)

    entity_list = sorted(all_entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_list)}

    emb_dim = len(next(iter(node2emb.values())))
    X_features = np.zeros((len(entity_list), emb_dim), dtype=np.float32)
    for e, idx in entity_to_idx.items():
        X_features[idx] = node2emb[e]

    # StandardScaler first (same as KGE baselines)
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    if pretrain:
        print(f"Pre-training MLP projector: {emb_dim} -> {proj_dim}")
        from sklearn.preprocessing import LabelEncoder
        rel_enc = LabelEncoder()
        rel_enc.fit(train_triples['relation'].unique())

        head_idx = np.array([entity_to_idx[row['head']] for _, row in train_triples.iterrows()], dtype=np.int32)
        tail_idx = np.array([entity_to_idx[row['tail']] for _, row in train_triples.iterrows()], dtype=np.int32)
        rel_labels = rel_enc.transform(train_triples['relation']).astype(np.int32)

        X_cp = cp.asarray(X_features, dtype=cp.float32)
        head_cp = cp.asarray(head_idx, dtype=cp.int32)
        tail_cp = cp.asarray(tail_idx, dtype=cp.int32)
        rel_cp = cp.asarray(rel_labels, dtype=cp.int32)

        num_relations = len(rel_enc.classes_)
        hidden_dim = max(emb_dim, proj_dim)
        X_proj_cp = _pretrain_mlp_projector(
            X_cp, head_cp, tail_cp, rel_cp, num_relations,
            proj_dim, hidden_dim, dropout,
            pretrain_epochs, pretrain_lr, pretrain_batch_size, random_seed + 1,
        )
        X_features = cp.asnumpy(X_proj_cp)
        print(f"Projected feature shape: {X_features.shape}")

    X_features = cp.asarray(X_features, dtype=cp.float32)

    print(f"Feature matrix shape: {X_features.shape}")
    return entity_to_idx, X_features


# ============================================================
# IMC training functions
# ============================================================

def trust_region_newton(w, grad, delta, maxiter, cg_maxiter, loss_func, grad_func, hvp_func):
    eta0 = cp.float32(1e-3)
    eta1 = cp.float32(0.25)
    eta2 = cp.float32(0.75)
    sigma1 = cp.float32(0.25)
    sigma2 = cp.float32(0.5)
    sigma3 = cp.float32(4.0)
    min_pred = cp.float32(1e-10)
    min_delta = cp.float32(1e-6)
    max_delta = cp.float32(10.0)

    initial_grad_norm = cp.linalg.norm(grad)
    if initial_grad_norm == 0:
        initial_grad_norm = cp.float32(1e-10)

    inner_loss_history = []

    for t in range(maxiter):
        r = -grad
        d = r.copy()
        s_current = cp.zeros_like(w)
        r_current = r.copy()

        for cg_iter in range(cg_maxiter):
            Hd = hvp_func(w, d)
            Hd_norm = d.T @ Hd
            if Hd_norm < 1e-14:
                alpha = cp.float32(0)
            else:
                alpha = (r_current.T @ r_current) / Hd_norm

            s_new = s_current + alpha * d
            s_norm = cp.linalg.norm(s_new)

            if s_norm > delta:
                tau = delta / s_norm
                s_new = s_current + tau * alpha * d
                s_current = s_new
                break

            s_current = s_new
            r_new = r_current - alpha * Hd

            if cp.linalg.norm(r_new) < 1e-12:
                break

            beta = (r_new.T @ r_new) / (r_current.T @ r_current)
            d = r_new + beta * d
            r_current = r_new

        s = s_current
        step_norm = cp.linalg.norm(s)

        loss_old = loss_func(w)
        loss_new = loss_func(w + s)
        actual_reduction = loss_old - loss_new
        loss_new = float(loss_new)
        inner_loss_history.append(loss_new)

        Hs = hvp_func(w, s)
        pred_reduction = -grad.T @ s - 0.5 * (s.T @ Hs)

        if cp.abs(pred_reduction) < min_pred:
            pred_reduction = min_pred

        rho = actual_reduction / pred_reduction if pred_reduction != 0 else cp.float32(0)
        rho = cp.clip(rho, -1e10, 1e10)

        if rho > eta0:
            w = w + s
            grad = grad_func(w)

        if rho <= eta1:
            min_val = cp.minimum(step_norm, delta)
            new_delta = (sigma1 * min_val + sigma2 * delta) / cp.float32(2)
            delta = cp.maximum(new_delta, min_delta)
        elif rho < eta2:
            new_delta = (sigma1 * delta + sigma3 * delta) / cp.float32(2)
            delta = cp.clip(new_delta, sigma1 * delta, sigma3 * delta)
        else:
            delta = cp.minimum(sigma3 * delta, max_delta)

        grad_norm = cp.linalg.norm(grad)
        if grad_norm < 1e-4 * initial_grad_norm:
            break
        elif step_norm < 1e-6:
            break
    return w, inner_loss_history


def IMC(sparse_R, X, Y, k, lambda_, maxiter, C=10, valid_triples=None,
        entity_to_idx=None, relation_encoder=None, eval_interval=10, random_seed=28, best_model=None):
    msg = ""
    cp.random.seed(random_seed)
    np.random.seed(random_seed)

    m, d1 = X.shape
    n, d2 = Y.shape

    if not isinstance(X, cp.ndarray):
        X_cp = cp.asarray(X, dtype=cp.float32)
    else:
        X_cp = X
    if not isinstance(Y, cp.ndarray):
        Y_cp = cp.asarray(Y, dtype=cp.float32)
    else:
        Y_cp = Y

    sparse_mask = create_sparse_mask(sparse_R)
    sparse_onehot_data = create_sparse_R_onehot(sparse_R, C)

    rows_np, cols_np, onehot_vectors_np = sparse_onehot_data
    rows = cp.asarray(rows_np, dtype=cp.int32)
    cols = cp.asarray(cols_np, dtype=cp.int32)
    onehot_vectors = cp.asarray(onehot_vectors_np, dtype=cp.float32)
    sparse_onehot_data_cp = (rows, cols, onehot_vectors)

    W = cp.random.randn(d1, k, dtype=cp.float32) * 0.1
    cp.random.seed(random_seed + 1)
    H = cp.random.randn(k, d2, dtype=cp.float32) * 0.1
    cp.random.seed(random_seed + 2)
    C_tensor = cp.random.randn(k, C, dtype=cp.float32) * 0.1

    val_mrrs = []
    if valid_triples is not None:
        val_mrr = compute_mrr(
            valid_triples, W, H, C_tensor, entity_to_idx,
            relation_encoder, X_cp
        )
        val_mrrs.append(val_mrr)
        best_val_mrr = val_mrr
        print(f"Initial valid MRR={val_mrr:.4f}")
    else:
        val_mrr = 0.0
        best_val_mrr = 0.0
    best_W, best_H, best_C = W.copy(), H.copy(), C_tensor.copy()
    best_iter = 0

    eps_abs = cp.float32(5e-3)
    eps_rel = cp.float32(5e-4)
    grad_threshold = cp.float32(5e-3)
    min_delta = cp.float32(1e-4)

    start_time = time.time()
    outer_losses = []
    inner_losses = []

    current_loss, _ = optimized_categorical_loss_sparse(
        cp.zeros(1, dtype=cp.float32),
        sparse_R, X_cp, Y_cp, lambda_, C, None, None,
        {'W': W, 'H': H, 'C_tensor': C_tensor},
        sparse_mask, sparse_onehot_data_cp
    )
    current_loss = float(current_loss)
    print(f"Initial-Loss={current_loss}")
    outer_losses.append(current_loss)

    for i in range(1, maxiter + 1):
        W0 = W.copy()
        H0 = H.copy()
        C0 = C_tensor.copy()

        # optimize W
        def loss_W(w_vec):
            W_cur = w_vec.reshape(d1, k)
            return compute_W_grad_and_loss(W_cur, H, C_tensor, X_cp, Y_cp,
                                           rows, cols, onehot_vectors, lambda_)[0]

        def grad_W(w_vec):
            W_cur = w_vec.reshape(d1, k)
            _, grad = compute_W_grad_and_loss(W_cur, H, C_tensor, X_cp, Y_cp,
                                              rows, cols, onehot_vectors, lambda_)
            return grad.ravel()

        def hvp_W(w_vec, v_vec):
            W_cur = w_vec.reshape(d1, k)
            V = v_vec.reshape(d1, k)
            Hv = compute_W_hvp(W_cur, H, C_tensor, X_cp, Y_cp, rows, cols,
                               onehot_vectors, lambda_, V)
            return Hv.ravel()

        w_init = W.ravel()
        g_init = grad_W(w_init)
        W_new, inner_losses_w = trust_region_newton(
            w_init, g_init, delta=1.0, maxiter=5, cg_maxiter=10,
            loss_func=loss_W, grad_func=grad_W, hvp_func=hvp_W)
        W = W_new.reshape(d1, k)
        inner_losses.extend(inner_losses_w)

        # optimize H
        def loss_H(h_vec):
            H_cur = h_vec.reshape(k, d2)
            return compute_H_grad_and_loss(H_cur, W, C_tensor, X_cp, Y_cp,
                                           rows, cols, onehot_vectors, lambda_)[0]

        def grad_H(h_vec):
            H_cur = h_vec.reshape(k, d2)
            _, grad = compute_H_grad_and_loss(H_cur, W, C_tensor, X_cp, Y_cp,
                                              rows, cols, onehot_vectors, lambda_)
            return grad.ravel()

        def hvp_H(h_vec, v_vec):
            H_cur = h_vec.reshape(k, d2)
            V = v_vec.reshape(k, d2)
            Hv = compute_H_hvp(H_cur, W, C_tensor, X_cp, Y_cp, rows, cols,
                               onehot_vectors, lambda_, V)
            return Hv.ravel()

        h_init = H.ravel()
        g_init = grad_H(h_init)
        H_new, inner_losses_h = trust_region_newton(
            h_init, g_init, delta=1.0, maxiter=5, cg_maxiter=10,
            loss_func=loss_H, grad_func=grad_H, hvp_func=hvp_H)
        H = H_new.reshape(k, d2)
        inner_losses.extend(inner_losses_h)

        # optimize C
        def loss_C(c_vec):
            C_cur = c_vec.reshape(k, C)
            return compute_C_grad_and_loss(C_cur, W, H, X_cp, Y_cp,
                                           rows, cols, onehot_vectors, lambda_)[0]

        def grad_C(c_vec):
            C_cur = c_vec.reshape(k, C)
            _, grad = compute_C_grad_and_loss(C_cur, W, H, X_cp, Y_cp,
                                              rows, cols, onehot_vectors, lambda_)
            return grad.ravel()

        def hvp_C(c_vec, v_vec):
            C_cur = c_vec.reshape(k, C)
            V = v_vec.reshape(k, C)
            Hv = compute_C_hvp(C_cur, W, H, X_cp, Y_cp, rows, cols,
                               onehot_vectors, lambda_, V)
            return Hv.ravel()

        c_init = C_tensor.ravel()
        g_init = grad_C(c_init)
        C_new, inner_losses_c = trust_region_newton(
            c_init, g_init, delta=1.0, maxiter=5, cg_maxiter=10,
            loss_func=loss_C, grad_func=grad_C, hvp_func=hvp_C)
        C_tensor = C_new.reshape(k, C)
        inner_losses.extend(inner_losses_c)

        current_loss, _ = optimized_categorical_loss_sparse(
            cp.zeros(1, dtype=cp.float32),
            sparse_R, X_cp, Y_cp, lambda_, C, None, None,
            {'W': W, 'H': H, 'C_tensor': C_tensor},
            sparse_mask, sparse_onehot_data_cp
        )
        current_loss = float(current_loss)
        print(f"Iter{i}-Loss={current_loss}")

        delta_W = float(cp.linalg.norm(W - W0)) if W0 is not None else 0
        delta_H = float(cp.linalg.norm(H - H0)) if H0 is not None else 0
        delta_C = float(cp.linalg.norm(C_tensor - C0)) if C0 is not None else 0
        print(f"Iter{i}-Parameter changes: W={delta_W:.6f}, H={delta_H:.6f}, C={delta_C:.6f}")

        outer_losses.append(current_loss)

        if (i % eval_interval == 0) and valid_triples is not None:
            val_mrr = compute_mrr(
                valid_triples, W, H, C_tensor, entity_to_idx,
                relation_encoder, X_cp
            )
            val_mrrs.append(val_mrr)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_W, best_H, best_C = W.copy(), H.copy(), C_tensor.copy()
                best_iter = i
                print(f"  -> New best model at iter {i}, valid_MRR={best_val_mrr:.4f}")

        current_grad_norm = optimized_compute_full_gradient_sparse(
            W, H, C_tensor, sparse_R, X_cp, Y_cp,
            lambda_, C, sparse_mask, sparse_onehot_data_cp
        )
        print(f"Iter{i}-grad_norm={current_grad_norm}")

    msg = f"Reach max iteration {maxiter}"
    print(msg)

    total_time = time.time() - start_time

    # Revert to best validation model
    if valid_triples is not None and best_iter > 0:
        print(f"Using best model from iter {best_iter} (valid_MRR={best_val_mrr:.4f})")
        W, H, C_tensor = best_W, best_H, best_C
        msg += f" | best_iter={best_iter}"

    def predict_proba(head_indices, tail_indices):
        logits = compute_relation_scores(head_indices, tail_indices, W, H, C_tensor, X)
        probabilities = softmax(cp.asnumpy(logits), axis=1)
        return probabilities

    return W, H, C_tensor, predict_proba, total_time, msg, outer_losses, inner_losses, val_mrrs


# ============================================================
# Evaluation functions (unchanged from main-fb15k.py)
# ============================================================

def reconstruct_accuracy(R, X, W, H, C):
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X, dtype=cp.float32)
    if not isinstance(W, cp.ndarray):
        W = cp.asarray(W, dtype=cp.float32)
    if not isinstance(H, cp.ndarray):
        H = cp.asarray(H, dtype=cp.float32)
    if not isinstance(C, cp.ndarray):
        C = cp.asarray(C, dtype=cp.float32)

    if hasattr(R, 'nnz'):
        if hasattr(R, 'get'):
            rows = R.row.get()
            cols = R.col.get()
            R_values = R.data.get()
        else:
            rows = R.row
            cols = R.col
            R_values = R.data
        rows = cp.asarray(rows, dtype=cp.int32)
        cols = cp.asarray(cols, dtype=cp.int32)
        R_values = cp.asarray(R_values, dtype=cp.float32)
    else:
        rows, cols = cp.where(R != 0)
        R_values = R[rows, cols]

    all_preds = []
    all_true = []
    batch_size = 1000
    k = W.shape[1]
    num_classes = C.shape[1]

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_cols = cols[i:i + batch_size]
        batch_values = R_values[i:i + batch_size]

        batch_logits = cp.zeros((len(batch_rows), num_classes), dtype=cp.float32)
        X_batch = X[batch_rows]
        Y_batch = X[batch_cols]

        for r in range(k):
            user_terms = cp.sum(X_batch * W[:, r], axis=1, keepdims=True)
            item_terms = cp.sum(Y_batch * H[r, :], axis=1, keepdims=True)
            interaction = user_terms * item_terms
            batch_logits += interaction * C[r, :]

        pred_classes = cp.argmax(batch_logits, axis=1) + 1
        all_preds.extend(cp.asnumpy(pred_classes).tolist())
        all_true.extend(cp.asnumpy(batch_values).tolist())

    accuracy = accuracy_score(all_true, all_preds)
    print(f"Train accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_on_subset(triples_subset, W, H, C, entity_to_idx, relation_encoder, X_features, subset_name):
    if not isinstance(X_features, cp.ndarray):
        X_features = cp.asarray(X_features, dtype=cp.float32)
    if not isinstance(W, cp.ndarray):
        W = cp.asarray(W, dtype=cp.float32)
    if not isinstance(H, cp.ndarray):
        H = cp.asarray(H, dtype=cp.float32)
    if not isinstance(C, cp.ndarray):
        C = cp.asarray(C, dtype=cp.float32)

    all_preds = []
    all_true = []
    batch_size = 256
    triples_list = list(triples_subset.to_dict('records'))

    for i in range(0, len(triples_list), batch_size):
        batch_triples = triples_list[i:i + batch_size]
        head_indices = []
        tail_indices = []
        true_relations = []

        for row in batch_triples:
            head = row['head']
            tail = row['tail']
            if head in entity_to_idx and tail in entity_to_idx:
                head_indices.append(entity_to_idx[head])
                tail_indices.append(entity_to_idx[tail])
                true_relations.append(row['relation'])

        if not head_indices:
            continue

        head_indices = cp.array(head_indices, dtype=cp.int32)
        tail_indices = cp.array(tail_indices, dtype=cp.int32)
        pred_relations = predict_relation(head_indices, tail_indices, W, H, C, X_features, relation_encoder)
        all_preds.extend(pred_relations)
        all_true.extend(true_relations)

    accuracy = accuracy_score(all_true, all_preds)
    print(f"{subset_name}accuracy: {accuracy:.4f}")
    return accuracy


def predict_relation(head_indices, tail_indices, W, H, C, X_features, relation_encoder):
    if not isinstance(X_features, cp.ndarray):
        X_features = cp.asarray(X_features, dtype=cp.float32)
    if not isinstance(W, cp.ndarray):
        W = cp.asarray(W, dtype=cp.float32)
    if not isinstance(H, cp.ndarray):
        H = cp.asarray(H, dtype=cp.float32)
    if not isinstance(C, cp.ndarray):
        C = cp.asarray(C, dtype=cp.float32)

    batch_size = len(head_indices)
    k = W.shape[1]
    num_classes = C.shape[1]

    x_heads = X_features[head_indices]
    x_tails = X_features[tail_indices]

    logits = cp.zeros((batch_size, num_classes), dtype=cp.float32)
    for r in range(k):
        user_terms = cp.sum(x_heads * W[:, r], axis=1, keepdims=True)
        item_terms = cp.sum(x_tails * H[r, :], axis=1, keepdims=True)
        interaction = user_terms * item_terms
        logits += interaction * C[r, :]

    pred_class_indices = cp.argmax(logits, axis=1)
    pred_class_indices_np = cp.asnumpy(pred_class_indices)
    return relation_encoder.inverse_transform(pred_class_indices_np)


def compute_mrr(test_triples, W, H, C, entity_to_idx, relation_encoder, X_features):
    if not isinstance(X_features, cp.ndarray):
        X_features = cp.asarray(X_features, dtype=cp.float32)
    if not isinstance(W, cp.ndarray):
        W = cp.asarray(W, dtype=cp.float32)
    if not isinstance(H, cp.ndarray):
        H = cp.asarray(H, dtype=cp.float32)
    if not isinstance(C, cp.ndarray):
        C = cp.asarray(C, dtype=cp.float32)

    reciprocal_ranks = []
    batch_size = 128
    triples_list = list(test_triples.to_dict('records'))

    for i in range(0, len(triples_list), batch_size):
        batch_triples = triples_list[i:i + batch_size]
        head_indices = []
        tail_indices = []
        true_relation_indices = []

        for row in batch_triples:
            head = row['head']
            tail = row['tail']
            relation = row['relation']
            if head in entity_to_idx and tail in entity_to_idx:
                head_indices.append(entity_to_idx[head])
                tail_indices.append(entity_to_idx[tail])
                true_relation_indices.append(relation_encoder.transform([relation])[0])

        if not head_indices:
            continue

        head_indices = cp.array(head_indices, dtype=cp.int32)
        tail_indices = cp.array(tail_indices, dtype=cp.int32)
        all_scores_batch = compute_relation_scores(head_indices, tail_indices, W, H, C, X_features)
        true_relation_indices = cp.array(true_relation_indices, dtype=cp.int32)
        ranks = compute_ranks(all_scores_batch, true_relation_indices)
        reciprocal_ranks.extend((1.0 / ranks).get().tolist())

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0


def compute_relation_scores(head_indices, tail_indices, W, H, C, X_features):
    batch_size = len(head_indices)
    num_relations = C.shape[1]
    x_heads = X_features[head_indices]
    x_tails = X_features[tail_indices]
    scores = cp.zeros((batch_size, num_relations), dtype=cp.float32)
    k = W.shape[1]
    for r in range(k):
        user_terms = cp.sum(x_heads * W[:, r], axis=1, keepdims=True)
        item_terms = cp.sum(x_tails * H[r, :], axis=1, keepdims=True)
        interaction = user_terms * item_terms
        scores += interaction * C[r, :]
    return scores


def compute_ranks(all_scores, true_indices):
    sorted_indices = cp.argsort(all_scores, axis=1)[:, ::-1]
    ranks = cp.zeros(len(true_indices), dtype=cp.int32)
    for i in range(len(true_indices)):
        rank_pos = cp.where(sorted_indices[i] == true_indices[i])[0][0] + 1
        ranks[i] = rank_pos
    return ranks


def compute_hits_at_k(test_triples, W, H, C, entity_to_idx, relation_encoder, X_features, k_values=[1, 3, 10]):
    if not isinstance(X_features, cp.ndarray):
        X_features = cp.asarray(X_features, dtype=cp.float32)
    if not isinstance(W, cp.ndarray):
        W = cp.asarray(W, dtype=cp.float32)
    if not isinstance(H, cp.ndarray):
        H = cp.asarray(H, dtype=cp.float32)
    if not isinstance(C, cp.ndarray):
        C = cp.asarray(C, dtype=cp.float32)

    hits = {k: 0 for k in k_values}
    total = 0
    batch_size = 128
    triples_list = list(test_triples.to_dict('records'))
    max_k = max(k_values)

    for i in range(0, len(triples_list), batch_size):
        batch_triples = triples_list[i:i + batch_size]
        head_indices = []
        tail_indices = []
        true_relation_indices = []

        for row in batch_triples:
            head = row['head']
            tail = row['tail']
            relation = row['relation']
            if head in entity_to_idx and tail in entity_to_idx:
                head_indices.append(entity_to_idx[head])
                tail_indices.append(entity_to_idx[tail])
                true_relation_indices.append(relation_encoder.transform([relation])[0])

        if not head_indices:
            continue

        head_indices = cp.array(head_indices, dtype=cp.int32)
        tail_indices = cp.array(tail_indices, dtype=cp.int32)
        all_scores_batch = compute_relation_scores(head_indices, tail_indices, W, H, C, X_features)
        true_relation_indices = cp.array(true_relation_indices, dtype=cp.int32)
        top_k_indices = cp.argsort(all_scores_batch, axis=1)[:, -max_k:][:, ::-1]

        for k in k_values:
            hits_mask = cp.any(top_k_indices[:, :k] == true_relation_indices[:, cp.newaxis], axis=1)
            hits[k] += cp.sum(hits_mask).get()
        total += len(head_indices)

    hits_at_k = {f"Hits@{k}": hits[k] / total if total > 0 else 0 for k in k_values}
    return hits_at_k


# ============================================================
# Conformal prediction (unchanged from main-fb15k.py)
# ============================================================

def generate_conformal_prediction_sets(calib_triples, test_triples, predict_proba,
                                       entity_to_idx, relation_encoder, alpha=0.1):
    calib_head_indices = []
    calib_tail_indices = []
    calib_true_labels = []

    for _, row in calib_triples.iterrows():
        if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
            calib_head_indices.append(entity_to_idx[row['head']])
            calib_tail_indices.append(entity_to_idx[row['tail']])
            calib_true_labels.append(relation_encoder.transform([row['relation']])[0])

    calib_head_indices = cp.array(calib_head_indices, dtype=cp.int32)
    calib_tail_indices = cp.array(calib_tail_indices, dtype=cp.int32)
    calib_probs = predict_proba(calib_head_indices, calib_tail_indices)
    calib_probs_np = cp.asnumpy(calib_probs)

    calib_scores = []
    for i, true_label in enumerate(calib_true_labels):
        score = 1 - calib_probs_np[i, true_label]
        calib_scores.append(score)

    n_calib = len(calib_scores)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(calib_scores, q_level, method='higher')
    print(f"Calibration set size: {n_calib}, Quantiles: {q_hat:.4f} (alpha={alpha})")

    test_head_indices = []
    test_tail_indices = []
    test_true_labels = []

    for _, row in test_triples.iterrows():
        if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
            test_head_indices.append(entity_to_idx[row['head']])
            test_tail_indices.append(entity_to_idx[row['tail']])
            test_true_labels.append(relation_encoder.transform([row['relation']])[0])

    test_head_indices = cp.array(test_head_indices, dtype=cp.int32)
    test_tail_indices = cp.array(test_tail_indices, dtype=cp.int32)
    test_probs = predict_proba(test_head_indices, test_tail_indices)
    test_probs_np = cp.asnumpy(test_probs)

    prediction_sets = []
    for probs in test_probs_np:
        prediction_set = set()
        for class_idx, prob in enumerate(probs):
            if prob >= 1 - q_hat:
                prediction_set.add(class_idx)
        prediction_sets.append(prediction_set)

    return prediction_sets


def evaluate_conformal_prediction(prediction_sets, true_labels, alpha=0.1):
    coverage = np.mean([true_label in pred_set
                        for true_label, pred_set in zip(true_labels, prediction_sets)])
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return {'marginal_coverage': coverage, 'average_set_size': avg_set_size}


def setup_gpu_environment():
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    return pool


# ============================================================
# Main training entry point
# ============================================================

def train_fb237_version(version, k=70, lambda_cat=1000.0, bias=32,
                        model="roberta", aggregation="sum", random_seed=28,
                        pretrain=False, proj_dim=512, pretrain_epochs=20,
                        pretrain_lr=1e-3, pretrain_batch_size=4096, dropout=0.3,
                        maxiter_cat=30):
    """Train IMC on a single fb237 dataset version."""
    pretrain_flag = " +pretrain" if pretrain else ""
    print("=" * 70)
    print(f"Training IMC on {version} ({model}/{aggregation}), k={k}, lambda={lambda_cat}, bias={bias}{pretrain_flag}")
    print("=" * 70)

    train_triples, valid_triples, test_triples, node2emb = load_fb237_data(
        version, model=model, aggregation=aggregation)
    entity_to_idx_base, X_features_base = prepare_features_fb237(
        train_triples, valid_triples, test_triples, node2emb, random_seed,
        pretrain=pretrain, proj_dim=proj_dim, pretrain_epochs=pretrain_epochs,
        pretrain_lr=pretrain_lr, pretrain_batch_size=pretrain_batch_size,
        dropout=dropout,
    )

    all_relations = sorted(set(train_triples['relation'].unique())
                          | set(valid_triples['relation'].unique())
                          | set(test_triples['relation'].unique()))
    R_train, relation_encoder, num_relations = create_sparse_relation_matrices(
        train_triples, entity_to_idx_base, all_relations=all_relations
    )
    print(f"Entities={X_features_base.shape[0]}, "
          f"Feature dim={X_features_base.shape[1]}, "
          f"Relation classes={num_relations}")

    eval_interval = 1

    if bias == 0:
        X_features = X_features_base
        entity_to_idx = entity_to_idx_base
    else:
        original_dim = X_features_base.shape[1]
        total_dim = original_dim + bias
        X_features_extended = cp.zeros((X_features_base.shape[0], total_dim), dtype=cp.float32)
        X_features_extended[:, :original_dim] = X_features_base
        X_features_extended[:, -bias:] = 1.0
        X_features = X_features_extended
        entity_to_idx = entity_to_idx_base

    print(f"Extended feature matrix shape: {X_features.shape}")

    W_cat, H_cat, C_cat, predict_proba, time_cat, msg_cat, loss_outer_cat, loss_inner_cat, val_mrrs = IMC(
        R_train, X_features, X_features, k, lambda_cat, maxiter_cat, C=num_relations,
        valid_triples=valid_triples, entity_to_idx=entity_to_idx,
        relation_encoder=relation_encoder, eval_interval=eval_interval,
        random_seed=random_seed
    )
    print(f"Training time: {time_cat:.2f} seconds")
    print(f"Convergence: {msg_cat}")

    # Conformal prediction
    alpha_values = [0.01, 0.05, 0.1]
    conformal_results = {}
    for alpha in alpha_values:
        print(f"Conformal prediction (alpha={alpha})")
        prediction_sets = generate_conformal_prediction_sets(
            valid_triples, test_triples, predict_proba,
            entity_to_idx, relation_encoder, alpha
        )
        true_labels = []
        for _, row in test_triples.iterrows():
            if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
                true_labels.append(relation_encoder.transform([row['relation']])[0])
        eval_results = evaluate_conformal_prediction(prediction_sets, true_labels, alpha)
        conformal_results[alpha] = eval_results
        print(f"  Coverage: {eval_results['marginal_coverage']:.4f}, "
              f"Avg Set Size: {eval_results['average_set_size']:.2f}")

    # Final evaluation
    results = {
        'model': model,
        'aggregation': aggregation,
        'version': version,
        'k': k,
        'lambda': lambda_cat,
        'bias': bias,
        'train_time': time_cat,
        'convergence_msg': msg_cat,
        'feature_dim': X_features_base.shape[1],
        'num_relations': num_relations,
        'conformal': conformal_results,
    }

    if not cp.isnan(W_cat).any().get() and not cp.isnan(H_cat).any().get() and not cp.isnan(C_cat).any().get():
        results['train_acc'] = reconstruct_accuracy(R_train, X_features, W_cat, H_cat, C_cat)
        results['valid_acc'] = evaluate_on_subset(
            valid_triples, W_cat, H_cat, C_cat, entity_to_idx, relation_encoder, X_features, "Valid "
        )
        results['test_acc'] = evaluate_on_subset(
            test_triples, W_cat, H_cat, C_cat, entity_to_idx, relation_encoder, X_features, "Test "
        )
        results['mrr'] = compute_mrr(test_triples, W_cat, H_cat, C_cat, entity_to_idx, relation_encoder, X_features)
        results['hits'] = compute_hits_at_k(test_triples, W_cat, H_cat, C_cat, entity_to_idx, relation_encoder, X_features)
        print(f"Test MRR: {results['mrr']:.4f}")
        for metric, value in results['hits'].items():
            print(f"Test {metric}: {value:.4f}")
    else:
        print("Optimization failed (NaN detected)")
        results['train_acc'] = None
        results['valid_acc'] = None
        results['test_acc'] = None
        results['mrr'] = None
        results['hits'] = None

    return results, loss_outer_cat, loss_inner_cat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None, help="Specific version (e.g. fb237_v1)")
    parser.add_argument("--all", action="store_true", help="Train on all 4 inductive versions")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"],
                        help="Which PLM embeddings to use")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"],
                        help="Aggregation method used in generate_plm_embeddings.py")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--k", type=int, default=70,
                        help="Latent factor rank (default: 70)")
    parser.add_argument("--lambda", type=float, default=1000.0, dest="lambda_cat",
                        help="L2 regularization strength (default: 1000.0)")
    parser.add_argument("--bias", type=int, default=32,
                        help="Number of bias dimensions appended to features (default: 32)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Pre-train MLP feature projector before IMC")
    parser.add_argument("--proj_dim", type=int, default=512,
                        help="Output dimension of pre-training MLP (default: 512)")
    parser.add_argument("--pretrain_epochs", type=int, default=20,
                        help="Number of pre-training epochs (default: 20)")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                        help="Learning rate for pre-training (default: 1e-3)")
    parser.add_argument("--pretrain_batch_size", type=int, default=4096,
                        help="Batch size for pre-training (default: 4096)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in pre-training MLP (default: 0.3)")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Number of IMC outer iterations (default: 50)")
    args = parser.parse_args()

    if args.all:
        versions = VERSIONS
    elif args.version:
        versions = [args.version]
    else:
        print("Specify --version <name> or --all")
        return

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    pool = setup_gpu_environment()
    all_results = []

    try:
        for v in versions:
            results, _, _ = train_fb237_version(v, k=args.k, lambda_cat=args.lambda_cat, bias=args.bias,
                                           model=args.model,
                                           aggregation=args.aggregation,
                                           random_seed=args.seed,
                                           pretrain=args.pretrain,
                                           proj_dim=args.proj_dim,
                                           pretrain_epochs=args.pretrain_epochs,
                                           pretrain_lr=args.pretrain_lr,
                                           pretrain_batch_size=args.pretrain_batch_size,
                                           dropout=args.dropout,
                                           maxiter_cat=args.maxiter)
            all_results.append(results)
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    finally:
        pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"{r['version']} ({r['model']}/{r['aggregation']}):")
        print(f"  k={r['k']}, lambda={r['lambda']}, bias={r['bias']}, "
              f"Feat dim={r['feature_dim']}, Time={r['train_time']:.1f}s")
        if r['test_acc'] is not None:
            print(f"  Test Acc={r['test_acc']:.6f}, MRR={r['mrr']:.6f}, "
                  f"Hits@1={r['hits']['Hits@1']:.6f}, "
                  f"Hits@3={r['hits']['Hits@3']:.6f}, "
                  f"Hits@10={r['hits']['Hits@10']:.6f}")
        else:
            print(f"  FAILED: {r['convergence_msg']}")

    # Save results to unified CSV (shared with baseline and kge-baseline)
    csv_path = os.path.join(os.path.dirname(__file__), "results", "fb237_unified_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            'method': 'IMC',
            'model': r['model'],
            'aggregation': r['aggregation'],
            'version': r['version'],
            'train_time_s': round(r['train_time'], 1),
            'train_acc': round(r['train_acc'], 6) if r['train_acc'] is not None else None,
            'valid_acc': round(r['valid_acc'], 6) if r['valid_acc'] is not None else None,
            'mrr': round(r['mrr'], 6) if r['mrr'] is not None else None,
        }
        if r.get('hits'):
            for k, v in r['hits'].items():
                row[k.replace('@', '_').lower()] = round(v, 6)
        for alpha in [0.01, 0.05, 0.1]:
            if alpha in r.get('conformal', {}):
                cr = r['conformal'][alpha]
                row[f'conformal_cov_a{alpha}'] = round(cr['marginal_coverage'], 6)
                row[f'conformal_size_a{alpha}'] = round(cr['average_set_size'], 4)
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    # Append to existing CSV if it exists, replacing duplicate entries
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        keys = df[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).tolist()
        mask = existing[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).isin(keys)
        existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

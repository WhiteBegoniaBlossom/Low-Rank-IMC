# Calculate loss function, gradient and hessian matrix.
import cupy as cp
import numpy as np
from typing import Tuple, Optional, Dict, Any

def optimized_categorical_loss_sparse(
        params: cp.ndarray,
        sparse_R,
        X: cp.ndarray,
        Y: cp.ndarray,
        lambda_: float,
        C: int,
        is_W: Optional[str],
        shape: Tuple[int, int],
        other: Dict[str, Any],
        sparse_mask,
        sparse_onehot_data
) -> Tuple[float, Optional[cp.ndarray]]:
    """
    Loss function for sparse matrix optimization
    """
    rows, cols, onehot_vectors = sparse_onehot_data
    num_nonzero = len(rows)

    # Parameter unpacking
    W, H, C_tensor, r_index = _prepare_parameters_sparse(params, is_W, shape, other)
    k = W.shape[1]

    # Batch feature extraction
    X_nonzero = X[rows]  # (num_nonzero, d1)
    Y_nonzero = Y[cols]  # (num_nonzero, d2)

    # Vectorized calculation of logits for all factors
    logits_nonzero = compute_logits_fully_vectorized(X_nonzero, Y_nonzero, W, H, C_tensor)

    # Calculate probability (vectorized)
    logits_max = cp.max(logits_nonzero, axis=1, keepdims=True)
    exp_logits = cp.exp(logits_nonzero - logits_max)
    probs_nonzero = exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + 1e-10)

    # Calculate cross-entropy loss
    log_probs = cp.log(probs_nonzero + 1e-10)
    cross_entropy = -cp.sum(onehot_vectors * log_probs)

    # Regularization term
    reg_term = 0.5 * lambda_ * (
            cp.sum(W ** 2) + cp.sum(H ** 2) + cp.sum(C_tensor ** 2)
    )
    total_loss = cross_entropy + reg_term

    # Gradient Calculation
    if is_W is not None:
        grad = _compute_gradient_sparse(
            params, probs_nonzero, onehot_vectors, rows, cols,
            X, Y, W, H, C_tensor, lambda_, is_W, r_index,
            X_nonzero, Y_nonzero
        )
        return float(total_loss), grad
    else:
        return float(total_loss), None


def _prepare_parameters_sparse(
        params: cp.ndarray,
        is_W: Optional[str],
        shape: Tuple[int, int],
        other: Dict[str, Any]
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, int]:
    """
    Prepare parameters for the sparse matrix
    """
    W = other.get('W', cp.zeros((1, 1), dtype=cp.float32)).copy()
    H = other.get('H', cp.zeros((1, 1), dtype=cp.float32)).copy()
    C_tensor = other.get('C_tensor', cp.zeros((1, 1), dtype=cp.float32)).copy()
    r_index = other.get('current_r', 0)

    if is_W == 'W':
        W[:, r_index] = params.reshape(-1)
    elif is_W == 'H':
        H[r_index, :] = params.reshape(-1)
    elif is_W is not None:
        C_tensor[r_index, :] = params.reshape(-1)

    return W, H, C_tensor, r_index


def _compute_gradient_sparse(
        params: cp.ndarray,
        probs_nonzero: cp.ndarray,
        onehot_vectors: cp.ndarray,
        rows: cp.ndarray,
        cols: cp.ndarray,
        X: cp.ndarray,
        Y: cp.ndarray,
        W: cp.ndarray,
        H: cp.ndarray,
        C_tensor: cp.ndarray,
        lambda_: float,
        is_W: str,
        r_index: int,
        X_nonzero: cp.ndarray,
        Y_nonzero: cp.ndarray
) -> cp.ndarray:
    """
    Calculate the gradient
    """
    dL_dlogits = (probs_nonzero - onehot_vectors)  # (num_nonzero, C)

    # Directly call the vectorized version
    return _compute_gradient_fully_vectorized(
        dL_dlogits, X_nonzero, Y_nonzero, W, H, C_tensor, lambda_, is_W, r_index
    )


def compute_logits_fully_vectorized(X_nonzero, Y_nonzero, W, H, C_tensor):
    """Fully vectorized logits computation"""
    # X_nonzero: (num_nonzero, d1), W: (d1, k) -> (num_nonzero, k)
    user_terms = X_nonzero @ W
    # Y_nonzero: (num_nonzero, d2), H: (k, d2) -> (num_nonzero, k)
    item_terms = Y_nonzero @ H.T
    # Interactive items: (num_nonzero, k)
    interactions = user_terms * item_terms
    # Batch matrix multiplication: (num_nonzero, k) @ (k, C) -> (num_nonzero, C)
    logits = interactions @ C_tensor
    return logits


def _compute_gradient_fully_vectorized(dL_dlogits, X_nonzero, Y_nonzero, W, H, C_tensor, lambda_, is_W, r_index):
    """Fully vectorized gradient computation"""
    if is_W == 'W':
        # Pre-calculation items
        Y_nonzero_H = cp.sum(Y_nonzero * H[r_index, :], axis=1)  # (num_nonzero,)

        # Vectorize all categories
        dL_dz_all = dL_dlogits * C_tensor[r_index, :]  # (num_nonzero, C)
        total_dL_dz = cp.sum(dL_dz_all, axis=1)  # (num_nonzero,)

        # Matrix multiplication
        grad = X_nonzero.T @ (total_dL_dz * Y_nonzero_H) + lambda_ * W[:, r_index]

    elif is_W == 'H':
        # Similar vectorization
        X_nonzero_W = cp.sum(X_nonzero * W[:, r_index], axis=1)  # (num_nonzero,)
        dL_dz_all = dL_dlogits * C_tensor[r_index, :]  # (num_nonzero, C)
        total_dL_dz = cp.sum(dL_dz_all, axis=1)  # (num_nonzero,)
        grad = Y_nonzero.T @ (total_dL_dz * X_nonzero_W) + lambda_ * H[r_index, :]

    else:  # C_tensor
        # Pre-compute interaction items
        user_terms = cp.sum(X_nonzero * W[:, r_index], axis=1)  # (num_nonzero,)
        item_terms = cp.sum(Y_nonzero * H[r_index, :], axis=1)  # (num_nonzero,)
        interaction = user_terms * item_terms  # (num_nonzero,)

        # Vectorized computation
        grad = dL_dlogits.T @ interaction + lambda_ * C_tensor[r_index, :]

    return grad


def optimized_compute_full_gradient_fully_vectorized(W, H, C_tensor, sparse_R, X, Y, lambda_, C, sparse_mask,
                                                     sparse_onehot_data):
    """Fully vectorized complete gradient norm calculation"""
    rows, cols, onehot_vectors = sparse_onehot_data
    num_nonzero = len(rows)

    # 批量提取特征
    X_nonzero = X[rows]  # (num_nonzero, d1)
    Y_nonzero = Y[cols]  # (num_nonzero, d2)

    # Calculate logits and probabilities (using vectorized version)
    logits = compute_logits_fully_vectorized(X_nonzero, Y_nonzero, W, H, C_tensor)
    logits_max = cp.max(logits, axis=1, keepdims=True)
    exp_logits = cp.exp(logits - logits_max)
    probs = exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + 1e-10)

    # Gradient
    dL_dlogits = probs - onehot_vectors  # (num_nonzero, C)

    total_grad_norm = 0.0
    k = W.shape[1]

    # Pre-calculate common terms
    XW = X_nonzero @ W  # (num_nonzero, k)
    YH = Y_nonzero @ H.T  # (num_nonzero, k)
    interactions = XW * YH  # (num_nonzero, k)

    for r in range(k):
        # W gradient
        Y_nonzero_H = cp.sum(Y_nonzero * H[r, :], axis=1)
        dL_dz_all = dL_dlogits * C_tensor[r, :]
        total_dL_dz = cp.sum(dL_dz_all, axis=1)
        grad_W = X_nonzero.T @ (total_dL_dz * Y_nonzero_H) + lambda_ * W[:, r]

        # W gradient
        X_nonzero_W = cp.sum(X_nonzero * W[:, r], axis=1)
        grad_H = Y_nonzero.T @ (total_dL_dz * X_nonzero_W) + lambda_ * H[r, :]

        # C_tensor gradient
        interaction_r = (cp.sum(X_nonzero * W[:, r], axis=1) *
                         cp.sum(Y_nonzero * H[r, :], axis=1))
        grad_C = dL_dlogits.T @ interaction_r + lambda_ * C_tensor[r, :]

        total_grad_norm += (cp.sum(grad_W ** 2) + cp.sum(grad_H ** 2) + cp.sum(grad_C ** 2))

    return float(cp.sqrt(total_grad_norm))


def optimized_compute_full_gradient_sparse(
        W: cp.ndarray,
        H: cp.ndarray,
        C_tensor: cp.ndarray,
        sparse_R,
        X: cp.ndarray,
        Y: cp.ndarray,
        lambda_: float,
        C: int,
        sparse_mask,
        sparse_onehot_data
) -> float:
    """Full gradient calculation using fully vectorized methods"""
    return optimized_compute_full_gradient_fully_vectorized(
        W, H, C_tensor, sparse_R, X, Y, lambda_, C, sparse_mask, sparse_onehot_data
    )


def compute_W_grad_and_loss(W, H, C_tensor, X, Y, rows, cols, onehot_vectors, lambda_):
    """
    Return loss (float), grad_W (d1, k)
    """
    X_nz = X[rows]
    Y_nz = Y[cols]
    logits = compute_logits_fully_vectorized(X_nz, Y_nz, W, H, C_tensor)
    logits_max = cp.max(logits, axis=1, keepdims=True)
    exp_logits = cp.exp(logits - logits_max)
    probs = exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    log_probs = cp.log(probs + 1e-10)
    cross_entropy = -cp.sum(onehot_vectors * log_probs)
    reg = 0.5 * lambda_ * (cp.sum(W**2) + cp.sum(H**2) + cp.sum(C_tensor**2))
    loss = cross_entropy + reg

    delta = probs - onehot_vectors          # (N, C)
    YH = Y_nz @ H.T                         # (N, k)
    grad = cp.zeros_like(W)
    for r in range(W.shape[1]):
        delta_dot_Cr = delta @ C_tensor[r, :]   # (N,)
        grad[:, r] = X_nz.T @ (delta_dot_Cr * YH[:, r]) + lambda_ * W[:, r]
    return loss, grad


def compute_W_hvp(W, H, C_tensor, X, Y, rows, cols, onehot_vectors, lambda_, V, batch_size=500):
    """
    Hessian-vector product for W.
    V: (d1, k) direction matrix.
    """
    d1, k = W.shape
    Hv = cp.zeros_like(W)
    N = len(rows)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = slice(start, end)
        Xb = X[rows[idx]]
        Yb = Y[cols[idx]]
        onehot_b = onehot_vectors[idx]

        # forward for this batch
        logits_b = compute_logits_fully_vectorized(Xb, Yb, W, H, C_tensor)
        logits_max_b = cp.max(logits_b, axis=1, keepdims=True)
        exp_b = cp.exp(logits_b - logits_max_b)
        probs_b = exp_b / (cp.sum(exp_b, axis=1, keepdims=True) + 1e-10)

        XW_b = Xb @ W            # (bs, k)
        YH_b = Yb @ H.T          # (bs, k)
        a_b = Xb @ V             # (bs, k)
        b_b = a_b * YH_b         # (bs, k)

        Cp_b = probs_b @ C_tensor.T            # (bs, k)
        t1_b = b_b @ C_tensor                   # (bs, C)

        # temp1 = sum_c C_{rc} * p_c * t1_c  (vectorized)
        weighted_C_b = probs_b[:, None, :] * C_tensor[None, :, :]   # (bs, k, C)
        temp1_b = cp.sum(weighted_C_b * t1_b[:, None, :], axis=2)   # (bs, k)

        # temp2 = Cp * (Cp · b)
        temp2_b = Cp_b * cp.sum(Cp_b * b_b, axis=1, keepdims=True)  # (bs, k)

        temp_b = temp1_b - temp2_b

        # accumulate Hv
        for r in range(k):
            Hv[:, r] += Xb.T @ (temp_b[:, r] * YH_b[:, r])

    Hv += lambda_ * V
    return Hv


def compute_H_grad_and_loss(H, W, C_tensor, X, Y, rows, cols, onehot_vectors, lambda_):
    """
    Return (loss, grad_H)
    H : (k, d2)
    """
    X_nz = X[rows]          # (N, d1)
    Y_nz = Y[cols]          # (N, d2)

    logits = compute_logits_fully_vectorized(X_nz, Y_nz, W, H, C_tensor)  # (N, C)
    logits_max = cp.max(logits, axis=1, keepdims=True)
    exp_logits = cp.exp(logits - logits_max)
    probs = exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    log_probs = cp.log(probs + 1e-10)
    cross_entropy = -cp.sum(onehot_vectors * log_probs)
    reg = 0.5 * lambda_ * (cp.sum(W**2) + cp.sum(H**2) + cp.sum(C_tensor**2))
    loss = cross_entropy + reg

    delta = probs - onehot_vectors          # (N, C)
    XW = X_nz @ W                           # (N, k)
    grad = cp.zeros_like(H)
    for r in range(H.shape[0]):
        delta_dot_Cr = delta @ C_tensor[r, :]   # (N,)
        grad[r, :] = Y_nz.T @ (delta_dot_Cr * XW[:, r]) + lambda_ * H[r, :]
    return loss, grad


def compute_H_hvp(H, W, C_tensor, X, Y, rows, cols, onehot_vectors, lambda_, V, batch_size=500):
    """
    Hessian-vector product for H.
    H, V shape: (k, d2)
    """
    k, d2 = H.shape
    Hv = cp.zeros_like(H)
    N = len(rows)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = slice(start, end)
        Xb = X[rows[idx]]          # (bs, d1)
        Yb = Y[cols[idx]]          # (bs, d2)
        onehot_b = onehot_vectors[idx]

        # 前向传播
        logits_b = compute_logits_fully_vectorized(Xb, Yb, W, H, C_tensor)  # (bs, C)
        logits_max_b = cp.max(logits_b, axis=1, keepdims=True)
        exp_b = cp.exp(logits_b - logits_max_b)
        probs_b = exp_b / (cp.sum(exp_b, axis=1, keepdims=True) + 1e-10)

        XW_b = Xb @ W                      # (bs, k)
        YVt_b = Yb @ V.T                   # (bs, k)   (因为 V 是 k×d2，Yb 是 bs×d2)
        a_b = YVt_b                        # (bs, k)
        b_b = a_b * XW_b                   # (bs, k)

        Cp_b = probs_b @ C_tensor.T        # (bs, k)
        t1_b = b_b @ C_tensor               # (bs, C)

        # temp1 = sum_c C_{rc} * p_c * t1_c
        weighted_C_b = probs_b[:, None, :] * C_tensor[None, :, :]   # (bs, k, C)
        temp1_b = cp.sum(weighted_C_b * t1_b[:, None, :], axis=2)   # (bs, k)

        # temp2 = Cp * (Cp · b)
        temp2_b = Cp_b * cp.sum(Cp_b * b_b, axis=1, keepdims=True)  # (bs, k)

        temp_b = temp1_b - temp2_b

        # 累加到 Hv：对每个 r，Hv[r,:] += Yb.T @ (temp_b[:,r] * XW_b[:,r])
        for r in range(k):
            Hv[r, :] += Yb.T @ (temp_b[:, r] * XW_b[:, r])

    Hv += lambda_ * V
    return Hv


def compute_C_grad_and_loss(C_tensor, W, H, X, Y, rows, cols, onehot_vectors, lambda_):
    """
    返回 (loss, grad_C)
    C_tensor : (k, C)
    """
    X_nz = X[rows]
    Y_nz = Y[cols]

    logits = compute_logits_fully_vectorized(X_nz, Y_nz, W, H, C_tensor)
    logits_max = cp.max(logits, axis=1, keepdims=True)
    exp_logits = cp.exp(logits - logits_max)
    probs = exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    log_probs = cp.log(probs + 1e-10)
    cross_entropy = -cp.sum(onehot_vectors * log_probs)
    reg = 0.5 * lambda_ * (cp.sum(W**2) + cp.sum(H**2) + cp.sum(C_tensor**2))
    loss = cross_entropy + reg

    delta = probs - onehot_vectors          # (N, C)
    interactions = (X_nz @ W) * (Y_nz @ H.T)  # (N, k)
    grad = interactions.T @ delta + lambda_ * C_tensor   # (k, C)
    return loss, grad


def compute_C_hvp(C_tensor, W, H, X, Y, rows, cols, onehot_vectors, lambda_, V, batch_size=500):
    """
    Hessian-vector product for C.
    C_tensor, V 形状均为 (k, C)
    """
    k, C = C_tensor.shape
    Hv = cp.zeros_like(C_tensor)
    N = len(rows)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = slice(start, end)
        Xb = X[rows[idx]]
        Yb = Y[cols[idx]]
        onehot_b = onehot_vectors[idx]

        logits_b = compute_logits_fully_vectorized(Xb, Yb, W, H, C_tensor)
        logits_max_b = cp.max(logits_b, axis=1, keepdims=True)
        exp_b = cp.exp(logits_b - logits_max_b)
        probs_b = exp_b / (cp.sum(exp_b, axis=1, keepdims=True) + 1e-10)

        interactions_b = (Xb @ W) * (Yb @ H.T)   # (bs, k)
        M_b = interactions_b @ V                  # (bs, C)

        # temp = probs * M - probs * (probs · M)
        temp_b = probs_b * M_b - probs_b * cp.sum(probs_b * M_b, axis=1, keepdims=True)

        Hv += interactions_b.T @ temp_b

    Hv += lambda_ * V
    return Hv
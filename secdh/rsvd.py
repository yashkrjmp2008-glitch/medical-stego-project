"""Randomized Singular Value Decomposition."""

from __future__ import annotations
import numpy as np

def rsvd(A: np.ndarray, rank: int | None = None, n_oversamples: int = 10,
          n_power_iter: int = 2, random_state: int | None = None):
    A = np.asarray(A, dtype=np.float64)
    i, j = A.shape
    if rank is None:
        rank = min(i, j)
    r = min(rank, min(i, j))
    k = min(r + n_oversamples, j)

    rng = np.random.default_rng(random_state)
    Omega = rng.standard_normal((j, k))
    Y = A @ Omega
    W, _ = np.linalg.qr(Y)

    for _ in range(n_power_iter):
        W2, _ = np.linalg.qr(A.T @ W)
        W, _ = np.linalg.qr(A @ W2)

    X = W.T @ A
    Ux, S, Vt = np.linalg.svd(X, full_matrices=False)
    U = W @ Ux
    return U[:, :r], S[:r], Vt[:r, :]

def rsvd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    return (U * S) @ Vt

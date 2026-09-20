"""PCA-based fusion and normalized principal components."""

from __future__ import annotations
import numpy as np

def normalized_principal_components(vec1: np.ndarray, vec2: np.ndarray):
    v1 = np.asarray(vec1, dtype=np.float64).ravel()
    v2 = np.asarray(vec2, dtype=np.float64).ravel()
    cov = np.cov(np.vstack([v1, v2]))
    eigvals, eigvecs = np.linalg.eigh(cov)

    eig11, eig21 = eigvecs[0, 0], eigvecs[1, 0]
    eig12, eig22 = eigvecs[0, 1], eigvecs[1, 1]
    diag11, diag22 = eigvals[0], eigvals[1]

    if diag11 > diag22:
        denom = eig11 + eig21
        pc1 = eig11 / denom if denom != 0 else 0.5
        pc2 = eig21 / denom if denom != 0 else 0.5
    else:
        denom = eig12 + eig22
        pc1 = eig12 / denom if denom != 0 else 0.5
        pc2 = eig22 / denom if denom != 0 else 0.5

    total = abs(pc1) + abs(pc2)
    if total == 0:
        return 0.5, 0.5
    return float(abs(pc1) / total), float(abs(pc2) / total)

def pca_fuse(img1: np.ndarray, img2: np.ndarray):
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    pc1, pc2 = normalized_principal_components(img1, img2)
    return pc1, pc2, pc1 * img1 + pc2 * img2

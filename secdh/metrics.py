"""Image-quality metrics used by the SecDH project."""

from __future__ import annotations
import numpy as np

def psnr(original, compared, data_range=255.0):
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(compared, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))

def normalized_correlation(original, recovered):
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(recovered, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)

def ssim(original, compared, data_range=255.0):
    """Global SSIM approximation using image means, variances and covariance."""
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(compared, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape.")
    L = float(data_range)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    value = ((2 * mu_a * mu_b + C1) * (2 * cov + C2) /
             ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)))
    return float(value)

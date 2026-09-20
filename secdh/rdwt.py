"""Redundant Haar wavelet transform (RDWT/SWT), single level."""

from __future__ import annotations
import numpy as np

_SQRT2 = np.sqrt(2.0)

def _haar_forward_1d(x: np.ndarray, axis: int):
    x0 = x
    x1 = np.roll(x, -1, axis=axis)
    return (x0 + x1) / _SQRT2, (x0 - x1) / _SQRT2

def _haar_inverse_1d(low: np.ndarray, high: np.ndarray, axis: int):
    return (low + high) / _SQRT2

def rdwt2(image: np.ndarray, level: int = 1):
    """Forward single-level 2-D undecimated Haar transform."""
    if level != 1:
        raise NotImplementedError("Only single-level RDWT is implemented.")
    image = np.asarray(image, dtype=np.float64)
    L, H = _haar_forward_1d(image, axis=0)
    LL, LH = _haar_forward_1d(L, axis=1)
    HL, HH = _haar_forward_1d(H, axis=1)
    return LL, LH, HL, HH

def irdwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray):
    """Inverse single-level RDWT."""
    L = _haar_inverse_1d(LL, LH, axis=1)
    H = _haar_inverse_1d(HL, HH, axis=1)
    return _haar_inverse_1d(L, H, axis=0)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    bands = rdwt2(img)
    rec = irdwt2(*bands)
    print("max reconstruction error:", np.max(np.abs(img - rec)))

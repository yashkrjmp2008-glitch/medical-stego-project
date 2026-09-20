"""Watermark extraction for the SecDH scheme."""

from __future__ import annotations
import numpy as np

from .rdwt import rdwt2, irdwt2
from .arnold_cat_map import arnold_unscramble
from .embed import WatermarkKey

def extract_watermark(enc_marked_img: np.ndarray, key: WatermarkKey) -> np.ndarray:
    dec_marked_img = arnold_unscramble(
        enc_marked_img, a=key.arnold_a, b=key.arnold_b,
        iterations=key.arnold_iterations
    )
    _A3, _B3, _C3, D3 = rdwt2(dec_marked_img)
    M = key.U1.T @ D3 @ key.Vt1.T
    S_emb_hat = np.diag(M)
    if abs(key.pc2) < 1e-12:
        raise ZeroDivisionError("Watermark PCA coefficient PC2 is too close to zero.")
    S2_hat = (S_emb_hat - key.pc1 * key.S1) / key.pc2
    D2_hat = (key.U2 * S2_hat) @ key.Vt2
    return irdwt2(key.A2, key.B2, key.C2, D2_hat)

"""Watermark embedding for the SecDH scheme."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .normalization import normalize_image
from .rdwt import rdwt2, irdwt2
from .rsvd import rsvd
from .pca_fusion import normalized_principal_components
from .arnold_cat_map import arnold_scramble

@dataclass
class WatermarkKey:
    pc1: float
    pc2: float
    U1: np.ndarray
    S1: np.ndarray
    Vt1: np.ndarray
    U2: np.ndarray
    S2: np.ndarray
    Vt2: np.ndarray
    A2: np.ndarray
    B2: np.ndarray
    C2: np.ndarray
    arnold_a: int
    arnold_b: int
    arnold_iterations: int
    watermark_shape: tuple

def embed_watermark(cover_img, watermark_img, arnold_a=1, arnold_b=1,
                     arnold_iterations=12, normalize_cover=False,
                     rsvd_random_state=0):
    cover_img = np.asarray(cover_img, dtype=np.float64)
    watermark_img = np.asarray(watermark_img, dtype=np.float64)
    if cover_img.shape != watermark_img.shape:
        raise ValueError("cover_img and watermark_img must have the same shape.")

    norm_img = normalize_image(cover_img) if normalize_cover else cover_img
    A1, B1, C1, D1 = rdwt2(norm_img)
    U1, S1, Vt1 = rsvd(D1, random_state=rsvd_random_state)

    A2, B2, C2, D2 = rdwt2(watermark_img)
    U2, S2, Vt2 = rsvd(D2, random_state=rsvd_random_state)

    pc1, pc2 = normalized_principal_components(S1, S2)
    S_emb = pc1 * S1 + pc2 * S2
    D_new = (U1 * S_emb) @ Vt1
    marked_img = irdwt2(A1, B1, C1, D_new)
    enc_marked_img = arnold_scramble(marked_img, a=arnold_a, b=arnold_b,
                                     iterations=arnold_iterations)

    key = WatermarkKey(
        pc1=pc1, pc2=pc2, U1=U1, S1=S1, Vt1=Vt1,
        U2=U2, S2=S2, Vt2=Vt2, A2=A2, B2=B2, C2=C2,
        arnold_a=arnold_a, arnold_b=arnold_b,
        arnold_iterations=arnold_iterations,
        watermark_shape=watermark_img.shape,
    )
    return enc_marked_img, marked_img, key

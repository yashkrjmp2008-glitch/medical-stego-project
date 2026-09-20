"""Basic sanity / round-trip tests."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from secdh.rdwt import rdwt2, irdwt2
from secdh.rsvd import rsvd, rsvd_reconstruct
from secdh.arnold_cat_map import arnold_scramble, arnold_unscramble
from secdh.pca_fusion import normalized_principal_components
from secdh.embed import embed_watermark
from secdh.extract import extract_watermark
from secdh.metrics import psnr, normalized_correlation

def test_rdwt_perfect_reconstruction():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)) * 255
    assert np.allclose(img, irdwt2(*rdwt2(img)), atol=1e-8)

def test_rsvd_reconstruction_quality():
    rng = np.random.default_rng(1)
    A = rng.random((64, 64)) * 255
    U, S, Vt = rsvd(A, random_state=1)
    rel_err = np.linalg.norm(A - rsvd_reconstruct(U, S, Vt)) / np.linalg.norm(A)
    assert rel_err < 1e-6

def test_arnold_cat_map_is_invertible():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(32, 32)).astype(np.float64)
    restored = arnold_unscramble(arnold_scramble(img, 1, 1, 7), 1, 1, 7)
    assert np.allclose(img, restored)

def test_pca_fusion_sums_to_one():
    rng = np.random.default_rng(3)
    pc1, pc2 = normalized_principal_components(rng.random(40)*100, rng.random(40)*5)
    assert 0 <= pc1 <= 1 and 0 <= pc2 <= 1
    assert abs(pc1 + pc2 - 1) < 1e-8

def test_embed_extract_roundtrip_no_attack():
    rng = np.random.default_rng(4)
    size = 64
    cover = np.clip(127 + 40*rng.standard_normal((size, size)), 0, 255)
    watermark = np.clip(127 + 40*rng.standard_normal((size, size)), 0, 255)
    enc_marked, marked, key = embed_watermark(cover, watermark, arnold_iterations=5)
    recovered = extract_watermark(enc_marked, key)
    assert normalized_correlation(watermark, recovered) > 0.99
    assert psnr(cover, marked) > 20

"""Arnold cat map scrambling and descrambling."""

from __future__ import annotations
import numpy as np

def arnold_scramble(image: np.ndarray, a: int = 1, b: int = 1, iterations: int = 1) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold cat map requires a square 2-D image.")
    n = image.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = xs.copy(), ys.copy()
    out = image.copy()
    for _ in range(iterations):
        new_x = (x + a * y) % n
        new_y = (b * x + (a * b + 1) * y) % n
        scrambled = np.zeros_like(out)
        scrambled[new_x, new_y] = out
        out = scrambled
    return out

def arnold_unscramble(image: np.ndarray, a: int = 1, b: int = 1, iterations: int = 1) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold cat map requires a square 2-D image.")
    n = image.shape[0]
    inv = np.array([[a * b + 1, -a], [-b, 1]])
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = xs.copy(), ys.copy()
    out = image.copy()
    for _ in range(iterations):
        coords = inv @ np.vstack([x.ravel(), y.ravel()])
        orig_x = coords[0].reshape(n, n) % n
        orig_y = coords[1].reshape(n, n) % n
        restored = np.zeros_like(out)
        restored[orig_x, orig_y] = out[x, y]
        out = restored
    return out

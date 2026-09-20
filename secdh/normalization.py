"""Moment-based image normalization."""

from __future__ import annotations
import numpy as np
from scipy.ndimage import affine_transform

def _raw_moment(img, p, q, x=None, y=None):
    if x is None or y is None:
        y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float(np.sum((x ** p) * (y ** q) * img))

def _central_moments(img):
    m00 = _raw_moment(img, 0, 0)
    if m00 == 0:
        m00 = 1e-8
    m10 = _raw_moment(img, 1, 0)
    m01 = _raw_moment(img, 0, 1)
    x_bar, y_bar = m10 / m00, m01 / m00
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xc, yc = x - x_bar, y - y_bar
    def mu(p, q):
        return float(np.sum((xc ** p) * (yc ** q) * img))
    return {
        "m00": m00, "x_bar": x_bar, "y_bar": y_bar,
        "mu20": mu(2, 0), "mu02": mu(0, 2), "mu11": mu(1, 1),
        "mu30": mu(3, 0), "mu03": mu(0, 3),
        "mu21": mu(2, 1), "mu12": mu(1, 2),
    }

def _solve_beta(mu):
    coeffs = [mu["mu03"], 3 * mu["mu12"], 3 * mu["mu21"], mu["mu30"]]
    if abs(coeffs[0]) < 1e-12:
        coeffs = [c for c in coeffs if abs(c) > 1e-12] or [1.0, 0.0]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if len(real_roots) == 0:
        return 0.0
    return float(np.clip(real_roots[np.argmin(np.abs(real_roots))], -1.0, 1.0))

def normalize_image(image):
    img = np.asarray(image, dtype=np.float64)
    out_shape = img.shape
    mu = _central_moments(img)
    cy, cx = out_shape[0] / 2.0, out_shape[1] / 2.0
    shift = (mu["y_bar"] - cy, mu["x_bar"] - cx)
    translated = affine_transform(img, matrix=np.eye(2), offset=shift, order=1, mode="constant")
    mu = _central_moments(translated)

    beta = _solve_beta(mu)
    Sx_inv = np.array([[1.0, -beta], [0.0, 1.0]])
    sheared_x = affine_transform(translated, matrix=Sx_inv, order=1, mode="constant")

    mu = _central_moments(sheared_x)
    lam = mu["mu11"] / mu["mu20"] if mu["mu20"] != 0 else 0.0
    lam = float(np.clip(lam, -1.0, 1.0))
    Sy_inv = np.array([[1.0, 0.0], [-lam, 1.0]])
    sheared_y = affine_transform(sheared_x, matrix=Sy_inv, order=1, mode="constant")

    mu = _central_moments(sheared_y)
    target_variance = (min(out_shape) ** 2) / 12.0
    var20 = mu["mu20"] / mu["m00"] if mu["m00"] != 0 else 0.0
    var02 = mu["mu02"] / mu["m00"] if mu["m00"] != 0 else 0.0
    alpha = np.sqrt(target_variance / var20) if var20 > 0 else 1.0
    delta = np.sqrt(target_variance / var02) if var02 > 0 else 1.0
    alpha = float(np.clip(alpha, 0.25, 4.0))
    delta = float(np.clip(delta, 0.25, 4.0))
    Sscale_inv = np.array([[1.0 / delta, 0.0], [0.0, 1.0 / alpha]])
    return affine_transform(sheared_y, matrix=Sscale_inv, order=1,
                            mode="constant", output_shape=out_shape)

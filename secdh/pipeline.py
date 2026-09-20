"""High-level SecDH convenience wrapper."""

from __future__ import annotations
import numpy as np
from .embed import embed_watermark, WatermarkKey
from .extract import extract_watermark
from .metrics import psnr, ssim, normalized_correlation

class SecDH:
    def __init__(self, arnold_a=1, arnold_b=1, arnold_iterations=12,
                 normalize_cover=False, rsvd_random_state=0):
        self.arnold_a = arnold_a
        self.arnold_b = arnold_b
        self.arnold_iterations = arnold_iterations
        self.normalize_cover = normalize_cover
        self.rsvd_random_state = rsvd_random_state

    def embed(self, cover_img: np.ndarray, watermark_img: np.ndarray):
        return embed_watermark(
            cover_img, watermark_img,
            arnold_a=self.arnold_a, arnold_b=self.arnold_b,
            arnold_iterations=self.arnold_iterations,
            normalize_cover=self.normalize_cover,
            rsvd_random_state=self.rsvd_random_state,
        )

    def extract(self, enc_marked_img: np.ndarray, key: WatermarkKey):
        return extract_watermark(enc_marked_img, key)

    @staticmethod
    def report(cover_img, marked_img, original_watermark, recovered_watermark,
               data_range=255.0):
        return {
            "PSNR": psnr(cover_img, marked_img, data_range=data_range),
            "SSIM": ssim(cover_img, marked_img, data_range=data_range),
            "NC": normalized_correlation(original_watermark, recovered_watermark),
        }

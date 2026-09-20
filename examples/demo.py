"""End-to-end SecDH demonstration."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from secdh import SecDH
from secdh import attacks
from secdh.metrics import normalized_correlation

def load_or_synthesize(path, size, seed):
    if path:
        img = Image.open(path).convert("L").resize((size, size))
        return np.array(img, dtype=np.float64)
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    img = 127 + 80 * np.sin(x) * np.cos(y) + 20 * rng.standard_normal((size, size))
    return np.clip(img, 0, 255)

def main():
    parser = argparse.ArgumentParser(description="SecDH watermarking demo")
    parser.add_argument("--cover", default=None)
    parser.add_argument("--watermark", default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cover = load_or_synthesize(args.cover, args.size, 1)
    watermark = load_or_synthesize(args.watermark, args.size, 2)

    secdh = SecDH(arnold_a=1, arnold_b=1, arnold_iterations=12)
    print("Embedding watermark ...")
    enc_marked, marked, key = secdh.embed(cover, watermark)

    recovered = secdh.extract(enc_marked, key)
    print("\n=== No attack ===")
    for k, v in secdh.report(cover, marked, watermark, recovered).items():
        print(f"  {k}: {v:.4f}")

    attack_fns = {
        "salt_and_pepper_0.01": lambda im: attacks.salt_and_pepper_noise(im, density=0.01, rng=np.random.default_rng(0)),
        "speckle_0.01": lambda im: attacks.speckle_noise(im, variance=0.01, rng=np.random.default_rng(0)),
        "gaussian_noise_0.001": lambda im: attacks.gaussian_noise(im, variance=0.001, rng=np.random.default_rng(0)),
        "median_filter_3x3": lambda im: attacks.median_filter(im, size=3),
        "jpeg_q50": lambda im: attacks.jpeg_compression(im, quality=50),
        "rotation_5deg": lambda im: attacks.rotate(im, angle_degrees=5),
        "scaling_0.5": lambda im: attacks.scale(im, factor=0.5),
        "cropping": lambda im: attacks.crop(im, box=(20, 20, args.size - 40, args.size - 20)),
        "histogram_equalization": attacks.histogram_equalization,
    }

    print("\n=== Robustness under attacks (NC) ===")
    for name, fn in attack_fns.items():
        attacked = fn(enc_marked.copy())
        recovered = secdh.extract(attacked, key)
        print(f"  {name:28s} NC = {normalized_correlation(watermark, recovered):.4f}")

    Image.fromarray(np.clip(cover, 0, 255).astype(np.uint8)).save(out_dir / "cover.png")
    Image.fromarray(np.clip(watermark, 0, 255).astype(np.uint8)).save(out_dir / "watermark.png")
    Image.fromarray(np.clip(marked, 0, 255).astype(np.uint8)).save(out_dir / "marked.png")
    Image.fromarray(np.clip(enc_marked, 0, 255).astype(np.uint8)).save(out_dir / "encrypted_marked.png")
    Image.fromarray(np.clip(recovered, 0, 255).astype(np.uint8)).save(out_dir / "recovered_watermark.png")
    print(f"\nSaved demo images to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

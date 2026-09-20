"""Attack simulators for robustness testing."""

from __future__ import annotations
import numpy as np
from scipy import ndimage

def salt_and_pepper_noise(img, density=0.01, rng=None):
    rng = rng or np.random.default_rng()
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < density / 2] = img.min()
    out[(mask >= density / 2) & (mask < density)] = img.max()
    return out

def speckle_noise(img, variance=0.01, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, np.sqrt(variance), img.shape)
    return img + img * noise

def gaussian_noise(img, mean=0.0, variance=0.001, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(mean, np.sqrt(variance) * 255.0, img.shape)
    return img + noise

def poisson_noise(img, rng=None):
    rng = rng or np.random.default_rng()
    scaled = np.clip(img, 0, None)
    vals = 2 ** np.ceil(np.log2(max(scaled.max(), 1)))
    return rng.poisson(scaled * vals) / float(vals)

def median_filter(img, size=3):
    return ndimage.median_filter(img, size=size)

def average_filter(img, size=3):
    return ndimage.uniform_filter(img, size=size)

def gaussian_low_pass(img, sigma=1.0):
    return ndimage.gaussian_filter(img, sigma=sigma)

def sharpen(img, amount=0.5):
    blurred = ndimage.gaussian_filter(img, sigma=1.0)
    return img + amount * (img - blurred)

def rotate(img, angle_degrees=5.0):
    return ndimage.rotate(img, angle_degrees, reshape=False, order=1, mode="reflect")

def scale(img, factor=0.5):
    original_shape = img.shape
    zoomed = ndimage.zoom(img, factor, order=1)
    out = np.zeros(original_shape, dtype=img.dtype)
    h = min(original_shape[0], zoomed.shape[0])
    w = min(original_shape[1], zoomed.shape[1])
    out[:h, :w] = zoomed[:h, :w]
    return out

def crop(img, box=(20, 20, 400, 480)):
    out = img.copy()
    r0, c0, r1, c1 = box
    out[r0:r1, c0:c1] = 0
    return out

def histogram_equalization(img):
    flat = img.ravel()
    hist, bins = np.histogram(flat, bins=256, range=(0, 255), density=True)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    equalized = np.interp(flat, bins[:-1], cdf)
    return equalized.reshape(img.shape)

def jpeg_compression(img, quality=50):
    from PIL import Image
    import io
    arr = np.clip(img, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.float64)

def motion_blur(img, size=9):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return ndimage.convolve(img, kernel, mode="reflect")

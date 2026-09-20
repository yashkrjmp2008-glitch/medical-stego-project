# SecDH-Watermarking

A Python implementation of a PCA-based image watermarking/data-hiding workflow inspired by the SecDH scheme described in:

> O.P. Singh, A.K. Singh, A.K. Agrawal, H. Zhou, “SecDH: Security of COVID-19 images based on data hiding with PCA,” Computer Communications, 191 (2022), 368–377.

## Features

- Haar Redundant Discrete Wavelet Transform (RDWT)
- Randomized Singular Value Decomposition (RSVD)
- PCA-based singular-value fusion
- Arnold cat-map scrambling/descrambling
- Moment-based image normalization
- Watermark embedding and extraction
- PSNR, SSIM and Normalized Correlation (NC)
- Noise, filtering, geometric and JPEG attack simulations
- End-to-end demonstration
- Basic automated tests

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python examples/demo.py
```

Use custom grayscale images:

```bash
python examples/demo.py --cover path/to/cover.png --watermark path/to/logo.png --size 256
```

Generated images are written to the directory specified by `--out-dir`.

## Run tests

```bash
python -m pytest tests/
```

You can also run:

```bash
python tests/test_basic.py
```

## Package layout

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

## Important note about medical images

This repository is intended for academic and research experimentation. Do not upload confidential, identifiable, or otherwise sensitive medical images to a public GitHub repository.

This implementation is not a clinical system and has not been validated or certified for clinical use.

## License

Released under the MIT License. See `LICENSE`.

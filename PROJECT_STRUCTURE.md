# Project Structure

```text
SecDH-Watermarking/
├── README.md
├── PROJECT_STRUCTURE.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── examples/
│   └── demo.py
├── secdh/
│   ├── __init__.py
│   ├── rdwt.py
│   ├── rsvd.py
│   ├── pca_fusion.py
│   ├── arnold_cat_map.py
│   ├── normalization.py
│   ├── embed.py
│   ├── extract.py
│   ├── metrics.py
│   ├── attacks.py
│   └── pipeline.py
├── tests/
│   └── test_basic.py
└── outputs/
```

## Main workflow

```text
Cover Image
    ↓
Normalization (optional)
    ↓
RDWT
    ↓
HH Sub-band
    ↓
RSVD
    ↓
PCA Fusion ← Watermark → RDWT → RSVD
    ↓
Modified Singular Values
    ↓
Inverse RDWT
    ↓
Watermarked Image
    ↓
Arnold Cat Map
    ↓
Encrypted / Transmitted Image
    ↓
Descrambling + Extraction
    ↓
Recovered Watermark
```

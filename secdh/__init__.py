"""
SecDH: PCA-based medical image watermarking / data hiding.

Python re-implementation of the scheme described in:
O.P. Singh, A.K. Singh, A.K. Agrawal, H. Zhou,
"SecDH: Security of COVID-19 images based on data hiding with PCA",
Computer Communications 191 (2022) 368-377.
"""

from .pipeline import SecDH

__all__ = ["SecDH"]
__version__ = "1.0.0"

"""Standalone Stage 2 pseudo-text generation package."""

from .categories import CATEGORY_SPECS, DEFAULT_CATEGORY_KEYS
from .schema import GeometryLabels, MaskQualityMetrics, VLMResult

__all__ = [
    "CATEGORY_SPECS",
    "DEFAULT_CATEGORY_KEYS",
    "GeometryLabels",
    "MaskQualityMetrics",
    "VLMResult",
]

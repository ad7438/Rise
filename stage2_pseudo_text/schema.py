"""Dataclasses shared by the Stage 2 standalone scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MaskQualityMetrics:
    area_ratio: float
    bbox: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    original_component_count: int
    kept_area_pixels: int
    touches_edge: bool
    score: float
    flags: List[str] = field(default_factory=list)
    is_empty: bool = False


@dataclass
class GeometryLabels:
    location_key: str
    location_label_zh: str
    size_key: str
    size_label_zh: str


@dataclass
class VLMResult:
    category_key: str
    category_confidence: float
    raw_response: str
    evidence: str = ""

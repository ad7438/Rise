"""Mask cleanup, geometry extraction, and visualization helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .categories import LOCATION_KEY_TO_ZH, SIZE_KEY_TO_ZH
from .schema import GeometryLabels, MaskQualityMetrics


def load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_binary_mask(mask_path: str, threshold: int = 127) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask >= threshold).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    if mask.shape[1] == target_width and mask.shape[0] == target_height:
        return mask
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    resized = mask_image.resize((target_width, target_height), resample=Image.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _touches_edge(bbox: Tuple[int, int, int, int], width: int, height: int) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= 0 or y0 <= 0 or x1 >= width - 1 or y1 >= height - 1


def score_mask_quality(
    area_ratio: float,
    original_component_count: int,
    touches_edge: bool,
    is_empty: bool,
) -> Tuple[float, list[str]]:
    if is_empty:
        return 0.0, ["empty_mask"]

    score = 1.0
    flags: list[str] = []

    if area_ratio < 0.002:
        score -= 0.45
        flags.append("tiny_mask")
    elif area_ratio < 0.01:
        score -= 0.2
        flags.append("small_mask")
    elif area_ratio > 0.45:
        score -= 0.2
        flags.append("oversized_mask")

    if original_component_count > 3:
        score -= 0.15
        flags.append("fragmented_mask")
    elif original_component_count > 1:
        score -= 0.05
        flags.append("multi_component_mask")

    if touches_edge:
        score -= 0.1
        flags.append("touching_edge")

    return float(max(0.0, min(1.0, score))), flags


def preprocess_mask(
    mask: np.ndarray,
    min_component_area_pixels: int = 64,
    min_component_area_ratio: float = 0.0005,
) -> tuple[np.ndarray, MaskQualityMetrics]:
    """Keep the largest component and summarize quality-related geometry."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    total_pixels = height * width
    min_pixels = max(int(total_pixels * min_component_area_ratio), min_component_area_pixels)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(0, num_labels - 1)

    if component_count == 0:
        cleaned = np.zeros_like(binary)
        score, flags = score_mask_quality(0.0, 0, False, True)
        metrics = MaskQualityMetrics(
            area_ratio=0.0,
            bbox=(0, 0, 0, 0),
            centroid_xy=(0.5, 0.5),
            original_component_count=0,
            kept_area_pixels=0,
            touches_edge=False,
            score=score,
            flags=flags,
            is_empty=True,
        )
        return cleaned, metrics

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = 1 + int(np.argmax(component_areas))
    cleaned = (labels == largest_index).astype(np.uint8)

    kept_area = int(component_areas[largest_index - 1])
    bbox = _compute_bbox(cleaned)
    centroid = centroids[largest_index]
    centroid_xy = (float(centroid[0] / max(width - 1, 1)), float(centroid[1] / max(height - 1, 1)))
    area_ratio = kept_area / float(total_pixels)
    touches_edge = _touches_edge(bbox, width, height)
    score, flags = score_mask_quality(area_ratio, component_count, touches_edge, False)
    if kept_area < min_pixels:
        flags.append("below_min_area")
        score = max(0.0, score - 0.15)

    metrics = MaskQualityMetrics(
        area_ratio=area_ratio,
        bbox=bbox,
        centroid_xy=centroid_xy,
        original_component_count=component_count,
        kept_area_pixels=kept_area,
        touches_edge=touches_edge,
        score=score,
        flags=sorted(set(flags)),
        is_empty=False,
    )
    return cleaned, metrics


def derive_geometry_labels(
    metrics: MaskQualityMetrics,
    small_threshold: float = 0.03,
    large_threshold: float = 0.15,
) -> GeometryLabels:
    x_norm, y_norm = metrics.centroid_xy
    x_bucket = 0 if x_norm < (1.0 / 3.0) else 2 if x_norm > (2.0 / 3.0) else 1
    y_bucket = 0 if y_norm < (1.0 / 3.0) else 2 if y_norm > (2.0 / 3.0) else 1

    location_key = (
        ("top_left", "top_center", "top_right"),
        ("middle_left", "middle_center", "middle_right"),
        ("bottom_left", "bottom_center", "bottom_right"),
    )[y_bucket][x_bucket]

    if metrics.area_ratio < small_threshold:
        size_key = "small"
    elif metrics.area_ratio < large_threshold:
        size_key = "medium"
    else:
        size_key = "large"

    return GeometryLabels(
        location_key=location_key,
        location_label_zh=LOCATION_KEY_TO_ZH[location_key],
        size_key=size_key,
        size_label_zh=SIZE_KEY_TO_ZH[size_key],
    )


def create_highlight_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> Image.Image:
    base = np.array(image).astype(np.float32)
    overlay = base.copy()
    overlay[mask > 0] = (1.0 - alpha) * overlay[mask > 0] + alpha * np.array(color, dtype=np.float32)

    mask_uint8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.dilate(mask_uint8, kernel, iterations=1) - cv2.erode(mask_uint8, kernel, iterations=1)
    overlay[edge > 0] = np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def crop_with_padding(image: Image.Image, bbox: Tuple[int, int, int, int], padding_ratio: float) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    crop_box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x + 1),
        min(height, y1 + pad_y + 1),
    )
    return image.crop(crop_box)


def save_mask_png(mask: np.ndarray, output_path: str) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(output_path)

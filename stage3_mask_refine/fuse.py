from __future__ import annotations

import math

import cv2
import numpy as np

from .common import mask_to_soft
from .visual_refine import adaptive_radii, dilate_mask


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[np.ndarray] = []
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        components.append((labels == label).astype(np.uint8))
    return components


def _mask_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.max() <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def _component_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _confidence_band(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def is_edge_preserving_target(init_mask: np.ndarray, category: str) -> bool:
    if init_mask.max() <= 0:
        return False
    if category == "arthropod":
        return True
    area = float(init_mask.sum())
    if area <= 0:
        return False
    perimeter = _mask_perimeter(init_mask)
    complexity = (perimeter * perimeter) / max(area * 4.0 * math.pi, 1.0)
    x0, y0, x1, y1 = _bbox_from_mask(init_mask)
    bbox_area = float(max(1, (x1 - x0 + 1) * (y1 - y0 + 1)))
    fill_ratio = area / bbox_area
    skeleton_ratio = float(_skeletonize(init_mask).sum()) / max(area, 1.0)
    return bool(complexity >= 7.5 or fill_ratio <= 0.22 or skeleton_ratio >= 0.22)


def _band_thresholds(band: str) -> tuple[float, tuple[float, float, float]]:
    if band == "high":
        return 0.42, (0.25, 0.50, 0.25)
    if band == "medium":
        return 0.50, (0.40, 0.45, 0.15)
    return 0.60, (0.65, 0.30, 0.05)


def _proximity_score(component: np.ndarray, anchor_components: list[np.ndarray]) -> float:
    if not anchor_components:
        return 0.0
    cx, cy = _component_center(component)
    best = 0.0
    for anchor in anchor_components:
        ax, ay = _component_center(anchor)
        x0, y0, x1, y1 = _bbox_from_mask(anchor)
        diag = max(1.0, math.hypot(x1 - x0 + 1, y1 - y0 + 1))
        distance = math.hypot(cx - ax, cy - ay)
        best = max(best, float(max(0.0, 1.0 - distance / diag)))
    return best


def _postprocess(mask: np.ndarray, min_area: int, edge_preserving: bool = False) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == label] = 1
    if kept.max() <= 0:
        return kept
    if edge_preserving:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN, kernel)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    return kept.astype(np.uint8)


def fuse_masks(
    init_mask: np.ndarray,
    text_prior: np.ndarray,
    vis_mask: np.ndarray,
    vis_soft: np.ndarray,
    final_confidence: float,
    category: str,
    low_confidence: bool,
    low_confidence_reasons: list[str],
    edge_preserving: bool = False,
) -> tuple[np.ndarray, dict]:
    init_mask = init_mask.astype(np.uint8)
    if init_mask.max() <= 0:
        return init_mask, {
            "refine_mode": "skip_empty",
            "r_base": 0,
            "r_expand": 0,
            "kept_components": 0,
            "candidate_components": 0,
            "band": "low",
            "allow_text_expand": False,
        }

    band = _confidence_band(float(final_confidence))
    score_threshold, (w0, wv, wt) = _band_thresholds(band)
    low_quality = "low_mask_quality" in low_confidence_reasons
    conservative = low_quality or (low_confidence and category == "unknown")
    anchor_components = _component_masks(init_mask)

    r_base, r_expand, _ = adaptive_radii(init_mask)
    base_region = dilate_mask(init_mask, r_base)
    expand_region = dilate_mask(init_mask, r_expand)
    allow_text_expand = not conservative

    if conservative:
        wt = min(wt, 0.05)
    if edge_preserving:
        w0 = min(0.75, w0 + 0.15)
        wv = max(0.20, wv - 0.05)
        wt = max(0.03, wt - 0.10)

    candidate = (vis_mask > 0).astype(np.uint8)
    if conservative:
        candidate &= base_region

    kept_components: list[np.ndarray] = []
    candidate_components = _component_masks(candidate)
    for component in candidate_components:
        area = float(component.sum())
        if area <= 0:
            continue
        overlap = float((component & base_region).sum()) / area
        text_mean = float(text_prior[component > 0].mean()) if np.any(component > 0) else 0.0
        proximity = _proximity_score(component, anchor_components)
        score = 0.50 * overlap + 0.35 * text_mean + 0.15 * proximity
        if score >= score_threshold:
            kept_components.append(component)

    if kept_components:
        keep_mask = np.clip(np.sum(kept_components, axis=0), 0, 1).astype(np.uint8)
        refine_mode = "refined"
    else:
        keep_mask = init_mask.copy()
        refine_mode = "fallback_init"

    allowed_region = dilate_mask(((keep_mask > 0) | (init_mask > 0)).astype(np.uint8), max(1, r_base // 2))
    init_soft = mask_to_soft(init_mask, sigma=2.0)
    p_text = np.clip(text_prior, 0.0, 1.0) * expand_region.astype(np.float32)
    p_vis = np.clip(vis_soft, 0.0, 1.0) * expand_region.astype(np.float32)
    edge_soft = mask_to_soft(_skeletonize(init_mask), sigma=1.0) if edge_preserving else np.zeros_like(init_soft)

    final_soft = w0 * init_soft + wv * p_vis + wt * p_text
    if edge_preserving:
        final_soft = np.maximum(final_soft, 0.62 * edge_soft)
    final_soft *= expand_region.astype(np.float32)
    final_soft *= allowed_region.astype(np.float32)

    threshold = 0.42 if edge_preserving else 0.5
    refined = (final_soft > threshold).astype(np.uint8)
    min_area = max(8 if edge_preserving else 16, int(init_mask.size * (0.00012 if edge_preserving else 0.00025)))
    refined = _postprocess(refined, min_area=min_area, edge_preserving=edge_preserving)

    init_area = int(init_mask.sum())
    refined_area = int(refined.sum())
    min_area_ratio = 0.25 if len(anchor_components) > 1 else 0.40
    if refined_area <= 0:
        refined = init_mask.copy()
        refine_mode = "fallback_empty"
    elif refined_area < int(min_area_ratio * init_area):
        refined = init_mask.copy()
        refine_mode = "fallback_too_small"
    elif refined_area > int(1.35 * init_area):
        added_region = (refined > 0) & (init_mask == 0)
        strong_add = (
            allow_text_expand
            and band == "high"
            and np.any(added_region)
            and float(p_text[added_region].mean()) > 0.65
            and float(p_vis[added_region].mean()) > 0.65
        )
        if not strong_add:
            refined = _postprocess((refined & base_region).astype(np.uint8), min_area=min_area)
            if refined.sum() <= 0:
                refined = init_mask.copy()
            refine_mode = "fallback_too_large"

    change_ratio = float(np.not_equal(refined, init_mask).sum()) / float(init_mask.size)
    info = {
        "refine_mode": refine_mode,
        "r_base": int(r_base),
        "r_expand": int(r_expand),
        "kept_components": int(len(kept_components)),
        "candidate_components": int(len(candidate_components)),
        "band": band,
        "allow_text_expand": bool(allow_text_expand),
        "change_ratio": change_ratio,
        "init_area_pixels": init_area,
        "refined_area_pixels": int(refined.sum()),
        "text_prior_mean": float(p_text[expand_region > 0].mean()) if np.any(expand_region > 0) else 0.0,
        "vis_mean": float(p_vis[expand_region > 0].mean()) if np.any(expand_region > 0) else 0.0,
        "refine_submode": "edge_preserving" if edge_preserving else "normal",
    }
    return refined.astype(np.uint8), info

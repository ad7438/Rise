"""Confidence aggregation helpers for Stage 2 pseudo text."""

from __future__ import annotations

from typing import Dict, Tuple


def clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_clip_score(value: float | None) -> float | None:
    if value is None:
        return None
    return clamp01((float(value) + 1.0) / 2.0)


def combine_confidence(
    mask_quality: float,
    category_confidence: float,
    clip_text_crop_score: float | None,
    clip_text_full_score: float | None,
) -> Tuple[float, Dict[str, float | None]]:
    components = {
        "mask_quality": clamp01(mask_quality),
        "vlm_category_confidence": clamp01(category_confidence),
        "clip_text_crop_score": normalize_clip_score(clip_text_crop_score),
        "clip_text_full_score": normalize_clip_score(clip_text_full_score),
    }
    weights = {
        "mask_quality": 0.35,
        "vlm_category_confidence": 0.30,
        "clip_text_crop_score": 0.20,
        "clip_text_full_score": 0.15,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        value = components[key]
        if value is None:
            continue
        weighted_sum += weight * value
        total_weight += weight

    final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
    return float(final_confidence), components


def evaluate_low_confidence(
    final_confidence: float,
    mask_quality: float,
    category_confidence: float,
    category_key: str | None = None,
    mask_flags: list[str] | None = None,
    final_threshold: float = 0.55,
    mask_threshold: float = 0.45,
    category_threshold: float = 0.50,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    flags = set(mask_flags or [])
    if final_confidence < final_threshold:
        reasons.append("low_final_confidence")
    if mask_quality < mask_threshold:
        reasons.append("low_mask_quality")
    if category_confidence < category_threshold:
        reasons.append("low_category_confidence")
    if category_key in {"unknown", "other_animal", "other_non_animal"} and category_confidence < 0.70:
        reasons.append("fallback_category")
    if "fragmented_mask" in flags and final_confidence < 0.70:
        reasons.append("fragmented_mask_semantics")
    if "touching_edge" in flags and category_confidence < 0.80 and final_confidence < 0.75:
        reasons.append("touching_edge_semantics")
    return bool(reasons), reasons

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .common import mask_to_soft
from .visual_refine import dilate_mask


def _clip01(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if float(value.max()) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _normalize(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(value.min())
    max_value = float(value.max())
    if max_value - min_value <= eps:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - min_value) / (max_value - min_value + eps), 0.0, 1.0).astype(np.float32)


def _resize_2d(value: np.ndarray, shape: tuple[int, int], interpolation: int, name: str) -> np.ndarray:
    value = np.squeeze(np.asarray(value))
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got shape {value.shape}")
    if value.shape == shape:
        return value.astype(np.float32)
    return cv2.resize(value.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[np.ndarray] = []
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        components.append((labels == label).astype(np.uint8))
    return components


def _bbox_diagonal(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 1.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return max(1.0, math.hypot(width, height))


def _postprocess(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == label] = 1
    if kept.max() <= 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    return kept.astype(np.uint8)


def _confidence_band(value: float, high_conf: float, mid_conf: float) -> str:
    if value >= high_conf:
        return "high"
    if value >= mid_conf:
        return "medium"
    return "low"


def _weights_for_band(
    band: str,
    high_weights: tuple[float, float, float],
    mid_weights: tuple[float, float, float],
    low_weights: tuple[float, float, float],
    eps: float,
) -> tuple[float, float, float]:
    if band == "high":
        weights = high_weights
    elif band == "medium":
        weights = mid_weights
    else:
        weights = low_weights
    weights_np = np.maximum(np.asarray(weights, dtype=np.float32), 0.0)
    total = float(weights_np.sum())
    if total <= eps:
        return 0.50, 0.35, 0.15
    weights_np = weights_np / total
    return float(weights_np[0]), float(weights_np[1]), float(weights_np[2])


def _mr_m0_iou(refined: np.ndarray, m0_bin: np.ndarray, eps: float) -> float:
    refined_bool = refined > 0
    m0_bool = m0_bin > 0
    union = np.logical_or(refined_bool, m0_bool).sum()
    if union <= 0:
        return 1.0
    intersection = np.logical_and(refined_bool, m0_bool).sum()
    return float(intersection) / float(union + eps)


def _empty_debug(
    shape: tuple[int, int],
    score_mode: str,
    fusion_mode: str,
    use_anchor_score: bool = True,
    use_semantic_score: bool = True,
    use_spatial_score: bool = True,
) -> dict[str, Any]:
    zeros = np.zeros(shape, dtype=np.float32)
    zeros_u8 = np.zeros(shape, dtype=np.uint8)
    return {
        "refine_module": "svac",
        "refine_mode": "skip_empty",
        "refine_submode": "svac",
        "r_base": 0,
        "r_expand": 0,
        "kept_components": 0,
        "candidate_components": 0,
        "band": "low",
        "allow_text_expand": False,
        "change_ratio": 0.0,
        "init_area_pixels": 0,
        "refined_area_pixels": 0,
        "text_prior_mean": 0.0,
        "vis_mean": 0.0,
        "svac_base_region_B": zeros_u8,
        "svac_expand_region_E": zeros_u8,
        "svac_candidate_from_pv": zeros_u8,
        "svac_retained_components_K": zeros_u8,
        "svac_local_region_Omega": zeros_u8,
        "svac_fused_prior_Pr": zeros,
        "svac_num_candidate_components": 0,
        "svac_num_retained_components": 0,
        "svac_mean_eta": 0.0,
        "svac_max_eta": 0.0,
        "svac_mean_anchor_consistency": 0.0,
        "svac_mean_semantic_support": 0.0,
        "svac_mean_spatial_consistency": 0.0,
        "svac_area_ratio": 0.0,
        "svac_pr_mean": 0.0,
        "svac_pr_max": 0.0,
        "svac_mr_area": 0,
        "svac_omega_area": 0,
        "svac_w0": 0.0,
        "svac_wv": 0.0,
        "svac_ws": 0.0,
        "score_mode": score_mode,
        "fusion_mode": fusion_mode,
        "svac_use_anchor_score": bool(use_anchor_score),
        "svac_use_semantic_score": bool(use_semantic_score),
        "svac_use_spatial_score": bool(use_spatial_score),
        "svac_score_terms": ",".join(
            term
            for term, enabled in (
                ("anchor", use_anchor_score),
                ("semantic", use_semantic_score),
                ("spatial", use_spatial_score),
            )
            if enabled
        ),
        "eta_mean": 0.0,
        "eta_max": 0.0,
        "eta_min": 0.0,
        "candidate_component_count": 0,
        "retained_component_count": 0,
        "Omega_area": 0,
        "Mr_area": 0,
        "M0_area": 0,
        "Mr_M0_iou": 0.0,
        "Mr_M0_change_ratio": 0.0,
    }


def build_svac_refined_mask(
    m0: np.ndarray,
    ps_agsp: np.ndarray,
    pv: np.ndarray,
    semantic_confidence: float,
    mf0: np.ndarray | None = None,
    base_radius: int = 10,
    expand_radius: int = 35,
    local_radius: int = 7,
    visual_threshold: float = 0.5,
    component_threshold: float = 0.45,
    binarize_threshold: float = 0.5,
    alpha_o: float = 0.50,
    alpha_s: float = 0.35,
    alpha_d: float = 0.15,
    high_conf: float = 0.75,
    mid_conf: float = 0.55,
    high_weights: tuple[float, float, float] = (0.25, 0.30, 0.45),
    mid_weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    low_weights: tuple[float, float, float] = (0.50, 0.35, 0.15),
    score_mode: str = "geometric_mean",
    fusion_mode: str = "confidence_modulated",
    use_anchor_score: bool = True,
    use_semantic_score: bool = True,
    use_spatial_score: bool = True,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build refined pseudo mask using semantic-visual anchor constraints.

    SVAC is intentionally limited to producing M_r. It does not decide
    whether a sample should use M_r, keep M0, or be dropped.
    """

    m0_np = np.squeeze(np.asarray(m0))
    if m0_np.ndim != 2:
        raise ValueError(f"m0 must be 2D after squeeze, got shape {m0_np.shape}")
    if score_mode not in {"weighted_sum", "geometric_mean"}:
        raise ValueError(f"score_mode must be weighted_sum or geometric_mean, got {score_mode}")
    if fusion_mode not in {"tiered", "confidence_modulated"}:
        raise ValueError(f"fusion_mode must be tiered or confidence_modulated, got {fusion_mode}")
    if not (use_anchor_score or use_semantic_score or use_spatial_score):
        raise ValueError("At least one SVAC score term must be enabled.")
    shape = m0_np.shape
    m0_bin = (_clip01(m0_np) > 0.5).astype(np.uint8)
    ps_np = _clip01(_resize_2d(ps_agsp, shape, cv2.INTER_LINEAR, "ps_agsp"))
    pv_np = _clip01(_resize_2d(pv, shape, cv2.INTER_LINEAR, "pv"))
    if mf0 is None:
        mf0_np = mask_to_soft(m0_bin, sigma=2.0)
    else:
        mf0_np = _clip01(_resize_2d(mf0, shape, cv2.INTER_LINEAR, "mf0"))

    if m0_bin.max() <= 0:
        debug = _empty_debug(
            shape,
            score_mode=score_mode,
            fusion_mode=fusion_mode,
            use_anchor_score=use_anchor_score,
            use_semantic_score=use_semantic_score,
            use_spatial_score=use_spatial_score,
        )
        return m0_bin, debug["svac_fused_prior_Pr"], debug["svac_local_region_Omega"], debug

    base_radius = max(0, int(base_radius))
    expand_radius = max(base_radius, int(expand_radius))
    local_radius = max(1, int(local_radius))
    base_region = dilate_mask(m0_bin, base_radius).astype(np.uint8)
    expand_region = dilate_mask(m0_bin, expand_radius).astype(np.uint8)

    candidate = ((pv_np >= float(visual_threshold)) & (expand_region > 0)).astype(np.uint8)
    components = _component_masks(candidate)

    alpha_total = max(eps, float(alpha_o + alpha_s + alpha_d))
    alpha_o = float(alpha_o) / alpha_total
    alpha_s = float(alpha_s) / alpha_total
    alpha_d = float(alpha_d) / alpha_total

    distance_source = (m0_bin <= 0).astype(np.uint8)
    distance_map = cv2.distanceTransform(distance_source, cv2.DIST_L2, 3)
    spatial_scale = max(1.0, min(float(expand_radius), 0.25 * _bbox_diagonal(m0_bin)))

    retained_components: list[np.ndarray] = []
    eta_values: list[float] = []
    anchor_values: list[float] = []
    semantic_values: list[float] = []
    spatial_values: list[float] = []
    for component in components:
        area = float(component.sum())
        if area <= 0:
            continue
        region = component > 0
        anchor_consistency = float((component & base_region).sum()) / (area + eps)
        semantic_support = float(ps_np[region].mean()) if np.any(region) else 0.0
        min_distance = float(distance_map[region].min()) if np.any(region) else spatial_scale
        spatial_consistency = float(math.exp(-min_distance / (spatial_scale + eps)))
        active_terms: list[tuple[str, float, float]] = []
        if use_anchor_score:
            active_terms.append(("anchor", alpha_o, anchor_consistency))
        if use_semantic_score:
            active_terms.append(("semantic", alpha_s, semantic_support))
        if use_spatial_score:
            active_terms.append(("spatial", alpha_d, spatial_consistency))
        if score_mode == "geometric_mean":
            product = 1.0
            for _, _, value in active_terms:
                product *= max(float(value), 0.0)
            eta = (product + eps) ** (1.0 / float(len(active_terms)))
        else:
            active_weight_sum = max(eps, sum(weight for _, weight, _ in active_terms))
            eta = sum((weight / active_weight_sum) * value for _, weight, value in active_terms)

        eta_values.append(float(eta))
        anchor_values.append(anchor_consistency)
        semantic_values.append(semantic_support)
        spatial_values.append(spatial_consistency)
        if eta >= float(component_threshold):
            retained_components.append(component)

    if retained_components:
        retained_mask = np.clip(np.sum(retained_components, axis=0), 0, 1).astype(np.uint8)
        refine_mode = "refined"
        svac_refine_mode_detail = "svac_refined"
        local_seed = ((retained_mask > 0) | (m0_bin > 0)).astype(np.uint8)
    else:
        retained_mask = np.zeros_like(m0_bin, dtype=np.uint8)
        refine_mode = "fallback_init"
        svac_refine_mode_detail = "svac_no_retained_fuse_m0_region"
        local_seed = m0_bin.copy()

    local_region = (dilate_mask(local_seed, local_radius) & expand_region).astype(np.uint8)
    if local_region.max() <= 0:
        local_region = (dilate_mask(m0_bin, local_radius) & expand_region).astype(np.uint8)
    if local_region.max() <= 0:
        local_region = m0_bin.copy()

    band = _confidence_band(float(semantic_confidence), high_conf=float(high_conf), mid_conf=float(mid_conf))
    if fusion_mode == "confidence_modulated":
        semantic_weight = float(np.clip(semantic_confidence, 0.0, 1.0))
        w0, wv, ws = 1.0, 1.0, semantic_weight
        fused_prior = mf0_np + pv_np + semantic_weight * ps_np
    else:
        w0, wv, ws = _weights_for_band(
            band,
            high_weights=high_weights,
            mid_weights=mid_weights,
            low_weights=low_weights,
            eps=eps,
        )
        fused_prior = w0 * mf0_np + wv * pv_np + ws * ps_np
    fused_prior = fused_prior * local_region.astype(np.float32)
    fused_prior = _normalize(fused_prior, eps=eps)

    refined_raw = (fused_prior >= float(binarize_threshold)).astype(np.uint8)
    min_area = max(8, int(m0_bin.size * 0.00025))
    refined = _postprocess(refined_raw, min_area=min_area)
    refined = (refined & local_region).astype(np.uint8)

    init_area = int(m0_bin.sum())
    refined_area = int(refined.sum())
    change_ratio = float(np.not_equal(refined, m0_bin).sum()) / float(m0_bin.size)
    mr_m0_iou = _mr_m0_iou(refined, m0_bin, eps=eps)
    omega_region = local_region > 0
    refined_region = refined > 0
    debug_info: dict[str, Any] = {
        "refine_module": "svac",
        "refine_mode": refine_mode,
        "refine_submode": "svac",
        "svac_refine_mode_detail": svac_refine_mode_detail,
        "r_base": int(base_radius),
        "r_expand": int(expand_radius),
        "kept_components": int(len(retained_components)),
        "candidate_components": int(len(components)),
        "band": band,
        "allow_text_expand": True,
        "change_ratio": change_ratio,
        "init_area_pixels": init_area,
        "refined_area_pixels": refined_area,
        "text_prior_mean": float(ps_np[omega_region].mean()) if np.any(omega_region) else 0.0,
        "vis_mean": float(pv_np[omega_region].mean()) if np.any(omega_region) else 0.0,
        "svac_base_region_B": base_region.astype(np.uint8),
        "svac_expand_region_E": expand_region.astype(np.uint8),
        "svac_candidate_from_pv": candidate.astype(np.uint8),
        "svac_retained_components_K": retained_mask.astype(np.uint8),
        "svac_local_region_Omega": local_region.astype(np.uint8),
        "svac_fused_prior_Pr": fused_prior.astype(np.float32),
        "svac_num_candidate_components": int(len(components)),
        "svac_num_retained_components": int(len(retained_components)),
        "svac_mean_eta": float(np.mean(eta_values)) if eta_values else 0.0,
        "svac_max_eta": float(np.max(eta_values)) if eta_values else 0.0,
        "svac_min_eta": float(np.min(eta_values)) if eta_values else 0.0,
        "svac_mean_anchor_consistency": float(np.mean(anchor_values)) if anchor_values else 0.0,
        "svac_mean_semantic_support": float(np.mean(semantic_values)) if semantic_values else 0.0,
        "svac_mean_spatial_consistency": float(np.mean(spatial_values)) if spatial_values else 0.0,
        "svac_area_ratio": float(refined_area) / float(max(init_area, 1)),
        "svac_pr_mean": float(fused_prior[omega_region].mean()) if np.any(omega_region) else 0.0,
        "svac_pr_max": float(fused_prior.max()) if fused_prior.size else 0.0,
        "svac_mr_area": refined_area,
        "svac_omega_area": int(local_region.sum()),
        "svac_w0": float(w0),
        "svac_wv": float(wv),
        "svac_ws": float(ws),
        "score_mode": score_mode,
        "fusion_mode": fusion_mode,
        "svac_use_anchor_score": bool(use_anchor_score),
        "svac_use_semantic_score": bool(use_semantic_score),
        "svac_use_spatial_score": bool(use_spatial_score),
        "svac_score_terms": ",".join(
            term
            for term, enabled in (
                ("anchor", use_anchor_score),
                ("semantic", use_semantic_score),
                ("spatial", use_spatial_score),
            )
            if enabled
        ),
        "eta_mean": float(np.mean(eta_values)) if eta_values else 0.0,
        "eta_max": float(np.max(eta_values)) if eta_values else 0.0,
        "eta_min": float(np.min(eta_values)) if eta_values else 0.0,
        "candidate_component_count": int(len(components)),
        "retained_component_count": int(len(retained_components)),
        "Omega_area": int(local_region.sum()),
        "Mr_area": refined_area,
        "M0_area": init_area,
        "Mr_M0_iou": mr_m0_iou,
        "Mr_M0_change_ratio": change_ratio,
        "s_sem_svac": float(ps_np[refined_region].mean()) if np.any(refined_region) else 0.0,
        "s_vis_svac": float(pv_np[refined_region].mean()) if np.any(refined_region) else 0.0,
    }
    return refined.astype(np.uint8), fused_prior.astype(np.float32), local_region.astype(np.uint8), debug_info


class SemanticVisualAnchorConstrainedRefinement:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(
        self,
        m0: np.ndarray,
        ps_agsp: np.ndarray,
        pv: np.ndarray,
        semantic_confidence: float,
        mf0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        return build_svac_refined_mask(
            m0=m0,
            ps_agsp=ps_agsp,
            pv=pv,
            semantic_confidence=semantic_confidence,
            mf0=mf0,
            **self.kwargs,
        )

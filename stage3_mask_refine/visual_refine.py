from __future__ import annotations

import math

import cv2
import numpy as np

from .common import mask_to_soft


def _elliptic_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), _elliptic_kernel(radius), iterations=1)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    return cv2.erode(mask.astype(np.uint8), _elliptic_kernel(radius), iterations=1)


def adaptive_radii(mask: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, (0, 0, 0, 0)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    diagonal = math.sqrt(width * width + height * height)
    r_base = int(round(min(15, max(3, 0.05 * diagonal))))
    r_expand = int(round(min(35, max(6, 0.12 * diagonal))))
    return r_base, r_expand, (x0, y0, x1, y1)


def run_grabcut_refine(image_rgb: np.ndarray, mask: np.ndarray, r_base: int, r_expand: int, iterations: int = 3) -> np.ndarray:
    if mask.max() <= 0:
        return mask.astype(np.uint8)

    expand_region = dilate_mask(mask, r_expand)
    ys, xs = np.where(expand_region > 0)
    if xs.size == 0:
        return mask.astype(np.uint8)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = image_rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)

    sure_fg = erode_mask(mask_crop, max(1, r_base // 2))
    likely_fg = mask_crop
    local_expand = dilate_mask(mask_crop, r_expand)

    gc_mask = np.full(mask_crop.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[local_expand == 0] = cv2.GC_BGD
    gc_mask[likely_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.astype(np.uint8)

    result_crop = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)
    result = np.zeros_like(mask, dtype=np.uint8)
    result[y0 : y1 + 1, x0 : x1 + 1] = result_crop
    result &= expand_region.astype(np.uint8)
    return result.astype(np.uint8)


def run_grabcut_refine_edge_preserving(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    r_base: int,
    r_expand: int,
    iterations: int = 2,
) -> np.ndarray:
    if mask.max() <= 0:
        return mask.astype(np.uint8)

    local_r_base = max(1, int(round(r_base * 0.5)))
    local_r_expand = max(2, int(round(r_expand * 0.7)))
    expand_region = dilate_mask(mask, local_r_expand)
    ys, xs = np.where(expand_region > 0)
    if xs.size == 0:
        return mask.astype(np.uint8)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = image_rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)

    sure_fg = erode_mask(mask_crop, max(1, local_r_base // 3))
    local_expand = dilate_mask(mask_crop, local_r_expand)

    gc_mask = np.full(mask_crop.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[local_expand == 0] = cv2.GC_BGD
    gc_mask[mask_crop > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.astype(np.uint8)

    result_crop = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)
    result = np.zeros_like(mask, dtype=np.uint8)
    result[y0 : y1 + 1, x0 : x1 + 1] = result_crop
    result &= expand_region.astype(np.uint8)
    return result.astype(np.uint8)


def visual_soft_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    r_base: int,
    r_expand: int,
    edge_preserving: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if edge_preserving:
        refined = run_grabcut_refine_edge_preserving(image_rgb, mask, r_base=r_base, r_expand=r_expand)
        return refined, mask_to_soft(refined, sigma=1.0)
    refined = run_grabcut_refine(image_rgb, mask, r_base=r_base, r_expand=r_expand)
    return refined, mask_to_soft(refined, sigma=2.0)

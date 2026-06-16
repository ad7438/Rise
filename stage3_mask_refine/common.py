from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from stage2_pseudo_text.mask_utils import (
    create_highlight_overlay,
    load_binary_mask,
    load_rgb_image,
    resize_binary_mask,
    save_mask_png,
    score_mask_quality,
)
from stage2_pseudo_text.schema import MaskQualityMetrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
WORKSPACE_ROOT = DATASET_ROOT / "RISE_Workspace"
AGSP_DEBUG_PANEL_SIZE = (160, 120)
AGSP_DEBUG_LABEL_HEIGHT = 22


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_stage2_records(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def ensure_dirs(
    output_root: Path,
    save_visuals: bool,
    save_agsp_vis: bool = False,
    save_svpm_vis: bool = False,
    save_svac_vis: bool = False,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": output_root,
        "mask_dir": output_root / "pseudo_mask_refined",
        "results_jsonl": output_root / "results.jsonl",
        "results_csv": output_root / "results.csv",
        "summary_json": output_root / "summary.json",
        "stage4_ready": output_root / "stage4_ready",
        "stage4_manifest": output_root / "stage4_ready" / "manifest.jsonl",
        "stage4_summary": output_root / "stage4_ready" / "manifest_summary.json",
    }
    paths["mask_dir"].mkdir(parents=True, exist_ok=True)
    paths["stage4_ready"].mkdir(parents=True, exist_ok=True)
    if save_visuals:
        visuals_root = output_root / "visuals"
        paths["visuals_root"] = visuals_root
        for name in ("overlay_init", "overlay_refined", "text_prior", "vis_mask"):
            path = visuals_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path
    if save_agsp_vis:
        agsp_root = output_root / "agsp_visuals"
        paths["agsp_visuals_root"] = agsp_root
        for name in (
            "semantic_prior_raw",
            "agsp_anchor",
            "agsp_mf0",
            "semantic_prior_agsp",
            "agsp_debug",
        ):
            path = agsp_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path
    if save_svpm_vis:
        svpm_root = output_root / "svpm_visuals"
        paths["svpm_visuals_root"] = svpm_root
        for name in (
            "svpm_superpixels",
            "svpm_local_region",
            "svpm_anchor_support",
            "svpm_semantic_support",
            "visual_prior_svpm",
            "svpm_debug",
        ):
            path = svpm_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path
    if save_svac_vis:
        svac_root = output_root / "svac_visuals"
        paths["svac_visuals_root"] = svac_root
        for name in (
            "svac_base_region",
            "svac_expand_region",
            "svac_candidate_components",
            "svac_retained_components",
            "svac_local_region",
            "svac_fused_prior",
            "svac_refined_mask",
            "svac_debug",
        ):
            path = svac_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path
    return paths


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def mask_to_soft(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if mask.max() <= 0:
        return mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    max_value = float(blurred.max())
    if max_value <= 1e-8:
        return np.zeros_like(mask, dtype=np.float32)
    return np.clip(blurred / max_value, 0.0, 1.0).astype(np.float32)


def save_visual_assets(paths: dict[str, Path], sample_id: str, image: Image.Image, init_mask: np.ndarray, refined_mask: np.ndarray, text_prior: np.ndarray, vis_mask: np.ndarray) -> dict[str, str]:
    overlay_init = create_highlight_overlay(image, init_mask)
    overlay_refined = create_highlight_overlay(image, refined_mask)
    overlay_init_path = paths["overlay_init"] / f"{sample_id}.png"
    overlay_refined_path = paths["overlay_refined"] / f"{sample_id}.png"
    text_prior_path = paths["text_prior"] / f"{sample_id}.png"
    vis_mask_path = paths["vis_mask"] / f"{sample_id}.png"
    overlay_init.save(overlay_init_path)
    overlay_refined.save(overlay_refined_path)
    Image.fromarray((np.clip(text_prior, 0.0, 1.0) * 255).astype(np.uint8)).save(text_prior_path)
    save_mask_png(vis_mask, str(vis_mask_path))
    return {
        "overlay_init_path": str(overlay_init_path.relative_to(PROJECT_ROOT)),
        "overlay_refined_path": str(overlay_refined_path.relative_to(PROJECT_ROOT)),
        "text_prior_path": str(text_prior_path.relative_to(PROJECT_ROOT)),
        "vis_mask_path": str(vis_mask_path.relative_to(PROJECT_ROOT)),
    }


def save_gray_array(path: Path, value: np.ndarray) -> None:
    Image.fromarray((np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)


def _soft_preview(value: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    return cv2.applyColorMap((np.clip(value, 0.0, 1.0) * 255).astype(np.uint8), colormap)[:, :, ::-1]


def _label_map_preview(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    red = (labels * 37 + 17) % 255
    green = (labels * 67 + 79) % 255
    blue = (labels * 97 + 131) % 255
    return np.stack([red, green, blue], axis=2).astype(np.uint8)


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    panel_image = Image.fromarray(panel.astype(np.uint8)).convert("RGB")
    panel_image.thumbnail(AGSP_DEBUG_PANEL_SIZE, Image.Resampling.BILINEAR)
    canvas = Image.new(
        "RGB",
        (AGSP_DEBUG_PANEL_SIZE[0], AGSP_DEBUG_PANEL_SIZE[1] + AGSP_DEBUG_LABEL_HEIGHT),
        "white",
    )
    x = (AGSP_DEBUG_PANEL_SIZE[0] - panel_image.width) // 2
    y = (AGSP_DEBUG_PANEL_SIZE[1] - panel_image.height) // 2
    canvas.paste(panel_image, (x, y))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.text((4, AGSP_DEBUG_PANEL_SIZE[1] + 4), label, fill=(0, 0, 0))
    return np.array(canvas)


def save_agsp_assets(
    paths: dict[str, Path],
    sample_id: str,
    image: Image.Image,
    init_mask: np.ndarray,
    ps_raw: np.ndarray,
    anchor_map: np.ndarray,
    mf0: np.ndarray,
    ps_agsp: np.ndarray,
    vis_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> dict[str, str]:
    ps_raw_path = paths["semantic_prior_raw"] / f"{sample_id}.png"
    anchor_path = paths["agsp_anchor"] / f"{sample_id}.png"
    mf0_path = paths["agsp_mf0"] / f"{sample_id}.png"
    ps_agsp_path = paths["semantic_prior_agsp"] / f"{sample_id}.png"
    debug_path = paths["agsp_debug"] / f"{sample_id}_agsp_debug.png"

    save_gray_array(ps_raw_path, ps_raw)
    save_gray_array(anchor_path, anchor_map)
    save_gray_array(mf0_path, mf0)
    save_gray_array(ps_agsp_path, ps_agsp)

    panels = [
        (np.array(image.convert("RGB")), "image"),
        (np.repeat(_mask_preview(init_mask)[:, :, None], 3, axis=2), "M0"),
        (cv2.applyColorMap((np.clip(ps_raw, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1], "P_s_raw"),
        (cv2.applyColorMap((np.clip(anchor_map, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1], "anchor A"),
        (cv2.applyColorMap((np.clip(mf0, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_BONE)[:, :, ::-1], "M_f0"),
        (cv2.applyColorMap((np.clip(ps_agsp, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1], "P_s_a"),
        (np.repeat(_mask_preview(vis_mask)[:, :, None], 3, axis=2), "P_v"),
        (np.repeat(_mask_preview(refined_mask)[:, :, None], 3, axis=2), "M_r"),
    ]
    panels = [_label_panel(panel, label) for panel, label in panels]
    Image.fromarray(np.concatenate(panels, axis=1)).save(debug_path)

    return {
        "ps_raw_path": str(ps_raw_path.relative_to(PROJECT_ROOT)),
        "agsp_anchor_path": str(anchor_path.relative_to(PROJECT_ROOT)),
        "agsp_mf0_path": str(mf0_path.relative_to(PROJECT_ROOT)),
        "ps_agsp_path": str(ps_agsp_path.relative_to(PROJECT_ROOT)),
        "agsp_debug_path": str(debug_path.relative_to(PROJECT_ROOT)),
    }


def save_svpm_assets(
    paths: dict[str, Path],
    sample_id: str,
    image: Image.Image,
    init_mask: np.ndarray,
    ps_raw: np.ndarray,
    ps_agsp: np.ndarray,
    superpixel_map: np.ndarray,
    local_region: np.ndarray,
    anchor_support: np.ndarray,
    semantic_support: np.ndarray,
    pv_svpm: np.ndarray,
    refined_mask: np.ndarray,
    selected_mask: np.ndarray | None = None,
    pv_grabcut: np.ndarray | None = None,
) -> dict[str, str]:
    superpixel_path = paths["svpm_superpixels"] / f"{sample_id}.png"
    local_region_path = paths["svpm_local_region"] / f"{sample_id}.png"
    anchor_support_path = paths["svpm_anchor_support"] / f"{sample_id}.png"
    semantic_support_path = paths["svpm_semantic_support"] / f"{sample_id}.png"
    visual_prior_path = paths["visual_prior_svpm"] / f"{sample_id}.png"
    debug_path = paths["svpm_debug"] / f"{sample_id}_svpm_debug.png"

    Image.fromarray(_label_map_preview(superpixel_map)).save(superpixel_path)
    np.save(superpixel_path.with_suffix(".npy"), np.asarray(superpixel_map, dtype=np.int32))
    save_gray_array(local_region_path, local_region)
    save_gray_array(anchor_support_path, anchor_support)
    save_gray_array(semantic_support_path, semantic_support)
    save_gray_array(visual_prior_path, pv_svpm)

    panels: list[tuple[np.ndarray, str]] = [
        (np.array(image.convert("RGB")), "image"),
        (np.repeat(_mask_preview(init_mask)[:, :, None], 3, axis=2), "M0"),
        (_soft_preview(ps_raw, cv2.COLORMAP_INFERNO), "P_s_raw"),
        (_soft_preview(ps_agsp, cv2.COLORMAP_INFERNO), "P_s_a"),
        (_label_map_preview(superpixel_map), "superpixels"),
        (_soft_preview(local_region, cv2.COLORMAP_BONE), "local E"),
        (_soft_preview(anchor_support, cv2.COLORMAP_VIRIDIS), "anchor r"),
        (_soft_preview(semantic_support, cv2.COLORMAP_INFERNO), "semantic s"),
        (_soft_preview(pv_svpm, cv2.COLORMAP_OCEAN), "P_v_svpm"),
    ]
    if pv_grabcut is not None:
        panels.append((_soft_preview(pv_grabcut, cv2.COLORMAP_OCEAN), "P_v_grabcut"))
    panels.extend(
        [
            (np.repeat(_mask_preview(refined_mask)[:, :, None], 3, axis=2), "M_r"),
            (
                np.repeat(_mask_preview(selected_mask if selected_mask is not None else refined_mask)[:, :, None], 3, axis=2),
                "M*",
            ),
        ]
    )
    labelled_panels = [_label_panel(panel, label) for panel, label in panels]
    Image.fromarray(np.concatenate(labelled_panels, axis=1)).save(debug_path)

    return {
        "svpm_superpixel_path": str(superpixel_path.relative_to(PROJECT_ROOT)),
        "svpm_superpixel_npy_path": str(superpixel_path.with_suffix(".npy").relative_to(PROJECT_ROOT)),
        "svpm_local_region_path": str(local_region_path.relative_to(PROJECT_ROOT)),
        "svpm_anchor_support_path": str(anchor_support_path.relative_to(PROJECT_ROOT)),
        "svpm_semantic_support_path": str(semantic_support_path.relative_to(PROJECT_ROOT)),
        "svpm_visual_prior_path": str(visual_prior_path.relative_to(PROJECT_ROOT)),
        "svpm_debug_path": str(debug_path.relative_to(PROJECT_ROOT)),
    }


def save_svac_assets(
    paths: dict[str, Path],
    sample_id: str,
    image: Image.Image,
    init_mask: np.ndarray,
    ps_raw: np.ndarray,
    ps_agsp: np.ndarray,
    pv: np.ndarray,
    base_region: np.ndarray,
    expand_region: np.ndarray,
    candidate_components: np.ndarray,
    retained_components: np.ndarray,
    local_region: np.ndarray,
    fused_prior: np.ndarray,
    refined_mask: np.ndarray,
    selected_mask: np.ndarray | None = None,
) -> dict[str, str]:
    base_path = paths["svac_base_region"] / f"{sample_id}.png"
    expand_path = paths["svac_expand_region"] / f"{sample_id}.png"
    candidate_path = paths["svac_candidate_components"] / f"{sample_id}.png"
    retained_path = paths["svac_retained_components"] / f"{sample_id}.png"
    local_path = paths["svac_local_region"] / f"{sample_id}.png"
    fused_path = paths["svac_fused_prior"] / f"{sample_id}.png"
    refined_path = paths["svac_refined_mask"] / f"{sample_id}.png"
    debug_path = paths["svac_debug"] / f"{sample_id}_svac_debug.png"

    save_mask_png(base_region, str(base_path))
    save_mask_png(expand_region, str(expand_path))
    save_mask_png(candidate_components, str(candidate_path))
    save_mask_png(retained_components, str(retained_path))
    save_mask_png(local_region, str(local_path))
    save_gray_array(fused_path, fused_prior)
    save_mask_png(refined_mask, str(refined_path))

    panels: list[tuple[np.ndarray, str]] = [
        (np.array(image.convert("RGB")), "image"),
        (np.repeat(_mask_preview(init_mask)[:, :, None], 3, axis=2), "M0"),
        (_soft_preview(ps_raw, cv2.COLORMAP_INFERNO), "P_s_raw"),
        (_soft_preview(ps_agsp, cv2.COLORMAP_INFERNO), "P_s_a"),
        (_soft_preview(pv, cv2.COLORMAP_OCEAN), "P_v"),
        (np.repeat(_mask_preview(base_region)[:, :, None], 3, axis=2), "B_i"),
        (np.repeat(_mask_preview(expand_region)[:, :, None], 3, axis=2), "E_i"),
        (np.repeat(_mask_preview(candidate_components)[:, :, None], 3, axis=2), "candidate"),
        (np.repeat(_mask_preview(retained_components)[:, :, None], 3, axis=2), "K_i"),
        (np.repeat(_mask_preview(local_region)[:, :, None], 3, axis=2), "Omega"),
        (_soft_preview(fused_prior, cv2.COLORMAP_VIRIDIS), "P_r"),
        (np.repeat(_mask_preview(refined_mask)[:, :, None], 3, axis=2), "M_r"),
        (
            np.repeat(_mask_preview(selected_mask if selected_mask is not None else refined_mask)[:, :, None], 3, axis=2),
            "M*",
        ),
    ]
    labelled_panels = [_label_panel(panel, label) for panel, label in panels]
    Image.fromarray(np.concatenate(labelled_panels, axis=1)).save(debug_path)

    return {
        "svac_base_region_path": str(base_path.relative_to(PROJECT_ROOT)),
        "svac_expand_region_path": str(expand_path.relative_to(PROJECT_ROOT)),
        "svac_candidate_components_path": str(candidate_path.relative_to(PROJECT_ROOT)),
        "svac_retained_components_path": str(retained_path.relative_to(PROJECT_ROOT)),
        "svac_local_region_path": str(local_path.relative_to(PROJECT_ROOT)),
        "svac_fused_prior_path": str(fused_path.relative_to(PROJECT_ROOT)),
        "svac_refined_mask_path": str(refined_path.relative_to(PROJECT_ROOT)),
        "svac_debug_path": str(debug_path.relative_to(PROJECT_ROOT)),
    }


def load_image_and_clean_mask(record: dict, min_component_area_pixels: int, min_component_area_ratio: float):
    image_path = resolve_path(record["image_path"])
    mask_path = resolve_path(record["mask_path"])
    image = load_rgb_image(str(image_path))
    raw_mask = load_binary_mask(str(mask_path))
    raw_mask = resize_binary_mask(raw_mask, image.size)
    clean_mask, metrics = preprocess_mask_preserve_components(
        raw_mask,
        min_component_area_pixels=min_component_area_pixels,
        min_component_area_ratio=min_component_area_ratio,
    )
    return image, clean_mask.astype(np.uint8), metrics


def _compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _touches_edge(bbox: tuple[int, int, int, int], width: int, height: int) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= 0 or y0 <= 0 or x1 >= width - 1 or y1 >= height - 1


def preprocess_mask_preserve_components(
    mask: np.ndarray,
    min_component_area_pixels: int = 64,
    min_component_area_ratio: float = 0.0005,
) -> tuple[np.ndarray, MaskQualityMetrics]:
    """Remove tiny noise while preserving all meaningful connected components."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    total_pixels = max(1, height * width)
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

    kept_labels = [label for label in range(1, num_labels) if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels]
    if not kept_labels:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept_labels = [largest_label]

    cleaned = np.isin(labels, kept_labels).astype(np.uint8)
    kept_area = int(cleaned.sum())
    bbox = _compute_bbox(cleaned)
    ys, xs = np.where(cleaned > 0)
    centroid_xy = (
        float(xs.mean() / max(width - 1, 1)) if xs.size > 0 else 0.5,
        float(ys.mean() / max(height - 1, 1)) if ys.size > 0 else 0.5,
    )
    area_ratio = kept_area / float(total_pixels)
    touches_edge = _touches_edge(bbox, width, height)
    score, flags = score_mask_quality(area_ratio, component_count, touches_edge, False)

    dropped_small = component_count - len(kept_labels)
    if dropped_small > 0:
        flags.append("dropped_small_components")
    if len(kept_labels) > 1:
        flags.append("preserved_multi_components")

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


def summarize(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total_samples": 0}
    changed = [float(record["change_ratio"]) for record in records]
    refine_counts: dict[str, int] = {}
    dropped = 0
    for record in records:
        mode = str(record["refine_mode"])
        refine_counts[mode] = refine_counts.get(mode, 0) + 1
        if bool(record.get("dropped_from_stage4")):
            dropped += 1
    return {
        "total_samples": total,
        "average_change_ratio": float(np.mean(changed)),
        "median_change_ratio": float(np.median(changed)),
        "max_change_ratio": float(np.max(changed)),
        "samples_above_1pct_change": int(sum(value > 0.01 for value in changed)),
        "samples_above_2pct_change": int(sum(value > 0.02 for value in changed)),
        "refine_mode_counts": refine_counts,
        "dropped_from_stage4": dropped,
    }

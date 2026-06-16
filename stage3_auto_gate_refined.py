#!/usr/bin/env python3
"""Automatic rule-based gate for Stage3 refined supervision.

This script converts the current manual workflow into a reproducible
three-way decision:

- use_refined
- use_old
- drop

It can also compare the automatic decisions against the current manual
selected subset and bad-refined blacklist, then optionally build a training
dataset directory with Image/GT symlinks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


HIGH_RISK_CATEGORIES = {"plant", "human", "unknown", "other_non_animal"}
DROP_MODES = {"skip_empty", "fallback_empty", "fallback_too_small"}
DEBUG_PANEL_SIZE = (160, 120)
DEBUG_LABEL_HEIGHT = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an automatic Stage3 gating decision set.")
    parser.add_argument(
        "--results_jsonl",
        default="Dataset/Stage3MaskRefine_v3_edge/results.jsonl",
        help="Stage3 results JSONL.",
    )
    parser.add_argument(
        "--dev_manifest_jsonl",
        default="Dataset/DevMini_stage3_v3_edge/dev_manifest.jsonl",
        help="Dev manifest to exclude from the train pool.",
    )
    parser.add_argument(
        "--selected_manifest_jsonl",
        default="Dataset/TrainSelected_stage3_v3_edge/train_manifest.jsonl",
        help="Optional selected refined manifest for agreement analysis.",
    )
    parser.add_argument(
        "--bad_blacklist_csv",
        default="Dataset/Review/train_pool_cod10k_blacklist/trainpool_cod10k_bad_refined_blacklist_selected.csv",
        help="Optional bad refined blacklist CSV for agreement analysis.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output directory for decisions/summary and optional built dataset.",
    )
    parser.add_argument(
        "--build_dataset",
        action="store_true",
        help="When set, build Image/GT symlink dataset under output_root/TrainDatasetAuto.",
    )
    parser.add_argument(
        "--save_gate_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save AGSP debug montages with selected supervision M* after auto gate.",
    )
    return parser.parse_args()


def load_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["sample_id"])
    return ids


def load_csv_keep_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = (row.get("sample_id") or "").strip()
            keep = (row.get("keep") or "").strip()
            if not sample_id:
                continue
            if keep and keep not in {"1", "true", "True", "yes", "YES"}:
                continue
            ids.add(sample_id)
    return ids


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _placeholder_panel(label: str) -> Image.Image:
    panel = Image.new("RGB", DEBUG_PANEL_SIZE, (245, 245, 245))
    return _with_label(panel, label)


def _with_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (DEBUG_PANEL_SIZE[0], DEBUG_PANEL_SIZE[1] + DEBUG_LABEL_HEIGHT), "white")
    image = image.convert("RGB")
    image.thumbnail(DEBUG_PANEL_SIZE, Image.Resampling.BILINEAR)
    x = (DEBUG_PANEL_SIZE[0] - image.width) // 2
    y = (DEBUG_PANEL_SIZE[1] - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, DEBUG_PANEL_SIZE[1] + 4), label, fill=(0, 0, 0))
    return canvas


def _load_panel(repo_root: Path, path_value: Any, label: str, grayscale: bool = False) -> Image.Image:
    if not path_value:
        return _placeholder_panel(label)
    path = (repo_root / str(path_value)).resolve()
    if not path.exists():
        return _placeholder_panel(label)
    with Image.open(path) as image:
        if grayscale:
            image = image.convert("L").convert("RGB")
        else:
            image = image.convert("RGB")
        return _with_label(image.copy(), label)


def save_gate_debug_montage(repo_root: Path, output_root: Path, row: dict[str, Any], mstar_path: Path) -> str:
    debug_root = output_root / "agsp_gate_debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    panels = [
        _load_panel(repo_root, row.get("image_path"), "image"),
        _load_panel(repo_root, row.get("init_mask_path"), "M0", grayscale=True),
        _load_panel(repo_root, row.get("ps_raw_path"), "P_s_raw", grayscale=True),
        _load_panel(repo_root, row.get("agsp_anchor_path"), "anchor A", grayscale=True),
        _load_panel(repo_root, row.get("agsp_mf0_path"), "M_f0", grayscale=True),
        _load_panel(repo_root, row.get("ps_agsp_path"), "P_s_a", grayscale=True),
        _load_panel(repo_root, row.get("svpm_visual_prior_path") or row.get("vis_mask_path"), "P_v", grayscale=True),
        _load_panel(repo_root, row.get("refined_mask_path"), "M_r", grayscale=True),
        _with_label(Image.open(mstar_path).convert("L").convert("RGB"), f"M*={row['decision']}"),
    ]
    canvas = Image.new("RGB", (len(panels) * DEBUG_PANEL_SIZE[0], DEBUG_PANEL_SIZE[1] + DEBUG_LABEL_HEIGHT), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * DEBUG_PANEL_SIZE[0], 0))
    debug_path = debug_root / f"{row['sample_id']}_agsp_gate_debug.png"
    canvas.save(debug_path)
    return str(debug_path.relative_to(repo_root))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def decide_row(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = row["sample_id"]
    is_camo = sample_id.startswith("camourflage_")
    final_conf = as_float(row, "final_confidence")
    text_prior_mean = as_float(row, "text_prior_mean")
    vis_mean = as_float(row, "vis_mean")
    change_ratio = as_float(row, "change_ratio")
    init_area = max(as_float(row, "init_area_pixels", 1.0), 1.0)
    refined_area = as_float(row, "refined_area_pixels")
    area_growth = refined_area / init_area
    candidate_components = int(as_float(row, "candidate_components"))
    refine_mode = str(row.get("refine_mode", ""))
    refine_submode = str(row.get("refine_submode", ""))
    category = str(row.get("category", ""))
    mask_is_empty = bool(row.get("mask_is_empty", False))

    if mask_is_empty or refine_mode in DROP_MODES:
        return "drop", "hard_failure_mode"
    if final_conf < 0.60:
        return "drop", "very_low_confidence"
    if text_prior_mean < 0.12 and vis_mean < 0.06:
        return "drop", "very_low_text_and_visual_support"

    if is_camo:
        if refine_mode == "refined" and final_conf >= 0.76 and text_prior_mean >= 0.40 and change_ratio < 0.12:
            return "refined", "camo_refined_gate"
        if (
            refine_mode == "fallback_too_large"
            and refine_submode == "normal"
            and final_conf >= 0.82
            and text_prior_mean >= 0.52
            and vis_mean >= 0.38
            and area_growth < 1.35
            and candidate_components <= 6
        ):
            return "refined", "camo_fallback_normal_gate"
        return "old", "camo_fallback_old"

    # COD10K: only drop obviously dangerous refined proposals.
    if refine_mode == "fallback_too_large" and refine_submode == "edge_preserving" and final_conf < 0.80 and text_prior_mean < 0.55:
        return "drop", "cod_fallback_edge_preserving_high_risk"
    if change_ratio >= 0.15:
        return "drop", "cod_extreme_change_ratio"
    if area_growth >= 2.20:
        return "drop", "cod_extreme_area_growth"
    if candidate_components >= 20:
        return "drop", "cod_extreme_component_count"
    if category in HIGH_RISK_CATEGORIES and final_conf < 0.72 and text_prior_mean < 0.35:
        return "drop", "cod_high_risk_category_low_support"

    if (
        refine_mode == "refined"
        and refine_submode == "normal"
        and final_conf >= 0.79
        and text_prior_mean >= 0.42
        and change_ratio < 0.08
        and area_growth < 1.60
        and candidate_components < 16
    ):
        return "refined", "cod_refined_normal_gate"
    if (
        refine_mode == "refined"
        and refine_submode == "edge_preserving"
        and final_conf >= 0.82
        and text_prior_mean >= 0.50
        and change_ratio < 0.04
        and area_growth < 1.35
        and candidate_components < 8
        and category not in HIGH_RISK_CATEGORIES
    ):
        return "refined", "cod_refined_edge_gate"
    if (
        refine_mode == "fallback_too_large"
        and refine_submode == "normal"
        and final_conf >= 0.82
        and text_prior_mean >= 0.55
        and vis_mean >= 0.38
        and change_ratio < 0.04
        and area_growth < 1.30
        and candidate_components < 6
    ):
        return "refined", "cod_fallback_normal_gate"
    return "old", "cod_fallback_old"


def summarize_numeric(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [as_float(row, field) for row in rows]
        if not values:
            continue
        summary[field] = {
            "mean": round(sum(values) / len(values), 6),
            "q25": round(percentile(values, 0.25), 6),
            "median": round(percentile(values, 0.5), 6),
            "q75": round(percentile(values, 0.75), 6),
        }
    return summary


def summarize_categorical(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, list[list[Any]]]:
    summary: dict[str, list[list[Any]]] = {}
    for field in fields:
        counter = Counter(str(row.get(field, "")) for row in rows)
        summary[field] = [[key, value] for key, value in counter.most_common(12)]
    return summary


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    results_path = (repo_root / args.results_jsonl).resolve()
    dev_manifest_path = (repo_root / args.dev_manifest_jsonl).resolve()
    selected_manifest_path = (repo_root / args.selected_manifest_jsonl).resolve()
    bad_blacklist_path = (repo_root / args.bad_blacklist_csv).resolve()
    output_root = (repo_root / args.output_root).resolve()
    ensure_clean_dir(output_root)

    dev_ids = load_jsonl_ids(dev_manifest_path)
    selected_ids = load_jsonl_ids(selected_manifest_path)
    bad_ids = load_csv_keep_ids(bad_blacklist_path)

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["sample_id"] in dev_ids:
                continue
            rows.append(row)

    train_pool_ids = {row["sample_id"] for row in rows}
    selected_ids &= train_pool_ids
    bad_ids &= train_pool_ids

    decisions: list[dict[str, Any]] = []
    decision_counts = Counter()
    source_decision_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts = Counter()

    for row in rows:
        decision, reason = decide_row(row)
        source = "CAMO" if row["sample_id"].startswith("camourflage_") else "COD10K"
        decision_counts[decision] += 1
        source_decision_counts[source][decision] += 1
        reason_counts[reason] += 1

        decision_row = {
            "sample_id": row["sample_id"],
            "source": source,
            "decision": decision,
            "reason": reason,
            "image_path": row["image_path"],
            "init_mask_path": row["init_mask_path"],
            "refined_mask_path": row["refined_mask_path"],
            "refine_mode": row.get("refine_mode", ""),
            "refine_submode": row.get("refine_submode", ""),
            "category": row.get("category", ""),
            "final_confidence": round(as_float(row, "final_confidence"), 6),
            "text_prior_mean": round(as_float(row, "text_prior_mean"), 6),
            "vis_mean": round(as_float(row, "vis_mean"), 6),
            "change_ratio": round(as_float(row, "change_ratio"), 6),
            "mask_quality": round(as_float(row, "mask_quality"), 6),
            "candidate_components": int(as_float(row, "candidate_components")),
            "kept_components": int(as_float(row, "kept_components")),
            "area_growth": round(as_float(row, "refined_area_pixels") / max(as_float(row, "init_area_pixels", 1.0), 1.0), 6),
            "semantic_prior_mode": row.get("semantic_prior_mode", ""),
            "ps_raw_mean": round(as_float(row, "ps_raw_mean"), 6),
            "ps_agsp_mean": round(as_float(row, "ps_agsp_mean"), 6),
            "anchor_mean": round(as_float(row, "anchor_mean"), 6),
            "s_sem_raw": round(as_float(row, "s_sem_raw"), 6),
            "s_sem_agsp": round(as_float(row, "s_sem_agsp"), 6),
            "visual_prior_mode": row.get("visual_prior_mode", ""),
            "svpm_n_segments": int(as_float(row, "svpm_n_segments")),
            "svpm_compactness": round(as_float(row, "svpm_compactness"), 6),
            "svpm_dilate_radius": int(as_float(row, "svpm_dilate_radius")),
            "svpm_alpha": round(as_float(row, "svpm_alpha"), 6),
            "svpm_beta": round(as_float(row, "svpm_beta"), 6),
            "svpm_pv_mean": round(as_float(row, "svpm_pv_mean"), 6),
            "svpm_pv_max": round(as_float(row, "svpm_pv_max"), 6),
            "svpm_pv_min": round(as_float(row, "svpm_pv_min"), 6),
            "s_vis_svpm": round(as_float(row, "s_vis_svpm"), 6),
            "s_vis_grabcut_if_available": round(as_float(row, "s_vis_grabcut_if_available"), 6),
            "ps_raw_path": row.get("ps_raw_path", ""),
            "agsp_anchor_path": row.get("agsp_anchor_path", ""),
            "agsp_mf0_path": row.get("agsp_mf0_path", ""),
            "ps_agsp_path": row.get("ps_agsp_path", ""),
            "vis_mask_path": row.get("vis_mask_path", ""),
            "agsp_debug_path": row.get("agsp_debug_path", ""),
            "svpm_superpixel_path": row.get("svpm_superpixel_path", ""),
            "svpm_local_region_path": row.get("svpm_local_region_path", ""),
            "svpm_anchor_support_path": row.get("svpm_anchor_support_path", ""),
            "svpm_semantic_support_path": row.get("svpm_semantic_support_path", ""),
            "svpm_visual_prior_path": row.get("svpm_visual_prior_path", ""),
            "svpm_debug_path": row.get("svpm_debug_path", ""),
            "manual_selected_refined": row["sample_id"] in selected_ids,
            "manual_bad_refined": row["sample_id"] in bad_ids,
        }
        decisions.append(decision_row)

    decisions_jsonl = output_root / "auto_gate_decisions.jsonl"
    with decisions_jsonl.open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisions_csv = output_root / "auto_gate_decisions.csv"
    with decisions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0].keys()))
        writer.writeheader()
        writer.writerows(decisions)

    decision_by_id = {row["sample_id"]: row for row in decisions}
    selected_rows = [decision_by_id[sid] for sid in selected_ids if sid in decision_by_id]
    bad_rows = [decision_by_id[sid] for sid in bad_ids if sid in decision_by_id]
    other_rows = [
        row
        for row in decisions
        if row["sample_id"] not in selected_ids and row["sample_id"] not in bad_ids
    ]

    numeric_fields = [
        "final_confidence",
        "mask_quality",
        "change_ratio",
        "text_prior_mean",
        "vis_mean",
        "svpm_pv_mean",
        "s_vis_svpm",
        "area_growth",
        "candidate_components",
        "kept_components",
    ]
    categorical_fields = ["source", "category", "refine_mode", "refine_submode", "visual_prior_mode", "decision", "reason"]

    manual_agreement = {
        "selected_total": len(selected_ids),
        "selected_auto_refined": sum(row["decision"] == "refined" for row in selected_rows),
        "selected_auto_old": sum(row["decision"] == "old" for row in selected_rows),
        "selected_auto_drop": sum(row["decision"] == "drop" for row in selected_rows),
        "bad_total": len(bad_ids),
        "bad_auto_refined": sum(row["decision"] == "refined" for row in bad_rows),
        "bad_auto_old": sum(row["decision"] == "old" for row in bad_rows),
        "bad_auto_drop": sum(row["decision"] == "drop" for row in bad_rows),
    }

    summary = {
        "train_pool_total": len(decisions),
        "decision_counts": dict(decision_counts),
        "source_decision_counts": {key: dict(value) for key, value in source_decision_counts.items()},
        "reason_counts_top20": [[key, value] for key, value in reason_counts.most_common(20)],
        "manual_agreement": manual_agreement,
        "selected_distribution": {
            "numeric": summarize_numeric(selected_rows, numeric_fields),
            "categorical": summarize_categorical(selected_rows, categorical_fields),
        },
        "bad_distribution": {
            "numeric": summarize_numeric(bad_rows, numeric_fields),
            "categorical": summarize_categorical(bad_rows, categorical_fields),
        },
        "other_distribution": {
            "numeric": summarize_numeric(other_rows, numeric_fields),
            "categorical": summarize_categorical(other_rows, categorical_fields),
        },
        "rule_notes": [
            "Use refined on high-confidence normal refined masks and on a small subset of safe fallback-too-large normal cases.",
            "Prefer old for borderline masks rather than dropping them.",
            "Drop only hard failures and high-risk COD10K refined cases with extreme change/area growth/component count.",
            "Apply a stricter gate to COD10K edge_preserving cases because bad refined samples are heavily concentrated there.",
        ],
    }

    with (output_root / "auto_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.build_dataset:
        dataset_root = output_root / "TrainDatasetAuto"
        ensure_clean_dir(dataset_root)
        image_root = dataset_root / "Image"
        gt_root = dataset_root / "GT"
        image_root.mkdir(parents=True, exist_ok=True)
        gt_root.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, Any]] = []
        for row in decisions:
            if row["decision"] == "drop":
                continue
            image_src = (repo_root / row["image_path"]).resolve()
            mask_rel = row["refined_mask_path"] if row["decision"] == "refined" else row["init_mask_path"]
            gt_src = (repo_root / mask_rel).resolve()
            image_dst = image_root / image_src.name
            gt_dst = gt_root / gt_src.name
            symlink_file(image_src, image_dst)
            symlink_file(gt_src, gt_dst)
            mstar_debug_path = ""
            if args.save_gate_debug:
                mstar_debug_path = save_gate_debug_montage(repo_root, output_root, row, gt_src)
            manifest_row = {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "image_path": str(image_dst),
                "gt_path": str(gt_dst),
                "supervision_type": row["decision"],
                "reason": row["reason"],
                "final_confidence": row["final_confidence"],
                "change_ratio": row["change_ratio"],
                "text_prior_mean": row["text_prior_mean"],
                "vis_mean": row["vis_mean"],
                "area_growth": row["area_growth"],
                "semantic_prior_mode": row.get("semantic_prior_mode", ""),
                "s_sem_raw": row.get("s_sem_raw", 0.0),
                "s_sem_agsp": row.get("s_sem_agsp", 0.0),
                "visual_prior_mode": row.get("visual_prior_mode", ""),
                "svpm_pv_mean": row.get("svpm_pv_mean", 0.0),
                "s_vis_svpm": row.get("s_vis_svpm", 0.0),
                "mstar_debug_path": mstar_debug_path,
            }
            manifest_rows.append(manifest_row)

        with (dataset_root / "train_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with (dataset_root / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        refined_root = output_root / "TrainDatasetAutoRefined"
        ensure_clean_dir(refined_root)
        refined_image_root = refined_root / "Image"
        refined_gt_root = refined_root / "GT"
        refined_image_root.mkdir(parents=True, exist_ok=True)
        refined_gt_root.mkdir(parents=True, exist_ok=True)

        refined_rows_for_write: list[dict[str, Any]] = []
        for row in decisions:
            if row["decision"] != "refined":
                continue
            image_src = (repo_root / row["image_path"]).resolve()
            gt_src = (repo_root / row["refined_mask_path"]).resolve()
            image_dst = refined_image_root / image_src.name
            gt_dst = refined_gt_root / gt_src.name
            symlink_file(image_src, image_dst)
            symlink_file(gt_src, gt_dst)
            refined_rows_for_write.append(
                {
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "image_path": str(image_dst),
                    "gt_path": str(gt_dst),
                    "supervision_type": "refined",
                    "reason": row["reason"],
                    "final_confidence": row["final_confidence"],
                    "change_ratio": row["change_ratio"],
                    "text_prior_mean": row["text_prior_mean"],
                    "vis_mean": row["vis_mean"],
                    "area_growth": row["area_growth"],
                    "semantic_prior_mode": row.get("semantic_prior_mode", ""),
                    "visual_prior_mode": row.get("visual_prior_mode", ""),
                    "s_sem_agsp": row.get("s_sem_agsp", 0.0),
                    "s_vis_svpm": row.get("s_vis_svpm", 0.0),
                }
            )

        with (refined_root / "train_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in refined_rows_for_write:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if refined_rows_for_write:
            with (refined_root / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(refined_rows_for_write[0].keys()))
                writer.writeheader()
                writer.writerows(refined_rows_for_write)

        with (refined_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "total": len(refined_rows_for_write),
                    "source": "auto_gate_refined_decisions",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        subset_summary = {
            "total": len(manifest_rows),
            "refined_total": sum(row["supervision_type"] == "refined" for row in manifest_rows),
            "old_total": sum(row["supervision_type"] == "old" for row in manifest_rows),
            "drop_total": decision_counts["drop"],
            "camo": sum(row["source"] == "CAMO" for row in manifest_rows),
            "cod10k": sum(row["source"] == "COD10K" for row in manifest_rows),
        }
        with (dataset_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(subset_summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary["manual_agreement"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["decision_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

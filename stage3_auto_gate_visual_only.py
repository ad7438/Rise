#!/usr/bin/env python3
"""Visual-only Stage-3 gate for G1 text contribution ablation.

This gate intentionally avoids text-derived semantic scores. Decisions are
based on M0 quality, SVPM/SVAC visual support, geometric change, and component
statistics. It builds the same TrainDatasetAuto layout as the main gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DROP_MODES = {"skip_empty", "fallback_empty", "fallback_too_small"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build G1 visual-only gate decisions and dataset.")
    parser.add_argument("--results_jsonl", required=True, help="Visual-only Stage-3 results JSONL.")
    parser.add_argument(
        "--dev_manifest_jsonl",
        default="Dataset/DevMini_stage3_v3_edge/dev_manifest.jsonl",
        help="Dev manifest to exclude from the training pool.",
    )
    parser.add_argument("--output_root", required=True, help="Output directory for decisions and datasets.")
    parser.add_argument("--build_dataset", action="store_true", help="Build Image/GT symlink dataset.")
    return parser.parse_args()


def load_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["sample_id"])
    return ids


def ensure_clean_dir(path: Path) -> None:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    if path == cwd or cwd not in path.parents:
        raise RuntimeError(f"refuse to reset path outside project: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), dst)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def source_name(sample_id: str) -> str:
    return "CAMO" if sample_id.startswith("camourflage_") else "COD10K"


def area_growth(row: dict[str, Any]) -> float:
    init_area = max(as_float(row, "init_area_pixels", 1.0), 1.0)
    return as_float(row, "refined_area_pixels") / init_area


def visual_score(row: dict[str, Any]) -> float:
    return (
        0.45 * as_float(row, "vis_mean")
        + 0.30 * as_float(row, "s_vis_svpm")
        + 0.15 * as_float(row, "svac_mean_anchor_consistency")
        + 0.10 * as_float(row, "svac_mean_spatial_consistency")
        - 0.35 * as_float(row, "change_ratio")
        - 0.06 * abs(area_growth(row) - 1.0)
        - 0.004 * as_float(row, "candidate_components")
    )


def decide_row(row: dict[str, Any]) -> tuple[str, str, float]:
    score = visual_score(row)
    refine_mode = str(row.get("refine_mode", ""))
    refine_submode = str(row.get("refine_submode", ""))
    mask_quality = as_float(row, "mask_quality")
    vis_mean = as_float(row, "vis_mean")
    s_vis = as_float(row, "s_vis_svpm")
    change_ratio = as_float(row, "change_ratio")
    growth = area_growth(row)
    components = int(as_float(row, "candidate_components"))
    retained = int(as_float(row, "kept_components"))
    is_camo = source_name(row["sample_id"]) == "CAMO"

    if bool(row.get("mask_is_empty", False)) or refine_mode in DROP_MODES:
        return "drop", "hard_failure_or_empty", score
    if mask_quality < 0.50 and vis_mean < 0.08:
        return "drop", "low_mask_quality_low_visual_support", score
    if change_ratio >= 0.18:
        return "drop", "extreme_change_ratio", score
    if growth >= 2.20:
        return "drop", "extreme_area_growth", score
    if components >= 24:
        return "drop", "extreme_component_count", score

    if is_camo:
        if (
            refine_mode == "refined"
            and retained > 0
            and score >= 0.62
            and vis_mean >= 0.22
            and s_vis >= 0.42
            and change_ratio < 0.12
            and growth < 1.55
            and components <= 12
        ):
            return "refined", "camo_visual_refined_gate", score
        if (
            refine_mode == "fallback_init"
            and score >= 0.62
            and vis_mean >= 0.25
            and s_vis >= 0.45
            and change_ratio < 0.08
            and growth < 1.35
            and components <= 8
        ):
            return "refined", "camo_visual_fallback_init_gate", score
        return "old", "camo_visual_fallback_old", score

    if (
        refine_mode == "refined"
        and refine_submode == "svac"
        and retained > 0
        and score >= 0.85
        and vis_mean >= 0.28
        and s_vis >= 0.50
        and change_ratio < 0.07
        and growth < 1.35
        and components <= 8
    ):
        return "refined", "cod_visual_refined_gate", score
    if (
        refine_mode == "fallback_init"
        and score >= 0.85
        and vis_mean >= 0.30
        and s_vis >= 0.55
        and change_ratio < 0.05
        and growth < 1.25
        and components <= 6
    ):
        return "refined", "cod_visual_fallback_init_gate", score
    return "old", "cod_visual_fallback_old", score


def decision_row(row: dict[str, Any], decision: str, reason: str, score: float) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "source": source_name(row["sample_id"]),
        "decision": decision,
        "reason": reason,
        "visual_gate_score": round(score, 6),
        "image_path": row["image_path"],
        "init_mask_path": row["init_mask_path"],
        "refined_mask_path": row["refined_mask_path"],
        "refine_mode": row.get("refine_mode", ""),
        "refine_submode": row.get("refine_submode", ""),
        "mask_quality": round(as_float(row, "mask_quality"), 6),
        "vis_mean": round(as_float(row, "vis_mean"), 6),
        "change_ratio": round(as_float(row, "change_ratio"), 6),
        "candidate_components": int(as_float(row, "candidate_components")),
        "kept_components": int(as_float(row, "kept_components")),
        "area_growth": round(area_growth(row), 6),
        "semantic_prior_mode": row.get("semantic_prior_mode", ""),
        "visual_prior_mode": row.get("visual_prior_mode", ""),
        "svpm_pv_mean": round(as_float(row, "svpm_pv_mean"), 6),
        "s_vis_svpm": round(as_float(row, "s_vis_svpm"), 6),
        "svac_mean_anchor_consistency": round(as_float(row, "svac_mean_anchor_consistency"), 6),
        "svac_mean_spatial_consistency": round(as_float(row, "svac_mean_spatial_consistency"), 6),
        "svac_num_candidate_components": int(as_float(row, "svac_num_candidate_components")),
        "svac_num_retained_components": int(as_float(row, "svac_num_retained_components")),
        "svac_score_terms": row.get("svac_score_terms", ""),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(repo_root: Path, output_root: Path, decisions: list[dict[str, Any]]) -> None:
    dataset_root = output_root / "TrainDatasetAuto"
    refined_root = output_root / "TrainDatasetAutoRefined"
    ensure_clean_dir(dataset_root)
    ensure_clean_dir(refined_root)
    image_root = dataset_root / "Image"
    gt_root = dataset_root / "GT"
    refined_image_root = refined_root / "Image"
    refined_gt_root = refined_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    refined_image_root.mkdir(parents=True, exist_ok=True)
    refined_gt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    refined_rows: list[dict[str, Any]] = []
    for row in decisions:
        if row["decision"] == "drop":
            continue
        image_src = (repo_root / row["image_path"]).resolve()
        gt_rel = row["refined_mask_path"] if row["decision"] == "refined" else row["init_mask_path"]
        gt_src = (repo_root / gt_rel).resolve()
        image_dst = image_root / f"{row['sample_id']}{image_src.suffix.lower() or '.jpg'}"
        gt_dst = gt_root / f"{row['sample_id']}.png"
        symlink_file(image_src, image_dst)
        symlink_file(gt_src, gt_dst)
        manifest_row = {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "image_path": str(image_dst),
            "gt_path": str(gt_dst),
            "supervision_type": row["decision"],
            "reason": row["reason"],
            "visual_gate_score": row["visual_gate_score"],
            "mask_quality": row["mask_quality"],
            "change_ratio": row["change_ratio"],
            "vis_mean": row["vis_mean"],
            "area_growth": row["area_growth"],
            "semantic_prior_mode": row["semantic_prior_mode"],
            "visual_prior_mode": row["visual_prior_mode"],
            "s_vis_svpm": row["s_vis_svpm"],
        }
        manifest_rows.append(manifest_row)
        if row["decision"] == "refined":
            refined_image_dst = refined_image_root / image_dst.name
            refined_gt_dst = refined_gt_root / gt_dst.name
            symlink_file(image_src, refined_image_dst)
            symlink_file((repo_root / row["refined_mask_path"]).resolve(), refined_gt_dst)
            refined_rows.append({**manifest_row, "image_path": str(refined_image_dst), "gt_path": str(refined_gt_dst)})

    write_rows_csv(dataset_root / "train_manifest.csv", manifest_rows)
    write_rows_jsonl(dataset_root / "train_manifest.jsonl", manifest_rows)
    write_rows_csv(refined_root / "train_manifest.csv", refined_rows)
    write_rows_jsonl(refined_root / "train_manifest.jsonl", refined_rows)
    (dataset_root / "subset_summary.json").write_text(
        json.dumps(
            {
                "total": len(manifest_rows),
                "refined_total": sum(row["supervision_type"] == "refined" for row in manifest_rows),
                "old_total": sum(row["supervision_type"] == "old" for row in manifest_rows),
                "drop_total": sum(row["decision"] == "drop" for row in decisions),
                "camo": sum(row["source"] == "CAMO" for row in manifest_rows),
                "cod10k": sum(row["source"] == "COD10K" for row in manifest_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (refined_root / "subset_summary.json").write_text(
        json.dumps({"total": len(refined_rows), "source": "g1_visual_only_gate"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    results_path = (repo_root / args.results_jsonl).resolve()
    output_root = (repo_root / args.output_root).resolve()
    ensure_clean_dir(output_root)
    dev_ids = load_jsonl_ids((repo_root / args.dev_manifest_jsonl).resolve())

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["sample_id"] not in dev_ids:
                rows.append(row)

    decisions = [decision_row(row, *decide_row(row)) for row in rows]
    decision_counts = Counter(row["decision"] for row in decisions)
    source_decision_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts = Counter(row["reason"] for row in decisions)
    for row in decisions:
        source_decision_counts[row["source"]][row["decision"]] += 1

    write_rows_csv(output_root / "auto_gate_decisions.csv", decisions)
    write_rows_jsonl(output_root / "auto_gate_decisions.jsonl", decisions)
    summary = {
        "train_pool_total": len(decisions),
        "decision_counts": dict(decision_counts),
        "source_decision_counts": {key: dict(value) for key, value in source_decision_counts.items()},
        "reason_counts_top20": [[key, value] for key, value in reason_counts.most_common(20)],
        "gate_type": "visual_only",
        "rule_notes": [
            "No text_prior_mean, s_sem_agsp, category, or text confidence is used for decisions.",
            "Refined masks are accepted only when visual support, anchor consistency, spatial consistency, and geometry are stable.",
        ],
    }
    (output_root / "auto_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.build_dataset:
        build_dataset(repo_root, output_root, decisions)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

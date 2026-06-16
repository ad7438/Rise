#!/usr/bin/env python3
"""Balanced Stage-3 gate for higher refined-mask coverage.

This experiment keeps the gate unsupervised: decisions are based only on
Stage-3 confidence, text/visual support, geometric change, and component
statistics. Ground-truth masks are not read by this script.
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
    parser = argparse.ArgumentParser(description="Build balanced Stage-3 gate decisions and dataset.")
    parser.add_argument(
        "--results_jsonl",
        default="Dataset/Stage3MaskRefine_AGSP_SVPM_SVAC_v1/results.jsonl",
        help="Stage-3 refinement results JSONL.",
    )
    parser.add_argument(
        "--dev_manifest_jsonl",
        default="Dataset/DevMini_stage3_v3_edge/dev_manifest.jsonl",
        help="Dev manifest to exclude from the training pool.",
    )
    parser.add_argument("--output_root", required=True, help="Output root for gate files and datasets.")
    parser.add_argument("--build_dataset", action="store_true", help="Build Image/GT symlink datasets.")
    parser.add_argument(
        "--target_refined_total",
        type=int,
        default=1000,
        help="Target refined coverage after accepting all eligible CAMO and top-scored COD10K samples.",
    )
    parser.add_argument(
        "--min_cod_refined",
        type=int,
        default=150,
        help="Minimum number of eligible COD10K refined samples to admit when available.",
    )
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


def balanced_score(row: dict[str, Any]) -> float:
    area = area_growth(row)
    low_conf = 1.0 if row.get("low_confidence") else 0.0
    empty = 1.0 if row.get("mask_is_empty") else 0.0
    return (
        as_float(row, "final_confidence")
        + 0.18 * as_float(row, "text_prior_mean")
        + 0.12 * as_float(row, "vis_mean")
        + 0.14 * as_float(row, "s_sem_agsp")
        + 0.10 * as_float(row, "s_vis_svpm")
        - 0.45 * as_float(row, "change_ratio")
        - 0.08 * abs(area - 1.0)
        - 0.006 * as_float(row, "candidate_components")
        - 0.20 * low_conf
        - 0.50 * empty
    )


def hard_drop_reason(row: dict[str, Any]) -> str:
    refine_mode = str(row.get("refine_mode", ""))
    if bool(row.get("mask_is_empty", False)) or refine_mode in DROP_MODES:
        return "hard_failure_or_empty"
    if as_float(row, "final_confidence") < 0.55:
        return "very_low_confidence"
    if as_float(row, "text_prior_mean") < 0.10 and as_float(row, "vis_mean") < 0.05:
        return "very_low_text_and_visual_support"
    if as_float(row, "change_ratio") >= 0.22:
        return "extreme_change_ratio"
    if area_growth(row) >= 2.60:
        return "extreme_area_growth"
    if int(as_float(row, "candidate_components")) >= 30:
        return "extreme_component_count"
    return ""


def eligible_refined(row: dict[str, Any]) -> tuple[bool, str]:
    if str(row.get("refine_mode", "")) != "refined":
        return False, "not_refined_mode"
    if as_float(row, "final_confidence") < 0.72:
        return False, "candidate_low_confidence"
    if as_float(row, "text_prior_mean") < 0.25:
        return False, "candidate_low_text_support"
    if as_float(row, "change_ratio") >= 0.18:
        return False, "candidate_large_change"
    if area_growth(row) >= 2.20:
        return False, "candidate_large_area_growth"
    if int(as_float(row, "candidate_components")) >= 25:
        return False, "candidate_many_components"

    if source_name(row["sample_id"]) == "COD10K":
        if as_float(row, "final_confidence") < 0.82:
            return False, "cod_low_confidence"
        if as_float(row, "text_prior_mean") < 0.55:
            return False, "cod_low_text_support"
        if as_float(row, "s_sem_agsp") < 0.82:
            return False, "cod_low_semantic_score"
        if as_float(row, "s_vis_svpm") < 0.50:
            return False, "cod_low_visual_score"
        if as_float(row, "change_ratio") >= 0.06:
            return False, "cod_large_change"
        if area_growth(row) >= 1.45:
            return False, "cod_large_area_growth"
        if int(as_float(row, "candidate_components")) >= 10:
            return False, "cod_many_components"

    return True, "eligible_refined"


def decision_common_fields(row: dict[str, Any], decision: str, reason: str, score: float) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "source": source_name(row["sample_id"]),
        "decision": decision,
        "reason": reason,
        "balanced_score": round(score, 6),
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
        "area_growth": round(area_growth(row), 6),
        "semantic_prior_mode": row.get("semantic_prior_mode", ""),
        "s_sem_agsp": round(as_float(row, "s_sem_agsp"), 6),
        "visual_prior_mode": row.get("visual_prior_mode", ""),
        "svpm_pv_mean": round(as_float(row, "svpm_pv_mean"), 6),
        "s_vis_svpm": round(as_float(row, "s_vis_svpm"), 6),
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
            "balanced_score": row["balanced_score"],
            "final_confidence": row["final_confidence"],
            "change_ratio": row["change_ratio"],
            "text_prior_mean": row["text_prior_mean"],
            "vis_mean": row["vis_mean"],
            "area_growth": row["area_growth"],
            "semantic_prior_mode": row["semantic_prior_mode"],
            "s_sem_agsp": row["s_sem_agsp"],
            "visual_prior_mode": row["visual_prior_mode"],
            "s_vis_svpm": row["s_vis_svpm"],
        }
        manifest_rows.append(manifest_row)

        if row["decision"] == "refined":
            refined_image_dst = refined_image_root / f"{row['sample_id']}{image_src.suffix.lower() or '.jpg'}"
            refined_gt_dst = refined_gt_root / f"{row['sample_id']}.png"
            symlink_file(image_src, refined_image_dst)
            symlink_file((repo_root / row["refined_mask_path"]).resolve(), refined_gt_dst)
            refined_row = dict(manifest_row)
            refined_row["image_path"] = str(refined_image_dst)
            refined_row["gt_path"] = str(refined_gt_dst)
            refined_rows.append(refined_row)

    write_rows_jsonl(dataset_root / "train_manifest.jsonl", manifest_rows)
    write_rows_csv(dataset_root / "train_manifest.csv", manifest_rows)
    write_rows_jsonl(refined_root / "train_manifest.jsonl", refined_rows)
    write_rows_csv(refined_root / "train_manifest.csv", refined_rows)

    subset_summary = {
        "total": len(manifest_rows),
        "refined_total": sum(row["supervision_type"] == "refined" for row in manifest_rows),
        "old_total": sum(row["supervision_type"] == "old" for row in manifest_rows),
        "drop_total": sum(row["decision"] == "drop" for row in decisions),
        "camo": sum(row["source"] == "CAMO" for row in manifest_rows),
        "cod10k": sum(row["source"] == "COD10K" for row in manifest_rows),
        "definition": "balanced gate v1: all eligible CAMO refined plus top-scored eligible COD10K refined until target coverage.",
    }
    with (dataset_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(subset_summary, handle, ensure_ascii=False, indent=2)
    with (refined_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"total": len(refined_rows), "source": "balanced_gate_v1"}, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    results_path = (repo_root / args.results_jsonl).resolve()
    dev_manifest_path = (repo_root / args.dev_manifest_jsonl).resolve()
    output_root = (repo_root / args.output_root).resolve()
    ensure_clean_dir(output_root)

    dev_ids = load_jsonl_ids(dev_manifest_path)
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["sample_id"] not in dev_ids:
                rows.append(row)

    candidate_rows: list[tuple[str, float, dict[str, Any]]] = []
    pre_decisions: dict[str, tuple[str, str]] = {}
    eligibility_reasons = Counter()

    for row in rows:
        drop_reason = hard_drop_reason(row)
        if drop_reason:
            pre_decisions[row["sample_id"]] = ("drop", drop_reason)
            continue
        eligible, reason = eligible_refined(row)
        eligibility_reasons[reason] += 1
        if eligible:
            candidate_rows.append((source_name(row["sample_id"]), balanced_score(row), row))
        else:
            pre_decisions[row["sample_id"]] = ("old", reason)

    camo_candidates = sorted(
        [item for item in candidate_rows if item[0] == "CAMO"],
        key=lambda item: item[1],
        reverse=True,
    )
    cod_candidates = sorted(
        [item for item in candidate_rows if item[0] == "COD10K"],
        key=lambda item: item[1],
        reverse=True,
    )

    cod_target = max(args.min_cod_refined, args.target_refined_total - len(camo_candidates))
    cod_target = max(0, min(cod_target, len(cod_candidates)))
    accepted_ids = {item[2]["sample_id"] for item in camo_candidates}
    accepted_ids.update(item[2]["sample_id"] for item in cod_candidates[:cod_target])

    decisions: list[dict[str, Any]] = []
    reason_counts = Counter()
    source_decision_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in accepted_ids:
            decision, reason = "refined", "balanced_refined_selected"
        else:
            decision, reason = pre_decisions.get(sample_id, ("old", "eligible_but_below_balanced_cut"))
        score = balanced_score(row)
        decision_row = decision_common_fields(row, decision, reason, score)
        decisions.append(decision_row)
        reason_counts[reason] += 1
        source_decision_counts[decision_row["source"]][decision] += 1

    write_rows_jsonl(output_root / "auto_gate_decisions.jsonl", decisions)
    write_rows_csv(output_root / "auto_gate_decisions.csv", decisions)

    summary = {
        "train_pool_total": len(decisions),
        "target_refined_total": args.target_refined_total,
        "min_cod_refined": args.min_cod_refined,
        "decision_counts": dict(Counter(row["decision"] for row in decisions)),
        "source_decision_counts": {key: dict(value) for key, value in source_decision_counts.items()},
        "reason_counts_top20": [[key, value] for key, value in reason_counts.most_common(20)],
        "eligibility_reason_counts_top20": [[key, value] for key, value in eligibility_reasons.most_common(20)],
        "eligible_candidates": {
            "CAMO": len(camo_candidates),
            "COD10K": len(cod_candidates),
        },
        "accepted_refined": {
            "CAMO": sum(row["decision"] == "refined" and row["source"] == "CAMO" for row in decisions),
            "COD10K": sum(row["decision"] == "refined" and row["source"] == "COD10K" for row in decisions),
        },
        "rule_notes": [
            "No ground-truth masks are used by this balanced gate.",
            "All eligible CAMO refined masks are accepted because CAMO refinement was empirically the stable source.",
            "COD10K admission is score-ranked and capped by target coverage because broad COD10K refinement is noisy.",
            "The default target is 1000 refined samples, which increases coverage without reducing train-pool pseudo-label quality in diagnostic analysis.",
            "Hard failures and extreme geometric changes are still dropped.",
        ],
    }
    with (output_root / "auto_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.build_dataset:
        build_dataset(repo_root, output_root, decisions)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

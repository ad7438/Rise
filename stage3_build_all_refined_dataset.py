#!/usr/bin/env python3
"""Build an all-refined supervision dataset for the E1 no-quality-gate ablation.

This ablation removes the quality gate Q by assigning every train-pool sample
to the refined mask M_r. It reuses an existing Stage-3 refinement results.jsonl
and excludes the dev split so the protocol stays aligned with the main run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-refined E1 ablation dataset.")
    parser.add_argument(
        "--results_jsonl",
        default=str(PROJECT_ROOT / "Dataset/Stage3MaskRefine_AGSP_SVPM_SVAC_v1/results.jsonl"),
        help="Stage-3 refinement results JSONL.",
    )
    parser.add_argument(
        "--dev_manifest_jsonl",
        default=str(PROJECT_ROOT / "Dataset/DevMini_stage3_v3_edge/dev_manifest.jsonl"),
        help="Dev manifest JSONL to exclude from the training pool.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output dataset root with Image/ and GT/ directories.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild output_root if it exists.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_reset_dir(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    project = PROJECT_ROOT.resolve()
    if path == project or project not in path.parents:
        raise RuntimeError(f"refuse to reset path outside project: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def load_dev_ids(path: Path) -> set[str]:
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


def read_results(path: Path, dev_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["sample_id"] in dev_ids:
                continue
            rows.append(row)
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def source_name(sample_id: str) -> str:
    return "CAMO" if sample_id.startswith("camourflage_") else "COD10K"


def output_image_name(sample_id: str, image_src: Path) -> str:
    suffix = image_src.suffix.lower() or ".jpg"
    return f"{sample_id}{suffix}"


def output_mask_name(sample_id: str) -> str:
    return f"{sample_id}.png"


def write_manifest_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_jsonl)
    dev_manifest_path = resolve_path(args.dev_manifest_jsonl)
    output_root = resolve_path(args.output_root)

    dev_ids = load_dev_ids(dev_manifest_path)
    result_rows = read_results(results_path, dev_ids)
    if not result_rows:
        raise RuntimeError(f"no training rows loaded from {results_path}")

    safe_reset_dir(output_root, overwrite=args.overwrite)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    refine_mode_counts: Counter[str] = Counter()
    refine_submode_counts: Counter[str] = Counter()
    low_confidence_total = 0
    empty_mask_total = 0
    missing: list[str] = []
    seen_names: set[str] = set()

    for row in result_rows:
        sample_id = row["sample_id"]
        image_src = resolve_path(row["image_path"])
        refined_src = resolve_path(row["refined_mask_path"])
        if not image_src.exists():
            missing.append(str(image_src))
            continue
        if not refined_src.exists():
            missing.append(str(refined_src))
            continue

        image_name = output_image_name(sample_id, image_src)
        mask_name = output_mask_name(sample_id)
        if image_name in seen_names or mask_name in seen_names:
            raise RuntimeError(f"duplicate output name for sample_id={sample_id}")
        seen_names.add(image_name)
        seen_names.add(mask_name)

        image_dst = image_root / image_name
        gt_dst = gt_root / mask_name
        link_file(image_src, image_dst)
        link_file(refined_src, gt_dst)

        source = source_name(sample_id)
        source_counts[source] += 1
        refine_mode_counts[str(row.get("refine_mode", ""))] += 1
        refine_submode_counts[str(row.get("refine_submode", ""))] += 1
        low_confidence_total += int(bool(row.get("low_confidence", False)))
        empty_mask_total += int(bool(row.get("mask_is_empty", False)))

        init_area = max(as_float(row, "init_area_pixels", 1.0), 1.0)
        refined_area = as_float(row, "refined_area_pixels")
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source": source,
                "image_path": str(image_dst),
                "gt_path": str(gt_dst),
                "supervision_type": "all_refined",
                "no_quality_gate": "true",
                "original_image_path": str(image_src),
                "init_mask_path": str(resolve_path(row["init_mask_path"])),
                "refined_mask_path": str(refined_src),
                "refine_mode": row.get("refine_mode", ""),
                "refine_submode": row.get("refine_submode", ""),
                "category": row.get("category", ""),
                "final_confidence": round(as_float(row, "final_confidence"), 6),
                "low_confidence": bool(row.get("low_confidence", False)),
                "mask_is_empty": bool(row.get("mask_is_empty", False)),
                "mask_quality": round(as_float(row, "mask_quality"), 6),
                "change_ratio": round(as_float(row, "change_ratio"), 6),
                "text_prior_mean": round(as_float(row, "text_prior_mean"), 6),
                "vis_mean": round(as_float(row, "vis_mean"), 6),
                "area_growth": round(refined_area / init_area, 6),
                "semantic_prior_mode": row.get("semantic_prior_mode", ""),
                "visual_prior_mode": row.get("visual_prior_mode", ""),
                "s_sem_agsp": round(as_float(row, "s_sem_agsp"), 6),
                "s_vis_svpm": round(as_float(row, "s_vis_svpm"), 6),
            }
        )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} source files, first entries:\n{preview}")

    fieldnames = list(manifest_rows[0].keys())
    write_manifest_csv(output_root / "train_manifest.csv", fieldnames, manifest_rows)
    write_manifest_jsonl(output_root / "train_manifest.jsonl", manifest_rows)

    summary = {
        "total": len(manifest_rows),
        "source_counts": dict(source_counts),
        "refine_mode_counts": dict(refine_mode_counts),
        "refine_submode_counts": dict(refine_submode_counts),
        "low_confidence_total": low_confidence_total,
        "empty_mask_total": empty_mask_total,
        "results_jsonl": str(results_path),
        "dev_manifest_jsonl": str(dev_manifest_path),
        "output_root": str(output_root),
        "definition": "E1 no quality gate: every train-pool sample uses refined mask M_r as supervision.",
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

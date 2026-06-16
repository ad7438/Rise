"""Build Stage 4 manifests from Stage 2 outputs and mask sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    compute_sample_weight,
    find_file_by_stem,
    load_jsonl_records,
    maybe_bool,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 4 CRIS-lite manifest.")
    parser.add_argument("--stage2_results_jsonl", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--text_field", default="training_text", choices=["training_text", "clip_text", "pseudo_text"])
    parser.add_argument("--min_weight", type=float, default=0.3)
    parser.add_argument("--low_conf_scale", type=float, default=0.6)
    parser.add_argument("--processing_error_cap", type=float, default=0.35)
    parser.add_argument("--drop_empty_masks", action="store_true")
    parser.add_argument("--drop_low_quality_masks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_manifest_records(
    stage2_records: list[dict],
    *,
    mask_dir: Path,
    gt_dir: Path | None,
    text_field: str,
    min_weight: float,
    low_conf_scale: float,
    processing_error_cap: float,
    drop_empty_masks: bool,
    drop_low_quality_masks: bool,
    limit: int | None,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    missing_masks = 0
    missing_images = 0
    missing_gts = 0
    dropped_empty_masks = 0
    dropped_low_quality_masks = 0

    for stage2_record in stage2_records:
        sample_id = stage2_record["sample_id"]
        image_path = Path(stage2_record["image_path"])
        mask_path = find_file_by_stem(mask_dir, sample_id)
        gt_path = find_file_by_stem(gt_dir, sample_id)

        if not image_path.exists():
            missing_images += 1
            continue
        if mask_path is None:
            missing_masks += 1
            continue
        if gt_dir is not None and gt_path is None:
            missing_gts += 1

        low_confidence_reasons = stage2_record.get("low_confidence_reasons") or []
        if drop_empty_masks and maybe_bool(stage2_record.get("mask_is_empty")):
            dropped_empty_masks += 1
            continue
        if drop_low_quality_masks and "low_mask_quality" in low_confidence_reasons:
            dropped_low_quality_masks += 1
            continue

        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "gt_path": str(gt_path) if gt_path is not None else None,
                "text": stage2_record.get(text_field) or "",
                "clip_text": stage2_record.get("clip_text") or "",
                "pseudo_text": stage2_record.get("pseudo_text") or "",
                "category": stage2_record.get("category"),
                "category_label_zh": stage2_record.get("category_label_zh"),
                "location": stage2_record.get("location"),
                "location_key": stage2_record.get("location_key"),
                "size": stage2_record.get("size"),
                "size_key": stage2_record.get("size_key"),
                "final_confidence": stage2_record.get("final_confidence"),
                "category_confidence": stage2_record.get("category_confidence"),
                "mask_quality": stage2_record.get("mask_quality"),
                "mask_area_ratio": stage2_record.get("mask_area_ratio"),
                "mask_is_empty": stage2_record.get("mask_is_empty"),
                "mask_flags": stage2_record.get("mask_flags") or [],
                "low_confidence": stage2_record.get("low_confidence"),
                "low_confidence_reasons": low_confidence_reasons,
                "processing_errors": stage2_record.get("processing_errors") or [],
                "sample_weight": compute_sample_weight(
                    stage2_record.get("final_confidence"),
                    low_confidence=maybe_bool(stage2_record.get("low_confidence")),
                    processing_errors=stage2_record.get("processing_errors") or [],
                    min_weight=min_weight,
                    low_conf_scale=low_conf_scale,
                    processing_error_cap=processing_error_cap,
                ),
            }
        )
        if limit is not None and len(records) >= limit:
            break

    summary = {
        "total_records": len(records),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "missing_gts": missing_gts,
        "dropped_empty_masks": dropped_empty_masks,
        "dropped_low_quality_masks": dropped_low_quality_masks,
        "mask_dir": str(mask_dir),
        "gt_dir": str(gt_dir) if gt_dir is not None else None,
        "text_field": text_field,
        "min_weight": min_weight,
        "low_conf_scale": low_conf_scale,
        "processing_error_cap": processing_error_cap,
        "drop_empty_masks": drop_empty_masks,
        "drop_low_quality_masks": drop_low_quality_masks,
    }
    return records, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_records = load_jsonl_records(Path(args.stage2_results_jsonl))
    records, summary = build_manifest_records(
        stage2_records,
        mask_dir=Path(args.mask_dir),
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        text_field=args.text_field,
        min_weight=args.min_weight,
        low_conf_scale=args.low_conf_scale,
        processing_error_cap=args.processing_error_cap,
        drop_empty_masks=args.drop_empty_masks,
        drop_low_quality_masks=args.drop_low_quality_masks,
        limit=args.limit,
    )

    write_jsonl(output_dir / "manifest.jsonl", records)
    write_csv(output_dir / "manifest.csv", records)
    write_json(output_dir / "manifest_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

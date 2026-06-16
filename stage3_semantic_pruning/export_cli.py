"""Export Stage 3 outputs into a Stage 4 ready manifest."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .common import load_stage2_records, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 3 results into Stage 4 ready assets.")
    parser.add_argument("--stage2_results_jsonl", default="Dataset/Stage2PseudoText_full_v3_coarse/results.jsonl")
    parser.add_argument("--refined_mask_dir", default="Dataset/Stage3Semantic_coarse/pseudo_mask_refined")
    parser.add_argument("--output_dir", default="Dataset/Stage3Semantic_coarse/stage4_ready")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.symlink(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    stage2_records = load_stage2_records(Path(args.stage2_results_jsonl))
    refined_mask_dir = Path(args.refined_mask_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "Image"
    mask_dir = output_dir / "Mask"
    text_dir = output_dir / "Text"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    missing_stage2 = 0
    missing_images = 0

    mask_paths = [path for path in sorted(refined_mask_dir.glob("*.png")) if path.is_file()]
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    for mask_path in mask_paths:
        sample_id = mask_path.stem
        stage2_record = stage2_records.get(sample_id)
        if stage2_record is None:
            missing_stage2 += 1
            continue

        source_image_path = Path(stage2_record["image_path"])
        if not source_image_path.exists():
            missing_images += 1
            continue

        linked_image_path = image_dir / f"{sample_id}{source_image_path.suffix.lower()}"
        linked_mask_path = mask_dir / f"{sample_id}.png"
        text_path = text_dir / f"{sample_id}.txt"

        _link_or_copy(source_image_path, linked_image_path)
        _link_or_copy(mask_path, linked_mask_path)
        text_path.write_text(stage2_record.get("pseudo_text", ""), encoding="utf-8")

        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(linked_image_path),
                "mask_path": str(linked_mask_path),
                "text_path": str(text_path),
                "source_image_path": str(source_image_path),
                "source_mask_path": str(mask_path),
                "pseudo_text": stage2_record.get("pseudo_text"),
                "clip_text": stage2_record.get("clip_text"),
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
                "low_confidence": stage2_record.get("low_confidence"),
                "low_confidence_reasons": stage2_record.get("low_confidence_reasons"),
                "processing_errors": stage2_record.get("processing_errors"),
            }
        )

    write_jsonl(output_dir / "train_manifest.jsonl", records)
    write_csv(output_dir / "train_manifest.csv", records)
    summary = {
        "total_exported": len(records),
        "missing_stage2_records": missing_stage2,
        "missing_source_images": missing_images,
        "refined_mask_dir": str(refined_mask_dir),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "manifest_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

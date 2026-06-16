#!/usr/bin/env python3
"""Build a refined-only subset from a train manifest.

Expected input manifest rows contain:
- sample_id
- image_path
- gt_path
- supervision_type

Rows with supervision_type == "refined" are exported into a new dataset root
with Image/GT symlinks plus train_manifest.{jsonl,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build refined-only subset from an auto train manifest.")
    parser.add_argument(
        "--input_manifest_jsonl",
        required=True,
        help="Input train manifest JSONL containing supervision_type.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output dataset root with Image/GT subdirectories.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest_jsonl).resolve()
    output_root = Path(args.output_root).resolve()

    ensure_clean_dir(output_root)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    source_counts = {"CAMO": 0, "COD10K": 0}

    with input_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("supervision_type") != "refined":
                continue

            sample_id = row["sample_id"]
            image_src = Path(row["image_path"]).resolve()
            gt_src = Path(row["gt_path"]).resolve()

            image_dst = image_root / image_src.name
            gt_dst = gt_root / gt_src.name
            symlink_file(image_src, image_dst)
            symlink_file(gt_src, gt_dst)

            source = row.get("source", "CAMO" if sample_id.startswith("camourflage_") else "COD10K")
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "image_path": str(image_dst),
                    "gt_path": str(gt_dst),
                    "supervision_type": "refined",
                    "reason": str(row.get("reason", "")),
                    "final_confidence": str(row.get("final_confidence", "")),
                    "change_ratio": str(row.get("change_ratio", "")),
                    "text_prior_mean": str(row.get("text_prior_mean", "")),
                    "vis_mean": str(row.get("vis_mean", "")),
                    "area_growth": str(row.get("area_growth", "")),
                }
            )

    jsonl_path = output_root / "train_manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_root / "train_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "total": len(manifest_rows),
        "camo": source_counts.get("CAMO", 0),
        "cod10k": source_counts.get("COD10K", 0),
        "supervision_type": "refined",
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

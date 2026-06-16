#!/usr/bin/env python3
"""Materialize a Stage3 training subset from a results JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage3 train subset directory.")
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="JSONL containing selected Stage3 samples.",
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
    repo_root = Path.cwd().resolve()
    input_jsonl = (repo_root / args.input_jsonl).resolve()
    output_root = (repo_root / args.output_root).resolve()

    ensure_clean_dir(output_root)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            image_src = (repo_root / row["image_path"]).resolve()
            gt_src = (repo_root / row["refined_mask_path"]).resolve()
            image_dst = image_root / image_src.name
            gt_dst = gt_root / gt_src.name
            symlink_file(image_src, image_dst)
            symlink_file(gt_src, gt_dst)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_dst),
                    "gt_path": str(gt_dst),
                    "source": "CAMO" if sample_id.startswith("camourflage_") else "COD10K",
                    "refine_mode": str(row.get("refine_mode", "")),
                    "refine_submode": str(row.get("refine_submode", "")),
                    "final_confidence": f"{float(row.get('final_confidence', 0.0)):.6f}",
                    "change_ratio": f"{float(row.get('change_ratio', 0.0)):.6f}",
                }
            )

    manifest_jsonl = output_root / "train_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_csv = output_root / "train_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source",
                "image_path",
                "gt_path",
                "refine_mode",
                "refine_submode",
                "final_confidence",
                "change_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "total": len(manifest_rows),
        "camo": sum(1 for row in manifest_rows if row["source"] == "CAMO"),
        "cod10k": sum(1 for row in manifest_rows if row["source"] == "COD10K"),
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

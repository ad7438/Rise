"""Manual review page for Stage 3 original-vs-refined pseudo masks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np

from stage2_pseudo_text.mask_utils import create_highlight_overlay, load_binary_mask, load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Stage 3 original-vs-refined pseudo mask comparisons.")
    parser.add_argument("--manifest_jsonl", default="Dataset/Stage3Semantic/stage4_ready/train_manifest.jsonl")
    parser.add_argument("--original_mask_dir", default="Dataset/RISE_Workspace/pseudo_mask")
    parser.add_argument("--output_dir", default="Dataset/Stage3Semantic/review_compare")
    parser.add_argument("--high_change_n", type=int, default=36)
    parser.add_argument("--low_change_n", type=int, default=18)
    parser.add_argument("--high_change_pool_ratio", type=float, default=1.0)
    parser.add_argument("--randomize_high_change", action="store_true")
    parser.add_argument("--seed", type=int, default=20260404)
    return parser.parse_args()


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def take_unique(records: List[dict], count: int, selected_ids: set[str]) -> List[dict]:
    if count <= 0:
        return []
    chosen: List[dict] = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in selected_ids:
            continue
        chosen.append(record)
        selected_ids.add(sample_id)
        if len(chosen) >= count:
            break
    return chosen


def copy_asset(source: Path, asset_dir: Path, stem: str, suffix: str) -> str:
    target = asset_dir / f"{stem}_{suffix}{source.suffix}"
    shutil.copy2(source, target)
    return str(target.relative_to(asset_dir.parent))


def save_image_asset(image, asset_dir: Path, stem: str, suffix: str) -> str:
    target = asset_dir / f"{stem}_{suffix}.png"
    image.save(target)
    return str(target.relative_to(asset_dir.parent))


def save_mask_asset(mask: np.ndarray, asset_dir: Path, stem: str, suffix: str) -> str:
    from PIL import Image

    target = asset_dir / f"{stem}_{suffix}.png"
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(target)
    return str(target.relative_to(asset_dir.parent))


def write_manifest(records: List[dict], output_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def img_tag(relative_path: str | None) -> str:
    if not relative_path:
        return ""
    return f"<img src='{html.escape(relative_path)}' />"


def build_html(records: List[dict], output_path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['group'])}</td>"
            f"<td>{html.escape(record['sample_id'])}</td>"
            f"<td>{record['change_ratio']:.4f}</td>"
            f"<td>{record['old_foreground_ratio']:.4f}</td>"
            f"<td>{record['refined_foreground_ratio']:.4f}</td>"
            f"<td>{record['final_confidence']:.4f}</td>"
            f"<td>{html.escape(record['pseudo_text'])}</td>"
            f"<td>{img_tag(record['image_asset'])}</td>"
            f"<td>{img_tag(record['old_overlay_asset'])}</td>"
            f"<td>{img_tag(record['refined_overlay_asset'])}</td>"
            f"<td>{img_tag(record['old_mask_asset'])}</td>"
            f"<td>{img_tag(record['refined_mask_asset'])}</td>"
            "</tr>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'><title>Stage 3 Mask Compare</title>"
        "<style>body{font-family:Arial,sans-serif;}table{border-collapse:collapse;width:100%;}"
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top;}img{max-width:240px;max-height:180px;}</style>"
        "</head><body><h1>Stage 3 Original vs Refined Pseudo Mask Review</h1><table><thead><tr>"
        "<th>Group</th><th>ID</th><th>Change</th><th>Old FG</th><th>Refined FG</th><th>Final</th><th>Text</th>"
        "<th>Image</th><th>Old Overlay</th><th>Refined Overlay</th><th>Old Mask</th><th>Refined Mask</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_jsonl)
    original_mask_dir = Path(args.original_mask_dir)
    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest_path)
    if not records:
        raise ValueError(f"No records found in {manifest_path}")

    enriched: List[dict] = []
    for record in records:
        sample_id = record["sample_id"]
        original_mask_path = original_mask_dir / f"{sample_id}.png"
        refined_mask_path = Path(record["source_mask_path"])
        image_path = Path(record["source_image_path"])
        if not original_mask_path.exists() or not refined_mask_path.exists() or not image_path.exists():
            continue

        original_mask = load_binary_mask(str(original_mask_path))
        refined_mask = load_binary_mask(str(refined_mask_path))
        image = load_rgb_image(str(image_path))
        height, width = original_mask.shape
        total = max(height * width, 1)
        diff = np.not_equal(original_mask, refined_mask).sum()
        change_ratio = float(diff / total)
        old_ratio = float(original_mask.sum() / total)
        refined_ratio = float(refined_mask.sum() / total)
        enriched.append(
            {
                **record,
                "original_mask_path": str(original_mask_path),
                "change_ratio": change_ratio,
                "old_foreground_ratio": old_ratio,
                "refined_foreground_ratio": refined_ratio,
                "_image": image,
                "_old_mask": original_mask,
                "_refined_mask": refined_mask,
            }
        )

    if not enriched:
        raise ValueError("No comparable Stage 3 records found.")

    rng = random.Random(args.seed)
    high_change = sorted(enriched, key=lambda record: record["change_ratio"], reverse=True)
    low_change = sorted(enriched, key=lambda record: record["change_ratio"])
    rng.shuffle(low_change)

    if args.randomize_high_change:
        pool_ratio = min(max(args.high_change_pool_ratio, 0.0), 1.0)
        pool_count = max(args.high_change_n, math.ceil(len(high_change) * pool_ratio))
        high_change = high_change[:pool_count]
        rng.shuffle(high_change)

    selected_ids: set[str] = set()
    selected: List[dict] = []
    for group_name, candidates, count in (
        ("high_change", high_change, args.high_change_n),
        ("low_change", low_change, args.low_change_n),
    ):
        for record in take_unique(candidates, count, selected_ids):
            review_record = {key: value for key, value in record.items() if not key.startswith("_")}
            review_record["group"] = group_name
            stem = f"{group_name}_{review_record['sample_id']}"
            review_record["image_asset"] = copy_asset(Path(review_record["source_image_path"]), asset_dir, stem, "image")
            review_record["old_overlay_asset"] = save_image_asset(
                create_highlight_overlay(record["_image"], record["_old_mask"]),
                asset_dir,
                stem,
                "old_overlay",
            )
            review_record["refined_overlay_asset"] = save_image_asset(
                create_highlight_overlay(record["_image"], record["_refined_mask"]),
                asset_dir,
                stem,
                "refined_overlay",
            )
            review_record["old_mask_asset"] = save_mask_asset(record["_old_mask"], asset_dir, stem, "old_mask")
            review_record["refined_mask_asset"] = save_mask_asset(record["_refined_mask"], asset_dir, stem, "refined_mask")
            selected.append(review_record)

    manifest_output = output_dir / "compare_manifest.csv"
    html_output = output_dir / "compare_review.html"
    write_manifest(selected, manifest_output)
    build_html(selected, html_output)
    print(json.dumps({"selected_samples": len(selected), "manifest": str(manifest_output), "html": str(html_output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

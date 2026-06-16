"""File IO helpers for the standalone Stage 2 pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from stage2_pseudo_text.mask_utils import save_mask_png


def ensure_output_dirs(output_dir: Path, save_visuals: bool) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "output_dir": output_dir,
        "results_jsonl": output_dir / "results.jsonl",
        "results_csv": output_dir / "results.csv",
        "summary_json": output_dir / "summary.json",
    }
    if save_visuals:
        visuals_dir = output_dir / "visuals"
        paths.update(
            {
                "visuals_dir": visuals_dir,
                "overlay_dir": visuals_dir / "overlay",
                "tight_crop_dir": visuals_dir / "tight_crop",
                "context_crop_dir": visuals_dir / "context_crop",
                "clean_mask_dir": visuals_dir / "clean_mask",
            }
        )
        for key, value in paths.items():
            if key.endswith("_dir"):
                value.mkdir(parents=True, exist_ok=True)
    return paths


def serialize_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def save_visuals(paths: Dict[str, Path], sample_id: str, overlay, tight_crop, context_crop, clean_mask):
    overlay_path = paths["overlay_dir"] / f"{sample_id}.png"
    tight_path = paths["tight_crop_dir"] / f"{sample_id}.png"
    context_path = paths["context_crop_dir"] / f"{sample_id}.png"
    clean_mask_path = paths["clean_mask_dir"] / f"{sample_id}.png"
    overlay.save(overlay_path)
    tight_crop.save(tight_path)
    context_crop.save(context_path)
    save_mask_png(clean_mask, str(clean_mask_path))
    return overlay_path, tight_path, context_path, clean_mask_path


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: List[dict], output_path: Path) -> None:
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


def build_summary(records: List[dict]) -> dict:
    total = len(records)
    low_count = sum(1 for record in records if record["low_confidence"])
    avg_final_conf = sum(record["final_confidence"] for record in records) / total if total else 0.0
    avg_mask_quality = sum(record["mask_quality"] for record in records) / total if total else 0.0
    category_histogram: Dict[str, int] = {}
    for record in records:
        category_histogram[record["category"]] = category_histogram.get(record["category"], 0) + 1
    return {
        "total_samples": total,
        "low_confidence_samples": low_count,
        "average_final_confidence": avg_final_conf,
        "average_mask_quality": avg_mask_quality,
        "category_histogram": category_histogram,
    }

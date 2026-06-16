"""CLI for manual review sampling of Stage 2 results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
from pathlib import Path
from typing import List


EDGE_FLAGS = {"tiny_mask", "small_mask", "fragmented_mask", "multi_component_mask", "touching_edge", "below_min_area"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Stage 2 outputs for manual inspection.")
    parser.add_argument("--results_jsonl", default="Dataset/Stage2PseudoText/results.jsonl")
    parser.add_argument("--output_dir", default="Dataset/Stage2PseudoText/review")
    parser.add_argument("--high_n", type=int, default=20)
    parser.add_argument("--low_n", type=int, default=20)
    parser.add_argument("--edge_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
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


def copy_asset(asset_path: str | None, asset_dir: Path, stem: str, suffix: str) -> str | None:
    if not asset_path:
        return None
    source = Path(asset_path)
    if not source.exists():
        return None
    target = asset_dir / f"{stem}_{suffix}{source.suffix}"
    shutil.copy2(source, target)
    return str(target.relative_to(asset_dir.parent))


def write_manifest(records: List[dict], output_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in record.items()})


def build_html(records: List[dict], output_path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['group'])}</td>"
            f"<td>{html.escape(record['sample_id'])}</td>"
            f"<td>{html.escape(record['pseudo_text'])}</td>"
            f"<td>{record['final_confidence']:.4f}</td>"
            f"<td>{record['mask_quality']:.4f}</td>"
            f"<td>{record['category_confidence']:.4f}</td>"
            f"<td>{html.escape(', '.join(record.get('mask_flags', [])))}</td>"
            f"<td>{img_tag(record.get('overlay_asset'))}</td>"
            f"<td>{img_tag(record.get('tight_crop_asset'))}</td>"
            f"<td>{img_tag(record.get('context_crop_asset'))}</td>"
            "</tr>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'><title>Stage 2 Review Samples</title>"
        "<style>body{font-family:Arial,sans-serif;}table{border-collapse:collapse;width:100%;}"
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top;}img{max-width:240px;max-height:180px;}</style>"
        "</head><body><h1>Stage 2 Manual Review Samples</h1><table><thead><tr>"
        "<th>Group</th><th>ID</th><th>Text</th><th>Final</th><th>Mask</th><th>Category</th><th>Flags</th><th>Overlay</th><th>Tight</th><th>Context</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


def img_tag(relative_path: str | None) -> str:
    if not relative_path:
        return ""
    return f"<img src='{html.escape(relative_path)}' />"


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_jsonl)
    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    rng = random.Random(args.seed)
    selected_ids: set[str] = set()
    high_candidates = sorted([record for record in records if not record.get("low_confidence", False)], key=lambda record: record.get("final_confidence", 0.0), reverse=True)
    low_candidates = sorted(records, key=lambda record: record.get("final_confidence", 0.0))
    edge_candidates = [record for record in records if EDGE_FLAGS.intersection(record.get("mask_flags", []))]
    rng.shuffle(edge_candidates)

    selected: List[dict] = []
    for group_name, candidates, count in [("high_confidence", high_candidates, args.high_n), ("low_confidence", low_candidates, args.low_n), ("edge_case", edge_candidates, args.edge_n)]:
        for record in take_unique(candidates, count, selected_ids):
            review_record = dict(record)
            review_record["group"] = group_name
            stem = f"{group_name}_{review_record['sample_id']}"
            review_record["overlay_asset"] = copy_asset(review_record.get("overlay_path"), asset_dir, stem, "overlay")
            review_record["tight_crop_asset"] = copy_asset(review_record.get("tight_crop_path"), asset_dir, stem, "tight")
            review_record["context_crop_asset"] = copy_asset(review_record.get("context_crop_path"), asset_dir, stem, "context")
            selected.append(review_record)

    manifest_path = output_dir / "review_manifest.csv"
    html_path = output_dir / "review_samples.html"
    write_manifest(selected, manifest_path)
    build_html(selected, html_path)
    print(json.dumps({"selected_samples": len(selected), "manifest": str(manifest_path), "html": str(html_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

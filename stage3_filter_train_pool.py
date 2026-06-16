#!/usr/bin/env python3
"""Auto-filter the 3640 train pool (4040 - DevMini) for manual Stage3 review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Stage3 train pool for manual review.")
    parser.add_argument(
        "--results_jsonl",
        default="Dataset/Stage3MaskRefine_v3_edge/results.jsonl",
        help="Full Stage3 results JSONL.",
    )
    parser.add_argument(
        "--dev_manifest_jsonl",
        default="Dataset/DevMini_stage3_v3_edge/dev_manifest.jsonl",
        help="DevMini manifest used to exclude validation samples.",
    )
    parser.add_argument(
        "--output_dir",
        default="Dataset/Stage3MaskRefine_v3_edge/train_pool_auto_filtered",
        help="Output directory for filtered manifests/results.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def source_of(sample_id: str) -> str:
    return "CAMO" if sample_id.startswith("camourflage_") else "COD10K"


def keep_decision(record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    sample_id = record["sample_id"]
    source = source_of(sample_id)
    refine_mode = str(record.get("refine_mode", ""))
    final_conf = maybe_float(record.get("final_confidence"))
    text_prior = maybe_float(record.get("text_prior_mean"))
    vis_mean = maybe_float(record.get("vis_mean"))
    low_conf = bool(record.get("low_confidence"))

    if bool(record.get("mask_is_empty")):
        reasons.append("mask_is_empty")
    if refine_mode in {"skip_empty", "fallback_empty", "fallback_too_small"}:
        reasons.append(f"refine_mode={refine_mode}")
    if source == "COD10K" and refine_mode == "fallback_too_large":
        reasons.append("cod10k_fallback_too_large")
    if final_conf < 0.70:
        reasons.append("final_confidence_lt_0.70")
    if low_conf and final_conf < 0.75:
        reasons.append("low_confidence_with_low_final_conf")
    if text_prior < 0.18:
        reasons.append("text_prior_mean_lt_0.18")
    if vis_mean < 0.08:
        reasons.append("vis_mean_lt_0.08")

    return len(reasons) == 0, reasons


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    results_path = (repo_root / args.results_jsonl).resolve()
    dev_manifest_path = (repo_root / args.dev_manifest_jsonl).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_jsonl(results_path)
    dev_rows = load_jsonl(dev_manifest_path)
    dev_ids = {row["sample_id"] for row in dev_rows}

    pool_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for row in results:
        sample_id = row["sample_id"]
        if sample_id in dev_ids:
            continue
        row = dict(row)
        row["source"] = source_of(sample_id)
        keep, reasons = keep_decision(row)
        row["auto_keep"] = keep
        row["auto_filter_reasons"] = reasons
        pool_rows.append(row)
        if keep:
            kept_rows.append(row)
        else:
            dropped_rows.append(row)

    kept_rows.sort(key=lambda row: maybe_float(row.get("change_ratio")), reverse=True)
    dropped_rows.sort(key=lambda row: maybe_float(row.get("change_ratio")), reverse=True)

    write_jsonl(output_dir / "train_pool_results.jsonl", pool_rows)
    write_jsonl(output_dir / "filtered_candidates.jsonl", kept_rows)
    write_jsonl(output_dir / "filtered_dropped.jsonl", dropped_rows)

    write_csv(
        output_dir / "filtered_dropped.csv",
        [
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "refine_mode": row.get("refine_mode", ""),
                "refine_submode": row.get("refine_submode", ""),
                "final_confidence": f"{maybe_float(row.get('final_confidence')):.6f}",
                "text_prior_mean": f"{maybe_float(row.get('text_prior_mean')):.6f}",
                "vis_mean": f"{maybe_float(row.get('vis_mean')):.6f}",
                "change_ratio": f"{maybe_float(row.get('change_ratio')):.6f}",
                "reasons": ";".join(row["auto_filter_reasons"]),
            }
            for row in dropped_rows
        ],
        [
            "sample_id",
            "source",
            "refine_mode",
            "refine_submode",
            "final_confidence",
            "text_prior_mean",
            "vis_mean",
            "change_ratio",
            "reasons",
        ],
    )

    summary = {
        "input_total": len(results),
        "dev_total": len(dev_ids),
        "train_pool_total": len(pool_rows),
        "kept_total": len(kept_rows),
        "dropped_total": len(dropped_rows),
        "kept_by_source": Counter(row["source"] for row in kept_rows),
        "dropped_by_source": Counter(row["source"] for row in dropped_rows),
        "kept_modes": Counter(row.get("refine_mode", "") for row in kept_rows),
        "dropped_modes": Counter(row.get("refine_mode", "") for row in dropped_rows),
        "drop_reasons": Counter(reason for row in dropped_rows for reason in row["auto_filter_reasons"]),
        "rule_notes": [
            "exclude dev samples",
            "drop empty/skip/fallback_empty/fallback_too_small",
            "drop COD10K fallback_too_large",
            "drop final_confidence < 0.70",
            "drop low_confidence with final_confidence < 0.75",
            "drop text_prior_mean < 0.18",
            "drop vis_mean < 0.08",
        ],
    }
    with (output_dir / "filter_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Shared helpers for Stage 4 CRIS-lite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def maybe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def maybe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def compute_sample_weight(
    final_confidence: float | None,
    *,
    low_confidence: bool,
    processing_errors: list[str] | None,
    min_weight: float = 0.3,
    low_conf_scale: float = 0.6,
    processing_error_cap: float = 0.35,
) -> float:
    confidence = clamp01(maybe_float(final_confidence, 0.0))
    weight = min_weight + (1.0 - min_weight) * confidence
    if low_confidence:
        weight = max(min_weight, weight * low_conf_scale)
    if processing_errors:
        weight = min(weight, max(min_weight, processing_error_cap))
    return float(weight)


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[dict]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def find_file_by_stem(directory: Path | None, stem: str) -> Path | None:
    if directory is None or not directory.exists():
        return None
    for candidate in sorted(directory.glob(f"{stem}.*")):
        if candidate.is_file():
            return candidate
    return None

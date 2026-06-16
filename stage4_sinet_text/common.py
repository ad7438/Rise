"""Shared helpers for Stage 4 SINet-text."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"

CATEGORY_VOCAB = [
    "human",
    "bird",
    "mammal",
    "reptile_amphibian",
    "aquatic_animal",
    "arthropod",
    "plant",
    "manmade_object",
    "other_animal",
    "other_non_animal",
    "unknown",
]
LOCATION_VOCAB = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]
SIZE_VOCAB = ["small", "medium", "large"]

CATEGORY_TO_ID = {key: index for index, key in enumerate(CATEGORY_VOCAB)}
LOCATION_TO_ID = {key: index for index, key in enumerate(LOCATION_VOCAB)}
SIZE_TO_ID = {key: index for index, key in enumerate(SIZE_VOCAB)}
LOCATION_HFLIP = {
    "top_left": "top_right",
    "top_center": "top_center",
    "top_right": "top_left",
    "middle_left": "middle_right",
    "middle_center": "middle_center",
    "middle_right": "middle_left",
    "bottom_left": "bottom_right",
    "bottom_center": "bottom_center",
    "bottom_right": "bottom_left",
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def maybe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def category_id(category: str | None) -> int:
    return CATEGORY_TO_ID.get((category or "unknown").strip(), CATEGORY_TO_ID["unknown"])


def location_id(location_key: str | None) -> int:
    return LOCATION_TO_ID.get((location_key or "middle_center").strip(), LOCATION_TO_ID["middle_center"])


def size_id(size_key: str | None) -> int:
    return SIZE_TO_ID.get((size_key or "medium").strip(), SIZE_TO_ID["medium"])


def structured_prompt(record: dict) -> str:
    category = record.get("category", "unknown")
    location = record.get("location_key", "middle_center")
    size = record.get("size_key", "medium")
    return f"[CAT] {category} [LOC] {location} [SIZE] {size}"


def curriculum_settings(epoch: int, total_epochs: int) -> tuple[float, float]:
    ratio = epoch / max(total_epochs, 1)
    if ratio <= 0.33:
        return 0.80, 0.15
    if ratio <= 0.66:
        return 0.60, 0.50
    return 0.0, 1.0

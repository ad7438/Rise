#!/usr/bin/env python3
"""Build a hybrid Stage3 training subset.

Selected sample ids use refined masks, the remaining train-pool samples use
their original Stage1 masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid Stage3 train subset directory.")
    parser.add_argument(
        "--train_pool_jsonl",
        required=True,
        help="JSONL for the full train pool after dev exclusion.",
    )
    parser.add_argument(
        "--selected_manifest_jsonl",
        required=True,
        help="JSONL manifest of selected refined samples.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output dataset root with Image/GT subdirectories.",
    )
    parser.add_argument(
        "--blacklist_csv",
        default="",
        help="Optional CSV of bad refined sample ids that should fall back to old masks.",
    )
    parser.add_argument(
        "--blacklist_action",
        default="drop",
        choices=["drop", "old"],
        help="How to handle blacklisted sample ids: drop them or force old masks.",
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


def load_selected_ids(path: Path) -> set[str]:
    selected_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            selected_ids.add(row["sample_id"])
    return selected_ids


def load_blacklist_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    blacklist_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            keep = (row.get("keep") or "").strip()
            if not sample_id:
                continue
            if keep and keep not in {"1", "true", "True", "yes", "YES"}:
                continue
            blacklist_ids.add(sample_id)
    return blacklist_ids


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    train_pool_jsonl = (repo_root / args.train_pool_jsonl).resolve()
    selected_manifest_jsonl = (repo_root / args.selected_manifest_jsonl).resolve()
    output_root = (repo_root / args.output_root).resolve()
    blacklist_csv = (repo_root / args.blacklist_csv).resolve() if args.blacklist_csv else None

    selected_ids = load_selected_ids(selected_manifest_jsonl)
    blacklist_ids = load_blacklist_ids(blacklist_csv) if blacklist_csv else set()

    ensure_clean_dir(output_root)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    source_counts = {"CAMO": 0, "COD10K": 0}
    supervision_counts = {"refined": 0, "old": 0}
    blacklist_hits = 0
    blacklist_dropped = 0

    with train_pool_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            image_src = (repo_root / row["image_path"]).resolve()
            force_old = sample_id in blacklist_ids
            if force_old and args.blacklist_action == "drop":
                blacklist_hits += 1
                blacklist_dropped += 1
                continue
            if sample_id in selected_ids and not force_old:
                gt_src = (repo_root / row["refined_mask_path"]).resolve()
                supervision = "refined"
            else:
                gt_src = (repo_root / row["init_mask_path"]).resolve()
                supervision = "old"
                if force_old:
                    blacklist_hits += 1

            image_dst = image_root / image_src.name
            gt_dst = gt_root / gt_src.name
            symlink_file(image_src, image_dst)
            symlink_file(gt_src, gt_dst)

            source = "CAMO" if sample_id.startswith("camourflage_") else "COD10K"
            source_counts[source] += 1
            supervision_counts[supervision] += 1

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "image_path": str(image_dst),
                    "gt_path": str(gt_dst),
                    "supervision_type": supervision,
                    "refine_mode": str(row.get("refine_mode", "")),
                    "refine_submode": str(row.get("refine_submode", "")),
                    "final_confidence": f"{float(row.get('final_confidence', 0.0)):.6f}",
                    "change_ratio": f"{float(row.get('change_ratio', 0.0)):.6f}",
                    "blacklist_forced_old": "1" if force_old else "0",
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
                "supervision_type",
                "refine_mode",
                "refine_submode",
                "final_confidence",
                "change_ratio",
                "blacklist_forced_old",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "total": len(manifest_rows),
        "selected_refined_total": supervision_counts["refined"],
        "remaining_old_total": supervision_counts["old"],
        "camo": source_counts["CAMO"],
        "cod10k": source_counts["COD10K"],
        "blacklist_total": len(blacklist_ids),
        "blacklist_hits": blacklist_hits,
        "blacklist_dropped": blacklist_dropped,
        "blacklist_action": args.blacklist_action,
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

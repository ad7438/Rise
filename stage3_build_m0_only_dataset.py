#!/usr/bin/env python3
"""Build M0-only ablation datasets from an existing training manifest.

The experiment keeps the image list and duplicate rehearsal samples from a
reference manifest, but replaces every supervision mask with the original M0
pseudo mask from Dataset/RISE_Workspace/pseudo_mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-protocol M0-only ablation dataset.")
    parser.add_argument("--manifest", required=True, help="Reference train_manifest.csv.")
    parser.add_argument("--output_root", required=True, help="Output dataset root with Image/ and GT/.")
    parser.add_argument(
        "--m0_root",
        default=str(PROJECT_ROOT / "Dataset/RISE_Workspace/pseudo_mask"),
        help="Directory containing original M0 masks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild output directories if present.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_reset_dir(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    project = PROJECT_ROOT.resolve()
    if path == project or project not in path.parents:
        raise RuntimeError(f"refuse to reset path outside project: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def base_sample_id(sample_id: str) -> str:
    return sample_id.split("__refboost", 1)[0]


def find_m0_mask(m0_root: Path, sample_id: str) -> Path:
    stem = base_sample_id(sample_id)
    for suffix in VALID_MASK_SUFFIXES:
        candidate = m0_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"missing M0 mask for {sample_id} under {m0_root}")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {path}")
        return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    output_root = resolve_path(args.output_root)
    m0_root = resolve_path(args.m0_root)

    fieldnames, rows = read_manifest(manifest_path)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    safe_reset_dir(output_root, overwrite=args.overwrite)
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    extra_fields = [
        "m0_only_base_sample_id",
        "m0_only_source_gt_path",
        "m0_only_reference_gt_path",
        "m0_only_reference_manifest",
    ]
    out_fieldnames = list(fieldnames)
    for field in extra_fields:
        if field not in out_fieldnames:
            out_fieldnames.append(field)

    out_rows: list[dict[str, str]] = []
    missing_images: list[str] = []
    duplicate_names: set[str] = set()
    seen_names: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        image_src = resolve_path(row["image_path"])
        if not image_src.exists():
            missing_images.append(str(image_src))
            continue
        m0_src = find_m0_mask(m0_root, sample_id)

        image_name = f"{sample_id}{image_src.suffix.lower() or '.jpg'}"
        mask_name = f"{sample_id}.png"
        if image_name in seen_names or mask_name in seen_names:
            duplicate_names.add(sample_id)
        seen_names.add(image_name)
        seen_names.add(mask_name)

        image_dst = image_root / image_name
        mask_dst = gt_root / mask_name
        link_file(image_src, image_dst)
        link_file(m0_src, mask_dst)

        out_row = dict(row)
        out_row["image_path"] = str(image_dst)
        out_row["gt_path"] = str(mask_dst)
        out_row["supervision_type"] = "m0_only"
        out_row["m0_only_base_sample_id"] = base_sample_id(sample_id)
        out_row["m0_only_source_gt_path"] = str(m0_src)
        out_row["m0_only_reference_gt_path"] = row.get("gt_path", "")
        out_row["m0_only_reference_manifest"] = str(manifest_path)
        out_rows.append(out_row)

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"missing {len(missing_images)} images, first entries:\n{preview}")
    if duplicate_names:
        raise RuntimeError(f"duplicate output names detected: {sorted(duplicate_names)[:10]}")

    write_csv(output_root / "train_manifest.csv", out_fieldnames, out_rows)
    write_jsonl(output_root / "train_manifest.jsonl", out_rows)

    summary = {
        "total": len(out_rows),
        "reference_manifest": str(manifest_path),
        "output_root": str(output_root),
        "m0_root": str(m0_root),
        "refboost_total": sum("__refboost" in row["sample_id"] for row in out_rows),
        "unique_base_samples": len({row["m0_only_base_sample_id"] for row in out_rows}),
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

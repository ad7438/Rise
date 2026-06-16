#!/usr/bin/env python3
"""Build a matched all-refined dataset from the final selected-sample manifest.

H5 keeps exactly the same retained sample set as the selective-refinement
dataset, but replaces every supervision mask with the corresponding refined
mask M_r from the Stage-3 gate decision table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H5 matched all-refined dataset.")
    parser.add_argument("--reference_manifest", required=True, help="H3 retained train_manifest.csv.")
    parser.add_argument("--decisions_csv", required=True, help="auto_gate_decisions.csv with refined_mask_path.")
    parser.add_argument("--output_root", required=True, help="Output dataset root with Image/ and GT/.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild output_root if it exists.")
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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
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
    reference_manifest = resolve_path(args.reference_manifest)
    decisions_csv = resolve_path(args.decisions_csv)
    output_root = resolve_path(args.output_root)

    ref_fieldnames, ref_rows = read_csv(reference_manifest)
    _, decision_rows = read_csv(decisions_csv)
    decision_by_id = {row["sample_id"]: row for row in decision_rows}

    safe_reset_dir(output_root, overwrite=args.overwrite)
    image_root = output_root / "Image"
    gt_root = output_root / "GT"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    extra_fields = [
        "matched_all_refined_source_gt_path",
        "matched_all_refined_reference_gt_path",
        "matched_all_refined_reference_manifest",
        "matched_all_refined_reference_decision",
        "matched_all_refined_reference_reason",
    ]
    out_fieldnames = list(ref_fieldnames)
    for field in extra_fields:
        if field not in out_fieldnames:
            out_fieldnames.append(field)

    out_rows: list[dict[str, str]] = []
    missing: list[str] = []
    decision_counts: dict[str, int] = {}
    seen_names: set[str] = set()

    for ref in ref_rows:
        sample_id = ref["sample_id"]
        decision = decision_by_id.get(sample_id)
        if decision is None:
            missing.append(f"missing decision for {sample_id}")
            continue
        refined_src = resolve_path(decision["refined_mask_path"])
        image_src = resolve_path(ref["image_path"])
        if not image_src.exists():
            missing.append(str(image_src))
            continue
        if not refined_src.exists():
            missing.append(str(refined_src))
            continue

        image_name = f"{sample_id}{image_src.suffix.lower() or '.jpg'}"
        mask_name = f"{sample_id}.png"
        if image_name in seen_names or mask_name in seen_names:
            raise RuntimeError(f"duplicate output name for sample_id={sample_id}")
        seen_names.add(image_name)
        seen_names.add(mask_name)

        image_dst = image_root / image_name
        gt_dst = gt_root / mask_name
        link_file(image_src, image_dst)
        link_file(refined_src, gt_dst)

        out = dict(ref)
        out["image_path"] = str(image_dst)
        out["gt_path"] = str(gt_dst)
        out["supervision_type"] = "matched_all_refined"
        out["matched_all_refined_source_gt_path"] = str(refined_src)
        out["matched_all_refined_reference_gt_path"] = ref.get("gt_path", "")
        out["matched_all_refined_reference_manifest"] = str(reference_manifest)
        out["matched_all_refined_reference_decision"] = decision.get("decision", "")
        out["matched_all_refined_reference_reason"] = decision.get("reason", "")
        out_rows.append(out)
        decision_counts[decision.get("decision", "")] = decision_counts.get(decision.get("decision", ""), 0) + 1

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} entries, first entries:\n{preview}")

    write_csv(output_root / "train_manifest.csv", out_fieldnames, out_rows)
    write_jsonl(output_root / "train_manifest.jsonl", out_rows)

    summary = {
        "total": len(out_rows),
        "reference_manifest": str(reference_manifest),
        "decisions_csv": str(decisions_csv),
        "output_root": str(output_root),
        "decision_counts_from_reference": decision_counts,
        "definition": "H5 matched all-refined: same retained samples as H3, all labels replaced by refined mask M_r.",
    }
    with (output_root / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

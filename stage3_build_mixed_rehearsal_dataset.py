#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def resolve_path(path):
    src = Path(path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    return src.resolve()


def quality_score(row):
    confidence = as_float(row, "final_confidence")
    text_prior = as_float(row, "text_prior_mean")
    vis_mean = as_float(row, "vis_mean")
    change_ratio = as_float(row, "change_ratio")
    area_growth = as_float(row, "area_growth", 1.0)
    return (
        confidence
        + 0.10 * text_prior
        + 0.05 * vis_mean
        - 0.15 * change_ratio
        - 0.05 * abs(area_growth - 1.0)
    )


def reset_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            raise RuntimeError("refuse to remove nested directory: {}".format(item))
        item.unlink()


def link_file(src, dst):
    if not src.exists():
        raise FileNotFoundError(str(src))
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def write_manifest(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest", default=str(
        PROJECT_ROOT / "Dataset/Stage3AutoGate_v1/TrainDatasetAuto/train_manifest.csv"
    ))
    parser.add_argument("--refined_manifest", default=str(
        PROJECT_ROOT / "Dataset/Stage3AutoGate_v1/TrainDatasetAutoRefined/train_manifest.csv"
    ))
    parser.add_argument("--output_root", default=str(
        PROJECT_ROOT / "Dataset/Stage3AutoGate_v1/TrainDatasetAutoRehearsal"
    ))
    parser.add_argument("--extra_refined_fraction", type=float, default=0.30,
                        help="target fraction of extra refined samples in the final rehearsal dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.extra_refined_fraction < 1.0):
        raise ValueError("--extra_refined_fraction must be in [0, 1)")

    output_root = Path(args.output_root)
    image_dir = output_root / "Image"
    gt_dir = output_root / "GT"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite to rebuild it".format(output_root))
    reset_dir(image_dir)
    reset_dir(gt_dir)

    base_rows = read_rows(args.base_manifest)
    refined_rows = read_rows(args.refined_manifest)
    extra_count = int(round(len(base_rows) * args.extra_refined_fraction / (1.0 - args.extra_refined_fraction)))
    extra_rows = sorted(refined_rows, key=quality_score, reverse=True)[:extra_count]

    manifest_rows = []
    for row in base_rows:
        sample_id = row["sample_id"]
        image_dst = image_dir / "{}.jpg".format(sample_id)
        gt_dst = gt_dir / "{}.png".format(sample_id)
        link_file(resolve_path(row["image_path"]), image_dst)
        link_file(resolve_path(row["gt_path"]), gt_dst)
        out_row = dict(row)
        out_row["stage2_source"] = "base_mixed"
        out_row["rehearsal_quality_score"] = "{:.6f}".format(quality_score(row))
        manifest_rows.append(out_row)

    for index, row in enumerate(extra_rows):
        sample_id = "{}__refboost{:04d}".format(row["sample_id"], index)
        image_dst = image_dir / "{}.jpg".format(sample_id)
        gt_dst = gt_dir / "{}.png".format(sample_id)
        link_file(resolve_path(row["image_path"]), image_dst)
        link_file(resolve_path(row["gt_path"]), gt_dst)
        out_row = dict(row)
        out_row["sample_id"] = sample_id
        out_row["stage2_source"] = "extra_refined"
        out_row["rehearsal_quality_score"] = "{:.6f}".format(quality_score(row))
        manifest_rows.append(out_row)

    fieldnames = list(base_rows[0].keys())
    for key in ("stage2_source", "rehearsal_quality_score"):
        if key not in fieldnames:
            fieldnames.append(key)
    write_manifest(output_root / "train_manifest.csv", fieldnames, manifest_rows)

    summary = {
        "base_total": len(base_rows),
        "refined_pool_total": len(refined_rows),
        "extra_refined_total": len(extra_rows),
        "total": len(manifest_rows),
        "extra_refined_fraction": len(extra_rows) / max(1, len(manifest_rows)),
        "output_root": str(output_root),
    }
    with open(output_root / "subset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

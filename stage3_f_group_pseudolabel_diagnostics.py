#!/usr/bin/env python3
"""F-group pseudo-label diagnostics for paper ablation evidence.

The script audits the final Stage-3 gate decisions and evaluates M0, Mr,
and M* against the training masks used only for offline quality diagnosis.
Ground-truth masks are never used to generate labels or gate decisions here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from PIL import Image


METRIC_NAMES = ("iou", "dice", "mae")
GROUP_ORDER = ("all", "retained", "accepted_refined", "fallback_old", "drop")
CORRELATION_FIELDS = (
    "final_confidence",
    "mask_quality",
    "text_prior_mean",
    "vis_mean",
    "s_sem_agsp",
    "s_vis_svpm",
    "change_ratio",
    "area_growth",
    "candidate_components",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F0-F5 pseudo-label quality diagnostics.")
    parser.add_argument(
        "--decisions_csv",
        default="Dataset/Stage3AutoGate_AGSP_SVPM_SVAC_v1/auto_gate_decisions.csv",
        help="Main-method auto-gate decisions CSV.",
    )
    parser.add_argument(
        "--gt_root",
        default="Dataset/TrainDataset/GT",
        help="Training GT root used only for offline pseudo-label evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/paper_ablation_metrics/F_group_pseudolabel_diagnostics",
        help="Output directory for F-group CSV/JSON/Markdown records.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel_or_empty(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def resolve_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def gt_path_for(sample_id: str, gt_root: Path) -> Path:
    return gt_root / f"{sample_id}.png"


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.float32) / 255.0


def metric_triplet(mask_path: Path, gt_path: Path) -> dict[str, float]:
    gt_img = Image.open(gt_path).convert("L")
    size = gt_img.size
    gt = np.asarray(gt_img, dtype=np.float32) / 255.0
    gt_img.close()
    pred = load_mask(mask_path, size=size)
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5
    intersection = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    iou = 1.0 if union == 0.0 else intersection / union
    dice = 1.0 if (pred_sum + gt_sum) == 0.0 else (2.0 * intersection) / (pred_sum + gt_sum)
    mae = float(np.mean(np.abs(pred - gt)))
    return {"iou": iou, "dice": dice, "mae": mae}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def decision_label(decision: str) -> str:
    if decision == "refined":
        return "accepted_refined"
    if decision == "old":
        return "fallback_old"
    if decision == "drop":
        return "drop"
    return decision or "unknown"


def final_source(decision: str) -> str:
    if decision == "refined":
        return "Mr"
    if decision == "old":
        return "M0"
    if decision == "drop":
        return "drop"
    return "unknown"


def add_metric_fields(row: dict[str, Any], prefix: str, metrics: dict[str, float] | None) -> None:
    for metric in METRIC_NAMES:
        row[f"{prefix}_{metric}"] = fmt_float(metrics[metric] if metrics else None)


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = values(rows, key)
    return mean(vals) if vals else None


def std(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = values(rows, key)
    return pstdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)


def count_condition(rows: list[dict[str, Any]], key: str, threshold: float = 0.0, op: str = "gt") -> int:
    count = 0
    for row in rows:
        value = safe_float(row.get(key))
        if value is None:
            continue
        if op == "gt" and value > threshold:
            count += 1
        elif op == "ge" and value >= threshold:
            count += 1
        elif op == "lt" and value < threshold:
            count += 1
        elif op == "le" and value <= threshold:
            count += 1
    return count


def make_summary_row(name: str, rows: list[dict[str, Any]], total_count: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group": name,
        "count": len(rows),
        "ratio_of_train_pool": fmt_float(len(rows) / total_count if total_count else None),
    }
    for label in ("m0", "mr", "mstar"):
        for metric in METRIC_NAMES:
            key = f"{label}_{metric}"
            out[f"{key}_mean"] = fmt_float(avg(rows, key))
            out[f"{key}_std"] = fmt_float(std(rows, key))
    for metric in METRIC_NAMES:
        out[f"delta_mr_minus_m0_{metric}_mean"] = fmt_float(avg(rows, f"delta_mr_minus_m0_{metric}"))
        out[f"delta_mstar_minus_m0_{metric}_mean"] = fmt_float(avg(rows, f"delta_mstar_minus_m0_{metric}"))
    out["mr_better_than_m0_iou_count"] = count_condition(rows, "delta_mr_minus_m0_iou", 0.0, "gt")
    out["mr_better_than_m0_iou_ratio"] = fmt_float(
        out["mr_better_than_m0_iou_count"] / len(rows) if rows else None
    )
    out["mstar_better_than_m0_iou_count"] = count_condition(rows, "delta_mstar_minus_m0_iou", 0.0, "gt")
    out["mstar_better_than_m0_iou_ratio"] = fmt_float(
        out["mstar_better_than_m0_iou_count"] / len(values(rows, "delta_mstar_minus_m0_iou"))
        if values(rows, "delta_mstar_minus_m0_iou")
        else None
    )
    return out


def rankdata(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda item: item[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def spearman(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[int, float | None]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return len(xs), None
    return len(xs), pearson(rankdata(xs), rankdata(ys))


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    decisions_path = resolve_path(repo_root, args.decisions_csv)
    gt_root = resolve_path(repo_root, args.gt_root)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions = read_csv(decisions_path)
    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for row in decisions:
        sample_id = row["sample_id"]
        decision = row.get("decision", "")
        group = decision_label(decision)
        m0_path = resolve_path(repo_root, row.get("init_mask_path", ""))
        mr_path = resolve_path(repo_root, row.get("refined_mask_path", ""))
        gt_path = gt_path_for(sample_id, gt_root)
        mstar_path = mr_path if decision == "refined" else (m0_path if decision == "old" else None)

        path_status = {
            "m0_exists": m0_path.exists(),
            "mr_exists": mr_path.exists(),
            "mstar_exists": bool(mstar_path and mstar_path.exists()),
            "gt_exists": gt_path.exists(),
        }
        for name, exists in path_status.items():
            if not exists and name != "mstar_exists":
                missing.append({"sample_id": sample_id, "missing": name, "decision": decision})

        m0_metrics = metric_triplet(m0_path, gt_path) if m0_path.exists() and gt_path.exists() else None
        mr_metrics = metric_triplet(mr_path, gt_path) if mr_path.exists() and gt_path.exists() else None
        mstar_metrics = (
            metric_triplet(mstar_path, gt_path)
            if mstar_path is not None and mstar_path.exists() and gt_path.exists()
            else None
        )

        metrics_row: dict[str, Any] = {
            "sample_id": sample_id,
            "source": row.get("source", ""),
            "decision": decision,
            "decision_group": group,
            "reason": row.get("reason", ""),
            "final_source": final_source(decision),
            "image_path": row.get("image_path", ""),
            "m0_path": rel_or_empty(m0_path, repo_root),
            "mr_path": rel_or_empty(mr_path, repo_root),
            "mstar_path": rel_or_empty(mstar_path, repo_root) if mstar_path is not None else "",
            "gt_path": rel_or_empty(gt_path, repo_root),
            **{key: str(value) for key, value in path_status.items()},
        }
        for key in CORRELATION_FIELDS:
            metrics_row[key] = row.get(key, "")
        add_metric_fields(metrics_row, "m0", m0_metrics)
        add_metric_fields(metrics_row, "mr", mr_metrics)
        add_metric_fields(metrics_row, "mstar", mstar_metrics)
        for metric in METRIC_NAMES:
            m0_value = m0_metrics[metric] if m0_metrics else None
            mr_value = mr_metrics[metric] if mr_metrics else None
            mstar_value = mstar_metrics[metric] if mstar_metrics else None
            metrics_row[f"delta_mr_minus_m0_{metric}"] = fmt_float(
                mr_value - m0_value if mr_value is not None and m0_value is not None else None
            )
            metrics_row[f"delta_mstar_minus_m0_{metric}"] = fmt_float(
                mstar_value - m0_value if mstar_value is not None and m0_value is not None else None
            )

        metrics_rows.append(metrics_row)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source": row.get("source", ""),
                "decision": decision,
                "decision_group": group,
                "reason": row.get("reason", ""),
                "final_source": final_source(decision),
                "image_path": row.get("image_path", ""),
                "m0_path": rel_or_empty(m0_path, repo_root),
                "mr_path": rel_or_empty(mr_path, repo_root),
                "mstar_path": rel_or_empty(mstar_path, repo_root) if mstar_path is not None else "",
                "gt_path": rel_or_empty(gt_path, repo_root),
                **{key: str(value) for key, value in path_status.items()},
            }
        )

    by_group = defaultdict(list)
    for row in metrics_rows:
        by_group["all"].append(row)
        if row["decision_group"] != "drop":
            by_group["retained"].append(row)
        by_group[row["decision_group"]].append(row)

    summary_rows = [make_summary_row(group, by_group[group], len(metrics_rows)) for group in GROUP_ORDER]
    accepted_rows = by_group["accepted_refined"]
    fallback_rows = by_group["fallback_old"]
    drop_rows = by_group["drop"]

    gate_rows = [
        {"metric": "train_pool_total", "value": len(metrics_rows), "definition": "all rows in auto_gate_decisions.csv"},
        {
            "metric": "accepted_refined_count",
            "value": len(accepted_rows),
            "definition": "decision=refined; final M*=Mr",
        },
        {
            "metric": "fallback_old_count",
            "value": len(fallback_rows),
            "definition": "decision=old; final M*=M0",
        },
        {"metric": "drop_count", "value": len(drop_rows), "definition": "decision=drop; excluded from retained training set"},
        {
            "metric": "accepted_rate",
            "value": fmt_float(len(accepted_rows) / len(metrics_rows) if metrics_rows else None),
            "definition": "accepted_refined_count / train_pool_total",
        },
        {
            "metric": "fallback_rate",
            "value": fmt_float(len(fallback_rows) / len(metrics_rows) if metrics_rows else None),
            "definition": "fallback_old_count / train_pool_total",
        },
        {
            "metric": "drop_rate",
            "value": fmt_float(len(drop_rows) / len(metrics_rows) if metrics_rows else None),
            "definition": "drop_count / train_pool_total",
        },
        {
            "metric": "accepted_hit_rate_iou",
            "value": fmt_float(
                count_condition(accepted_rows, "delta_mr_minus_m0_iou", 0.0, "gt") / len(accepted_rows)
                if accepted_rows
                else None
            ),
            "definition": "accepted samples where IoU(Mr)>IoU(M0)",
        },
        {
            "metric": "fallback_protection_rate_iou",
            "value": fmt_float(
                count_condition(fallback_rows, "delta_mr_minus_m0_iou", 0.0, "le") / len(fallback_rows)
                if fallback_rows
                else None
            ),
            "definition": "fallback samples where rejected Mr is not better than M0 in IoU",
        },
        {
            "metric": "drop_mr_worse_or_equal_rate_iou",
            "value": fmt_float(
                count_condition(drop_rows, "delta_mr_minus_m0_iou", 0.0, "le") / len(drop_rows) if drop_rows else None
            ),
            "definition": "drop samples where Mr is not better than M0 in IoU",
        },
    ]

    correlation_rows = []
    for field in CORRELATION_FIELDS:
        n, corr = spearman(metrics_rows, field, "delta_mr_minus_m0_iou")
        correlation_rows.append(
            {
                "field": field,
                "target": "delta_mr_minus_m0_iou",
                "n": n,
                "spearman_r": fmt_float(corr),
            }
        )

    manifest_fields = list(manifest_rows[0].keys()) if manifest_rows else []
    metric_fields = list(metrics_rows[0].keys()) if metrics_rows else []
    write_csv(output_dir / "sample_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_dir / "pseudo_label_metrics_per_sample.csv", metrics_rows, metric_fields)
    write_csv(output_dir / "pseudo_label_summary.csv", summary_rows)
    write_csv(output_dir / "accepted_subset_summary.csv", [make_summary_row("accepted_refined", accepted_rows, len(metrics_rows))])
    write_csv(output_dir / "fallback_subset_summary.csv", [make_summary_row("fallback_old", fallback_rows, len(metrics_rows))])
    write_csv(output_dir / "drop_subset_summary.csv", [make_summary_row("drop", drop_rows, len(metrics_rows))])
    write_csv(output_dir / "gate_diagnostic_summary.csv", gate_rows)
    write_csv(output_dir / "gate_score_correlation.csv", correlation_rows)
    write_csv(output_dir / "missing_files.csv", missing, ["sample_id", "missing", "decision"])

    decision_counts = Counter(row["decision_group"] for row in metrics_rows)
    source_decision_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in metrics_rows:
        source_decision_counts[row["source"]][row["decision_group"]] += 1

    summary = {
        "input_decisions_csv": rel_or_empty(decisions_path, repo_root),
        "output_dir": rel_or_empty(output_dir, repo_root),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "git_head": git_output(["git", "rev-parse", "--short", "HEAD"]),
        "train_pool_total": len(metrics_rows),
        "decision_counts": dict(decision_counts),
        "source_decision_counts": {source: dict(counts) for source, counts in source_decision_counts.items()},
        "summary_rows": summary_rows,
        "gate_diagnostics": gate_rows,
        "correlations": correlation_rows,
        "missing_files": missing[:20],
        "missing_file_count": len(missing),
    }
    write_json(output_dir / "f_group_summary.json", summary)

    retained = by_group["retained"]
    accepted_delta = avg(accepted_rows, "delta_mr_minus_m0_iou")
    fallback_delta = avg(fallback_rows, "delta_mr_minus_m0_iou")
    mstar_retained_delta = avg(retained, "delta_mstar_minus_m0_iou")
    notes = [
        "# F Group Pseudo-label Diagnostics",
        "",
        f"- Input decisions: `{rel_or_empty(decisions_path, repo_root)}`",
        f"- Output directory: `{rel_or_empty(output_dir, repo_root)}`",
        f"- Git branch: `{summary['git_branch']}`",
        f"- Git head: `{summary['git_head']}`",
        "",
        "## Counts",
        "",
        f"- Train-pool total: {len(metrics_rows)}",
        f"- Accepted refined: {len(accepted_rows)} ({len(accepted_rows) / len(metrics_rows):.2%})",
        f"- Fallback old: {len(fallback_rows)} ({len(fallback_rows) / len(metrics_rows):.2%})",
        f"- Drop: {len(drop_rows)} ({len(drop_rows) / len(metrics_rows):.2%})",
        "",
        "## Key Quality Signals",
        "",
        f"- Retained M* vs M0 mean Delta IoU: {fmt_float(mstar_retained_delta)}",
        f"- Accepted subset Mr vs M0 mean Delta IoU: {fmt_float(accepted_delta)}",
        f"- Fallback subset rejected Mr vs M0 mean Delta IoU: {fmt_float(fallback_delta)}",
        "",
        "## Files",
        "",
        "- `sample_manifest.csv`",
        "- `pseudo_label_metrics_per_sample.csv`",
        "- `pseudo_label_summary.csv`",
        "- `accepted_subset_summary.csv`",
        "- `fallback_subset_summary.csv`",
        "- `drop_subset_summary.csv`",
        "- `gate_diagnostic_summary.csv`",
        "- `gate_score_correlation.csv`",
    ]
    (output_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (output_dir / "command.sh").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_state.txt").write_text(
        git_output(["git", "status", "--short", "--branch"]) + "\n", encoding="utf-8"
    )

    print(f"Wrote F-group diagnostics to {output_dir}")
    print(f"Counts: {dict(decision_counts)}")
    print(f"Retained M* vs M0 mean Delta IoU: {fmt_float(mstar_retained_delta)}")
    print(f"Accepted Mr vs M0 mean Delta IoU: {fmt_float(accepted_delta)}")
    print(f"Fallback rejected Mr vs M0 mean Delta IoU: {fmt_float(fallback_delta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

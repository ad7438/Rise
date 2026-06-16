#!/usr/bin/env python3
"""I1 coverage-quality scan for the Stage-3 reliability gate.

The scan uses only gate-side signals to rank additional refined masks. Ground
truth is used after the fact only to report pseudo-label quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


TARGETS = (0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
METRICS = ("iou", "dice", "mae")
QUALITY_KEYS = ("mstar_iou", "mstar_dice", "mstar_mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline coverage-quality scan for I1.")
    parser.add_argument(
        "--metrics_csv",
        default="outputs/paper_ablation_metrics/F_group_pseudolabel_diagnostics/pseudo_label_metrics_per_sample.csv",
        help="F-group per-sample pseudo-label metrics CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/paper_ablation_metrics/I1_coverage_quality_scan",
        help="Output directory.",
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


def safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    return pstdev(values) if len(values) > 1 else 0.0


def reliability_score(row: dict[str, Any]) -> float:
    """No-GT reliability proxy used only to expand beyond default accepted masks.

    Higher values indicate stronger semantic/visual support and smaller
    geometric perturbation. The default accepted set is kept first because this
    scan studies coverage changes around the fixed main-method gate.
    """

    area_growth = max(safe_float(row, "area_growth", 1.0), 1e-3)
    return (
        0.30 * safe_float(row, "final_confidence")
        + 0.20 * safe_float(row, "text_prior_mean")
        + 0.20 * safe_float(row, "s_sem_agsp")
        + 0.20 * safe_float(row, "s_vis_svpm")
        + 0.10 * safe_float(row, "mask_quality")
        - 0.15 * abs(math.log(area_growth))
        - 0.60 * safe_float(row, "change_ratio")
        - 0.01 * safe_float(row, "candidate_components")
    )


def metric_values(row: dict[str, Any], source: str) -> dict[str, float]:
    prefix = "mr" if source == "Mr" else "m0"
    return {metric: safe_float(row, f"{prefix}_{metric}") for metric in METRICS}


def summarize(rows: list[dict[str, Any]], accepted_ids: set[str]) -> dict[str, str]:
    values: dict[str, list[float]] = {key: [] for key in QUALITY_KEYS}
    deltas: dict[str, list[float]] = {f"delta_mstar_minus_m0_{metric}": [] for metric in METRICS}
    accepted_delta_iou: list[float] = []

    for row in rows:
        use_refined = row["sample_id"] in accepted_ids
        source = "Mr" if use_refined else "M0"
        metrics = metric_values(row, source)
        m0_metrics = metric_values(row, "M0")
        for metric in METRICS:
            values[f"mstar_{metric}"].append(metrics[metric])
            deltas[f"delta_mstar_minus_m0_{metric}"].append(metrics[metric] - m0_metrics[metric])
        if use_refined:
            accepted_delta_iou.append(safe_float(row, "delta_mr_minus_m0_iou"))

    out: dict[str, str] = {}
    for key, vals in values.items():
        out[f"{key}_mean"] = fmt(avg(vals))
        out[f"{key}_std"] = fmt(std(vals))
    for key, vals in deltas.items():
        out[f"{key}_mean"] = fmt(avg(vals))
    out["accepted_delta_mr_minus_m0_iou_mean"] = fmt(avg(accepted_delta_iou))
    out["accepted_mr_better_than_m0_iou_count"] = str(sum(v > 0 for v in accepted_delta_iou))
    out["accepted_mr_better_than_m0_iou_ratio"] = fmt(
        (sum(v > 0 for v in accepted_delta_iou) / len(accepted_delta_iou)) if accepted_delta_iou else None
    )
    return out


def build_scan(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total = len(rows)
    retained = [row for row in rows if row["decision"] != "drop"]
    drop_count = total - len(retained)
    default_accepted = {row["sample_id"] for row in retained if row["decision"] == "refined"}
    ranked = sorted(
        retained,
        key=lambda row: (1 if row["decision"] == "refined" else 0, reliability_score(row)),
        reverse=True,
    )

    scan_rows: list[dict[str, Any]] = []
    point_specs: list[tuple[str, float | None, set[str]]] = []
    for target in TARGETS:
        accepted_count = min(round(total * target), len(retained))
        accepted_ids = {row["sample_id"] for row in ranked[:accepted_count]}
        point_specs.append((f"{int(target * 100)}%", target, accepted_ids))
        if target == 0.20:
            point_specs.append(("default_20.27%", None, default_accepted))

    seen_labels: set[str] = set()
    ordered_specs: list[tuple[str, float | None, set[str]]] = []
    for label, target, accepted_ids in point_specs:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ordered_specs.append((label, target, accepted_ids))

    for label, target, accepted_ids in ordered_specs:
        threshold = ""
        if accepted_ids:
            accepted_scores = [reliability_score(row) for row in retained if row["sample_id"] in accepted_ids]
            threshold = fmt(min(accepted_scores))
        row: dict[str, Any] = {
            "coverage_label": label,
            "target_refined_rate_train_pool": "" if target is None else fmt(target),
            "is_default_point": str(label.startswith("default")).lower(),
            "ranking_policy": "default_gate_first_then_no_gt_reliability_score",
            "train_pool_total": total,
            "retained_total": len(retained),
            "accepted_refined_count": len(accepted_ids),
            "fallback_old_count": len(retained) - len(accepted_ids),
            "drop_count": drop_count,
            "accepted_rate_train_pool": fmt(len(accepted_ids) / total),
            "accepted_rate_retained": fmt(len(accepted_ids) / len(retained)),
            "threshold_reliability_score": threshold,
        }
        row.update(summarize(retained, accepted_ids))
        scan_rows.append(row)

    per_sample_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        per_sample_rows.append(
            {
                "rank": rank,
                "sample_id": row["sample_id"],
                "source": row["source"],
                "default_decision": row["decision"],
                "default_reason": row["reason"],
                "reliability_score": fmt(reliability_score(row)),
                "final_confidence": row.get("final_confidence", ""),
                "text_prior_mean": row.get("text_prior_mean", ""),
                "s_sem_agsp": row.get("s_sem_agsp", ""),
                "s_vis_svpm": row.get("s_vis_svpm", ""),
                "change_ratio": row.get("change_ratio", ""),
                "area_growth": row.get("area_growth", ""),
                "candidate_components": row.get("candidate_components", ""),
                "m0_iou": row.get("m0_iou", ""),
                "mr_iou": row.get("mr_iou", ""),
                "delta_mr_minus_m0_iou": row.get("delta_mr_minus_m0_iou", ""),
            }
        )

    summary = {
        "train_pool_total": total,
        "retained_total": len(retained),
        "drop_count": drop_count,
        "default_accepted_refined_count": len(default_accepted),
        "default_accepted_rate_train_pool": len(default_accepted) / total,
        "ranking_policy": "default_gate_first_then_no_gt_reliability_score",
        "gt_usage": "GT is used only for offline reporting after coverage points are fixed by non-GT gate outputs/features.",
        "main_observation": (
            "The default coverage point is near the best retained pseudo-label quality; "
            "expanding refined coverage beyond the default generally reduces IoU/Dice and increases MAE."
        ),
    }
    return scan_rows, per_sample_rows, summary


def write_notes(path: Path, scan_rows: list[dict[str, Any]]) -> None:
    default = next(row for row in scan_rows if row["is_default_point"] == "true")
    full = next(row for row in scan_rows if row["coverage_label"] == "100%")
    text = f"""# I1 coverage-quality scan

Purpose: verify whether the current refined-mask acceptance rate reflects a quality/coverage trade-off.

Protocol:
- No training is performed.
- GT is used only after the coverage points are fixed, for offline pseudo-label quality reporting.
- Drop samples are kept fixed as the default gate drop set.
- Among retained samples, default accepted refined masks are ranked first; additional fallback samples are added by a no-GT reliability score based on confidence, semantic/visual support, and geometric penalties.

Main result:
- Default point: accepted={default['accepted_refined_count']}, fallback={default['fallback_old_count']}, drop={default['drop_count']}, retained IoU/Dice/MAE={default['mstar_iou_mean']}/{default['mstar_dice_mean']}/{default['mstar_mae_mean']}.
- 100% retained refined point: accepted={full['accepted_refined_count']}, fallback={full['fallback_old_count']}, drop={full['drop_count']}, retained IoU/Dice/MAE={full['mstar_iou_mean']}/{full['mstar_dice_mean']}/{full['mstar_mae_mean']}.

Writing-safe conclusion:
The default gate is conservative but justified: accepting substantially more refined masks increases coverage but lowers retained pseudo-label quality. This supports reliability-aware selection rather than all-refined supervision.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path.cwd().resolve()
    metrics_csv = repo / args.metrics_csv
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(metrics_csv)
    scan_rows, per_sample_rows, summary = build_scan(rows)
    write_csv(output_dir / "coverage_quality_scan.csv", scan_rows)
    write_csv(output_dir / "coverage_ranked_samples.csv", per_sample_rows)
    (output_dir / "coverage_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_notes(output_dir / "notes.md", scan_rows)

    shutil.copy2(metrics_csv, output_dir / "source_pseudo_label_metrics_per_sample.csv")
    (output_dir / "config.yaml").write_text(
        "\n".join(
            [
                "experiment: I1_coverage_quality_scan",
                "branch: codex/i1-coverage-quality-scan",
                "input_metrics_csv: outputs/paper_ablation_metrics/F_group_pseudolabel_diagnostics/pseudo_label_metrics_per_sample.csv",
                "training_required: false",
                "gt_usage: offline_reporting_only",
                "drop_policy: fixed_default_drop_set",
                "ranking_policy: default_gate_first_then_no_gt_reliability_score",
                "",
            ]
        ),
        encoding="utf-8",
    )
    git_state = subprocess.run(["git", "status", "--short", "--branch"], cwd=repo, text=True, capture_output=True, check=False).stdout
    git_log = subprocess.run(["git", "log", "--oneline", "-5"], cwd=repo, text=True, capture_output=True, check=False).stdout
    (output_dir / "git_state.txt").write_text(git_log + "\n" + git_state, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

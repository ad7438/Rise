"""Conservative foreground prototype pruning for Stage 3."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_jsonl_records, maybe_float, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively prune Stage 3 foreground prototypes.")
    parser.add_argument("--raw_proto_dir", default="Dataset/Stage3Semantic_coarse/prototype_raw")
    parser.add_argument("--output_dir", default="Dataset/Stage3Semantic_coarse/prototype_refined")
    parser.add_argument("--final_conf_threshold", type=float, default=0.50)
    parser.add_argument("--unknown_conf_threshold", type=float, default=0.50)
    parser.add_argument("--unknown_final_threshold", type=float, default=0.70)
    parser.add_argument("--fallback_conf_threshold", type=float, default=0.70)
    parser.add_argument("--fallback_final_threshold", type=float, default=0.78)
    parser.add_argument("--semantic_final_threshold", type=float, default=0.72)
    parser.add_argument("--min_keep", type=int, default=512)
    return parser.parse_args()


def _prune_reasons(meta: dict, args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    low_conf_reasons = set(meta.get("low_confidence_reasons") or [])
    category = str(meta.get("category") or "")
    final_confidence = maybe_float(meta.get("final_confidence"))
    category_confidence = maybe_float(meta.get("category_confidence"))
    if meta.get("processing_errors"):
        reasons.append("processing_error")
    if meta.get("mask_is_empty"):
        reasons.append("empty_mask")
    if final_confidence < args.final_conf_threshold:
        reasons.append("low_final_confidence")
    if (
        category == "unknown"
        and category_confidence < args.unknown_conf_threshold
        and final_confidence < args.unknown_final_threshold
    ):
        reasons.append("unknown_low_confidence")
    if (
        category in {"other_animal", "other_non_animal"}
        and category_confidence < args.fallback_conf_threshold
        and final_confidence < args.fallback_final_threshold
    ):
        reasons.append("fallback_semantic_category")
    if (
        meta.get("low_confidence")
        and low_conf_reasons.intersection({"fallback_category", "fragmented_mask_semantics", "touching_edge_semantics", "processing_error"})
        and final_confidence < args.semantic_final_threshold
    ):
        reasons.append("semantic_low_confidence")
    return reasons


def _rescue_sort_key(meta: dict) -> tuple[float, float, float, str]:
    return (
        maybe_float(meta.get("final_confidence")),
        maybe_float(meta.get("category_confidence")),
        maybe_float(meta.get("mask_quality")),
        str(meta.get("sample_id", "")),
    )


def _compute_raw_weight(meta: dict) -> float:
    final_conf = maybe_float(meta.get("final_confidence"))
    category_conf = maybe_float(meta.get("category_confidence"))
    mask_quality = maybe_float(meta.get("mask_quality"))
    flags = set(meta.get("mask_flags") or [])
    low_conf_reasons = set(meta.get("low_confidence_reasons") or [])
    category = str(meta.get("category") or "")

    weight = 0.35 + 0.45 * final_conf + 0.10 * category_conf + 0.10 * mask_quality
    if meta.get("low_confidence"):
        weight *= 0.85
    if category == "unknown":
        weight *= 0.60
    elif category in {"other_animal", "other_non_animal"}:
        weight *= 0.78
    if "fallback_category" in low_conf_reasons:
        weight *= 0.82
    if "fragmented_mask_semantics" in low_conf_reasons:
        weight *= 0.88
    if "touching_edge_semantics" in low_conf_reasons:
        weight *= 0.92
    if "fragmented_mask" in flags:
        weight *= 0.92
    if "multi_component_mask" in flags:
        weight *= 0.94
    if meta.get("mask_touches_edge"):
        weight *= 0.96
    if meta.get("processing_errors"):
        weight *= 0.50
    if meta.get("mask_is_empty"):
        weight *= 0.20
    return max(float(weight), 0.05)


def main() -> None:
    args = parse_args()
    raw_proto_dir = Path(args.raw_proto_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fore = np.load(raw_proto_dir / "fore.npy")
    back = np.load(raw_proto_dir / "back.npy")
    fore_meta = load_jsonl_records(raw_proto_dir / "fore_meta.jsonl")
    back_meta = load_jsonl_records(raw_proto_dir / "back_meta.jsonl")

    if fore.shape[0] != len(fore_meta):
        raise ValueError("Foreground prototype rows do not match metadata rows.")
    if back.shape[0] != len(back_meta):
        raise ValueError("Background prototype rows do not match metadata rows.")

    kept_indices: list[int] = []
    pruned_indices: list[int] = []
    kept_meta_map: dict[int, dict] = {}
    pruned_meta_map: dict[int, dict] = {}
    reason_counter: Counter[str] = Counter()

    for index, meta in enumerate(fore_meta):
        reasons = _prune_reasons(meta, args)
        if reasons:
            meta_with_status = {
                **meta,
                "prune_reasons": reasons,
                "rescued_for_min_keep": False,
            }
            pruned_indices.append(index)
            pruned_meta_map[index] = meta_with_status
            reason_counter.update(reasons)
        else:
            meta_with_status = {
                **meta,
                "prune_reasons": [],
                "rescued_for_min_keep": False,
            }
            kept_indices.append(index)
            kept_meta_map[index] = meta_with_status

    rescued_count = 0
    if len(kept_indices) < args.min_keep:
        missing = min(args.min_keep - len(kept_indices), len(pruned_indices))
        rescue_candidates = sorted(
            ((idx, pruned_meta_map[idx]) for idx in pruned_indices),
            key=lambda item: _rescue_sort_key(item[1]),
            reverse=True,
        )
        rescue_index_set = {idx for idx, _ in rescue_candidates[:missing]}
        if rescue_index_set:
            new_pruned_indices: list[int] = []
            for idx in pruned_indices:
                meta = pruned_meta_map[idx]
                if idx in rescue_index_set:
                    rescued_count += 1
                    kept_indices.append(idx)
                    kept_meta_map[idx] = {
                        **meta,
                        "original_prune_reasons": list(meta["prune_reasons"]),
                        "prune_reasons": [],
                        "rescued_for_min_keep": True,
                    }
                    pruned_meta_map.pop(idx, None)
                else:
                    new_pruned_indices.append(idx)
            pruned_indices = new_pruned_indices

    keep_order = np.array(sorted(kept_indices), dtype=np.int64)
    kept_raw_weights = [_compute_raw_weight(kept_meta_map[int(index)]) for index in keep_order.tolist()]
    mean_kept_raw_weight = float(np.mean(kept_raw_weights)) if kept_raw_weights else 1.0
    if mean_kept_raw_weight <= 0.0:
        mean_kept_raw_weight = 1.0

    kept_meta = []
    for index, raw_weight in zip(keep_order.tolist(), kept_raw_weights):
        normalized_weight = float(np.clip(raw_weight / mean_kept_raw_weight, 0.25, 1.75))
        kept_meta.append(
            {
                **kept_meta_map[int(index)],
                "prototype_weight_raw": float(raw_weight),
                "prototype_weight": normalized_weight,
            }
        )

    pruned_meta = []
    for index in sorted(pruned_indices):
        meta = pruned_meta_map[int(index)]
        raw_weight = _compute_raw_weight(meta)
        normalized_weight = float(np.clip(raw_weight / mean_kept_raw_weight, 0.25, 1.75))
        pruned_meta.append(
            {
                **meta,
                "prototype_weight_raw": float(raw_weight),
                "prototype_weight": normalized_weight,
            }
        )
    fore_refined = fore[keep_order]
    np.save(output_dir / "fore_refined.npy", fore_refined)
    np.save(output_dir / "fore.npy", fore_refined)
    np.save(output_dir / "back.npy", back)

    write_jsonl(output_dir / "kept_meta.jsonl", kept_meta)
    write_jsonl(output_dir / "pruned_meta.jsonl", pruned_meta)
    write_jsonl(output_dir / "fore_meta.jsonl", kept_meta)
    write_jsonl(output_dir / "back_meta.jsonl", back_meta)
    shutil.copy2(raw_proto_dir / "back.npy", output_dir / "back.npy")

    report = {
        "total_fore_prototypes": int(fore.shape[0]),
        "kept_fore_prototypes": int(fore_refined.shape[0]),
        "pruned_fore_prototypes": int(len(pruned_indices)),
        "rescued_for_min_keep": rescued_count,
        "back_prototypes": int(back.shape[0]),
        "min_keep": args.min_keep,
        "final_conf_threshold": args.final_conf_threshold,
        "unknown_conf_threshold": args.unknown_conf_threshold,
        "unknown_final_threshold": args.unknown_final_threshold,
        "fallback_conf_threshold": args.fallback_conf_threshold,
        "fallback_final_threshold": args.fallback_final_threshold,
        "semantic_final_threshold": args.semantic_final_threshold,
        "prototype_weight_mean_raw": mean_kept_raw_weight,
        "prototype_weight_min": min((record["prototype_weight"] for record in kept_meta), default=1.0),
        "prototype_weight_max": max((record["prototype_weight"] for record in kept_meta), default=1.0),
        "prune_reason_histogram": dict(reason_counter),
    }
    write_json(output_dir / "prune_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

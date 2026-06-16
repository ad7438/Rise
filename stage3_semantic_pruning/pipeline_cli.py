"""End-to-end Stage 3 pipeline runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Stage 3 semantic pruning pipeline.")
    parser.add_argument("--stage3_root", default="Dataset/Stage3Semantic_coarse")
    parser.add_argument("--stage2_results_jsonl", default="Dataset/Stage2PseudoText_full_v3_coarse/results.jsonl")
    parser.add_argument("--image_dir", default="Dataset/TrainDataset/Image")
    parser.add_argument("--cluster_dir", default="Dataset/RISE_Workspace/cluster_map")
    parser.add_argument("--dino", default="vit-l14")
    parser.add_argument("--imgsz", type=int, default=476)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--faiss_device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--top_k_ratio", type=float, default=0.33)
    parser.add_argument("--min_top_k", type=int, default=256)
    parser.add_argument("--vote_mode", choices=["count", "weighted"], default="weighted")
    parser.add_argument("--foreground_weight_power", type=float, default=1.5)
    parser.add_argument("--foreground_vote_scale", type=float, default=1.0)
    parser.add_argument("--background_vote_weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_keep", type=int, default=512)
    parser.add_argument("--final_conf_threshold", type=float, default=0.50)
    parser.add_argument("--unknown_conf_threshold", type=float, default=0.50)
    parser.add_argument("--unknown_final_threshold", type=float, default=0.70)
    parser.add_argument("--fallback_conf_threshold", type=float, default=0.70)
    parser.add_argument("--fallback_final_threshold", type=float, default=0.78)
    parser.add_argument("--semantic_final_threshold", type=float, default=0.72)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    stage3_root = Path(args.stage3_root)
    raw_dir = stage3_root / "prototype_raw"
    refined_dir = stage3_root / "prototype_refined"
    refined_mask_dir = stage3_root / "pseudo_mask_refined"
    stage4_dir = stage3_root / "stage4_ready"

    common_flags = []
    if args.limit is not None:
        common_flags.extend(["--limit", str(args.limit)])

    _run(
        [
            sys.executable,
            "stage3_build_prototypes.py",
            "--dino",
            args.dino,
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--faiss_device",
            args.faiss_device,
            "--image_dir",
            args.image_dir,
            "--cluster_dir",
            args.cluster_dir,
            "--stage2_results_jsonl",
            args.stage2_results_jsonl,
            "--output_dir",
            str(raw_dir),
            *common_flags,
        ]
    )
    _run(
        [
            sys.executable,
            "stage3_prune_prototypes.py",
            "--raw_proto_dir",
            str(raw_dir),
            "--output_dir",
            str(refined_dir),
            "--final_conf_threshold",
            str(args.final_conf_threshold),
            "--unknown_conf_threshold",
            str(args.unknown_conf_threshold),
            "--unknown_final_threshold",
            str(args.unknown_final_threshold),
            "--fallback_conf_threshold",
            str(args.fallback_conf_threshold),
            "--fallback_final_threshold",
            str(args.fallback_final_threshold),
            "--semantic_final_threshold",
            str(args.semantic_final_threshold),
            "--min_keep",
            str(args.min_keep),
        ]
    )
    _run(
        [
            sys.executable,
            "stage3_run_retrieval.py",
            "--top_k",
            str(args.top_k),
            "--top_k_ratio",
            str(args.top_k_ratio),
            "--min_top_k",
            str(args.min_top_k),
            "--dino",
            args.dino,
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--faiss_device",
            args.faiss_device,
            "--vote_mode",
            args.vote_mode,
            "--foreground_weight_power",
            str(args.foreground_weight_power),
            "--foreground_vote_scale",
            str(args.foreground_vote_scale),
            "--background_vote_weight",
            str(args.background_vote_weight),
            "--image_dir",
            args.image_dir,
            "--prototype_dir",
            str(refined_dir),
            "--output_dir",
            str(refined_mask_dir),
            *common_flags,
        ]
    )
    _run(
        [
            sys.executable,
            "stage3_export_stage4_ready.py",
            "--stage2_results_jsonl",
            args.stage2_results_jsonl,
            "--refined_mask_dir",
            str(refined_mask_dir),
            "--output_dir",
            str(stage4_dir),
            *common_flags,
        ]
    )

    summary = {
        "stage3_root": str(stage3_root),
        "prototype_raw_dir": str(raw_dir),
        "prototype_refined_dir": str(refined_dir),
        "pseudo_mask_refined_dir": str(refined_mask_dir),
        "stage4_ready_dir": str(stage4_dir),
        "dino": args.dino,
        "imgsz": args.imgsz,
        "device": args.device,
        "faiss_device": args.faiss_device,
        "top_k": args.top_k,
        "top_k_ratio": args.top_k_ratio,
        "min_top_k": args.min_top_k,
        "vote_mode": args.vote_mode,
        "foreground_weight_power": args.foreground_weight_power,
        "foreground_vote_scale": args.foreground_vote_scale,
        "background_vote_weight": args.background_vote_weight,
        "limit": args.limit,
        "min_keep": args.min_keep,
        "final_conf_threshold": args.final_conf_threshold,
        "unknown_conf_threshold": args.unknown_conf_threshold,
        "unknown_final_threshold": args.unknown_final_threshold,
        "fallback_conf_threshold": args.fallback_conf_threshold,
        "fallback_final_threshold": args.fallback_final_threshold,
        "semantic_final_threshold": args.semantic_final_threshold,
    }
    stage3_root.mkdir(parents=True, exist_ok=True)
    write_json(stage3_root / "pipeline_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

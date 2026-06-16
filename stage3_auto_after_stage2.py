#!/usr/bin/env python3
"""Wait for Stage 2 completion, then launch Stage 3 once."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Stage 2 completion and launch Stage 3.")
    parser.add_argument("--stage2_output_dir", default="Dataset/Stage2PseudoText_full_v3_coarse")
    parser.add_argument("--stage3_root", default="Dataset/Stage3Semantic_coarse")
    parser.add_argument("--image_dir", default="Dataset/TrainDataset/Image")
    parser.add_argument("--cluster_dir", default="Dataset/RISE_Workspace/cluster_map")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--faiss_device", default="cpu")
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--min_keep", type=int, default=512)
    parser.add_argument("--poll_seconds", type=int, default=120)
    return parser.parse_args()


def stage2_finished(stage2_output_dir: Path) -> bool:
    return (stage2_output_dir / "summary.json").exists() and (stage2_output_dir / "results.jsonl").exists()


def stage3_finished(stage3_root: Path) -> bool:
    return (stage3_root / "pipeline_summary.json").exists()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    stage2_output_dir = repo_root / args.stage2_output_dir
    stage3_root = repo_root / args.stage3_root

    print(f"[watcher] waiting for Stage 2 outputs in {stage2_output_dir}", flush=True)
    while not stage2_finished(stage2_output_dir):
        if stage3_finished(stage3_root):
            print(f"[watcher] Stage 3 already completed at {stage3_root}, exiting.", flush=True)
            return
        time.sleep(max(args.poll_seconds, 10))

    if stage3_finished(stage3_root):
        print(f"[watcher] Stage 3 already completed at {stage3_root}, exiting.", flush=True)
        return

    command = [
        sys.executable,
        "stage3_run_pipeline.py",
        "--stage3_root",
        args.stage3_root,
        "--stage2_results_jsonl",
        f"{args.stage2_output_dir}/results.jsonl",
        "--image_dir",
        args.image_dir,
        "--cluster_dir",
        args.cluster_dir,
        "--device",
        args.device,
        "--faiss_device",
        args.faiss_device,
        "--top_k",
        str(args.top_k),
        "--min_keep",
        str(args.min_keep),
    ]
    print("[watcher] launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

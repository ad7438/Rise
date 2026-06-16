"""Prepare Stage 4 eval texts and manifests from baseline coarse predictions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import load_jsonl_records, write_csv, write_json, write_jsonl
from .prepare_manifest_cli import build_manifest_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 4 eval text assets.")
    parser.add_argument("--dataset_root", default="Dataset/TestDataset")
    parser.add_argument("--coarse_pred_root", default="SINet-V2/pred/RISE_paper_split")
    parser.add_argument("--output_root", default="Dataset/Stage4CRISLite_baseline/eval")
    parser.add_argument("--datasets", nargs="+", default=["CAMO", "CHAMELEON", "COD10K", "NC4K"])
    parser.add_argument("--vlm_backend", default="qwen2_5_vl")
    parser.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm_device", default="auto")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=128)
    parser.add_argument("--vlm_dtype", default="bfloat16")
    parser.add_argument("--clip_backend", default="none")
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--hf_hub_download_timeout", type=int, default=None)
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=None)
    parser.add_argument("--min_weight", type=float, default=0.3)
    parser.add_argument("--low_conf_scale", type=float, default=0.6)
    parser.add_argument("--processing_error_cap", type=float, default=0.35)
    parser.add_argument("--text_field", default="training_text", choices=["training_text", "clip_text", "pseudo_text"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_stage2_for_dataset(args: argparse.Namespace, dataset_name: str, dataset_output_root: Path) -> Path:
    stage2_output = dataset_output_root / "stage2_text"
    results_jsonl = stage2_output / "results.jsonl"
    if results_jsonl.exists() and not args.force:
        return results_jsonl

    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "stage2_generate_pseudo_text.py"),
        "--image_dir",
        str(Path(args.dataset_root) / dataset_name / "Image"),
        "--mask_dir",
        str(Path(args.coarse_pred_root) / dataset_name),
        "--output_dir",
        str(stage2_output),
        "--no_save_visuals",
        "--vlm_backend",
        args.vlm_backend,
        "--vlm_model",
        args.vlm_model,
        "--vlm_device",
        args.vlm_device,
        "--vlm_max_new_tokens",
        str(args.vlm_max_new_tokens),
        "--vlm_dtype",
        args.vlm_dtype,
        "--clip_backend",
        args.clip_backend,
    ]
    if args.hf_endpoint:
        command.extend(["--hf_endpoint", args.hf_endpoint])
    if args.hf_hub_download_timeout is not None:
        command.extend(["--hf_hub_download_timeout", str(args.hf_hub_download_timeout)])
    if args.hf_hub_etag_timeout is not None:
        command.extend(["--hf_hub_etag_timeout", str(args.hf_hub_etag_timeout)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    subprocess.run(command, check=True)
    return results_jsonl


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset_output_root = output_root / dataset_name
        dataset_output_root.mkdir(parents=True, exist_ok=True)
        results_jsonl = run_stage2_for_dataset(args, dataset_name, dataset_output_root)

        stage2_records = load_jsonl_records(results_jsonl)
        records, summary = build_manifest_records(
            stage2_records,
            mask_dir=Path(args.coarse_pred_root) / dataset_name,
            gt_dir=Path(args.dataset_root) / dataset_name / "GT",
            text_field=args.text_field,
            min_weight=args.min_weight,
            low_conf_scale=args.low_conf_scale,
            processing_error_cap=args.processing_error_cap,
            limit=args.limit,
        )
        write_jsonl(dataset_output_root / "manifest.jsonl", records)
        write_csv(dataset_output_root / "manifest.csv", records)
        write_json(dataset_output_root / "manifest_summary.json", summary)
        print(f"{dataset_name}: {summary}")


if __name__ == "__main__":
    main()

"""End-to-end pipeline for Stage 4 CRIS-lite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 CRIS-lite end-to-end.")
    parser.add_argument("--train_stage2_results_jsonl", default="Dataset/Stage2PseudoText_full_v3_coarse/results.jsonl")
    parser.add_argument("--train_mask_dir", default="Dataset/RISE_Workspace/pseudo_mask")
    parser.add_argument("--train_manifest_dir", default="Dataset/Stage4CRISLite_baseline/train")
    parser.add_argument("--dataset_root", default="Dataset/TestDataset")
    parser.add_argument("--eval_coarse_pred_root", default="SINet-V2/pred/RISE_paper_split")
    parser.add_argument("--eval_root", default="Dataset/Stage4CRISLite_baseline/eval")
    parser.add_argument("--snapshot_dir", default="snapshot/CRIS_lite_baseline")
    parser.add_argument("--pred_dir", default="pred/CRIS_lite_baseline")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--backbone_name", default="resnet50")
    parser.add_argument("--text_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--vlm_backend", default="qwen2_5_vl")
    parser.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm_device", default="auto")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=128)
    parser.add_argument("--vlm_dtype", default="bfloat16")
    parser.add_argument("--clip_backend", default="none")
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--hf_hub_download_timeout", type=int, default=None)
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=None)
    parser.add_argument("--skip_prepare_eval_texts", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    python_exec = sys.executable

    run_command(
        [
            python_exec,
            str(PROJECT_ROOT / "stage4_prepare_manifest.py"),
            "--stage2_results_jsonl",
            args.train_stage2_results_jsonl,
            "--mask_dir",
            args.train_mask_dir,
            "--output_dir",
            args.train_manifest_dir,
        ]
    )

    if not args.skip_prepare_eval_texts:
        eval_command = [
            python_exec,
            str(PROJECT_ROOT / "stage4_prepare_eval_texts.py"),
            "--dataset_root",
            args.dataset_root,
            "--coarse_pred_root",
            args.eval_coarse_pred_root,
            "--output_root",
            args.eval_root,
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
            eval_command.extend(["--hf_endpoint", args.hf_endpoint])
        if args.hf_hub_download_timeout is not None:
            eval_command.extend(["--hf_hub_download_timeout", str(args.hf_hub_download_timeout)])
        if args.hf_hub_etag_timeout is not None:
            eval_command.extend(["--hf_hub_etag_timeout", str(args.hf_hub_etag_timeout)])
        run_command(eval_command)

    run_command(
        [
            python_exec,
            str(PROJECT_ROOT / "stage4_train_cris_lite.py"),
            "--train_manifest_jsonl",
            str(Path(args.train_manifest_dir) / "manifest.jsonl"),
            "--val_manifest_jsonl",
            str(Path(args.eval_root) / "CAMO" / "manifest.jsonl"),
            "--epoch",
            str(args.epoch),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--batchsize",
            str(args.batchsize),
            "--workers",
            str(args.workers),
            "--trainsize",
            str(args.trainsize),
            "--gpu_id",
            args.gpu_id,
            "--backbone_name",
            args.backbone_name,
            "--text_model_name",
            args.text_model_name,
            "--decoder_dim",
            str(args.decoder_dim),
            "--num_heads",
            str(args.num_heads),
            "--dropout",
            str(args.dropout),
            "--max_text_length",
            str(args.max_text_length),
            "--save_path",
            args.snapshot_dir,
        ]
    )

    if not args.skip_test:
        run_command(
            [
                python_exec,
                str(PROJECT_ROOT / "stage4_test_cris_lite.py"),
                "--pth_path",
                str(Path(args.snapshot_dir) / "Net_epoch_best.pth"),
                "--eval_root",
                args.eval_root,
                "--save_dir",
                args.pred_dir,
                "--testsize",
                str(args.trainsize),
                "--batchsize",
                str(min(args.batchsize, 4)),
                "--workers",
                str(min(args.workers, 2)),
                "--gpu_id",
                args.gpu_id,
                "--max_text_length",
                str(args.max_text_length),
            ]
        )


if __name__ == "__main__":
    main()

"""End-to-end pipeline for Stage 4 SINet-text."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 SINet-text pipeline.")
    parser.add_argument("--train_stage2_results_jsonl", default="Dataset/Stage2PseudoText_full_v4_rich/results.jsonl")
    parser.add_argument("--train_mask_dir", default="Dataset/RISE_Workspace/pseudo_mask")
    parser.add_argument("--train_output_dir", default="Dataset/Stage4SINetText_v3/train")
    parser.add_argument("--dataset_root", default="Dataset/TestDataset")
    parser.add_argument("--coarse_pred_root", default="SINet-V2/pred/RISE_paper_split")
    parser.add_argument("--eval_output_root", default="Dataset/Stage4SINetText_v3/eval")
    parser.add_argument("--teacher_path", default="SINet-V2/snapshot/RISE_paper_split/Net_epoch_best.pth")
    parser.add_argument("--save_path", default="snapshot/SINet_text_v3_refine")
    parser.add_argument("--pred_root", default="pred/SINet_text_v3_refine")
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--disable_text", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable

    if not args.skip_prepare:
        run_command(
            [
                python_executable,
                str(PROJECT_ROOT / "stage2_generate_pseudo_text.py"),
                "--image_dir",
                "Dataset/TrainDataset/Image",
                "--mask_dir",
                args.train_mask_dir,
                "--output_dir",
                str(Path(args.train_stage2_results_jsonl).parent),
                "--vlm_backend",
                "qwen2_5_vl",
                "--vlm_model",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "--vlm_device",
                "auto",
                "--vlm_max_new_tokens",
                "128",
                "--vlm_dtype",
                "bfloat16",
                "--clip_backend",
                "hf_clip",
                "--clip_model",
                "openai/clip-vit-base-patch32",
                *([] if not args.hf_endpoint else ["--hf_endpoint", args.hf_endpoint]),
            ]
        )
        run_command(
            [
                python_executable,
                str(PROJECT_ROOT / "stage4_prepare_manifest.py"),
                "--stage2_results_jsonl",
                args.train_stage2_results_jsonl,
                "--mask_dir",
                args.train_mask_dir,
                "--output_dir",
                args.train_output_dir,
                "--text_field",
                "training_text",
                "--drop_empty_masks",
                "--drop_low_quality_masks",
            ]
        )
        prepare_eval = [
            python_executable,
            str(PROJECT_ROOT / "stage4_prepare_eval_texts.py"),
            "--dataset_root",
            args.dataset_root,
            "--coarse_pred_root",
            args.coarse_pred_root,
            "--output_root",
            args.eval_output_root,
            "--vlm_backend",
            "qwen2_5_vl",
            "--vlm_model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--vlm_device",
            "auto",
            "--vlm_max_new_tokens",
            "128",
            "--vlm_dtype",
            "bfloat16",
            "--clip_backend",
            "none",
            "--text_field",
            "training_text",
            "--force",
        ]
        if args.hf_endpoint:
            prepare_eval.extend(["--hf_endpoint", args.hf_endpoint])
        run_command(prepare_eval)

    run_command(
        [
            python_executable,
            str(PROJECT_ROOT / "stage4_train_sinet_text.py"),
            "--train_manifest_jsonl",
            f"{args.train_output_dir}/manifest.jsonl",
            "--val_manifest_jsonl",
            f"{args.eval_output_root}/CAMO/manifest.jsonl",
            "--teacher_path",
            args.teacher_path,
            "--student_init",
            args.teacher_path,
            "--save_path",
            args.save_path,
            "--gpu_id",
            args.gpu_id,
            *([] if not args.disable_text else ["--disable_text"]),
        ]
    )
    run_command(
        [
            python_executable,
            str(PROJECT_ROOT / "stage4_test_sinet_text.py"),
            "--pth_path",
            f"{args.save_path}/Net_epoch_best.pth",
            "--eval_root",
            args.eval_output_root,
            "--save_dir",
            args.pred_root,
            "--gpu_id",
            args.gpu_id,
        ]
    )


if __name__ == "__main__":
    main()

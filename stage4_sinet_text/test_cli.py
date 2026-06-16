"""Testing CLI for Stage 4 SINet-text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from vis import EvaluationMetricsV2

from .dataset import SINetTextEvalDataset, build_text_collate_fn
from .model import SINetTextNetwork


def load_tokenizer(text_model_name: str) -> CLIPTokenizer:
    try:
        return CLIPTokenizer.from_pretrained(text_model_name, local_files_only=True)
    except OSError:
        return CLIPTokenizer.from_pretrained(text_model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Stage 4 SINet-text.")
    parser.add_argument("--pth_path", default="./snapshot/SINet_text_v2/Net_epoch_best.pth")
    parser.add_argument("--eval_root", default="Dataset/Stage4CRISLite_baseline/eval")
    parser.add_argument("--save_dir", default="./pred/SINet_text_v2")
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def configure_device(gpu_id: str) -> torch.device:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SINetTextNetwork, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = SINetTextNetwork(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval()
    return model, model_kwargs.get("text_model_name", "openai/clip-vit-base-patch32")


def main() -> None:
    args = parse_args()
    device = configure_device(args.gpu_id)
    model, text_model_name = load_model(Path(args.pth_path), device)
    tokenizer = load_tokenizer(text_model_name)
    collate_fn = build_text_collate_fn(tokenizer, args.max_text_length)

    eval_root = Path(args.eval_root)
    dataset_names = args.datasets or sorted(path.name for path in eval_root.iterdir() if path.is_dir())
    all_results: dict[str, dict] = {}

    for dataset_name in dataset_names:
        manifest_path = eval_root / dataset_name / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing eval manifest: {manifest_path}")

        dataset = SINetTextEvalDataset(str(manifest_path), testsize=args.testsize)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        metric = EvaluationMetricsV2()
        dataset_save_dir = Path(args.save_dir) / dataset_name
        dataset_save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Stage4 v2 test {dataset_name}", leave=False):
                preds = model(
                    images=batch["images"].to(device),
                    text_input_ids=batch["text_input_ids"].to(device),
                    text_attention_mask=batch["text_attention_mask"].to(device),
                    category_ids=batch["category_ids"].to(device),
                    location_ids=batch["location_ids"].to(device),
                    size_ids=batch["size_ids"].to(device),
                )
                probs = torch.sigmoid(preds[3]).cpu().numpy()

                for index, sample_id in enumerate(batch["sample_ids"]):
                    gt_path = batch["gt_paths"][index]
                    if gt_path is None:
                        raise FileNotFoundError(f"Missing gt_path for sample {sample_id} in {dataset_name}")
                    gt = np.asarray(Image.open(gt_path).convert("L"), np.float32)
                    pred = probs[index, 0]
                    pred = np.array(
                        Image.fromarray((pred * 255).astype(np.uint8)).resize(
                            (gt.shape[1], gt.shape[0]),
                            resample=Image.BILINEAR,
                        ),
                        np.float32,
                    )
                    pred /= 255.0
                    metric.step(pred=pred, gt=gt)
                    Image.fromarray((pred * 255).astype(np.uint8)).save(dataset_save_dir / f"{sample_id}.png")

        metric_dic = metric.get_results()
        result = {
            "sm": float(metric_dic["sm"]),
            "emMean": float(metric_dic["emMean"]),
            "emAdp": float(metric_dic["emAdp"]),
            "emMax": float(metric_dic["emMax"]),
            "fmMean": float(metric_dic["fmMean"]),
            "fmAdp": float(metric_dic["fmAdp"]),
            "fmMax": float(metric_dic["fmMax"]),
            "wfm": float(metric_dic["wfm"]),
            "mae": float(metric_dic["mae"]),
        }
        all_results[dataset_name] = result
        print(dataset_name)
        for key, value in result.items():
            print(f"{key}: {value}")

    summary_path = Path(args.save_dir) / "test_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

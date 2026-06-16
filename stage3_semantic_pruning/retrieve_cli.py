"""Run retrieval with refined Stage 3 prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .common import (
    DATASET_ROOT,
    build_transform,
    create_pair_index,
    ensure_float32,
    gather_samples,
    load_jsonl_records,
    load_dino_model,
    write_json,
)


AUG_TRANSFORMS = {
    "normal": lambda x: x,
    "r90": lambda x: TF.rotate(x, 90, expand=True),
    "r180": lambda x: TF.rotate(x, 180, expand=True),
    "r270": lambda x: TF.rotate(x, 270, expand=True),
    "hflip": lambda x: TF.hflip(x),
    "vflip": lambda x: TF.vflip(x),
}

INV_AUG_TRANSFORMS = {
    "normal": lambda x: x,
    "r90": lambda x: np.rot90(x, k=3),
    "r180": lambda x: np.rot90(x, k=2),
    "r270": lambda x: np.rot90(x, k=1),
    "hflip": lambda x: np.fliplr(x),
    "vflip": lambda x: np.flipud(x),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 retrieval with refined prototypes.")
    parser.add_argument("--top_k", type=int, default=512)
    parser.add_argument("--top_k_ratio", type=float, default=0.33)
    parser.add_argument("--min_top_k", type=int, default=256)
    parser.add_argument("--dino", default="vit-l14")
    parser.add_argument("--imgsz", type=int, default=476)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--faiss_device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--vote_mode", choices=["count", "weighted"], default="weighted")
    parser.add_argument("--foreground_weight_power", type=float, default=1.5)
    parser.add_argument("--foreground_vote_scale", type=float, default=1.0)
    parser.add_argument("--background_vote_weight", type=float, default=1.0)
    parser.add_argument("--image_dir", default=str(DATASET_ROOT / "TrainDataset" / "Image"))
    parser.add_argument("--prototype_dir", default="Dataset/Stage3Semantic_coarse/prototype_refined")
    parser.add_argument("--output_dir", default="Dataset/Stage3Semantic_coarse/pseudo_mask_refined")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _load_fore_weights(prototype_dir: Path, fore_count: int, power: float) -> np.ndarray:
    weights = np.ones(fore_count, dtype=np.float32)
    meta_path = prototype_dir / "fore_meta.jsonl"
    if not meta_path.exists():
        return weights

    meta_records = load_jsonl_records(meta_path)
    if len(meta_records) != fore_count:
        return weights

    for index, meta in enumerate(meta_records):
        weights[index] = max(float(meta.get("prototype_weight", 1.0)), 1e-3)

    if power != 1.0:
        weights = np.power(weights, power).astype(np.float32)
    mean_weight = float(np.mean(weights)) if weights.size else 1.0
    if mean_weight > 0.0:
        weights = weights / mean_weight
    return weights.astype(np.float32)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    prototype_dir = Path(args.prototype_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_fore_embed = ensure_float32(np.load(prototype_dir / "fore.npy"))
    pixel_back_embed = ensure_float32(np.load(prototype_dir / "back.npy"))
    if pixel_fore_embed.shape[0] == 0:
        raise RuntimeError("No refined foreground prototypes available for retrieval.")

    faiss.normalize_L2(pixel_fore_embed)
    faiss.normalize_L2(pixel_back_embed)
    total_proto_count = int(pixel_fore_embed.shape[0] + pixel_back_embed.shape[0])
    if args.top_k_ratio > 0:
        adaptive_top_k = max(args.min_top_k, int(round(pixel_fore_embed.shape[0] * args.top_k_ratio)))
        used_top_k = min(args.top_k, adaptive_top_k, total_proto_count)
    else:
        used_top_k = min(args.top_k, int(pixel_fore_embed.shape[0]), total_proto_count)
    used_top_k = max(1, used_top_k)
    index_device = args.faiss_device if used_top_k <= 2048 else "cpu"
    pixel_index = create_pair_index(pixel_fore_embed, pixel_back_embed, index_device)
    fore_weights = _load_fore_weights(
        prototype_dir=prototype_dir,
        fore_count=int(pixel_fore_embed.shape[0]),
        power=args.foreground_weight_power,
    )
    all_vote_weights = np.concatenate(
        [
            fore_weights * args.foreground_vote_scale,
            np.full(int(pixel_back_embed.shape[0]), args.background_vote_weight, dtype=np.float32),
        ],
        axis=0,
    )

    feat_h = int(args.imgsz / 14)
    embed_dim = int(pixel_fore_embed.shape[1])
    transform = build_transform(args.imgsz)
    dinov2 = load_dino_model(args.dino, args.device)

    samples = gather_samples(image_dir=image_dir, cluster_dir=None, stage2_records=None, limit=args.limit)
    for sample in tqdm(samples, desc="Stage3 retrieval"):
        image_path = Path(sample["image_path"])
        save_path = output_dir / f"{sample['sample_id']}.png"
        fusion_mask = None
        for aug_name, aug in AUG_TRANSFORMS.items():
            image = Image.open(image_path).convert("RGB")
            image = aug(image)
            width, height = image.size
            image_tensor = transform(image).unsqueeze(0).to(args.device)

            with torch.no_grad():
                feats = dinov2.get_intermediate_layers(image_tensor, reshape=True)[0]
            local_query = (
                feats.permute(2, 3, 1, 0)
                .reshape(feat_h * feat_h, embed_dim, -1)
                .contiguous()
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )
            faiss.normalize_L2(local_query)

            _, pixel_indices = pixel_index.search(local_query, used_top_k)
            is_fore = pixel_indices < pixel_fore_embed.shape[0]
            if args.vote_mode == "weighted":
                neighbor_weights = all_vote_weights[pixel_indices]
                fore_votes = np.sum(neighbor_weights * is_fore, axis=1)
                total_votes = np.sum(neighbor_weights, axis=1) + 1e-8
                pixel_mask = fore_votes / total_votes
            else:
                pixel_mask = np.sum(is_fore, axis=1) / used_top_k
            pixel_mask = pixel_mask.astype(np.float32).reshape(feat_h, feat_h)
            pixel_mask = (
                F.interpolate(
                    torch.tensor(pixel_mask).unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze()
                .numpy()
            )
            aug_mask = INV_AUG_TRANSFORMS[aug_name](pixel_mask)
            fusion_mask = aug_mask if fusion_mask is None else fusion_mask + aug_mask

        fusion_mask = fusion_mask / len(AUG_TRANSFORMS)
        fusion_mask = (fusion_mask > 0.5).astype(np.uint8)
        Image.fromarray(fusion_mask * 255, mode="L").save(save_path)

    summary = {
        "total_samples": len(samples),
        "requested_top_k": args.top_k,
        "top_k_ratio": args.top_k_ratio,
        "min_top_k": args.min_top_k,
        "used_top_k": used_top_k,
        "fore_prototype_count": int(pixel_fore_embed.shape[0]),
        "back_prototype_count": int(pixel_back_embed.shape[0]),
        "faiss_device": index_device,
        "vote_mode": args.vote_mode,
        "foreground_weight_power": args.foreground_weight_power,
        "foreground_vote_scale": args.foreground_vote_scale,
        "background_vote_weight": args.background_vote_weight,
        "fore_weight_mean": float(np.mean(fore_weights)) if fore_weights.size else 1.0,
        "fore_weight_min": float(np.min(fore_weights)) if fore_weights.size else 1.0,
        "fore_weight_max": float(np.max(fore_weights)) if fore_weights.size else 1.0,
        "prototype_dir": str(prototype_dir),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "retrieval_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

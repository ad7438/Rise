"""Build Stage 3 prototypes with per-row metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

from utils.adaptive_threshold import HIST_RISE

from .common import (
    DATASET_ROOT,
    DINO_DIM,
    WORKSPACE_ROOT,
    build_transform,
    cosine_similarity_batch,
    create_single_index,
    ensure_float32,
    gather_samples,
    load_cluster_fore_back,
    load_dino_model,
    load_rgb_image,
    load_stage2_records,
    maybe_float,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3 prototype bank with metadata.")
    parser.add_argument("--dino", default="vit-l14")
    parser.add_argument("--imgsz", type=int, default=476)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--faiss_device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--image_dir", default=str(DATASET_ROOT / "TrainDataset" / "Image"))
    parser.add_argument("--cluster_dir", default=str(WORKSPACE_ROOT / "cluster_map"))
    parser.add_argument("--stage2_results_jsonl", default=str(DATASET_ROOT / "Stage2PseudoText_full_v3_coarse" / "results.jsonl"))
    parser.add_argument("--output_dir", default=str(DATASET_ROOT / "Stage3Semantic_coarse" / "prototype_raw"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _extract_image_level_proto(dinov2, transform, image_path: Path, fore: np.ndarray, back: np.ndarray, device: str):
    image = load_rgb_image(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dinov2.get_intermediate_layers(image_tensor, reshape=True)[0]
    feats_np = feats.detach().cpu().numpy()
    fore_proto = np.sum(feats_np * fore, axis=(2, 3)) / (np.sum(fore) + 1e-8).astype(np.float32)
    back_proto = np.sum(feats_np * back, axis=(2, 3)) / (np.sum(back) + 1e-8).astype(np.float32)
    return feats, ensure_float32(fore_proto), ensure_float32(back_proto)


def _select_pixel_prototype(feats: torch.Tensor, fore: np.ndarray, back: np.ndarray, fore_proto: np.ndarray, back_proto: np.ndarray, faiss_device: str):
    new_feats = feats.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype("float32")
    pixel_fore_feats = ensure_float32(new_feats[fore == 1])
    pixel_back_feats = ensure_float32(new_feats[back == 1])

    faiss.normalize_L2(pixel_fore_feats)
    faiss.normalize_L2(pixel_back_feats)

    fore_index = create_single_index(pixel_fore_feats, faiss_device)
    back_index = create_single_index(pixel_back_feats, faiss_device)

    fore_query = ensure_float32(fore_proto.copy())
    back_query = ensure_float32(back_proto.copy())
    faiss.normalize_L2(fore_query)
    faiss.normalize_L2(back_query)

    fore_search_k = min(new_feats.shape[0], pixel_fore_feats.shape[0])
    back_search_k = min(new_feats.shape[0], pixel_back_feats.shape[0])
    _, fore_indices = fore_index.search(back_query, fore_search_k)
    _, back_indices = back_index.search(fore_query, back_search_k)

    fore_index_selected = int(fore_indices[0][-1])
    back_index_selected = int(back_indices[0][-1])

    fore_coords = np.argwhere(fore == 1)[fore_index_selected]
    back_coords = np.argwhere(back == 1)[back_index_selected]

    pixel_fore_proto = pixel_fore_feats[fore_index_selected : fore_index_selected + 1]
    pixel_back_proto = pixel_back_feats[back_index_selected : back_index_selected + 1]

    return {
        "pixel_fore_proto": pixel_fore_proto,
        "pixel_back_proto": pixel_back_proto,
        "fore_index_selected": fore_index_selected,
        "back_index_selected": back_index_selected,
        "fore_coords": fore_coords,
        "back_coords": back_coords,
        "fore_pixel_count": int(pixel_fore_feats.shape[0]),
        "back_pixel_count": int(pixel_back_feats.shape[0]),
        "fore_search_k": int(fore_search_k),
        "back_search_k": int(back_search_k),
    }


def _stage2_snapshot(record: dict) -> dict:
    return {
        "pseudo_text": record.get("pseudo_text"),
        "clip_text": record.get("clip_text"),
        "category": record.get("category"),
        "category_label_zh": record.get("category_label_zh"),
        "category_confidence": maybe_float(record.get("category_confidence")),
        "location": record.get("location"),
        "location_key": record.get("location_key"),
        "size": record.get("size"),
        "size_key": record.get("size_key"),
        "final_confidence": maybe_float(record.get("final_confidence")),
        "low_confidence": bool(record.get("low_confidence")),
        "low_confidence_reasons": list(record.get("low_confidence_reasons") or []),
        "mask_quality": maybe_float(record.get("mask_quality")),
        "mask_area_ratio": maybe_float(record.get("mask_area_ratio")),
        "mask_bbox": list(record.get("mask_bbox") or []),
        "mask_component_count": int(record.get("mask_component_count") or 0),
        "mask_kept_area_pixels": int(record.get("mask_kept_area_pixels") or 0),
        "mask_touches_edge": bool(record.get("mask_touches_edge")),
        "mask_is_empty": bool(record.get("mask_is_empty")),
        "mask_flags": list(record.get("mask_flags") or []),
        "clip_text_crop_score": record.get("clip_text_crop_score"),
        "clip_text_full_score": record.get("clip_text_full_score"),
        "processing_errors": list(record.get("processing_errors") or []),
    }


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    cluster_dir = Path(args.cluster_dir)
    stage2_records = load_stage2_records(Path(args.stage2_results_jsonl))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = gather_samples(image_dir=image_dir, cluster_dir=cluster_dir, stage2_records=stage2_records, limit=args.limit)
    if not samples:
        raise FileNotFoundError("No stage3 prototype samples found.")

    feat_h = int(args.imgsz / 14)
    embed_dim = DINO_DIM[args.dino]
    transform = build_transform(args.imgsz)
    dinov2 = load_dino_model(args.dino, args.device)

    valid_samples: list[dict] = []
    fore_proto_rows: list[np.ndarray] = []
    back_proto_rows: list[np.ndarray] = []
    skipped_degenerate = 0

    for sample in tqdm(samples, desc="Stage3 image-level proto"):
        fore, back, is_valid = load_cluster_fore_back(Path(sample["cluster_path"]), feat_h)
        if not is_valid:
            skipped_degenerate += 1
            continue
        _, fore_proto, back_proto = _extract_image_level_proto(
            dinov2=dinov2,
            transform=transform,
            image_path=Path(sample["image_path"]),
            fore=fore,
            back=back,
            device=args.device,
        )
        valid_samples.append(sample)
        fore_proto_rows.append(fore_proto)
        back_proto_rows.append(back_proto)

    if not valid_samples:
        raise RuntimeError("All candidate samples were degenerate after cluster-map validation.")

    image_level_fore = ensure_float32(np.concatenate(fore_proto_rows, axis=0))
    image_level_back = ensure_float32(np.concatenate(back_proto_rows, axis=0))
    cos_similarities = cosine_similarity_batch(image_level_fore, image_level_back)
    similarity_threshold = float(HIST_RISE(cos_similarities))

    pixel_fore_protos: list[np.ndarray] = []
    pixel_back_protos: list[np.ndarray] = []
    fore_meta: list[dict] = []
    back_meta: list[dict] = []
    skipped_by_similarity = 0

    for sample, cos_similarity in tqdm(
        list(zip(valid_samples, cos_similarities)),
        total=len(valid_samples),
        desc="Stage3 pixel-level proto",
    ):
        if float(cos_similarity) >= similarity_threshold:
            skipped_by_similarity += 1
            continue

        stage2_record = sample["stage2"]
        fore, back, _ = load_cluster_fore_back(Path(sample["cluster_path"]), feat_h)
        feats, fore_proto, back_proto = _extract_image_level_proto(
            dinov2=dinov2,
            transform=transform,
            image_path=Path(sample["image_path"]),
            fore=fore,
            back=back,
            device=args.device,
        )
        selection = _select_pixel_prototype(
            feats=feats,
            fore=fore,
            back=back,
            fore_proto=fore_proto,
            back_proto=back_proto,
            faiss_device=args.faiss_device,
        )

        prototype_index = len(pixel_fore_protos)
        pixel_fore_protos.append(selection["pixel_fore_proto"])
        pixel_back_protos.append(selection["pixel_back_proto"])

        common_meta = {
            "sample_id": sample["sample_id"],
            "image_path": str(sample["image_path"]),
            "cluster_path": str(sample["cluster_path"]),
            "image_level_cos_similarity": float(cos_similarity),
            "similarity_threshold": similarity_threshold,
            "feat_h": feat_h,
            "feat_w": feat_h,
            "prototype_index": prototype_index,
            "fore_pixel_count": selection["fore_pixel_count"],
            "back_pixel_count": selection["back_pixel_count"],
            "fore_search_k": selection["fore_search_k"],
            "back_search_k": selection["back_search_k"],
            **_stage2_snapshot(stage2_record),
        }
        fore_meta.append(
            {
                **common_meta,
                "prototype_role": "fore",
                "selected_patch_row": int(selection["fore_coords"][0]),
                "selected_patch_col": int(selection["fore_coords"][1]),
                "selected_patch_index_within_subset": selection["fore_index_selected"],
                "selected_patch_index_flat": int(selection["fore_coords"][0] * feat_h + selection["fore_coords"][1]),
            }
        )
        back_meta.append(
            {
                **common_meta,
                "prototype_role": "back",
                "selected_patch_row": int(selection["back_coords"][0]),
                "selected_patch_col": int(selection["back_coords"][1]),
                "selected_patch_index_within_subset": selection["back_index_selected"],
                "selected_patch_index_flat": int(selection["back_coords"][0] * feat_h + selection["back_coords"][1]),
            }
        )

    if not pixel_fore_protos:
        raise RuntimeError("No prototypes survived similarity filtering.")

    fore_array = ensure_float32(np.concatenate(pixel_fore_protos, axis=0))
    back_array = ensure_float32(np.concatenate(pixel_back_protos, axis=0))
    np.save(output_dir / "fore.npy", fore_array)
    np.save(output_dir / "back.npy", back_array)
    write_jsonl(output_dir / "fore_meta.jsonl", fore_meta)
    write_jsonl(output_dir / "back_meta.jsonl", back_meta)

    summary = {
        "total_candidate_samples": len(samples),
        "valid_cluster_samples": len(valid_samples),
        "skipped_degenerate_cluster_samples": skipped_degenerate,
        "skipped_by_similarity": skipped_by_similarity,
        "selected_prototype_count": int(fore_array.shape[0]),
        "dino": args.dino,
        "embed_dim": embed_dim,
        "imgsz": args.imgsz,
        "feat_h": feat_h,
        "faiss_device": args.faiss_device,
        "device": args.device,
        "stage2_results_jsonl": str(Path(args.stage2_results_jsonl)),
        "similarity_threshold": similarity_threshold,
        "image_level_cosine_mean": float(np.mean(cos_similarities)),
        "image_level_cosine_min": float(np.min(cos_similarities)),
        "image_level_cosine_max": float(np.max(cos_similarities)),
        "fore_shape": list(fore_array.shape),
        "back_shape": list(back_array.shape),
    }
    write_json(output_dir / "build_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

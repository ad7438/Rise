"""CLI for the standalone Stage 2 pseudo-text generation pipeline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .backends import build_vlm_backend
from .categories import category_label_zh
from .clip_scoring import build_clip_scorer
from .io_utils import build_summary, ensure_output_dirs, save_visuals, serialize_path, write_csv, write_jsonl
from .mask_utils import (
    create_highlight_overlay,
    crop_with_padding,
    derive_geometry_labels,
    load_binary_mask,
    load_rgb_image,
    preprocess_mask,
    resize_binary_mask,
)
from .prompting import build_category_prompt, compose_clip_text, compose_pseudo_text, compose_training_text
from .scoring import combine_confidence, evaluate_low_confidence
from .schema import VLMResult


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 pseudo text from RISE pseudo masks.")
    parser.add_argument("--image_dir", default="Dataset/TrainDataset/Image")
    parser.add_argument("--mask_dir", default="Dataset/RISE_Workspace/pseudo_mask")
    parser.add_argument("--output_dir", default="Dataset/Stage2PseudoText")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--small_area_threshold", type=float, default=0.03)
    parser.add_argument("--large_area_threshold", type=float, default=0.15)
    parser.add_argument("--min_component_area_pixels", type=int, default=64)
    parser.add_argument("--min_component_area_ratio", type=float, default=0.0005)
    parser.add_argument("--context_padding_ratio", type=float, default=0.15)
    parser.add_argument("--uncertain_threshold", type=float, default=0.5)
    parser.add_argument("--save_visuals", dest="save_visuals", action="store_true")
    parser.add_argument("--no_save_visuals", dest="save_visuals", action="store_false")
    parser.set_defaults(save_visuals=True)
    parser.add_argument(
        "--vlm_backend",
        choices=["mock", "json_lookup", "hf_vision2seq", "qwen2_5_vl"],
        default="mock",
    )
    parser.add_argument("--vlm_lookup_path", default=None)
    parser.add_argument("--vlm_model", default=None)
    parser.add_argument("--vlm_device", default="auto")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=128)
    parser.add_argument("--vlm_trust_remote_code", action="store_true")
    parser.add_argument("--vlm_dtype", default="auto")
    parser.add_argument("--vlm_attn_implementation", default="sdpa")
    parser.add_argument("--vlm_min_pixels", type=int, default=None)
    parser.add_argument("--vlm_max_pixels", type=int, default=None)
    parser.add_argument("--vlm_use_fast_processor", action="store_true")
    parser.add_argument("--clip_backend", choices=["none", "hf_clip"], default="none")
    parser.add_argument("--clip_model", default=None)
    parser.add_argument("--clip_device", default="auto")
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--hf_hub_download_timeout", type=int, default=None)
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=None)
    return parser.parse_args()


def find_image_for_mask(mask_path: Path, image_dir: Path) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{mask_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def gather_samples(image_dir: Path, mask_dir: Path, limit: int | None = None) -> List[Dict[str, Path]]:
    samples: List[Dict[str, Path]] = []
    for mask_path in sorted(mask_dir.glob("*")):
        if not mask_path.is_file():
            continue
        image_path = find_image_for_mask(mask_path, image_dir)
        if image_path is None:
            continue
        samples.append({"sample_id": Path(mask_path).stem, "image_path": image_path, "mask_path": mask_path})
        if limit is not None and len(samples) >= limit:
            break
    return samples


def build_empty_vlm_result() -> VLMResult:
    return VLMResult(
        category_key="unknown",
        category_confidence=0.0,
        raw_response='{"category":"unknown","category_confidence":0.0,"evidence":"empty mask"}',
        evidence="empty mask",
    )


def build_error_vlm_result(error_message: str) -> VLMResult:
    escaped = error_message.replace('"', "'")
    return VLMResult(
        category_key="unknown",
        category_confidence=0.0,
        raw_response=f'{{"category":"unknown","category_confidence":0.0,"evidence":"{escaped}"}}',
        evidence=error_message,
    )


def build_record(args: argparse.Namespace, sample: Dict[str, Path], paths: Dict[str, Path], vlm_backend, clip_scorer) -> dict:
    sample_id = str(sample["sample_id"])
    image = load_rgb_image(str(sample["image_path"]))
    raw_mask = load_binary_mask(str(sample["mask_path"]))
    raw_mask = resize_binary_mask(raw_mask, image.size)
    clean_mask, metrics = preprocess_mask(
        raw_mask,
        min_component_area_pixels=args.min_component_area_pixels,
        min_component_area_ratio=args.min_component_area_ratio,
    )
    geometry = derive_geometry_labels(metrics, args.small_area_threshold, args.large_area_threshold)
    overlay = create_highlight_overlay(image, clean_mask)
    tight_crop = crop_with_padding(image, metrics.bbox, 0.0)
    context_crop = crop_with_padding(image, metrics.bbox, args.context_padding_ratio)
    prompt = build_category_prompt(geometry.location_key, geometry.size_key)
    processing_errors = []
    if metrics.is_empty:
        vlm_result = build_empty_vlm_result()
    else:
        try:
            vlm_result = vlm_backend.describe_region([overlay, context_crop, tight_crop], prompt, sample_id)
        except Exception as exc:
            error_message = f"vlm_error: {type(exc).__name__}: {exc}"
            processing_errors.append(error_message)
            vlm_result = build_error_vlm_result(error_message)
    pseudo_text = compose_pseudo_text(vlm_result.category_key, geometry.location_label_zh, geometry.size_label_zh, vlm_result.category_confidence, args.uncertain_threshold)
    clip_text = compose_clip_text(
        vlm_result.category_key,
        geometry.location_key,
        geometry.size_key,
        vlm_result.category_confidence,
        args.uncertain_threshold,
    )
    training_text = compose_training_text(
        vlm_result.category_key,
        geometry.location_key,
        geometry.size_key,
        vlm_result.category_confidence,
        vlm_result.evidence,
        args.uncertain_threshold,
    )
    try:
        clip_scores = clip_scorer.score_text_images(clip_text, {"tight_crop": tight_crop, "overlay": overlay})
    except Exception as exc:
        processing_errors.append(f"clip_error: {type(exc).__name__}: {exc}")
        clip_scores = {"tight_crop": None, "overlay": None}
    final_confidence, normalized_scores = combine_confidence(metrics.score, vlm_result.category_confidence, clip_scores.get("tight_crop"), clip_scores.get("overlay"))
    low_confidence, low_confidence_reasons = evaluate_low_confidence(
        final_confidence,
        metrics.score,
        vlm_result.category_confidence,
        category_key=vlm_result.category_key,
        mask_flags=metrics.flags,
    )
    if processing_errors:
        low_confidence = True
        low_confidence_reasons = low_confidence_reasons + ["processing_error"]

    overlay_path = tight_path = context_path = clean_mask_path = None
    if args.save_visuals:
        overlay_path, tight_path, context_path, clean_mask_path = save_visuals(paths, sample_id, overlay, tight_crop, context_crop, clean_mask)

    return {
        "sample_id": sample_id,
        "image_path": str(sample["image_path"]),
        "mask_path": str(sample["mask_path"]),
        "clean_mask_path": serialize_path(clean_mask_path),
        "overlay_path": serialize_path(overlay_path),
        "tight_crop_path": serialize_path(tight_path),
        "context_crop_path": serialize_path(context_path),
        "category": vlm_result.category_key,
        "category_label_zh": category_label_zh(vlm_result.category_key),
        "category_confidence": vlm_result.category_confidence,
        "location": geometry.location_label_zh,
        "location_key": geometry.location_key,
        "size": geometry.size_label_zh,
        "size_key": geometry.size_key,
        "mask_quality": metrics.score,
        "mask_area_ratio": metrics.area_ratio,
        "mask_bbox": list(metrics.bbox),
        "mask_component_count": metrics.original_component_count,
        "mask_kept_area_pixels": metrics.kept_area_pixels,
        "mask_touches_edge": metrics.touches_edge,
        "mask_is_empty": metrics.is_empty,
        "mask_flags": metrics.flags,
        "clip_text_crop_score": clip_scores.get("tight_crop"),
        "clip_text_full_score": clip_scores.get("overlay"),
        "normalized_clip_text_crop_score": normalized_scores["clip_text_crop_score"],
        "normalized_clip_text_full_score": normalized_scores["clip_text_full_score"],
        "vlm_prompt": prompt,
        "vlm_raw_response": vlm_result.raw_response,
        "vlm_evidence": vlm_result.evidence,
        "pseudo_text": pseudo_text,
        "clip_text": clip_text,
        "training_text": training_text,
        "final_confidence": final_confidence,
        "low_confidence": low_confidence,
        "low_confidence_reasons": low_confidence_reasons,
        "processing_errors": processing_errors,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_hub_download_timeout is not None:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_hub_download_timeout)
    if args.hf_hub_etag_timeout is not None:
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.hf_hub_etag_timeout)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    samples = gather_samples(image_dir, mask_dir, args.limit)
    if not samples:
        raise FileNotFoundError(f"No matched samples found under {image_dir} and {mask_dir}")

    paths = ensure_output_dirs(Path(args.output_dir), args.save_visuals)
    vlm_backend = build_vlm_backend(
        backend_name=args.vlm_backend,
        model_name_or_path=args.vlm_model,
        lookup_path=args.vlm_lookup_path,
        device=args.vlm_device,
        max_new_tokens=args.vlm_max_new_tokens,
        trust_remote_code=args.vlm_trust_remote_code,
        torch_dtype=args.vlm_dtype,
        attn_implementation=args.vlm_attn_implementation,
        min_pixels=args.vlm_min_pixels,
        max_pixels=args.vlm_max_pixels,
        use_fast_processor=args.vlm_use_fast_processor,
    )
    clip_scorer = build_clip_scorer(args.clip_backend, args.clip_model, args.clip_device)
    records = [build_record(args, sample, paths, vlm_backend, clip_scorer) for sample in tqdm(samples, desc="Stage 2 pseudo text")]
    write_jsonl(records, paths["results_jsonl"])
    write_csv(records, paths["results_csv"])
    summary = build_summary(records)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to {paths['results_jsonl']}")


if __name__ == "__main__":
    main()

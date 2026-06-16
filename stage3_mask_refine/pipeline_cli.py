from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (
    PROJECT_ROOT,
    ensure_dirs,
    load_image_and_clean_mask,
    load_stage2_records,
    resolve_path,
    save_agsp_assets,
    save_svac_assets,
    save_svpm_assets,
    save_visual_assets,
    summarize,
    write_csv,
    write_json,
    write_jsonl,
)
from .agsp import build_agsp_prior
from .fuse import fuse_masks, is_edge_preserving_target
from .svac import build_svac_refined_mask
from .svpm import build_svpm_prior_with_debug
from .text_prior import CLIPSegTextPrior
from .visual_refine import adaptive_radii, visual_soft_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 text-guided mask refinement.")
    parser.add_argument("--stage2_results_jsonl", default="Dataset/Stage2PseudoText_full_v4_rich/results.jsonl")
    parser.add_argument("--output_dir", default="Dataset/Stage3MaskRefine_v1")
    parser.add_argument("--clipseg_model", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf_endpoint", default=None)
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--min_component_area_pixels", type=int, default=64)
    parser.add_argument("--min_component_area_ratio", type=float, default=0.0005)
    parser.add_argument("--drop_empty_from_stage4", action="store_true", default=True)
    parser.add_argument("--drop_low_quality_from_stage4", action="store_true", default=True)
    parser.add_argument("--use_agsp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--semantic_prior_mode",
        choices=["visual_only", "raw_clipseg", "agsp_no_mf0", "agsp_full"],
        default="agsp_full",
    )
    parser.add_argument("--anchor_radius", type=int, default=25)
    parser.add_argument("--anchor_blur", type=int, default=7)
    parser.add_argument("--mask_blur", type=int, default=5)
    parser.add_argument("--lambda_s", type=float, default=0.2)
    parser.add_argument("--save_agsp_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visual_prior_mode",
        choices=["grabcut", "svpm_m0_only", "svpm_full"],
        default="svpm_full",
    )
    parser.add_argument("--svpm_n_segments", type=int, default=300)
    parser.add_argument("--svpm_compactness", type=float, default=10.0)
    parser.add_argument("--svpm_dilate_radius", type=int, default=25)
    parser.add_argument("--svpm_alpha", type=float, default=0.6)
    parser.add_argument("--svpm_beta", type=float, default=0.4)
    parser.add_argument("--svpm_blur_ksize", type=int, default=5)
    parser.add_argument("--save_svpm_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--refine_module",
        choices=["old_dual_consistency", "svac"],
        default="old_dual_consistency",
    )
    parser.add_argument("--svac_base_radius", type=int, default=10)
    parser.add_argument("--svac_expand_radius", type=int, default=35)
    parser.add_argument("--svac_local_radius", type=int, default=7)
    parser.add_argument("--svac_visual_threshold", type=float, default=0.5)
    parser.add_argument("--svac_component_threshold", type=float, default=0.45)
    parser.add_argument("--svac_binarize_threshold", type=float, default=0.5)
    parser.add_argument("--svac_alpha_o", type=float, default=0.50)
    parser.add_argument("--svac_alpha_s", type=float, default=0.35)
    parser.add_argument("--svac_alpha_d", type=float, default=0.15)
    parser.add_argument("--svac_high_conf", type=float, default=0.75)
    parser.add_argument("--svac_mid_conf", type=float, default=0.55)
    parser.add_argument("--svac_high_weights", type=float, nargs=3, default=(0.25, 0.30, 0.45))
    parser.add_argument("--svac_mid_weights", type=float, nargs=3, default=(0.35, 0.35, 0.30))
    parser.add_argument("--svac_low_weights", type=float, nargs=3, default=(0.50, 0.35, 0.15))
    parser.add_argument(
        "--svac_score_mode",
        choices=["weighted_sum", "geometric_mean"],
        default="geometric_mean",
    )
    parser.add_argument(
        "--svac_fusion_mode",
        choices=["tiered", "confidence_modulated"],
        default="confidence_modulated",
    )
    parser.add_argument(
        "--svac_threshold_mode",
        choices=["fixed"],
        default="fixed",
    )
    parser.add_argument("--svac_use_anchor_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svac_use_semantic_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svac_use_spatial_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_svac_vis", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _text_for_record(record: dict) -> str:
    training_text = str(record.get("training_text") or "").strip()
    if training_text:
        return training_text
    clip_text = str(record.get("clip_text") or "").strip()
    if clip_text:
        return clip_text
    return str(record.get("pseudo_text") or "A camouflaged target.")


def _mean_on_mask(value: np.ndarray, mask: np.ndarray) -> float:
    region = mask > 0
    if not np.any(region):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)[region].mean())


def _svpm_candidate_mask(pv: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
    pv = np.clip(np.asarray(pv, dtype=np.float32), 0.0, 1.0)
    if float(pv.max()) <= 1e-6:
        return init_mask.astype(np.uint8)
    candidate = (pv >= 0.20).astype(np.uint8)
    if candidate.max() <= 0:
        candidate = (pv >= max(0.05, float(pv.max()) * 0.5)).astype(np.uint8)
    if candidate.max() <= 0:
        candidate = init_mask.astype(np.uint8)
    return candidate


def build_stage4_manifest(records: list[dict], output_path: Path) -> dict:
    manifest: list[dict] = []
    dropped = 0
    for record in records:
        if bool(record.get("dropped_from_stage4")):
            dropped += 1
            continue
        manifest.append(
            {
                "sample_id": record["sample_id"],
                "image_path": record["image_path"],
                "mask_path": record["refined_mask_path"],
                "pseudo_text": record.get("pseudo_text"),
                "clip_text": record.get("clip_text"),
                "training_text": record.get("training_text"),
                "category": record.get("category"),
                "location_key": record.get("location_key"),
                "size_key": record.get("size_key"),
                "final_confidence": record.get("final_confidence"),
                "low_confidence": record.get("low_confidence"),
                "low_confidence_reasons": record.get("low_confidence_reasons"),
                "refine_mode": record.get("refine_mode"),
                "change_ratio": record.get("change_ratio"),
            }
        )
    write_jsonl(output_path, manifest)
    return {
        "total_records": len(records),
        "kept_for_stage4": len(manifest),
        "dropped_for_stage4": dropped,
        "manifest_jsonl": str(output_path),
    }


def main() -> None:
    args = parse_args()
    if args.svpm_blur_ksize % 2 == 0:
        args.svpm_blur_ksize += 1
    svpm_weight_sum = float(args.svpm_alpha + args.svpm_beta)
    if svpm_weight_sum > 1e-6:
        args.svpm_alpha = float(args.svpm_alpha) / svpm_weight_sum
        args.svpm_beta = float(args.svpm_beta) / svpm_weight_sum
    else:
        args.svpm_alpha = 0.6
        args.svpm_beta = 0.4
    stage2_path = resolve_path(args.stage2_results_jsonl)
    output_dir = resolve_path(args.output_dir)
    records = load_stage2_records(stage2_path, limit=args.limit)
    if not records:
        raise FileNotFoundError(f"No Stage2 records found in {stage2_path}")

    effective_prior_mode = (
        "visual_only"
        if args.semantic_prior_mode == "visual_only"
        else (args.semantic_prior_mode if args.use_agsp else "raw_clipseg")
    )
    visual_only_mode = effective_prior_mode == "visual_only"
    paths = ensure_dirs(
        output_dir,
        save_visuals=args.save_visuals,
        save_agsp_vis=args.save_agsp_vis,
        save_svpm_vis=args.save_svpm_vis,
        save_svac_vis=args.save_svac_vis and args.refine_module == "svac",
    )
    text_prior_model = None
    if not visual_only_mode:
        text_prior_model = CLIPSegTextPrior(
            model_name=args.clipseg_model,
            device=args.device,
            hf_endpoint=args.hf_endpoint,
        )

    refined_records: list[dict] = []
    for record in tqdm(records, desc="Stage3 mask refine"):
        image, init_mask, metrics = load_image_and_clean_mask(
            record,
            min_component_area_pixels=args.min_component_area_pixels,
            min_component_area_ratio=args.min_component_area_ratio,
        )
        image_np = np.array(image)
        text = _text_for_record(record)

        if metrics.is_empty:
            ps_raw = np.zeros(init_mask.shape, dtype=np.float32)
            text_prior = np.zeros(init_mask.shape, dtype=np.float32)
            agsp_anchor = np.zeros(init_mask.shape, dtype=np.float32)
            agsp_mf0 = np.zeros(init_mask.shape, dtype=np.float32)
            svpm_superpixel_map = np.zeros(init_mask.shape, dtype=np.int32)
            svpm_local_region = np.zeros(init_mask.shape, dtype=np.float32)
            svpm_anchor_support = np.zeros(init_mask.shape, dtype=np.float32)
            svpm_semantic_support = np.zeros(init_mask.shape, dtype=np.float32)
            svac_pr = np.zeros(init_mask.shape, dtype=np.float32)
            svac_omega = np.zeros(init_mask.shape, dtype=np.uint8)
            svac_info = {
                "svac_base_region_B": np.zeros(init_mask.shape, dtype=np.uint8),
                "svac_expand_region_E": np.zeros(init_mask.shape, dtype=np.uint8),
                "svac_candidate_from_pv": np.zeros(init_mask.shape, dtype=np.uint8),
                "svac_retained_components_K": np.zeros(init_mask.shape, dtype=np.uint8),
                "svac_local_region_Omega": svac_omega,
                "svac_fused_prior_Pr": svac_pr,
            }
            vis_mask = init_mask.copy()
            vis_soft = np.zeros(init_mask.shape, dtype=np.float32)
            s_vis_grabcut = 0.0
            refined_mask = init_mask.copy()
            fuse_info = {
                "refine_mode": "skip_empty",
                "refine_submode": "skip_empty",
                "r_base": 0,
                "r_expand": 0,
                "kept_components": 0,
                "candidate_components": 0,
                "band": "low",
                "allow_text_expand": False,
                "change_ratio": 0.0,
                "init_area_pixels": 0,
                "refined_area_pixels": 0,
                "text_prior_mean": 0.0,
                "vis_mean": 0.0,
                "refine_module": args.refine_module,
            }
            fuse_info.update(svac_info)
        else:
            if visual_only_mode:
                ps_raw = np.zeros(init_mask.shape, dtype=np.float32)
            else:
                assert text_prior_model is not None
                ps_raw = text_prior_model.predict(image, text)
            text_prior, agsp_anchor, agsp_mf0 = build_agsp_prior(
                ps_raw=ps_raw,
                m0=init_mask,
                anchor_radius=args.anchor_radius,
                anchor_blur=args.anchor_blur,
                mask_blur=args.mask_blur,
                lambda_s=args.lambda_s,
                semantic_prior_mode=effective_prior_mode,
            )
            text_prior = np.asarray(text_prior, dtype=np.float32)
            agsp_anchor = np.asarray(agsp_anchor, dtype=np.float32)
            agsp_mf0 = np.asarray(agsp_mf0, dtype=np.float32)
            r_base, r_expand, _ = adaptive_radii(init_mask)
            edge_preserving = is_edge_preserving_target(init_mask, str(record.get("category") or "unknown"))
            svpm_superpixel_map = np.zeros(init_mask.shape, dtype=np.int32)
            svpm_local_region = np.zeros(init_mask.shape, dtype=np.float32)
            svpm_anchor_support = np.zeros(init_mask.shape, dtype=np.float32)
            svpm_semantic_support = np.zeros(init_mask.shape, dtype=np.float32)
            s_vis_grabcut = 0.0
            if args.visual_prior_mode == "grabcut":
                vis_mask, vis_soft = visual_soft_mask(
                    image_np,
                    init_mask,
                    r_base,
                    r_expand,
                    edge_preserving=edge_preserving,
                )
                s_vis_grabcut = _mean_on_mask(vis_soft, vis_mask)
            else:
                vis_soft, svpm_superpixel_map, svpm_debug = build_svpm_prior_with_debug(
                    image=image_np,
                    m0=init_mask,
                    ps_agsp=text_prior,
                    n_segments=args.svpm_n_segments,
                    compactness=args.svpm_compactness,
                    dilate_radius=args.svpm_dilate_radius,
                    alpha=args.svpm_alpha,
                    beta=args.svpm_beta,
                    blur_ksize=args.svpm_blur_ksize,
                    visual_prior_mode=args.visual_prior_mode,
                )
                vis_soft = np.asarray(vis_soft, dtype=np.float32)
                vis_mask = _svpm_candidate_mask(vis_soft, init_mask)
                svpm_local_region = svpm_debug.local_region
                svpm_anchor_support = svpm_debug.anchor_support
                svpm_semantic_support = svpm_debug.semantic_support
            if args.refine_module == "svac":
                semantic_confidence = (
                    float(record.get("mask_quality") or 0.0)
                    if visual_only_mode
                    else float(record.get("final_confidence") or 0.0)
                )
                refined_mask, svac_pr, svac_omega, fuse_info = build_svac_refined_mask(
                    m0=init_mask,
                    ps_agsp=text_prior,
                    pv=vis_soft,
                    semantic_confidence=semantic_confidence,
                    mf0=agsp_mf0,
                    base_radius=args.svac_base_radius,
                    expand_radius=args.svac_expand_radius,
                    local_radius=args.svac_local_radius,
                    visual_threshold=args.svac_visual_threshold,
                    component_threshold=args.svac_component_threshold,
                    binarize_threshold=args.svac_binarize_threshold,
                    alpha_o=args.svac_alpha_o,
                    alpha_s=args.svac_alpha_s,
                    alpha_d=args.svac_alpha_d,
                    high_conf=args.svac_high_conf,
                    mid_conf=args.svac_mid_conf,
                    high_weights=tuple(args.svac_high_weights),
                    mid_weights=tuple(args.svac_mid_weights),
                    low_weights=tuple(args.svac_low_weights),
                    score_mode=args.svac_score_mode,
                    fusion_mode=args.svac_fusion_mode,
                    use_anchor_score=args.svac_use_anchor_score,
                    use_semantic_score=args.svac_use_semantic_score,
                    use_spatial_score=args.svac_use_spatial_score,
                )
            else:
                refined_mask, fuse_info = fuse_masks(
                    init_mask=init_mask,
                    text_prior=text_prior,
                    vis_mask=vis_mask,
                    vis_soft=vis_soft,
                    final_confidence=float(record.get("final_confidence") or 0.0),
                    category=str(record.get("category") or "unknown"),
                    low_confidence=bool(record.get("low_confidence")),
                    low_confidence_reasons=list(record.get("low_confidence_reasons") or []),
                    edge_preserving=edge_preserving,
                )
                svac_pr = np.zeros(init_mask.shape, dtype=np.float32)
                svac_omega = np.zeros(init_mask.shape, dtype=np.uint8)
                fuse_info["refine_module"] = args.refine_module
                fuse_info.update(
                    {
                        "svac_base_region_B": np.zeros(init_mask.shape, dtype=np.uint8),
                        "svac_expand_region_E": np.zeros(init_mask.shape, dtype=np.uint8),
                        "svac_candidate_from_pv": np.zeros(init_mask.shape, dtype=np.uint8),
                        "svac_retained_components_K": np.zeros(init_mask.shape, dtype=np.uint8),
                        "svac_local_region_Omega": svac_omega,
                        "svac_fused_prior_Pr": svac_pr,
                    }
                )

        refined_mask_path = paths["mask_dir"] / f"{record['sample_id']}.png"
        Image.fromarray((refined_mask > 0).astype(np.uint8) * 255).save(refined_mask_path)

        visuals = {}
        if args.save_visuals:
            visuals = save_visual_assets(
                paths=paths,
                sample_id=record["sample_id"],
                image=image,
                init_mask=init_mask,
                refined_mask=refined_mask,
                text_prior=text_prior,
                vis_mask=vis_mask,
            )
        agsp_visuals = {}
        if args.save_agsp_vis:
            agsp_visuals = save_agsp_assets(
                paths=paths,
                sample_id=record["sample_id"],
                image=image,
                init_mask=init_mask,
                ps_raw=ps_raw,
                anchor_map=agsp_anchor,
                mf0=agsp_mf0,
                ps_agsp=text_prior,
                vis_mask=vis_mask,
                refined_mask=refined_mask,
            )
        svpm_visuals = {}
        if args.save_svpm_vis and args.visual_prior_mode.startswith("svpm"):
            svpm_visuals = save_svpm_assets(
                paths=paths,
                sample_id=record["sample_id"],
                image=image,
                init_mask=init_mask,
                ps_raw=ps_raw,
                ps_agsp=text_prior,
                superpixel_map=svpm_superpixel_map,
                local_region=svpm_local_region,
                anchor_support=svpm_anchor_support,
                semantic_support=svpm_semantic_support,
                pv_svpm=vis_soft,
                refined_mask=refined_mask,
            )
        svac_visuals = {}
        if args.save_svac_vis and args.refine_module == "svac":
            svac_visuals = save_svac_assets(
                paths=paths,
                sample_id=record["sample_id"],
                image=image,
                init_mask=init_mask,
                ps_raw=ps_raw,
                ps_agsp=text_prior,
                pv=vis_soft,
                base_region=fuse_info["svac_base_region_B"],
                expand_region=fuse_info["svac_expand_region_E"],
                candidate_components=fuse_info["svac_candidate_from_pv"],
                retained_components=fuse_info["svac_retained_components_K"],
                local_region=fuse_info["svac_local_region_Omega"],
                fused_prior=fuse_info["svac_fused_prior_Pr"],
                refined_mask=refined_mask,
            )

        dropped_from_stage4 = False
        stage4_drop_reasons: list[str] = []
        if args.drop_empty_from_stage4 and int(refined_mask.sum()) <= 0:
            dropped_from_stage4 = True
            stage4_drop_reasons.append("empty_refined_mask")
        if args.drop_low_quality_from_stage4 and (
            "low_mask_quality" in list(record.get("low_confidence_reasons") or [])
            or bool(record.get("mask_is_empty"))
        ):
            dropped_from_stage4 = True
            stage4_drop_reasons.append("stage2_low_mask_quality")

        refined_record = {
            "sample_id": record["sample_id"],
            "image_path": record["image_path"],
            "init_mask_path": record["mask_path"],
            "refined_mask_path": str(refined_mask_path.relative_to(PROJECT_ROOT)),
            "pseudo_text": record.get("pseudo_text"),
            "clip_text": record.get("clip_text"),
            "training_text": "" if visual_only_mode else text,
            "category": record.get("category"),
            "location_key": record.get("location_key"),
            "size_key": record.get("size_key"),
            "final_confidence": record.get("final_confidence"),
            "low_confidence": record.get("low_confidence"),
            "low_confidence_reasons": record.get("low_confidence_reasons"),
            "mask_quality": record.get("mask_quality"),
            "mask_is_empty": record.get("mask_is_empty"),
            "refine_module": fuse_info.get("refine_module", args.refine_module),
            "refine_mode": fuse_info["refine_mode"],
            "change_ratio": fuse_info["change_ratio"],
            "r_base": fuse_info["r_base"],
            "r_expand": fuse_info["r_expand"],
            "kept_components": fuse_info["kept_components"],
            "candidate_components": fuse_info["candidate_components"],
            "band": fuse_info["band"],
            "allow_text_expand": fuse_info["allow_text_expand"],
            "init_area_pixels": fuse_info["init_area_pixels"],
            "refined_area_pixels": fuse_info["refined_area_pixels"],
            "text_prior_mean": fuse_info["text_prior_mean"],
            "vis_mean": fuse_info["vis_mean"],
            "refine_submode": fuse_info.get("refine_submode", "normal"),
            "semantic_prior_mode": effective_prior_mode,
            "visual_only_mode": visual_only_mode,
            "anchor_radius": args.anchor_radius,
            "anchor_blur": args.anchor_blur,
            "mask_blur": args.mask_blur,
            "lambda_s": args.lambda_s,
            "ps_raw_mean": float(np.clip(ps_raw, 0.0, 1.0).mean()),
            "ps_agsp_mean": float(np.clip(text_prior, 0.0, 1.0).mean()),
            "anchor_mean": float(np.clip(agsp_anchor, 0.0, 1.0).mean()),
            "s_sem_raw": _mean_on_mask(ps_raw, refined_mask),
            "s_sem_agsp": _mean_on_mask(text_prior, refined_mask),
            "visual_prior_mode": args.visual_prior_mode,
            "svpm_n_segments": args.svpm_n_segments,
            "svpm_compactness": args.svpm_compactness,
            "svpm_dilate_radius": args.svpm_dilate_radius,
            "svpm_alpha": args.svpm_alpha,
            "svpm_beta": args.svpm_beta,
            "svpm_pv_mean": float(np.clip(vis_soft, 0.0, 1.0).mean()),
            "svpm_pv_max": float(np.clip(vis_soft, 0.0, 1.0).max()),
            "svpm_pv_min": float(np.clip(vis_soft, 0.0, 1.0).min()),
            "s_vis_svpm": _mean_on_mask(vis_soft, refined_mask) if args.visual_prior_mode.startswith("svpm") else "",
            "s_vis_grabcut_if_available": (
                _mean_on_mask(vis_soft, refined_mask) if args.visual_prior_mode == "grabcut" else ""
            ),
            "svac_num_candidate_components": fuse_info.get("svac_num_candidate_components", ""),
            "svac_refine_mode_detail": fuse_info.get("svac_refine_mode_detail", ""),
            "svac_num_retained_components": fuse_info.get("svac_num_retained_components", ""),
            "svac_mean_eta": fuse_info.get("svac_mean_eta", ""),
            "svac_max_eta": fuse_info.get("svac_max_eta", ""),
            "svac_min_eta": fuse_info.get("svac_min_eta", ""),
            "svac_mean_anchor_consistency": fuse_info.get("svac_mean_anchor_consistency", ""),
            "svac_mean_semantic_support": fuse_info.get("svac_mean_semantic_support", ""),
            "svac_mean_spatial_consistency": fuse_info.get("svac_mean_spatial_consistency", ""),
            "svac_area_ratio": fuse_info.get("svac_area_ratio", ""),
            "svac_pr_mean": fuse_info.get("svac_pr_mean", ""),
            "svac_pr_max": fuse_info.get("svac_pr_max", ""),
            "svac_mr_area": fuse_info.get("svac_mr_area", ""),
            "svac_omega_area": fuse_info.get("svac_omega_area", ""),
            "svac_w0": fuse_info.get("svac_w0", ""),
            "svac_wv": fuse_info.get("svac_wv", ""),
            "svac_ws": fuse_info.get("svac_ws", ""),
            "svac_score_mode": args.svac_score_mode,
            "svac_fusion_mode": args.svac_fusion_mode,
            "svac_threshold_mode": args.svac_threshold_mode,
            "svac_use_anchor_score": args.svac_use_anchor_score,
            "svac_use_semantic_score": args.svac_use_semantic_score,
            "svac_use_spatial_score": args.svac_use_spatial_score,
            "svac_score_terms": fuse_info.get("svac_score_terms", ""),
            "score_mode": fuse_info.get("score_mode", ""),
            "fusion_mode": fuse_info.get("fusion_mode", ""),
            "eta_mean": fuse_info.get("eta_mean", ""),
            "eta_max": fuse_info.get("eta_max", ""),
            "eta_min": fuse_info.get("eta_min", ""),
            "candidate_component_count": fuse_info.get("candidate_component_count", ""),
            "retained_component_count": fuse_info.get("retained_component_count", ""),
            "Omega_area": fuse_info.get("Omega_area", ""),
            "Mr_area": fuse_info.get("Mr_area", ""),
            "M0_area": fuse_info.get("M0_area", ""),
            "Mr_M0_iou": fuse_info.get("Mr_M0_iou", ""),
            "Mr_M0_change_ratio": fuse_info.get("Mr_M0_change_ratio", ""),
            "dropped_from_stage4": dropped_from_stage4,
            "stage4_drop_reasons": stage4_drop_reasons,
            **visuals,
            **agsp_visuals,
            **svpm_visuals,
            **svac_visuals,
        }
        refined_records.append(refined_record)

    write_jsonl(paths["results_jsonl"], refined_records)
    write_csv(paths["results_csv"], refined_records)
    summary = summarize(refined_records)
    summary["stage2_results_jsonl"] = str(stage2_path)
    summary["clipseg_model"] = args.clipseg_model
    summary["output_dir"] = str(output_dir)
    summary["semantic_prior_mode"] = effective_prior_mode
    summary["use_agsp"] = bool(args.use_agsp)
    summary["anchor_radius"] = args.anchor_radius
    summary["anchor_blur"] = args.anchor_blur
    summary["mask_blur"] = args.mask_blur
    summary["lambda_s"] = args.lambda_s
    summary["visual_prior_mode"] = args.visual_prior_mode
    summary["refine_module"] = args.refine_module
    summary["svpm_n_segments"] = args.svpm_n_segments
    summary["svpm_compactness"] = args.svpm_compactness
    summary["svpm_dilate_radius"] = args.svpm_dilate_radius
    summary["svpm_alpha"] = args.svpm_alpha
    summary["svpm_beta"] = args.svpm_beta
    summary["svpm_blur_ksize"] = args.svpm_blur_ksize
    summary["svac_base_radius"] = args.svac_base_radius
    summary["svac_expand_radius"] = args.svac_expand_radius
    summary["svac_local_radius"] = args.svac_local_radius
    summary["svac_visual_threshold"] = args.svac_visual_threshold
    summary["svac_component_threshold"] = args.svac_component_threshold
    summary["svac_binarize_threshold"] = args.svac_binarize_threshold
    summary["svac_score_mode"] = args.svac_score_mode
    summary["svac_fusion_mode"] = args.svac_fusion_mode
    summary["svac_threshold_mode"] = args.svac_threshold_mode
    summary["svac_use_anchor_score"] = bool(args.svac_use_anchor_score)
    summary["svac_use_semantic_score"] = bool(args.svac_use_semantic_score)
    summary["svac_use_spatial_score"] = bool(args.svac_use_spatial_score)
    write_json(paths["summary_json"], summary)

    manifest_summary = build_stage4_manifest(refined_records, paths["stage4_manifest"])
    write_json(paths["stage4_summary"], manifest_summary)

    print(summary)
    print(manifest_summary)


if __name__ == "__main__":
    main()

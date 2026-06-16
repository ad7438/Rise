#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"

"$PYTHON" -m stage3_mask_refine.pipeline_cli \
  --stage2_results_jsonl "$ROOT/Dataset/Stage2PseudoText_full_v4_rich/results.jsonl" \
  --output_dir "$ROOT/Dataset/Stage3MaskRefine_SVAC2_v1" \
  --semantic_prior_mode agsp_full \
  --use_agsp \
  --anchor_radius 25 \
  --anchor_blur 7 \
  --mask_blur 5 \
  --lambda_s 0.2 \
  --visual_prior_mode svpm_full \
  --svpm_n_segments 300 \
  --svpm_compactness 10 \
  --svpm_dilate_radius 25 \
  --svpm_alpha 0.6 \
  --svpm_beta 0.4 \
  --svpm_blur_ksize 5 \
  --refine_module svac \
  --svac_base_radius 10 \
  --svac_expand_radius 35 \
  --svac_local_radius 7 \
  --svac_visual_threshold 0.5 \
  --svac_component_threshold 0.45 \
  --svac_binarize_threshold 0.5 \
  --svac_alpha_o 0.50 \
  --svac_alpha_s 0.35 \
  --svac_alpha_d 0.15 \
  --svac_score_mode geometric_mean \
  --svac_fusion_mode confidence_modulated \
  --svac_threshold_mode fixed \
  --save_agsp_vis \
  --save_svpm_vis \
  --save_svac_vis

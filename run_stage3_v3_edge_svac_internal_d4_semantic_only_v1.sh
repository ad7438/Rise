#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REFINE_ROOT="$ROOT/Dataset/Stage3MaskRefine_SVAC_Internal_D4_semantic_only_v1"
GATE_ROOT="$ROOT/Dataset/Stage3AutoGate_SVAC_Internal_D4_semantic_only_v1"
STAGE1_DATA="$GATE_ROOT/TrainDatasetAuto"
STAGE1_REFINED_DATA="$GATE_ROOT/TrainDatasetAutoRefined"
STAGE2_DATA="$GATE_ROOT/TrainDatasetAutoRehearsal"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"
OLD_MASK_ROOT="$ROOT/Dataset/RISE_Workspace/pseudo_mask"
REFINED_MASK_ROOT="$REFINE_ROOT/pseudo_mask_refined"

EXP_NAME="stage3_v3_edge_svac_internal_d4_semantic_only_v1"
STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage1_auto"
STAGE2_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage2_rehearsal"
TEST_SAVE="$SINET_ROOT/pred/RISE_${EXP_NAME}_stage2_best"

mkdir -p "$STAGE1_SAVE" "$STAGE2_SAVE" "$TEST_SAVE"

cd "$ROOT"

$PYTHON -m stage3_mask_refine.pipeline_cli \
  --stage2_results_jsonl "$ROOT/Dataset/Stage2PseudoText_full_v4_rich/results.jsonl" \
  --output_dir "$REFINE_ROOT" \
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
  --svac_high_conf 0.75 \
  --svac_mid_conf 0.55 \
  --svac_high_weights 0.25 0.30 0.45 \
  --svac_mid_weights 0.35 0.35 0.30 \
  --svac_low_weights 0.50 0.35 0.15 \
  --svac_score_mode geometric_mean \
  --svac_fusion_mode confidence_modulated \
  --no-svac_use_anchor_score \
  --no-svac_use_spatial_score \
  --no-save_agsp_vis \
  --no-save_svpm_vis \
  --no-save_svac_vis \
  > "$ROOT/${EXP_NAME}_stage3_refine_out.log" \
  2> "$ROOT/${EXP_NAME}_stage3_refine_err.log"

$PYTHON "$ROOT/stage3_auto_gate_refined.py" \
  --results_jsonl "$REFINE_ROOT/results.jsonl" \
  --output_root "$GATE_ROOT" \
  --build_dataset \
  > "$ROOT/${EXP_NAME}_gate_out.log" \
  2> "$ROOT/${EXP_NAME}_gate_err.log"

$PYTHON "$ROOT/stage3_build_mixed_rehearsal_dataset.py" \
  --base_manifest "$STAGE1_DATA/train_manifest.csv" \
  --refined_manifest "$STAGE1_REFINED_DATA/train_manifest.csv" \
  --output_root "$STAGE2_DATA" \
  --extra_refined_fraction 0.30 \
  --overwrite \
  > "$ROOT/${EXP_NAME}_build_rehearsal_out.log" \
  2> "$ROOT/${EXP_NAME}_build_rehearsal_err.log"

cd "$SINET_ROOT"

COMMON_ARGS=(
  --gpu_id 0
  --test_dataset_root "$DEV_ROOT"
  --val_datasets CAMO,COD10K
  --best_mode joint_four_metrics
  --use_reliability_map
  --old_mask_root "$OLD_MASK_ROOT"
  --refined_mask_root "$REFINED_MASK_ROOT"
  --disagreement_weight 0.35
  --boundary_downweight 0.35
  --disable_gt_pepper
  --nc_weight 0.20
  --nc_q_early 2.0
  --nc_q_late 1.0
)

$PYTHON -u MyTrain_Val.py \
  --epoch 100 \
  --lr 1e-4 \
  --batchsize 16 \
  "${COMMON_ARGS[@]}" \
  --nc_q_switch 40 \
  --resume_last \
  --img_root "$STAGE1_DATA/Image" \
  --gt_root "$STAGE1_DATA/GT" \
  --save_path "$STAGE1_SAVE/" \
  > "$ROOT/${EXP_NAME}_stage1_train_out.log" \
  2> "$ROOT/${EXP_NAME}_stage1_train_err.log"

$PYTHON -u MyTrain_Val.py \
  --epoch 40 \
  --lr 5e-5 \
  --batchsize 16 \
  --load "$STAGE1_SAVE/Net_epoch_best.pth" \
  "${COMMON_ARGS[@]}" \
  --nc_q_switch 15 \
  --resume_last \
  --img_root "$STAGE2_DATA/Image" \
  --gt_root "$STAGE2_DATA/GT" \
  --save_path "$STAGE2_SAVE/" \
  > "$ROOT/${EXP_NAME}_stage2_train_out.log" \
  2> "$ROOT/${EXP_NAME}_stage2_train_err.log"

$PYTHON -u MyTesting.py \
  --pth_path "$STAGE2_SAVE/Net_epoch_best.pth" \
  --save_dir "$TEST_SAVE" \
  > "$ROOT/${EXP_NAME}_test_out.log" \
  2> "$ROOT/${EXP_NAME}_test_err.log"

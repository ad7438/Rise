#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

STAGE1_DATA="$ROOT/Dataset/Stage3AutoGate_v1/TrainDatasetAuto"
STAGE2_DATA="$ROOT/Dataset/Stage3AutoGate_v1/TrainDatasetAutoRehearsal"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"
OLD_MASK_ROOT="$ROOT/Dataset/RISE_Workspace/pseudo_mask"
REFINED_MASK_ROOT="$ROOT/Dataset/Stage3MaskRefine_v3_edge/pseudo_mask_refined"

EXP_NAME="stage3_v3_edge_robust_auto_rehearsal_v1"
STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage1_auto3370"
STAGE2_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage2_rehearsal"
TEST_SAVE="$SINET_ROOT/pred/RISE_${EXP_NAME}_stage2_best"

mkdir -p "$STAGE1_SAVE" "$STAGE2_SAVE" "$TEST_SAVE"

$PYTHON "$ROOT/stage3_build_mixed_rehearsal_dataset.py" \
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

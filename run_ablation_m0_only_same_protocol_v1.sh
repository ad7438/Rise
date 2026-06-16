#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

EXP_NAME="ablation_m0_only_same_protocol_v1"
OUT_ROOT="$ROOT/Dataset/Ablation_M0Only_SameProtocol_v1"
STAGE1_DATA="$OUT_ROOT/TrainDatasetM0Stage1"
STAGE2_DATA="$OUT_ROOT/TrainDatasetM0Stage2"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"
M0_ROOT="$ROOT/Dataset/RISE_Workspace/pseudo_mask"
REF_MANIFEST_STAGE1="$ROOT/Dataset/Stage3AutoGate_AGSP_SVPM_SVAC_v1/TrainDatasetAuto/train_manifest.csv"
REF_MANIFEST_STAGE2="$ROOT/Dataset/Stage3AutoGate_AGSP_SVPM_SVAC_v1/TrainDatasetAutoRehearsal/train_manifest.csv"

STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage1"
STAGE2_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage2"
TEST_SAVE="$SINET_ROOT/pred/RISE_${EXP_NAME}_stage2_best"

mkdir -p "$STAGE1_SAVE" "$STAGE2_SAVE" "$TEST_SAVE"

cd "$ROOT"

$PYTHON "$ROOT/stage3_build_m0_only_dataset.py" \
  --manifest "$REF_MANIFEST_STAGE1" \
  --output_root "$STAGE1_DATA" \
  --m0_root "$M0_ROOT" \
  --overwrite \
  > "$ROOT/${EXP_NAME}_build_stage1_out.log" \
  2> "$ROOT/${EXP_NAME}_build_stage1_err.log"

$PYTHON "$ROOT/stage3_build_m0_only_dataset.py" \
  --manifest "$REF_MANIFEST_STAGE2" \
  --output_root "$STAGE2_DATA" \
  --m0_root "$M0_ROOT" \
  --overwrite \
  > "$ROOT/${EXP_NAME}_build_stage2_out.log" \
  2> "$ROOT/${EXP_NAME}_build_stage2_err.log"

cd "$SINET_ROOT"

COMMON_ARGS=(
  --gpu_id 0
  --test_dataset_root "$DEV_ROOT"
  --val_datasets CAMO,COD10K
  --best_mode joint_four_metrics
  --use_reliability_map
  --old_mask_root "$M0_ROOT"
  --refined_mask_root "$M0_ROOT"
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

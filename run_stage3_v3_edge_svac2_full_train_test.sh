#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

EXP_NAME="stage3_v3_edge_svac2_full"
REFINE_ROOT="$ROOT/Dataset/Stage3MaskRefine_SVAC2_v1"
GATE_ROOT="$ROOT/Dataset/Stage3AutoGate_SVAC2_v1"
STAGE1_DATA="$GATE_ROOT/TrainDatasetAuto"
STAGE1_REFINED_DATA="$GATE_ROOT/TrainDatasetAutoRefined"
STAGE2_DATA="$GATE_ROOT/TrainDatasetAutoRehearsal"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"
OLD_MASK_ROOT="$ROOT/Dataset/RISE_Workspace/pseudo_mask"
REFINED_MASK_ROOT="$REFINE_ROOT/pseudo_mask_refined"

STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage1_auto"
STAGE2_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage2_rehearsal"
RESULT_ROOT="$ROOT/results/svac2_full"
PRED_SAVE="$RESULT_ROOT/predictions"
STAGE1_LOG_DIR="$ROOT/logs/stage1_svac2_full"
STAGE2_LOG_DIR="$ROOT/logs/stage2_svac2_full"

mkdir -p "$STAGE1_SAVE" "$STAGE2_SAVE" "$RESULT_ROOT" "$PRED_SAVE" "$STAGE1_LOG_DIR" "$STAGE2_LOG_DIR"

cd "$ROOT"

echo "[SVAC2] Build M* dataset with existing AutoGate selection logic"
$PYTHON "$ROOT/stage3_auto_gate_refined.py" \
  --results_jsonl "$REFINE_ROOT/results.jsonl" \
  --output_root "$GATE_ROOT" \
  --build_dataset \
  > "$RESULT_ROOT/auto_gate_out.log" \
  2> "$RESULT_ROOT/auto_gate_err.log"

echo "[SVAC2] Build Stage-2 rehearsal dataset"
$PYTHON "$ROOT/stage3_build_mixed_rehearsal_dataset.py" \
  --base_manifest "$STAGE1_DATA/train_manifest.csv" \
  --refined_manifest "$STAGE1_REFINED_DATA/train_manifest.csv" \
  --output_root "$STAGE2_DATA" \
  --extra_refined_fraction 0.30 \
  --overwrite \
  > "$RESULT_ROOT/build_rehearsal_out.log" \
  2> "$RESULT_ROOT/build_rehearsal_err.log"

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

echo "[SVAC2] Stage-1 training"
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
  > "$STAGE1_LOG_DIR/train_out.log" \
  2> "$STAGE1_LOG_DIR/train_err.log"

echo "[SVAC2] Stage-2 rehearsal fine-tuning"
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
  > "$STAGE2_LOG_DIR/train_out.log" \
  2> "$STAGE2_LOG_DIR/train_err.log"

echo "[SVAC2] Full test on CAMO/CHAMELEON/COD10K/NC4K"
$PYTHON -u MyTesting.py \
  --pth_path "$STAGE2_SAVE/Net_epoch_best.pth" \
  --save_dir "$PRED_SAVE" \
  > "$RESULT_ROOT/final_test_metrics.txt" \
  2> "$RESULT_ROOT/test_err.log"

cp "$STAGE1_LOG_DIR/train_out.log" "$RESULT_ROOT/stage1_train_out.log"
cp "$STAGE1_LOG_DIR/train_err.log" "$RESULT_ROOT/stage1_train_err.log"
cp "$STAGE2_LOG_DIR/train_out.log" "$RESULT_ROOT/stage2_train_out.log"
cp "$STAGE2_LOG_DIR/train_err.log" "$RESULT_ROOT/stage2_train_err.log"

echo "[SVAC2] Done"

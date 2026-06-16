#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

EXP_NAME="stage3_v3_edge_h4_matched_old_only_stage1_only_v1"
DATA_ROOT="$ROOT/Dataset/Stage3AutoGate_H4_matched_old_only_v1"
STAGE1_DATA="$DATA_ROOT/TrainDatasetAuto"
REF_MANIFEST="$ROOT/Dataset/Stage3AutoGate_AGSP_SVPM_SVAC_v1/TrainDatasetAuto/train_manifest.csv"
M0_ROOT="$ROOT/Dataset/RISE_Workspace/pseudo_mask"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"
STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_${EXP_NAME}_stage1_matched_old"
TEST_SAVE="$SINET_ROOT/pred/RISE_${EXP_NAME}_stage1_best"

mkdir -p "$STAGE1_SAVE" "$TEST_SAVE"
cd "$ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" || ! -f "$STAGE1_DATA/train_manifest.csv" ]]; then
  "$PYTHON" "$ROOT/stage3_build_m0_only_dataset.py" \
    --manifest "$REF_MANIFEST" \
    --output_root "$STAGE1_DATA" \
    --m0_root "$M0_ROOT" \
    --overwrite \
    > "$ROOT/${EXP_NAME}_build_matched_old_out.log" \
    2> "$ROOT/${EXP_NAME}_build_matched_old_err.log"
else
  echo "SKIP_BUILD=1 and $STAGE1_DATA/train_manifest.csv exists; reuse H4 matched old-only dataset." \
    > "$ROOT/${EXP_NAME}_build_matched_old_out.log"
  : > "$ROOT/${EXP_NAME}_build_matched_old_err.log"
fi

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

"$PYTHON" -u MyTrain_Val.py \
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

"$PYTHON" -u MyTesting.py \
  --pth_path "$STAGE1_SAVE/Net_epoch_best.pth" \
  --save_dir "$TEST_SAVE" \
  > "$ROOT/${EXP_NAME}_test_out.log" \
  2> "$ROOT/${EXP_NAME}_test_err.log"

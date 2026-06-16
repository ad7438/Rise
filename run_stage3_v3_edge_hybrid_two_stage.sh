#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

HYBRID_DATA="$ROOT/Dataset/TrainHybrid_stage3_v3_edge"
STAGE2_DATA="$ROOT/Dataset/TrainSelected_stage3_v3_edge"
DEV_ROOT="$ROOT/Dataset/DevMini_stage3_v3_edge"

STAGE1_SAVE="$SINET_ROOT/snapshot/RISE_stage3_v3_edge_hybrid_two_stage_stage1_hybrid3640"
STAGE2_SAVE="$SINET_ROOT/snapshot/RISE_stage3_v3_edge_hybrid_two_stage_stage2_selected1607"
TEST_SAVE="$SINET_ROOT/pred/RISE_stage3_v3_edge_hybrid_two_stage_stage2_best"

cd "$ROOT"

$PYTHON stage3_build_hybrid_train_subset.py \
  --train_pool_jsonl Dataset/Stage3MaskRefine_v3_edge/train_pool_auto_filtered/train_pool_results.jsonl \
  --selected_manifest_jsonl Dataset/TrainSelected_stage3_v3_edge/train_manifest.jsonl \
  --output_root Dataset/TrainHybrid_stage3_v3_edge

cd "$SINET_ROOT"

$PYTHON -u MyTrain_Val.py \
  --epoch 100 \
  --gpu_id 0 \
  --img_root "$HYBRID_DATA/Image" \
  --gt_root "$HYBRID_DATA/GT" \
  --test_dataset_root "$DEV_ROOT" \
  --val_datasets CAMO,COD10K \
  --best_mode joint_four_metrics \
  --save_path "$STAGE1_SAVE/" \
  > "$ROOT/stage3_v3_edge_hybrid_two_stage_stage1_train_out.log" \
  2> "$ROOT/stage3_v3_edge_hybrid_two_stage_stage1_train_err.log"

$PYTHON -u MyTrain_Val.py \
  --epoch 40 \
  --gpu_id 0 \
  --lr 5e-5 \
  --load "$STAGE1_SAVE/Net_epoch_best.pth" \
  --img_root "$STAGE2_DATA/Image" \
  --gt_root "$STAGE2_DATA/GT" \
  --test_dataset_root "$DEV_ROOT" \
  --val_datasets CAMO,COD10K \
  --best_mode joint_four_metrics \
  --save_path "$STAGE2_SAVE/" \
  > "$ROOT/stage3_v3_edge_hybrid_two_stage_stage2_train_out.log" \
  2> "$ROOT/stage3_v3_edge_hybrid_two_stage_stage2_train_err.log"

$PYTHON -u MyTesting.py \
  --pth_path "$STAGE2_SAVE/Net_epoch_best.pth" \
  --save_dir "$TEST_SAVE" \
  > "$ROOT/stage3_v3_edge_hybrid_two_stage_test_out.log" \
  2> "$ROOT/stage3_v3_edge_hybrid_two_stage_test_err.log"

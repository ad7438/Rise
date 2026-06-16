#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

SAVE_DIR="$SINET_ROOT/snapshot/RISE_stage3_v3_edge_hybrid_dropbadcod_stage1_30test_rerun"
OUT_LOG="$ROOT/stage3_v3_edge_hybrid_dropbadcod_stage1_30test_rerun_out.log"
ERR_LOG="$ROOT/stage3_v3_edge_hybrid_dropbadcod_stage1_30test_rerun_err.log"

mkdir -p "$SAVE_DIR"

cd "$SINET_ROOT"

$PYTHON -u MyTrain_Val.py \
  --epoch 30 \
  --gpu_id 0 \
  --img_root ../Dataset/TrainHybrid_stage3_v3_edge_dropbadcod/Image \
  --gt_root ../Dataset/TrainHybrid_stage3_v3_edge_dropbadcod/GT \
  --test_dataset_root ../Dataset/DevMini_stage3_v3_edge \
  --val_datasets CAMO,COD10K \
  --best_mode joint_four_metrics \
  --save_path "$SAVE_DIR/" \
  > "$OUT_LOG" \
  2> "$ERR_LOG"

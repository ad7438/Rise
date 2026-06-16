#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

EXP_NAME="stage3_v3_edge_training_e4_stage1_only_v1"
STAGE1_CKPT="$SINET_ROOT/snapshot/RISE_stage3_v3_edge_agsp_svpm_svac_v1_stage1_auto/Net_epoch_best.pth"
TEST_SAVE="$SINET_ROOT/pred/RISE_${EXP_NAME}_stage1_best"

mkdir -p "$TEST_SAVE"

cd "$SINET_ROOT"

$PYTHON -u MyTesting.py \
  --pth_path "$STAGE1_CKPT" \
  --save_dir "$TEST_SAVE" \
  > "$ROOT/${EXP_NAME}_test_out.log" \
  2> "$ROOT/${EXP_NAME}_test_err.log"

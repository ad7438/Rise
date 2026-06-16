#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ad/Rise/RISE-master"
SINET_ROOT="$ROOT/SINet-V2"
PYTHON="/home/ad/miniconda3/envs/rise/bin/python"

J1_CKPT="$SINET_ROOT/snapshot/RISE_ablation_m0_only_same_protocol_v1_stage1/Net_epoch_best.pth"
J2_CKPT="$SINET_ROOT/snapshot/RISE_stage3_v3_edge_agsp_svpm_svac_v1_stage1_auto/Net_epoch_best.pth"

J1_SAVE="$SINET_ROOT/pred/RISE_j1_rise_local_m0_stage1_fair_best"
J2_SAVE="$SINET_ROOT/pred/RISE_j2_ours_agsp_svpm_svac_stage1_final_best"

mkdir -p "$J1_SAVE" "$J2_SAVE"

cd "$SINET_ROOT"

"$PYTHON" -u MyTesting.py \
  --pth_path "$J1_CKPT" \
  --save_dir "$J1_SAVE" \
  > "$ROOT/j1_rise_local_m0_stage1_fair_test_out.log" \
  2> "$ROOT/j1_rise_local_m0_stage1_fair_test_err.log"

"$PYTHON" -u MyTesting.py \
  --pth_path "$J2_CKPT" \
  --save_dir "$J2_SAVE" \
  > "$ROOT/j2_ours_agsp_svpm_svac_stage1_final_test_out.log" \
  2> "$ROOT/j2_ours_agsp_svpm_svac_stage1_final_test_err.log"

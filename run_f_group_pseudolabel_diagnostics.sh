#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/ad/miniconda3/envs/rise/bin/python}"

"$PYTHON_BIN" stage3_f_group_pseudolabel_diagnostics.py \
  --decisions_csv Dataset/Stage3AutoGate_AGSP_SVPM_SVAC_v1/auto_gate_decisions.csv \
  --gt_root Dataset/TrainDataset/GT \
  --output_dir outputs/paper_ablation_metrics/F_group_pseudolabel_diagnostics

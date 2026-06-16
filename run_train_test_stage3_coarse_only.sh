#!/usr/bin/env bash
set -euo pipefail

cd /home/ad/Rise/RISE-master/SINet-V2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rise

python -u MyTrain_Val.py \
  --img_root ../Dataset/TrainDataset/Image/ \
  --gt_root ../Dataset/Stage3Semantic_coarse/pseudo_mask_refined/ \
  --val_root ../Dataset/TestDataset/CAMO/ \
  --save_path ./snapshot/RISE_stage3_coarse/

python -u MyTesting.py \
  --pth_path ./snapshot/RISE_stage3_coarse/Net_epoch_best.pth \
  --data_path ../Dataset/TestDataset \
  --save_dir ./pred/RISE_stage3_coarse

#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rise

cd /home/ad/Rise/RISE-master/SINet-V2

python -u MyTrain_Val.py \
  --epoch 100 \
  --gpu_id 0 \
  --img_root /home/ad/Rise/RISE-master/Dataset/TrainDataset/Image \
  --gt_root /home/ad/Rise/RISE-master/Dataset/Stage3MaskRefine_v3_edge/pseudo_mask_refined \
  --test_dataset_root /home/ad/Rise/RISE-master/Dataset/DevMini_stage3_v3_edge \
  --val_datasets CAMO,COD10K \
  --best_mode joint_four_metrics \
  --save_path /home/ad/Rise/RISE-master/SINet-V2/snapshot/RISE_stage3_v3_edge_100_devmini_full4040/

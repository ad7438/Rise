#!/usr/bin/env bash
set -euo pipefail

cd /home/ad/Rise/RISE-master/SINet-V2

/home/ad/miniconda3/envs/rise/bin/python -u /home/ad/Rise/RISE-master/SINet-V2/MyTrain_Val.py \
  --epoch 200 \
  --gpu_id 0 \
  --gt_root /home/ad/Rise/RISE-master/Dataset/Stage3MaskRefine_v3_edge/pseudo_mask_refined \
  --test_dataset_root /home/ad/Rise/RISE-master/Dataset/TestDataset \
  --val_datasets CAMO,COD10K \
  --best_mode joint_sm_mae \
  --save_path /home/ad/Rise/RISE-master/SINet-V2/snapshot/RISE_stage3_mask_refine_v3_edge_200_joint/ \
  > /home/ad/Rise/RISE-master/stage3_mask_refine_v3_edge_200_joint_train_out.log \
  2> /home/ad/Rise/RISE-master/stage3_mask_refine_v3_edge_200_joint_train_err.log

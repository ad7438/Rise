#!/usr/bin/env bash
set -euo pipefail

cd /home/ad/Rise/RISE-master/SINet-V2

/home/ad/miniconda3/envs/rise/bin/python -u /home/ad/Rise/RISE-master/SINet-V2/MyTrain_Val.py \
  --epoch 200 \
  --gpu_id 0 \
  --img_root /home/ad/Rise/RISE-master/Dataset/TrainSelected_stage3_v3_edge/Image \
  --gt_root /home/ad/Rise/RISE-master/Dataset/TrainSelected_stage3_v3_edge/GT \
  --test_dataset_root /home/ad/Rise/RISE-master/Dataset/DevMini_stage3_v3_edge \
  --val_datasets CAMO,COD10K \
  --best_mode joint_four_metrics \
  --save_path /home/ad/Rise/RISE-master/SINet-V2/snapshot/RISE_stage3_v3_edge_200_devmini_selected/ \
  > /home/ad/Rise/RISE-master/stage3_v3_edge_200_devmini_selected_train_out.log \
  2> /home/ad/Rise/RISE-master/stage3_v3_edge_200_devmini_selected_train_err.log

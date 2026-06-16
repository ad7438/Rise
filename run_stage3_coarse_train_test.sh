#!/usr/bin/env bash
set -euo pipefail

cd /home/ad/Rise/RISE-master
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rise

python -u stage3_run_pipeline.py \
  --stage3_root Dataset/Stage3Semantic_coarse \
  --stage2_results_jsonl Dataset/Stage2PseudoText_full_v3_coarse/results.jsonl \
  --image_dir Dataset/TrainDataset/Image \
  --cluster_dir Dataset/RISE_Workspace/cluster_map \
  --device cuda \
  --faiss_device cpu \
  --top_k 512 \
  --min_keep 512

cd /home/ad/Rise/RISE-master/SINet-V2

python -u MyTrain_Val.py \
  --img_root ../Dataset/TrainDataset/Image/ \
  --gt_root ../Dataset/Stage3Semantic_coarse/pseudo_mask_refined/ \
  --save_path ./snapshot/RISE_stage3_coarse/

python -u MyTesting.py \
  --pth_path ./snapshot/RISE_stage3_coarse/Net_epoch_best.pth \
  --data_path ../Dataset/TestDataset \
  --save_dir ./pred/RISE_stage3_coarse

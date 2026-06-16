#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p Dataset/NonPaperSplitBackup/TrainDataset/Image
mkdir -p Dataset/NonPaperSplitBackup/TrainDataset/GT
mkdir -p Dataset/NonPaperSplitBackup/TestDataset/COD10K/Image
mkdir -p Dataset/NonPaperSplitBackup/TestDataset/COD10K/GT
mkdir -p Dataset/NonPaperSplitBackup/RISE_Workspace/cluster_map
mkdir -p Dataset/NonPaperSplitBackup/RISE_Workspace/pseudo_mask
mkdir -p Dataset/NonPaperSplitBackup/RISE_Workspace/prototype
mkdir -p Dataset/RISE_Workspace/cluster_map
mkdir -p Dataset/RISE_Workspace/pseudo_mask
mkdir -p Dataset/RISE_Workspace/prototype

move_noncam() {
    local src_image_dir="$1"
    local src_gt_dir="$2"
    local dst_image_dir="$3"
    local dst_gt_dir="$4"
    local img

    shopt -s nullglob
    for img in "${src_image_dir}"/COD10K-NonCAM-*; do
        [ -f "${img}" ] || continue
        local name="${img##*/}"
        local stem="${name%.*}"
        if [ -f "${src_gt_dir}/${stem}.png" ]; then
            mv -n "${src_gt_dir}/${stem}.png" "${dst_gt_dir}/"
        fi
        mv -n "${img}" "${dst_image_dir}/"
    done
}

move_noncam_masks() {
    local src_dir="$1"
    local dst_dir="$2"
    local img

    shopt -s nullglob
    for img in "${src_dir}"/COD10K-NonCAM-*.png; do
        [ -f "${img}" ] || continue
        mv -n "${img}" "${dst_dir}/"
    done
}

move_all_files() {
    local src_dir="$1"
    local dst_dir="$2"
    local f

    shopt -s nullglob
    for f in "${src_dir}"/*; do
        [ -f "${f}" ] || continue
        mv -n "${f}" "${dst_dir}/"
    done
}

move_noncam \
    "Dataset/TrainDataset/Image" \
    "Dataset/TrainDataset/GT" \
    "Dataset/NonPaperSplitBackup/TrainDataset/Image" \
    "Dataset/NonPaperSplitBackup/TrainDataset/GT"

move_noncam \
    "Dataset/TestDataset/COD10K/Image" \
    "Dataset/TestDataset/COD10K/GT" \
    "Dataset/NonPaperSplitBackup/TestDataset/COD10K/Image" \
    "Dataset/NonPaperSplitBackup/TestDataset/COD10K/GT"

move_noncam_masks \
    "Dataset/RISE_Workspace/cluster_map" \
    "Dataset/NonPaperSplitBackup/RISE_Workspace/cluster_map"

move_all_files \
    "Dataset/RISE_Workspace/pseudo_mask" \
    "Dataset/NonPaperSplitBackup/RISE_Workspace/pseudo_mask"

move_all_files \
    "Dataset/RISE_Workspace/prototype" \
    "Dataset/NonPaperSplitBackup/RISE_Workspace/prototype"

printf "TrainDataset/Image: "
find Dataset/TrainDataset/Image -maxdepth 1 -type f | wc -l
printf "TrainDataset/GT: "
find Dataset/TrainDataset/GT -maxdepth 1 -type f | wc -l
printf "TestDataset/COD10K/Image: "
find Dataset/TestDataset/COD10K/Image -maxdepth 1 -type f | wc -l
printf "TestDataset/COD10K/GT: "
find Dataset/TestDataset/COD10K/GT -maxdepth 1 -type f | wc -l
printf "RISE cluster_map: "
find Dataset/RISE_Workspace/cluster_map -maxdepth 1 -type f | wc -l
printf "RISE pseudo_mask: "
find Dataset/RISE_Workspace/pseudo_mask -maxdepth 1 -type f | wc -l
printf "RISE prototype: "
find Dataset/RISE_Workspace/prototype -maxdepth 1 -type f | wc -l
printf "Backup Train NonCAM/Image: "
find Dataset/NonPaperSplitBackup/TrainDataset/Image -maxdepth 1 -type f | wc -l
printf "Backup Test NonCAM/Image: "
find Dataset/NonPaperSplitBackup/TestDataset/COD10K/Image -maxdepth 1 -type f | wc -l
printf "Backup cluster_map NonCAM: "
find Dataset/NonPaperSplitBackup/RISE_Workspace/cluster_map -maxdepth 1 -type f | wc -l
printf "Backup pseudo_mask old split: "
find Dataset/NonPaperSplitBackup/RISE_Workspace/pseudo_mask -maxdepth 1 -type f | wc -l
printf "Backup prototype old split: "
find Dataset/NonPaperSplitBackup/RISE_Workspace/prototype -maxdepth 1 -type f | wc -l
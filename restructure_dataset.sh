#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p Dataset/TrainDataset/Image
mkdir -p Dataset/TrainDataset/GT
mkdir -p Dataset/TestDataset/CAMO/Image
mkdir -p Dataset/TestDataset/CAMO/GT
mkdir -p Dataset/TestDataset/COD10K/Image
mkdir -p Dataset/TestDataset/COD10K/GT
mkdir -p Dataset/TestDataset/NC4K/Image
mkdir -p Dataset/TestDataset/NC4K/GT
mkdir -p Dataset/TestDataset/CHAMELEON/Image
mkdir -p Dataset/TestDataset/CHAMELEON/GT
mkdir -p Dataset/RISE_Workspace/cluster_map
mkdir -p Dataset/RISE_Workspace/prototype
mkdir -p Dataset/RISE_Workspace/pseudo_mask

shopt -s nullglob

move_split() {
    local image_glob="$1"
    local src_gt_dir="$2"
    local dst_image_dir="$3"
    local dst_gt_dir="$4"
    local img

    for img in ${image_glob}; do
        [ -f "${img}" ] || continue
        local name="${img##*/}"
        local stem="${name%.*}"
        if [ -f "${src_gt_dir}/${stem}.png" ]; then
            mv -n "${src_gt_dir}/${stem}.png" "${dst_gt_dir}/"
        fi
        mv -n "${img}" "${dst_image_dir}/"
    done
}

move_split "Dataset/CAMO/Imgs/Train/*" "Dataset/CAMO/GT" "Dataset/TrainDataset/Image" "Dataset/TrainDataset/GT"
move_split "Dataset/CAMO/Imgs/Test/*" "Dataset/CAMO/GT" "Dataset/TestDataset/CAMO/Image" "Dataset/TestDataset/CAMO/GT"
move_split "Dataset/COD10K/Train/Image/COD10K-CAM-*" "Dataset/COD10K/Train/GT_Object" "Dataset/TrainDataset/Image" "Dataset/TrainDataset/GT"
move_split "Dataset/COD10K/Test/Image/COD10K-CAM-*" "Dataset/COD10K/Test/GT_Object" "Dataset/TestDataset/COD10K/Image" "Dataset/TestDataset/COD10K/GT"
move_split "Dataset/NC4K/Imgs/*" "Dataset/NC4K/GT" "Dataset/TestDataset/NC4K/Image" "Dataset/TestDataset/NC4K/GT"
move_split "Dataset/CHAMELEON/Imgs/*" "Dataset/CHAMELEON/GT" "Dataset/TestDataset/CHAMELEON/Image" "Dataset/TestDataset/CHAMELEON/GT"

rmdir --ignore-fail-on-non-empty Dataset/CAMO/Imgs/Train || true
rmdir --ignore-fail-on-non-empty Dataset/CAMO/Imgs/Test || true
rmdir --ignore-fail-on-non-empty Dataset/CAMO/Imgs || true
rmdir --ignore-fail-on-non-empty Dataset/CAMO/GT || true
rmdir --ignore-fail-on-non-empty Dataset/COD10K/Train/Image || true
rmdir --ignore-fail-on-non-empty Dataset/COD10K/Train/GT_Object || true
rmdir --ignore-fail-on-non-empty Dataset/COD10K/Test/Image || true
rmdir --ignore-fail-on-non-empty Dataset/COD10K/Test/GT_Object || true
rmdir --ignore-fail-on-non-empty Dataset/NC4K/Imgs || true
rmdir --ignore-fail-on-non-empty Dataset/NC4K/GT || true
rmdir --ignore-fail-on-non-empty Dataset/CHAMELEON/Imgs || true
rmdir --ignore-fail-on-non-empty Dataset/CHAMELEON/GT || true

echo "Dataset migration complete."
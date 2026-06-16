"""Shared helpers for Stage 3 semantic prototype pruning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from hubconf import dinov2_vitb14, dinov2_vitl14, dinov2_vits14


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
WORKSPACE_ROOT = DATASET_ROOT / "RISE_Workspace"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DINO_DIM = {
    "vit-s14": 384,
    "vit-b14": 768,
    "vit-l14": 1024,
}


def build_transform(imgsz: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dino_model(dino: str, device: str):
    if dino == "vit-s14":
        return dinov2_vits14().to(device)
    if dino == "vit-b14":
        return dinov2_vitb14().to(device)
    if dino == "vit-l14":
        return dinov2_vitl14().to(device)
    raise ValueError(f"Unsupported DINO backbone: {dino}")


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b + 1e-8)


def _create_gpu_flat_index(dim: int):
    resources = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(dim)
    return faiss.index_cpu_to_gpu(resources, 0, index_flat)


def create_single_index(embeddings: np.ndarray, device: str) -> faiss.Index:
    if device == "cuda":
        index = _create_gpu_flat_index(int(embeddings.shape[1]))
    elif device == "cpu":
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    else:
        raise ValueError(f"Unsupported faiss device: {device}")
    index.add(embeddings)
    return index


def create_pair_index(fore: np.ndarray, back: np.ndarray, device: str) -> faiss.Index:
    if fore.shape[1] != back.shape[1]:
        raise ValueError("Foreground and background prototype dimensions do not match.")
    if device == "cuda":
        index = _create_gpu_flat_index(int(fore.shape[1]))
    elif device == "cpu":
        index = faiss.IndexFlatIP(int(fore.shape[1]))
    else:
        raise ValueError(f"Unsupported faiss device: {device}")
    index.add(fore)
    index.add(back)
    return index


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_stage2_records(path: Path) -> dict[str, dict]:
    return {record["sample_id"]: record for record in load_jsonl_records(path)}


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[dict]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_cluster_fore_back(cluster_path: Path, feat_h: int) -> tuple[np.ndarray, np.ndarray, bool]:
    mask = Image.open(cluster_path).convert("L").resize((feat_h, feat_h))
    mask_array = np.array(mask)
    fore = (mask_array / 255.0 > 0.5).astype(np.float32)
    back = 1.0 - fore
    is_valid = bool(fore.mean() > 0.0 and fore.mean() < 1.0)
    return fore, back, is_valid


def gather_samples(
    image_dir: Path,
    cluster_dir: Path | None = None,
    stage2_records: dict[str, dict] | None = None,
    limit: int | None = None,
) -> List[dict]:
    samples: List[dict] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sample_id = image_path.stem
        cluster_path = None
        if cluster_dir is not None:
            cluster_path = cluster_dir / f"{sample_id}.png"
            if not cluster_path.exists():
                continue
        if stage2_records is not None and sample_id not in stage2_records:
            continue
        sample = {
            "sample_id": sample_id,
            "image_path": image_path,
        }
        if cluster_path is not None:
            sample["cluster_path"] = cluster_path
        if stage2_records is not None:
            sample["stage2"] = stage2_records[sample_id]
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def ensure_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def maybe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

"""Datasets and collators for Stage 4 SINet-text."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image, ImageEnhance

from .common import CATEGORY_TO_ID, LOCATION_HFLIP, LOCATION_TO_ID, SIZE_TO_ID, category_id, load_jsonl_records, location_id, maybe_float, size_id


def _rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        return Image.open(handle).convert("RGB")


def _binary_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        return Image.open(handle).convert("L")


def _random_flip(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, bool]:
    flipped = random.random() < 0.5
    if flipped:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask, flipped


def _color_enhance(image: Image.Image) -> Image.Image:
    bright = random.randint(5, 15) / 10.0
    contrast = random.randint(5, 15) / 10.0
    color = random.randint(0, 20) / 10.0
    sharp = random.randint(0, 30) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    image = ImageEnhance.Sharpness(image).enhance(sharp)
    return image


class SINetTextTrainDataset(data.Dataset):
    def __init__(self, manifest_jsonl: str, trainsize: int):
        self.records = load_jsonl_records(Path(manifest_jsonl))
        self.trainsize = trainsize
        self.img_transform = T.Compose(
            [
                T.Resize((trainsize, trainsize)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize((trainsize, trainsize), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = _rgb_loader(record["image_path"])
        mask = _binary_loader(record["mask_path"])
        category_key = record.get("category", "unknown")
        location_key = record.get("location_key", "middle_center")
        size_key = record.get("size_key", "medium")

        image, mask, flipped = _random_flip(image, mask)
        image = _color_enhance(image)
        if flipped:
            location_key = LOCATION_HFLIP.get(location_key, location_key)

        return {
            "sample_id": record["sample_id"],
            "image": self.img_transform(image),
            "mask": self.mask_transform(mask),
            "text": record.get("training_text") or record.get("clip_text") or record.get("text") or "A camouflaged target in the image.",
            "category_id": CATEGORY_TO_ID.get(category_key, CATEGORY_TO_ID["unknown"]),
            "location_id": LOCATION_TO_ID.get(location_key, LOCATION_TO_ID["middle_center"]),
            "size_id": SIZE_TO_ID.get(size_key, SIZE_TO_ID["medium"]),
            "sample_weight": float(record.get("sample_weight", 1.0)),
            "final_confidence": maybe_float(record.get("final_confidence"), 0.0),
        }


class SINetTextEvalDataset(data.Dataset):
    def __init__(self, manifest_jsonl: str, testsize: int):
        self.records = load_jsonl_records(Path(manifest_jsonl))
        self.testsize = testsize
        self.img_transform = T.Compose(
            [
                T.Resize((testsize, testsize)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = _rgb_loader(record["image_path"])
        return {
            "sample_id": record["sample_id"],
            "image": self.img_transform(image),
            "text": record.get("training_text") or record.get("clip_text") or record.get("text") or "A camouflaged target in the image.",
            "category_id": category_id(record.get("category")),
            "location_id": location_id(record.get("location_key")),
            "size_id": size_id(record.get("size_key")),
            "image_path": record["image_path"],
            "gt_path": record.get("gt_path"),
        }


def build_text_collate_fn(tokenizer, max_length: int) -> Callable[[list[dict]], dict]:
    def collate(batch: list[dict]) -> dict:
        free_text = tokenizer(
            [sample["text"] for sample in batch],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        payload = {
            "sample_ids": [sample["sample_id"] for sample in batch],
            "images": torch.stack([sample["image"] for sample in batch], dim=0),
            "text_input_ids": free_text["input_ids"],
            "text_attention_mask": free_text["attention_mask"],
            "category_ids": torch.tensor([sample["category_id"] for sample in batch], dtype=torch.long),
            "location_ids": torch.tensor([sample["location_id"] for sample in batch], dtype=torch.long),
            "size_ids": torch.tensor([sample["size_id"] for sample in batch], dtype=torch.long),
        }
        if "mask" in batch[0]:
            payload["masks"] = torch.stack([sample["mask"] for sample in batch], dim=0)
            payload["sample_weights"] = torch.tensor(
                [sample["sample_weight"] for sample in batch],
                dtype=torch.float32,
            )
            payload["final_confidences"] = torch.tensor(
                [sample["final_confidence"] for sample in batch],
                dtype=torch.float32,
            )
        else:
            payload["image_paths"] = [sample["image_path"] for sample in batch]
            payload["gt_paths"] = [sample.get("gt_path") for sample in batch]
        return payload

    return collate

"""Datasets and collators for Stage 4 CRIS-lite."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image, ImageEnhance

from .common import load_jsonl_records


def _rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        return Image.open(handle).convert("RGB")


def _binary_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        return Image.open(handle).convert("L")


def _random_flip(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask


def _random_crop(image: Image.Image, mask: Image.Image, border: int = 30) -> tuple[Image.Image, Image.Image]:
    image_width, image_height = image.size
    crop_win_width = np.random.randint(max(1, image_width - border), image_width + 1)
    crop_win_height = np.random.randint(max(1, image_height - border), image_height + 1)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), mask.crop(random_region)


def _random_rotation(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() > 0.8:
        angle = np.random.randint(-15, 16)
        image = image.rotate(angle, Image.BICUBIC)
        mask = mask.rotate(angle, Image.NEAREST)
    return image, mask


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


class CRISLiteTrainDataset(data.Dataset):
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

        image, mask = _random_flip(image, mask)
        image, mask = _random_crop(image, mask)
        image, mask = _random_rotation(image, mask)
        image = _color_enhance(image)

        return {
            "sample_id": record["sample_id"],
            "image": self.img_transform(image),
            "mask": self.mask_transform(mask),
            "text": record.get("text") or record.get("clip_text") or "A camouflaged target in the image.",
            "sample_weight": float(record.get("sample_weight", 1.0)),
        }


class CRISLiteEvalDataset(data.Dataset):
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
            "text": record.get("text") or record.get("clip_text") or "A camouflaged target in the image.",
            "image_path": record["image_path"],
            "gt_path": record.get("gt_path"),
        }


def build_text_collate_fn(tokenizer, max_length: int) -> Callable[[list[dict]], dict]:
    def collate(batch: list[dict]) -> dict:
        tokenized = tokenizer(
            [sample["text"] for sample in batch],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        payload = {
            "sample_ids": [sample["sample_id"] for sample in batch],
            "images": torch.stack([sample["image"] for sample in batch], dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if "mask" in batch[0]:
            payload["masks"] = torch.stack([sample["mask"] for sample in batch], dim=0)
            payload["sample_weights"] = torch.tensor(
                [sample["sample_weight"] for sample in batch],
                dtype=torch.float32,
            )
        else:
            payload["image_paths"] = [sample["image_path"] for sample in batch]
            payload["gt_paths"] = [sample.get("gt_path") for sample in batch]
        return payload

    return collate

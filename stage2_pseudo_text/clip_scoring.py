"""Optional CLIP-based text-image scoring backends for Stage 2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from PIL import Image


class BaseClipScorer(ABC):
    @abstractmethod
    def score_text_images(self, text: str, images: Dict[str, Image.Image]) -> Dict[str, float | None]:
        raise NotImplementedError


class NoOpClipScorer(BaseClipScorer):
    def score_text_images(self, text: str, images: Dict[str, Image.Image]) -> Dict[str, float | None]:
        return {key: None for key in images}


class TransformersClipScorer(BaseClipScorer):
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        try:
            from transformers import AutoProcessor, CLIPModel  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers is required for hf_clip backend") from exc

        import torch

        self.torch = torch
        self.device = _resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def score_text_images(self, text: str, images: Dict[str, Image.Image]) -> Dict[str, float | None]:
        if not images:
            return {}
        names = list(images.keys())
        image_list = [images[name].convert("RGB") for name in names]
        inputs = self.processor(
            text=[text],
            images=image_list,
            return_tensors="pt",
            padding=True,
            input_data_format="channels_last",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (text_features @ image_features.T).squeeze(0).tolist()
        if isinstance(similarities, float):
            similarities = [similarities]
        return {name: float(score) for name, score in zip(names, similarities)}


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_clip_scorer(
    backend_name: str,
    model_name_or_path: str | None = None,
    device: str = "auto",
) -> BaseClipScorer:
    if backend_name == "none":
        return NoOpClipScorer()
    if backend_name == "hf_clip":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for hf_clip backend")
        return TransformersClipScorer(model_name_or_path, device=device)
    raise ValueError(f"Unsupported CLIP backend: {backend_name}")

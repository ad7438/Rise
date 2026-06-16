from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image


class CLIPSegTextPrior:
    def __init__(self, model_name: str, device: str = "cuda", hf_endpoint: str | None = None) -> None:
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device).eval()

    def _normalize_text(self, text: str) -> str:
        text = " ".join(str(text).strip().split())
        if not text:
            return "A camouflaged target."
        words = text.split(" ")
        if len(words) > 28:
            text = " ".join(words[:28])
        return text

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str) -> np.ndarray:
        text = self._normalize_text(text)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        if logits.ndim == 4:
            logits = logits[:, 0]
        probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        resized = cv2.resize(probs, image.size, interpolation=cv2.INTER_CUBIC)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

"""Pluggable VLM backends for Stage 2 pseudo-text generation."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence

from PIL import Image

from .categories import normalize_category_key
from .schema import VLMResult


class BaseVLMBackend(ABC):
    @abstractmethod
    def describe_region(self, images: Sequence[Image.Image], prompt: str, sample_id: str) -> VLMResult:
        raise NotImplementedError


class MockVLMBackend(BaseVLMBackend):
    def describe_region(self, images: Sequence[Image.Image], prompt: str, sample_id: str) -> VLMResult:
        return VLMResult(
            category_key="unknown",
            category_confidence=0.0,
            raw_response='{"category":"unknown","category_confidence":0.0,"evidence":"mock backend"}',
            evidence="mock backend",
        )


class JsonLookupVLMBackend(BaseVLMBackend):
    def __init__(self, lookup_path: str):
        self.records = _load_lookup(Path(lookup_path))

    def describe_region(self, images: Sequence[Image.Image], prompt: str, sample_id: str) -> VLMResult:
        record = self.records.get(sample_id)
        if record is None:
            return VLMResult(
                category_key="unknown",
                category_confidence=0.0,
                raw_response='{"category":"unknown","category_confidence":0.0,"evidence":"missing lookup record"}',
                evidence="missing lookup record",
            )
        return _vlm_result_from_payload(record, raw_response=json.dumps(record, ensure_ascii=False))


class HFVision2SeqBackend(BaseVLMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        trust_remote_code: bool = False,
    ):
        try:
            from transformers import AutoProcessor  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers is required for hf_vision2seq backend") from exc

        try:
            from transformers import AutoModelForVision2Seq  # type: ignore
            model_cls = AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText  # type: ignore
            model_cls = AutoModelForImageTextToText

        import torch

        self.torch = torch
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model = model_cls.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model.to(self.device)
        self.model.eval()

    def describe_region(self, images: Sequence[Image.Image], prompt: str, sample_id: str) -> VLMResult:
        if not images:
            return VLMResult(category_key="unknown", category_confidence=0.0, raw_response="", evidence="no images")

        prepared_images = list(images)
        try:
            inputs = self.processor(images=prepared_images, text=prompt, return_tensors="pt")
        except TypeError:
            inputs = self.processor(images=prepared_images[0], text=prompt, return_tensors="pt")

        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        raw_response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return parse_vlm_response(raw_response)


class Qwen25VLBackend(BaseVLMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_new_tokens: int = 128,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        attn_implementation: str | None = "sdpa",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        use_fast_processor: bool = False,
    ):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers with Qwen2.5-VL support is required for qwen2_5_vl backend") from exc

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError as exc:
            raise ImportError("qwen-vl-utils is required for qwen2_5_vl backend") from exc

        import torch

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = max_new_tokens
        self.device = _resolve_device(device)
        self.input_device = self.device

        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_processor,
        }
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["dtype"] = resolved_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if self.device == "cpu":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
            self.model.to(self.device)
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map="auto",
                **model_kwargs,
            )
            self.input_device = "cuda"
        self.model.eval()

    def describe_region(self, images: Sequence[Image.Image], prompt: str, sample_id: str) -> VLMResult:
        if not images:
            return VLMResult(category_key="unknown", category_confidence=0.0, raw_response="", evidence="no images")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)

        processor_kwargs: Dict[str, Any] = {
            "text": [chat_text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)
        inputs = {
            key: value.to(self.input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        raw_response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_vlm_response(raw_response)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _resolve_torch_dtype(torch_module, requested: str | None):
    if requested in (None, "", "auto"):
        return "auto"

    normalized = str(requested).lower()
    if normalized in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch_module.float16
    if normalized in ("float32", "fp32"):
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {requested}")


def _load_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        records: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                sample_id = str(payload.get("sample_id") or payload.get("image_id") or payload.get("stem"))
                records[sample_id] = payload
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}
        sample_id = str(payload.get("sample_id") or payload.get("image_id") or payload.get("stem"))
        return {sample_id: payload}
    if isinstance(payload, list):
        records = {}
        for item in payload:
            sample_id = str(item.get("sample_id") or item.get("image_id") or item.get("stem"))
            records[sample_id] = item
        return records
    raise ValueError(f"Unsupported lookup payload in {path}")


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _vlm_result_from_payload(payload: Dict[str, Any], raw_response: str) -> VLMResult:
    category_key = normalize_category_key(
        payload.get("category") or payload.get("label") or payload.get("class") or payload.get("prediction")
    )
    evidence = str(payload.get("evidence") or payload.get("reason") or payload.get("rationale") or "")
    confidence = _coerce_confidence(payload.get("category_confidence") or payload.get("confidence"))
    return VLMResult(
        category_key=category_key,
        category_confidence=confidence,
        raw_response=raw_response,
        evidence=evidence,
    )


def parse_vlm_response(raw_response: str) -> VLMResult:
    raw_response = raw_response.strip()
    payload: Dict[str, Any] | None = None

    json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None

    if payload is not None:
        return _vlm_result_from_payload(payload, raw_response=raw_response)

    lowered = raw_response.lower()
    category_key = normalize_category_key(lowered)

    confidence_match = re.search(r"([01](?:\.\d+)?)", lowered)
    confidence = _coerce_confidence(confidence_match.group(1) if confidence_match else 0.0)
    return VLMResult(
        category_key=category_key,
        category_confidence=confidence,
        raw_response=raw_response,
        evidence="",
    )


def build_vlm_backend(
    backend_name: str,
    model_name_or_path: str | None = None,
    lookup_path: str | None = None,
    device: str = "auto",
    max_new_tokens: int = 128,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    attn_implementation: str | None = "sdpa",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_fast_processor: bool = False,
) -> BaseVLMBackend:
    if backend_name == "mock":
        return MockVLMBackend()
    if backend_name == "json_lookup":
        if not lookup_path:
            raise ValueError("lookup_path is required for json_lookup backend")
        return JsonLookupVLMBackend(lookup_path)
    if backend_name == "hf_vision2seq":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for hf_vision2seq backend")
        return HFVision2SeqBackend(
            model_name_or_path=model_name_or_path,
            device=device,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )
    if backend_name == "qwen2_5_vl":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for qwen2_5_vl backend")
        return Qwen25VLBackend(
            model_name_or_path=model_name_or_path,
            device=device,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast_processor=use_fast_processor,
        )
    raise ValueError(f"Unsupported backend: {backend_name}")

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _to_numpy(value: Any) -> tuple[np.ndarray, Any | None]:
    if _is_torch_tensor(value):
        return value.detach().float().cpu().numpy(), value
    return np.asarray(value), None


def _restore_type(value: np.ndarray, reference: Any | None) -> Any:
    if reference is None:
        return value.astype(np.float32)
    import torch

    return torch.from_numpy(value.astype(np.float32)).to(device=reference.device, dtype=reference.dtype)


def _normalize(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(value.min())
    max_value = float(value.max())
    if max_value - min_value <= eps:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - min_value) / (max_value - min_value + eps), 0.0, 1.0).astype(np.float32)


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[-2:] == shape:
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _squeeze_to_2d(value: np.ndarray) -> np.ndarray:
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"AGSP expects 2D prior/mask arrays after squeeze, got shape {value.shape}")
    return value.astype(np.float32)


def build_agsp_prior(
    ps_raw: Any,
    m0: Any,
    anchor_radius: int = 25,
    anchor_blur: int = 7,
    mask_blur: int = 5,
    lambda_s: float = 0.2,
    eps: float = 1e-6,
    semantic_prior_mode: str = "agsp_full",
) -> tuple[Any, Any, Any]:
    """Build an anchor-guided semantic prior from CLIPSeg output and M0.

    Supported modes:
    - visual_only: P_s = 0, while still returning anchor A and M_f0
    - raw_clipseg: P_s = Normalize(P_s_raw)
    - agsp_no_mf0: P_s = Normalize(P_s_raw * A)
    - agsp_full: P_s = Normalize(P_s_raw * A + lambda_s * M_f0)
    """

    ps_np, ps_ref = _to_numpy(ps_raw)
    m0_np, _ = _to_numpy(m0)
    ps_np = _normalize(_squeeze_to_2d(ps_np), eps=eps)
    m0_np = _squeeze_to_2d(m0_np)
    m0_np = _resize_mask(m0_np, ps_np.shape)
    m0_np = _normalize((m0_np > 0.5).astype(np.float32), eps=eps)

    radius = max(0, int(anchor_radius))
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        anchor = cv2.dilate(m0_np.astype(np.float32), kernel, iterations=1)
    else:
        anchor = m0_np.copy()
    anchor_blur_size = _odd_kernel_size(anchor_blur)
    anchor = cv2.GaussianBlur(anchor, (anchor_blur_size, anchor_blur_size), sigmaX=0, sigmaY=0)
    anchor = _normalize(anchor, eps=eps)

    mask_blur_size = _odd_kernel_size(mask_blur)
    mf0 = cv2.GaussianBlur(m0_np.astype(np.float32), (mask_blur_size, mask_blur_size), sigmaX=0, sigmaY=0)
    mf0 = _normalize(mf0, eps=eps)

    if semantic_prior_mode == "visual_only":
        ps_used = np.zeros_like(ps_np, dtype=np.float32)
    elif semantic_prior_mode == "raw_clipseg":
        ps_used = ps_np
    elif semantic_prior_mode == "agsp_no_mf0":
        ps_used = _normalize(ps_np * anchor, eps=eps)
    elif semantic_prior_mode == "agsp_full":
        ps_used = _normalize(ps_np * anchor + float(lambda_s) * mf0, eps=eps)
    else:
        raise ValueError(
            "semantic_prior_mode must be one of visual_only, raw_clipseg, agsp_no_mf0, agsp_full; "
            f"got {semantic_prior_mode}"
        )

    return (
        _restore_type(ps_used, ps_ref),
        _restore_type(anchor, ps_ref),
        _restore_type(mf0, ps_ref),
    )


class AnchorGuidedSemanticPrior:
    def __init__(
        self,
        anchor_radius: int = 25,
        anchor_blur: int = 7,
        mask_blur: int = 5,
        lambda_s: float = 0.2,
        eps: float = 1e-6,
        semantic_prior_mode: str = "agsp_full",
    ) -> None:
        self.anchor_radius = anchor_radius
        self.anchor_blur = anchor_blur
        self.mask_blur = mask_blur
        self.lambda_s = lambda_s
        self.eps = eps
        self.semantic_prior_mode = semantic_prior_mode

    def __call__(self, ps_raw: Any, m0: Any) -> tuple[Any, Any, Any]:
        return build_agsp_prior(
            ps_raw=ps_raw,
            m0=m0,
            anchor_radius=self.anchor_radius,
            anchor_blur=self.anchor_blur,
            mask_blur=self.mask_blur,
            lambda_s=self.lambda_s,
            eps=self.eps,
            semantic_prior_mode=self.semantic_prior_mode,
        )

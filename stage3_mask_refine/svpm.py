from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.segmentation import slic


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


def _squeeze_to_2d(value: np.ndarray, name: str) -> np.ndarray:
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"{name} expects a 2D array after squeeze, got shape {value.shape}")
    return value.astype(np.float32)


def _image_to_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))
    else:
        image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"SVPM expects image shape HxWx3, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    if float(image.max()) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _clip01(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if float(value.max()) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _normalize(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(value.min())
    max_value = float(value.max())
    if max_value - min_value <= eps:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - min_value) / (max_value - min_value + eps), 0.0, 1.0).astype(np.float32)


def _resize_to_shape(value: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    if value.shape == shape:
        return value.astype(np.float32)
    return cv2.resize(value.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _elliptic_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0.5).astype(np.float32)
    return cv2.dilate((mask > 0.5).astype(np.uint8), _elliptic_kernel(radius), iterations=1).astype(np.float32)


def _normalize_alpha_beta(alpha: float, beta: float, eps: float) -> tuple[float, float]:
    alpha = max(0.0, float(alpha))
    beta = max(0.0, float(beta))
    total = alpha + beta
    if total <= eps:
        return 0.6, 0.4
    return alpha / total, beta / total


@dataclass
class SVPMDebug:
    local_region: np.ndarray
    anchor_support: np.ndarray
    semantic_support: np.ndarray
    visual_prior_preblur: np.ndarray
    alpha: float
    beta: float


def _build_svpm_numpy(
    image_np: np.ndarray,
    m0_np: np.ndarray,
    ps_np: np.ndarray,
    n_segments: int,
    compactness: float,
    dilate_radius: int,
    alpha: float,
    beta: float,
    blur_ksize: int,
    eps: float,
    visual_prior_mode: str,
) -> tuple[np.ndarray, np.ndarray, SVPMDebug]:
    height, width = image_np.shape[:2]
    shape = (height, width)
    m0_np = _resize_to_shape(_squeeze_to_2d(m0_np, "m0"), shape, cv2.INTER_NEAREST)
    ps_np = _resize_to_shape(_squeeze_to_2d(ps_np, "ps_agsp"), shape, cv2.INTER_LINEAR)
    m0_np = (_clip01(m0_np) > 0.5).astype(np.float32)
    ps_np = _normalize(_clip01(ps_np), eps=eps)

    image_float = image_np.astype(np.float32) / 255.0
    n_segments = max(2, int(n_segments))
    compactness = max(float(compactness), eps)
    superpixel_map = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)

    local_region = _dilate(m0_np, int(dilate_radius))
    flat_labels = superpixel_map.reshape(-1)
    label_count = int(flat_labels.max()) + 1
    counts = np.bincount(flat_labels, minlength=label_count).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    r_scores = np.bincount(flat_labels, weights=m0_np.reshape(-1), minlength=label_count) / counts
    s_scores = np.bincount(flat_labels, weights=ps_np.reshape(-1), minlength=label_count) / counts
    l_scores = np.bincount(flat_labels, weights=local_region.reshape(-1), minlength=label_count) / counts

    if visual_prior_mode == "svpm_m0_only":
        v_scores = l_scores * r_scores
        used_alpha, used_beta = 1.0, 0.0
    elif visual_prior_mode == "svpm_full":
        used_alpha, used_beta = _normalize_alpha_beta(alpha, beta, eps)
        v_scores = l_scores * (used_alpha * r_scores + used_beta * s_scores)
    else:
        raise ValueError(
            "visual_prior_mode for SVPM must be one of svpm_m0_only, svpm_full; "
            f"got {visual_prior_mode}"
        )

    anchor_support = r_scores[superpixel_map].astype(np.float32)
    semantic_support = s_scores[superpixel_map].astype(np.float32)
    visual_prior_preblur = v_scores[superpixel_map].astype(np.float32)

    blur_size = _odd_kernel_size(blur_ksize)
    pv = cv2.GaussianBlur(visual_prior_preblur, (blur_size, blur_size), sigmaX=0, sigmaY=0)
    pv = _normalize(pv, eps=eps)

    debug = SVPMDebug(
        local_region=np.clip(local_region, 0.0, 1.0).astype(np.float32),
        anchor_support=np.clip(anchor_support, 0.0, 1.0).astype(np.float32),
        semantic_support=np.clip(semantic_support, 0.0, 1.0).astype(np.float32),
        visual_prior_preblur=np.clip(visual_prior_preblur, 0.0, 1.0).astype(np.float32),
        alpha=float(used_alpha),
        beta=float(used_beta),
    )
    return pv.astype(np.float32), superpixel_map, debug


def build_svpm_prior(
    image: Any,
    m0: Any,
    ps_agsp: Any,
    n_segments: int = 300,
    compactness: float = 10,
    dilate_radius: int = 25,
    alpha: float = 0.6,
    beta: float = 0.4,
    blur_ksize: int = 5,
    eps: float = 1e-6,
) -> tuple[Any, np.ndarray]:
    """Build a superpixel-guided visual prior P_v from image, M0, and AGSP prior."""

    pv, superpixel_map, _ = build_svpm_prior_with_debug(
        image=image,
        m0=m0,
        ps_agsp=ps_agsp,
        n_segments=n_segments,
        compactness=compactness,
        dilate_radius=dilate_radius,
        alpha=alpha,
        beta=beta,
        blur_ksize=blur_ksize,
        eps=eps,
        visual_prior_mode="svpm_full",
    )
    return pv, superpixel_map


def build_svpm_prior_with_debug(
    image: Any,
    m0: Any,
    ps_agsp: Any,
    n_segments: int = 300,
    compactness: float = 10,
    dilate_radius: int = 25,
    alpha: float = 0.6,
    beta: float = 0.4,
    blur_ksize: int = 5,
    eps: float = 1e-6,
    visual_prior_mode: str = "svpm_full",
) -> tuple[Any, np.ndarray, SVPMDebug]:
    image_np = _image_to_numpy(image)
    m0_np, m0_ref = _to_numpy(m0)
    ps_np, _ = _to_numpy(ps_agsp)
    pv, superpixel_map, debug = _build_svpm_numpy(
        image_np=image_np,
        m0_np=m0_np,
        ps_np=ps_np,
        n_segments=n_segments,
        compactness=compactness,
        dilate_radius=dilate_radius,
        alpha=alpha,
        beta=beta,
        blur_ksize=blur_ksize,
        eps=eps,
        visual_prior_mode=visual_prior_mode,
    )
    return _restore_type(pv, m0_ref), superpixel_map, debug


class SuperpixelGuidedVisualPrior:
    def __init__(
        self,
        n_segments: int = 300,
        compactness: float = 10,
        dilate_radius: int = 25,
        alpha: float = 0.6,
        beta: float = 0.4,
        blur_ksize: int = 5,
        eps: float = 1e-6,
        visual_prior_mode: str = "svpm_full",
    ) -> None:
        self.n_segments = n_segments
        self.compactness = compactness
        self.dilate_radius = dilate_radius
        self.alpha = alpha
        self.beta = beta
        self.blur_ksize = blur_ksize
        self.eps = eps
        self.visual_prior_mode = visual_prior_mode

    def __call__(self, image: Any, m0: Any, ps_agsp: Any) -> tuple[Any, np.ndarray, SVPMDebug]:
        return build_svpm_prior_with_debug(
            image=image,
            m0=m0,
            ps_agsp=ps_agsp,
            n_segments=self.n_segments,
            compactness=self.compactness,
            dilate_radius=self.dilate_radius,
            alpha=self.alpha,
            beta=self.beta,
            blur_ksize=self.blur_ksize,
            eps=self.eps,
            visual_prior_mode=self.visual_prior_mode,
        )

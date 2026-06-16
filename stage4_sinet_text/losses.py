"""Losses for Stage 4 SINet-text."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(values.device)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_structure_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(masks, kernel_size=31, stride=1, padding=15) - masks)
    wbce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * masks) * weit).sum(dim=(2, 3))
    union = ((pred + masks) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return _weighted_mean(wbce + wiou, sample_weights)


def boundary_targets(masks: torch.Tensor) -> torch.Tensor:
    max_pool = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
    return (max_pool - min_pool).clamp(0.0, 1.0)


def weighted_boundary_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pred_edge = boundary_targets(pred)
    mask_edge = boundary_targets(masks)
    per_pixel = F.smooth_l1_loss(pred_edge, mask_edge, reduction="none")
    per_sample = per_pixel.mean(dim=(1, 2, 3))
    return _weighted_mean(per_sample, sample_weights)


def weighted_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student = torch.sigmoid(student_logits)
    teacher = torch.sigmoid(teacher_logits.detach())
    per_sample = F.mse_loss(student, teacher, reduction="none").mean(dim=(1, 2, 3))
    return _weighted_mean(per_sample, sample_weights)

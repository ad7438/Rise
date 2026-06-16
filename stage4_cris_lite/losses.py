"""Losses for Stage 4 CRIS-lite."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    per_sample = wbce + wiou

    if sample_weights is None:
        return per_sample.mean()

    weights = sample_weights.to(per_sample.device)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)

"""CRIS-lite image-text segmentation model."""

from __future__ import annotations

from contextlib import nullcontext

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, channels, height, width = feature.shape
        query = feature.flatten(2).transpose(1, 2)
        norm_query = self.query_norm(query)
        norm_text = self.text_norm(text_tokens)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        attended, _ = self.attn(
            norm_query,
            norm_text,
            norm_text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query.transpose(1, 2).reshape(batch_size, channels, height, width)


class CRISLiteModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        *,
        backbone_pretrained: bool = True,
        text_model_name: str = "openai/clip-vit-base-patch32",
        decoder_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        self.freeze_text_encoder = freeze_text_encoder

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        channels = self.backbone.feature_info.channels()

        self.text_encoder = self._load_text_encoder(text_model_name)
        text_width = int(self.text_encoder.config.hidden_size)
        self.text_projection = nn.Linear(text_width, decoder_dim)
        self.pooled_text_gate = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        self.visual_projections = nn.ModuleList([nn.Conv2d(ch, decoder_dim, kernel_size=1) for ch in channels])
        self.shallow_refine = ConvNormAct(decoder_dim, decoder_dim)
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(decoder_dim, num_heads, dropout=dropout) for _ in range(3)]
        )
        self.merge_blocks = nn.ModuleList([ConvNormAct(decoder_dim * 2, decoder_dim) for _ in range(3)])
        self.feature_refine_blocks = nn.ModuleList([ConvNormAct(decoder_dim, decoder_dim) for _ in range(4)])
        self.decoder_refine_blocks = nn.ModuleList([ConvNormAct(decoder_dim, decoder_dim) for _ in range(3)])
        self.head = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

        if freeze_text_encoder:
            self.text_encoder.eval()
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_text_encoder(text_model_name: str) -> CLIPTextModel:
        try:
            return CLIPTextModel.from_pretrained(text_model_name, local_files_only=True)
        except OSError:
            return CLIPTextModel.from_pretrained(text_model_name)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad if self.freeze_text_encoder else nullcontext
        with context():
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(text_outputs.last_hidden_state)

    @staticmethod
    def masked_mean(text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        text_tokens = self.encode_text(input_ids, attention_mask)
        pooled_text = self.masked_mean(text_tokens, attention_mask)

        image_features = self.backbone(images)
        projected = [proj(feature) for proj, feature in zip(self.visual_projections, image_features)]

        gate = torch.sigmoid(self.pooled_text_gate(pooled_text)).unsqueeze(-1).unsqueeze(-1)
        fused_features = [self.shallow_refine(projected[0] * gate)]
        for feature, block in zip(projected[1:], self.cross_blocks):
            fused_features.append(block(feature, text_tokens, attention_mask))

        fused_features = [block(feature) for block, feature in zip(self.feature_refine_blocks, fused_features)]

        x = fused_features[-1]
        for index in range(2, -1, -1):
            x = F.interpolate(x, size=fused_features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge_blocks[2 - index](torch.cat([fused_features[index], x], dim=1))
            x = self.decoder_refine_blocks[2 - index](x)

        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

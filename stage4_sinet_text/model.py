"""SINet-V2 with text fusion adapters for Stage 4."""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINET_ROOT = PROJECT_ROOT / "SINet-V2"
if str(SINET_ROOT) not in sys.path:
    sys.path.insert(0, str(SINET_ROOT))

from lib.Network_Res2Net_GRA_NCD import NeighborConnectionDecoder, RFB_modified, ReverseStage  # noqa: E402
from lib.Res2Net_v1b import res2net50_v1b_26w_4s  # noqa: E402


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StructuredFieldEncoder(nn.Module):
    def __init__(self, text_dim: int, category_count: int, location_count: int, size_count: int):
        super().__init__()
        self.category = nn.Embedding(category_count, text_dim)
        self.location = nn.Embedding(location_count, text_dim)
        self.size = nn.Embedding(size_count, text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
        )

    def forward(
        self,
        category_ids: torch.Tensor,
        location_ids: torch.Tensor,
        size_ids: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat(
            [
                self.category(category_ids),
                self.location(location_ids),
                self.size(size_ids),
            ],
            dim=1,
        )
        return self.proj(stacked)


class TextFiLMAdapter(nn.Module):
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.gamma = nn.Linear(text_dim, channels)
        self.beta = nn.Linear(text_dim, channels)
        self.refine = ConvBNReLU(channels, channels)

    def forward(self, feature: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        gamma = 0.2 * torch.tanh(self.gamma(text_context)).unsqueeze(-1).unsqueeze(-1)
        beta = 0.2 * torch.tanh(self.beta(text_context)).unsqueeze(-1).unsqueeze(-1)
        feature = self.norm(feature)
        feature = feature * (1 + gamma) + beta
        return self.refine(feature)


class TokenSpatialFusion(nn.Module):
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.query = nn.Conv2d(channels, text_dim, kernel_size=1, bias=False)
        self.value = nn.Linear(text_dim, channels)
        self.out = ConvBNReLU(channels, channels)

    def forward(
        self,
        feature: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        query = self.query(feature).flatten(2).transpose(1, 2)
        logits = torch.matmul(query, text_tokens.transpose(1, 2)) / math.sqrt(query.shape[-1])
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        context = torch.matmul(attn, text_tokens)
        context = self.value(context).transpose(1, 2).reshape(batch_size, feature.shape[1], height, width)
        return self.out(feature + context)


class TextRefinementHead(nn.Module):
    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.coarse_proj = ConvBNReLU(1, channels)
        self.fuse = ConvBNReLU(channels * 2, channels)
        self.norm = nn.BatchNorm2d(channels)
        self.gamma = nn.Linear(text_dim, channels)
        self.beta = nn.Linear(text_dim, channels)
        self.spatial = TokenSpatialFusion(channels, text_dim)
        self.delta = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        coarse_logit: torch.Tensor,
        text_context: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        coarse_prob = torch.sigmoid(coarse_logit)
        fused = self.fuse(torch.cat([feature, self.coarse_proj(coarse_prob)], dim=1))
        gamma = 0.15 * torch.tanh(self.gamma(text_context)).unsqueeze(-1).unsqueeze(-1)
        beta = 0.15 * torch.tanh(self.beta(text_context)).unsqueeze(-1).unsqueeze(-1)
        fused = self.norm(fused)
        fused = fused * (1 + gamma) + beta
        fused = self.spatial(fused, text_tokens, attention_mask)
        return coarse_logit + self.delta(fused)


class SINetTextNetwork(nn.Module):
    def __init__(
        self,
        channel: int = 32,
        *,
        imagenet_pretrained: bool = True,
        text_model_name: str = "openai/clip-vit-base-patch32",
        text_dim: int = 256,
        freeze_text_encoder: bool = True,
        category_count: int = 11,
        location_count: int = 9,
        size_count: int = 3,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.freeze_text_encoder = freeze_text_encoder
        self.text_model_name = text_model_name
        self.text_dim = text_dim

        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)
        if use_text:
            self.text_encoder = self._load_text_encoder(text_model_name)
            text_width = int(self.text_encoder.config.hidden_size)
            self.text_projection = nn.Linear(text_width, text_dim)
            self.structured_encoder = StructuredFieldEncoder(
                text_dim=text_dim,
                category_count=category_count,
                location_count=location_count,
                size_count=size_count,
            )
            self.context_norm = nn.LayerNorm(text_dim)
            self.token_norm = nn.LayerNorm(text_dim)

            self.refine_head = TextRefinementHead(channel, text_dim)

            if freeze_text_encoder:
                self.text_encoder.eval()
                for parameter in self.text_encoder.parameters():
                    parameter.requires_grad = False
        else:
            self.text_encoder = None
            self.text_projection = None
            self.structured_encoder = None
            self.context_norm = None
            self.token_norm = None
            self.refine_head = None

    @staticmethod
    def _load_text_encoder(text_model_name: str) -> CLIPTextModel:
        try:
            return CLIPTextModel.from_pretrained(text_model_name, local_files_only=True)
        except OSError:
            return CLIPTextModel.from_pretrained(text_model_name)

    def load_visual_state_dict(self, state_dict: dict) -> tuple[list[str], list[str]]:
        own_state = self.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        incompatible = self.load_state_dict(filtered, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def encode_text(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        category_ids: torch.Tensor,
        location_ids: torch.Tensor,
        size_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_text:
            raise RuntimeError("encode_text() should not be called when use_text=False.")
        context = torch.no_grad if self.freeze_text_encoder else nullcontext
        with context():
            free_text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )

        free_tokens = self.text_projection(free_text_outputs.last_hidden_state)
        free_mask = text_attention_mask.unsqueeze(-1).float()
        free_pooled = (free_tokens * free_mask).sum(dim=1) / free_mask.sum(dim=1).clamp_min(1.0)
        field_pooled = self.structured_encoder(category_ids, location_ids, size_ids)

        text_context = self.context_norm(free_pooled + field_pooled)
        text_tokens = self.token_norm(torch.cat([field_pooled.unsqueeze(1), free_tokens], dim=1))
        token_mask = torch.cat(
            [
                torch.ones(
                    (text_attention_mask.shape[0], 1),
                    device=text_attention_mask.device,
                    dtype=text_attention_mask.dtype,
                ),
                text_attention_mask,
            ],
            dim=1,
        )
        return text_context, text_tokens, token_mask

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        category_ids: torch.Tensor,
        location_ids: torch.Tensor,
        size_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_text:
            text_context, text_tokens, token_mask = self.encode_text(
                text_input_ids,
                text_attention_mask,
                category_ids,
                location_ids,
                size_ids,
            )
        else:
            text_context = None
            text_tokens = None
            token_mask = None

        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        s_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        s_g_pred = F.interpolate(s_g, scale_factor=8, mode="bilinear", align_corners=False)

        guidance_g = F.interpolate(s_g, scale_factor=0.25, mode="bilinear", align_corners=False)
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        s_5 = ra4_feat + guidance_g
        s_5_pred = F.interpolate(s_5, scale_factor=32, mode="bilinear", align_corners=False)

        guidance_5 = F.interpolate(s_5, scale_factor=2, mode="bilinear", align_corners=False)
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        s_4 = ra3_feat + guidance_5
        s_4_pred = F.interpolate(s_4, scale_factor=16, mode="bilinear", align_corners=False)

        guidance_4 = F.interpolate(s_4, scale_factor=2, mode="bilinear", align_corners=False)
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        s_3 = ra2_feat + guidance_4
        if self.use_text:
            s_3 = self.refine_head(x2_rfb, s_3, text_context, text_tokens, token_mask)
        s_3_pred = F.interpolate(s_3, scale_factor=8, mode="bilinear", align_corners=False)

        return s_g_pred, s_5_pred, s_4_pred, s_3_pred

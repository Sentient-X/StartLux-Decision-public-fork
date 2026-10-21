"""Small reward heads over frozen, cached visual and language features.

The three controls share their projection and outcome objective. Only the spatial
aggregation and temporal mixer differ; no encoder or language model is trained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HeadKind = Literal["late_temporal", "pooled_temporal", "late_orderless"]
HEAD_KINDS: tuple[HeadKind, ...] = (
    "late_temporal", "pooled_temporal", "late_orderless",
)


@dataclass(frozen=True)
class HeadConfig:
    feature_dim: int = 768
    width: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.1
    temperature: float = 0.1
    max_views: int = 3
    num_classes: int = 5

    def __post_init__(self) -> None:
        if min(self.feature_dim, self.width, self.num_heads, self.num_layers,
               self.feedforward_dim, self.max_views) <= 0:
            raise ValueError("head dimensions must be positive")
        if self.width % self.num_heads or self.width % 2:
            raise ValueError("width must be even and divisible by num_heads")
        if not 0 <= self.dropout < 1 or self.temperature <= 0:
            raise ValueError("invalid dropout or matching temperature")
        if self.num_classes != 5:
            raise ValueError("the RoboReward gate uses exactly five outcome classes")


class _ResidualMLP(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, width), nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(self.norm(value))


class RewardHead(nn.Module):
    """Outcome classifier; forward inputs are explicitly ordered below.

    visual: [B,T,V,P,D], text: [B,L,D], text_mask: [B,L],
    view_mask: [B,T,V], times: [B,T] in [0,1]. Mask True means valid.
    P must be a square raster grid. Camera slots must retain their identities.
    Every frame has at least one present camera and every example a text token.
    """

    def __init__(self, kind: HeadKind, config: HeadConfig) -> None:
        super().__init__()
        if kind not in HEAD_KINDS:
            raise ValueError(f"unknown head: {kind}")
        self.kind = kind
        self.config = config
        width = config.width
        self.visual_projection = nn.Linear(config.feature_dim, width)
        self.text_projection = nn.Linear(config.feature_dim, width)
        # Query, matched evidence, maximum similarity, x/y, camera, coverage.
        extra_features = 3 if kind == "pooled_temporal" else 5
        self.evidence_projection = nn.Sequential(
            nn.Linear(2 * width + extra_features, width), nn.GELU(), nn.LayerNorm(width),
        )
        if kind == "late_orderless":
            # Attention contributes ~4*w*w parameters; add 2*w to the MLP
            # hidden width to match the temporal control without unused weights.
            hidden = config.feedforward_dim + 2 * width
            self.mixer = nn.Sequential(*(
                _ResidualMLP(width, hidden, config.dropout)
                for _ in range(config.num_layers)
            ))
        else:
            self.mixer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=width, nhead=config.num_heads,
                    dim_feedforward=config.feedforward_dim,
                    dropout=config.dropout, activation="gelu",
                    batch_first=True, norm_first=True,
                ), num_layers=config.num_layers, enable_nested_tensor=False,
            )
            # TransformerEncoder clones identical initial weights; independently
            # initialise each layer so controls do not start with tied values.
            for layer in self.mixer.layers:
                for parameter in layer.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        self.outcome = nn.Linear(width, config.num_classes)

    def _validate(self, visual: Tensor, text: Tensor, text_mask: Tensor,
                  view_mask: Tensor, times: Tensor) -> None:
        if visual.ndim != 5 or text.ndim != 3:
            raise ValueError("visual must be [B,T,V,P,D] and text [B,L,D]")
        batch, frames, views, patches, features = visual.shape
        if min(batch, frames, views, patches, text.shape[1]) < 1:
            raise ValueError("empty feature axes are unsupported")
        if (features != self.config.feature_dim or text.shape[0] != batch
                or text.shape[2] != features):
            raise ValueError("feature shapes do not match HeadConfig")
        if (text_mask.shape != text.shape[:2]
                or view_mask.shape != (batch, frames, views)
                or times.shape != (batch, frames)):
            raise ValueError("mask or timestamp shape mismatch")
        if text_mask.dtype != torch.bool or view_mask.dtype != torch.bool:
            raise ValueError("text_mask and view_mask must be boolean")
        if views > self.config.max_views or math.isqrt(patches) ** 2 != patches:
            raise ValueError("too many cameras or non-square patch grid")
        if not bool(text_mask.any(dim=-1).all()) or not bool(view_mask.any(dim=-1).all()):
            raise ValueError("each example needs text and each frame a valid camera")
        if not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("times must be finite and normalized to [0,1]")

    def _evidence(self, visual: Tensor, text: Tensor, text_mask: Tensor,
                  view_mask: Tensor) -> Tensor:
        batch, frames, views, patches, _ = visual.shape
        # Mask before projection: arbitrary padded values, including NaN, must
        # never influence an example or a gradient.
        visual = visual.masked_fill(~view_mask[..., None, None], 0)
        text = text.masked_fill(~text_mask[..., None], 0)
        query = F.normalize(self.text_projection(text), dim=-1)
        grid = math.isqrt(patches)
        if self.kind == "pooled_temporal":
            visual = visual.mean(dim=-2, keepdim=True)
            patches = 1
            xy = visual.new_zeros((1, 2))
        else:
            axis = torch.linspace(-1, 1, grid, device=visual.device, dtype=visual.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        values = F.normalize(self.visual_projection(visual), dim=-1)
        values = values.flatten(2, 3)
        similarities = torch.einsum("bld,btkd->btlk", query, values)
        valid = view_mask[..., None].expand(batch, frames, views, patches).flatten(2, 3)
        similarities = similarities.masked_fill(~valid[:, :, None, :], -torch.inf)
        weights = torch.softmax(similarities / self.config.temperature, dim=-1)
        evidence = torch.einsum("btlk,btkd->btld", weights, values)
        maximum = similarities.amax(dim=-1, keepdim=True)
        xy = xy.repeat(views, 1)
        coordinates = torch.einsum("btlk,kd->btld", weights, xy)
        cameras = torch.arange(views, device=visual.device, dtype=visual.dtype)
        cameras = cameras / max(1, self.config.max_views - 1)
        camera = torch.einsum("btlk,k->btl", weights, cameras.repeat_interleave(patches))[..., None]
        coverage = (view_mask.sum(dim=-1).to(visual.dtype) / self.config.max_views)
        coverage = coverage[:, :, None, None].expand(batch, frames, text.shape[1], 1)
        query = query[:, None, :, :].expand(batch, frames, text.shape[1], -1)
        features = (query, evidence, maximum, camera, coverage)
        if self.kind != "pooled_temporal":
            features = (*features, coordinates)
        return self.evidence_projection(torch.cat(features, dim=-1))

    def forward(self, visual: Tensor, text: Tensor, text_mask: Tensor,
                view_mask: Tensor, times: Tensor) -> Tensor:
        self._validate(visual, text, text_mask, view_mask, times)
        # Cache storage may be bf16 while CPU head training uses fp32.
        dtype = self.visual_projection.weight.dtype
        visual, text = visual.to(dtype=dtype), text.to(dtype=dtype)
        evidence = self._evidence(visual, text, text_mask, view_mask)
        batch, frames, tokens, width = evidence.shape
        if self.kind == "late_orderless":
            mixed = self.mixer(evidence.mean(dim=1))
            valid = text_mask
        else:
            frequency = torch.exp(torch.arange(
                0, width, 2, device=evidence.device, dtype=dtype,
            ) * (-math.log(10000.0) / width))
            phase = times.to(dtype=dtype)[..., None] * frequency * (2 * math.pi)
            position = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
            sequence = (evidence + position[:, :, None, :]).reshape(batch, frames * tokens, width)
            valid = text_mask[:, None, :].expand(batch, frames, tokens).reshape(batch, frames * tokens)
            mixed = self.mixer(sequence, src_key_padding_mask=~valid)
        pooled = mixed.masked_fill(~valid[..., None], 0).sum(dim=1)
        pooled = pooled / valid.sum(dim=1, keepdim=True).to(dtype)
        return self.outcome(pooled)


def make_head(kind: HeadKind, config: HeadConfig = HeadConfig()) -> RewardHead:
    return RewardHead(kind, config)


def ranked_probability_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Normalized ranked probability score for zero-based ordinal class IDs."""
    if logits.ndim != 2 or logits.shape[1] != 5 or targets.shape != logits.shape[:1]:
        raise ValueError("expected five logits per example and one class target")
    probabilities = logits.softmax(dim=-1)
    observed = F.one_hot(targets, num_classes=5).to(probabilities.dtype)
    return (probabilities.cumsum(-1)[:, :-1] - observed.cumsum(-1)[:, :-1]).square().mean()


def outcome_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy + 0.25 normalized ordinal RPS; targets are 0..4."""
    return F.cross_entropy(logits, targets) + 0.25 * ranked_probability_loss(logits, targets)

"""Simple PointNet-style LiDAR adapter for AV stage-2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import AdapterConfig, EncoderAdapter
from ..types import AdapterOutput, TrainBatch


@dataclass
class LidarPointNetAdapterConfig(AdapterConfig):
    """Config for the built-in PointNet-style LiDAR adapter."""

    encoder_name: str = "pointnet_lidar"
    input_dim: int = 5
    hidden_dim: int = 128
    output_dim: int = 256


class LidarPointNetAdapter(EncoderAdapter):
    """Lightweight per-point MLP that emits token embeddings."""

    def __init__(self, config: LidarPointNetAdapterConfig):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def encode(self, batch: TrainBatch) -> AdapterOutput:
        points = batch.points
        if points.ndim != 3:
            raise ValueError(f"expected [B,N,C] points, got {points.shape}")
        if points.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"input dim mismatch: expected {self.config.input_dim}, got {points.shape[-1]}"
            )
        x = points.to(torch.float32)

        xyz = x[..., :3]
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz_centered = xyz - centroid
        scale = xyz_centered.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True).clamp_min(1e-6)
        x = x.clone()
        x[..., :3] = xyz_centered / scale

        tokens = self.norm(self.net(x))
        return AdapterOutput(tokens=tokens, belief_features=tokens)


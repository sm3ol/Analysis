"""PointRCNN adapter backed by a PointNet++ style point backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import AdapterConfig, EncoderAdapter
from .repo_backbones import PointNet2FeatureBackbone
from ..types import AdapterOutput, TrainBatch


@dataclass
class PointRCNNAdapterConfig(AdapterConfig):
    """PointRCNN backbone settings."""

    encoder_name: str = "pointrcnn"
    input_dim: int = 5
    hidden_dim: int = 256
    output_dim: int = 512


class PointRCNNAdapter(EncoderAdapter):
    """Expose PointRCNN-style set abstraction features as tokens."""

    def __init__(self, config: PointRCNNAdapterConfig):
        super().__init__(config)
        self.backbone = PointNet2FeatureBackbone(int(config.input_dim))
        self.freeze_module(self.backbone)
        self.checkpoint_status = self.load_optional_checkpoint(
            self.backbone,
            rename_rules=[
                ("backbone_net.SA_modules.0.", "sa1."),
                ("backbone_net.SA_modules.1.", "sa2."),
                ("backbone_3d.SA_modules.0.", "sa1."),
                ("backbone_3d.SA_modules.1.", "sa2."),
            ],
        )
        self.token_projection = nn.Sequential(
            nn.Linear(256, int(config.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim), int(config.output_dim)),
        )
        self.norm = nn.LayerNorm(int(config.output_dim))

    def encode(self, batch: TrainBatch) -> AdapterOutput:
        tokens, shapes = self.backbone(batch.points.to(dtype=self.norm.weight.dtype))
        tokens = self.norm(self.token_projection(tokens))
        extras = {"checkpoint": self.checkpoint_status}
        if bool(self.config.return_intermediate_shapes):
            extras["intermediate_shapes"] = shapes
        return AdapterOutput(
            tokens=tokens,
            token_mask=torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool),
            belief_features=tokens,
            extras=extras,
        )

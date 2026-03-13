"""OpenPCDet-style PointPillars adapter built from real pillar/BEV blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import AdapterConfig, EncoderAdapter
from .repo_backbones import EasyConfig, PillarBackboneConfig, PillarFeatureBackbone
from ..types import AdapterOutput, TrainBatch


@dataclass
class PointPillarsAdapterConfig(AdapterConfig):
    """PointPillars backbone settings."""

    encoder_name: str = "pointpillars"
    input_dim: int = 5
    hidden_dim: int = 256
    output_dim: int = 512
    grid_size_x: int = 128
    grid_size_y: int = 128
    max_pillars: int = 2048
    max_points_per_pillar: int = 32
    token_pool_h: int = 4
    token_pool_w: int = 4


class PointPillarsAdapter(EncoderAdapter):
    """Expose PointPillars backbone features as unified token embeddings."""

    def __init__(self, config: PointPillarsAdapterConfig):
        super().__init__(config)
        backbone_cfg = PillarBackboneConfig(
            input_dim=int(config.input_dim),
            grid_size_x=int(config.grid_size_x),
            grid_size_y=int(config.grid_size_y),
            max_pillars=int(config.max_pillars),
            max_points_per_pillar=int(config.max_points_per_pillar),
            x_range=(-54.0, 54.0),
            y_range=(-54.0, 54.0),
            z_range=(-5.0, 3.0),
            pfn_filters=(64,),
            bev_backbone_cfg=EasyConfig(
                LAYER_NUMS=[3, 5, 5],
                LAYER_STRIDES=[2, 2, 2],
                NUM_FILTERS=[64, 128, 256],
                UPSAMPLE_STRIDES=[1, 2, 4],
                NUM_UPSAMPLE_FILTERS=[128, 128, 128],
            ),
            token_pool_hw=(int(config.token_pool_h), int(config.token_pool_w)),
        )
        self.backbone = PillarFeatureBackbone(backbone_cfg)
        self.freeze_module(self.backbone)
        self.checkpoint_status = self.load_optional_checkpoint(self.backbone)
        self.token_projection = nn.Sequential(
            nn.Linear(int(self.backbone.backbone_2d.num_bev_features), int(config.hidden_dim)),
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

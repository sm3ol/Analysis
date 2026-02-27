"""Base class for encoder-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .types import AdapterOutput, TrainBatch


@dataclass
class AdapterConfig:
    """Adapter-specific runtime config."""

    encoder_name: str
    pretrained_path: str | None = None
    device: str = "cpu"
    freeze_encoder: bool = True


class EncoderAdapter(nn.Module, ABC):
    """Abstract adapter that normalizes encoder outputs."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, batch: TrainBatch) -> AdapterOutput:
        """Encode input batch into token features and optional masks."""

    def forward(self, batch: TrainBatch) -> AdapterOutput:
        return self.encode(batch)

    def build_belief_features(self, output: AdapterOutput) -> torch.Tensor:
        """Optional adapter-level belief feature extraction override."""
        if output.belief_features is not None:
            return output.belief_features
        return output.tokens

    def extra_state(self) -> dict[str, Any]:
        """Optional adapter metadata for checkpointing/logging."""
        return {}

"""Shared datatypes for the AV framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class AdapterOutput:
    """Encoder adapter output in a common shape contract."""

    tokens: torch.Tensor
    token_mask: Optional[torch.Tensor] = None
    belief_features: Optional[torch.Tensor] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainBatch:
    """Frame-level LiDAR batch payload."""

    points: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    stream_id: torch.Tensor
    corruption_family_id: Optional[torch.Tensor] = None
    is_corrupt: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityStepOutput:
    """Per-step reliability outputs from both brains plus controller state."""

    r_a: torch.Tensor
    r_b: torch.Tensor
    final_reliability: torch.Tensor
    alarm: torch.Tensor
    mode_name: str
    suspicious: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


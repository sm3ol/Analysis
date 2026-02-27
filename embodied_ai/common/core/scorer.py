"""Brain A / Brain B scoring interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .brain_b_stats import CleanReferenceStats, mahalanobis_distance


@dataclass
class BrainAScoreOutput:
    """Outputs from belief-vs-evidence scoring."""

    reliability: torch.Tensor
    raw_score: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class BrainBScoreOutput:
    """Outputs from MD-to-clean scoring."""

    reliability: torch.Tensor
    md_clean: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class BrainAScorer(nn.Module):
    """Belief-vs-evidence consistency scorer."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(latent_dim))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, belief: torch.Tensor, evidence: torch.Tensor) -> BrainAScoreOutput:
        """Compute Brain A reliability."""
        if belief.shape != evidence.shape:
            raise ValueError(f"belief shape {belief.shape} must match evidence shape {evidence.shape}")
        if belief.ndim != 2:
            raise ValueError(f"BrainAScorer expects [B,D], got {belief.shape}")
        if belief.shape[-1] != self.log_var.shape[0]:
            raise ValueError(
                f"latent dim mismatch: input D={belief.shape[-1]} vs log_var D={self.log_var.shape[0]}"
            )
        diff = evidence - belief
        inv_var = torch.exp(-torch.clamp(self.log_var, -10.0, 10.0))
        md = torch.sum(diff * diff * inv_var.unsqueeze(0), dim=-1)
        raw = -0.5 * md
        reliability = torch.sigmoid(self.beta * raw + self.bias)
        return BrainAScoreOutput(
            reliability=reliability,
            raw_score=raw,
            diagnostics={"md_belief_evidence": md},
        )


class BrainBScorer(nn.Module):
    """Frozen clean-reference Mahalanobis reliability scorer."""

    def __init__(self, stats: CleanReferenceStats, temperature: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.register_buffer("mu_clean", stats.mu_clean)
        self.register_buffer("precision_clean", stats.precision_clean)
        self.temperature = float(temperature)
        self.bias = float(bias)

    def forward(self, z: torch.Tensor) -> BrainBScoreOutput:
        """Compute Brain B reliability from MD-to-clean."""
        stats = CleanReferenceStats(
            mu_clean=self.mu_clean,
            cov_clean=torch.empty(0, device=z.device),
            precision_clean=self.precision_clean,
        )
        md = mahalanobis_distance(z, stats)
        reliability = torch.sigmoid(-md / self.temperature + self.bias)
        return BrainBScoreOutput(
            reliability=reliability,
            md_clean=md,
            diagnostics={"md_clean": md},
        )

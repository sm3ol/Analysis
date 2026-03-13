"""Supervised contrastive loss helpers."""

from __future__ import annotations

import torch
from torch import nn


def build_delta_embeddings(z_clean: torch.Tensor, z_corrupt: torch.Tensor) -> torch.Tensor:
    """Construct shift embeddings if a caller wants clean->corrupt deltas."""
    return z_corrupt - z_clean


class SupConLoss(nn.Module):
    """Supervised contrastive loss on projected embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B,D], got {embeddings.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be [B], got {labels.shape}")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(f"batch mismatch: {embeddings.shape[0]} vs {labels.shape[0]}")
        if embeddings.shape[0] < 2:
            raise ValueError("SupCon needs at least 2 samples.")

        z = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        eye = torch.eye(pos_mask.shape[0], device=pos_mask.device, dtype=pos_mask.dtype)
        pos_mask = pos_mask - eye

        logits_mask = 1.0 - eye
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not bool(valid.any().item()):
            raise ValueError("SupCon found no positive pairs in batch.")
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        return -mean_log_prob_pos[valid].mean()

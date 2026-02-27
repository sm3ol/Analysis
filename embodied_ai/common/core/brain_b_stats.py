"""Offline clean-distribution fitting utilities for Brain B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CleanReferenceStats:
    """Frozen clean distribution statistics."""

    mu_clean: torch.Tensor
    cov_clean: torch.Tensor
    precision_clean: torch.Tensor


def fit_clean_reference_stats(
    embeddings: torch.Tensor,
    shrinkage: float = 0.01,
    eps: float = 1e-5,
) -> CleanReferenceStats:
    """Fit clean mu/cov/precision from projected clean embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N,D], got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError("need at least 2 clean embeddings to estimate covariance")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError(f"shrinkage must be in [0,1], got {shrinkage}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    x = embeddings.float()
    n, d = x.shape
    mu = x.mean(dim=0)
    centered = x - mu.unsqueeze(0)
    cov = centered.T @ centered / float(max(1, n - 1))

    diag_cov = torch.diag(torch.diag(cov))
    cov_shrunk = (1.0 - float(shrinkage)) * cov + float(shrinkage) * diag_cov
    cov_reg = cov_shrunk + float(eps) * torch.eye(d, device=x.device, dtype=x.dtype)
    precision = torch.linalg.inv(cov_reg)

    return CleanReferenceStats(mu_clean=mu, cov_clean=cov_reg, precision_clean=precision)


def mahalanobis_distance(
    z: torch.Tensor,
    stats: CleanReferenceStats,
) -> torch.Tensor:
    """Compute squared Mahalanobis distance to the clean distribution."""
    delta = z - stats.mu_clean.unsqueeze(0)
    return torch.einsum("bi,ij,bj->b", delta, stats.precision_clean, delta)


def save_clean_reference_stats(stats: CleanReferenceStats, path: str | Path) -> None:
    """Serialize clean reference stats artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu_clean=stats.mu_clean.detach().cpu().numpy(),
        cov_clean=stats.cov_clean.detach().cpu().numpy(),
        precision_clean=stats.precision_clean.detach().cpu().numpy(),
    )


def load_clean_reference_stats(path: str | Path, device: str = "cpu") -> CleanReferenceStats:
    """Load clean reference stats artifact."""
    payload = np.load(Path(path))
    return CleanReferenceStats(
        mu_clean=torch.as_tensor(payload["mu_clean"], device=device, dtype=torch.float32),
        cov_clean=torch.as_tensor(payload["cov_clean"], device=device, dtype=torch.float32),
        precision_clean=torch.as_tensor(payload["precision_clean"], device=device, dtype=torch.float32),
    )

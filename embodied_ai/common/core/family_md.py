"""Mahalanobis family identification in delta-embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class FamilyMDStats:
    """Per-family Gaussian statistics in delta space."""

    family_ids: torch.Tensor
    mu: torch.Tensor
    covariance: torch.Tensor
    precision: torch.Tensor
    counts: torch.Tensor
    diag_cov: bool
    eps: float
    unseen_threshold: float = float("nan")
    threshold_source: str = "unset"

    def to(self, device: str | torch.device) -> "FamilyMDStats":
        """Move tensor fields to target device."""
        return FamilyMDStats(
            family_ids=self.family_ids.to(device),
            mu=self.mu.to(device),
            covariance=self.covariance.to(device),
            precision=self.precision.to(device),
            counts=self.counts.to(device),
            diag_cov=bool(self.diag_cov),
            eps=float(self.eps),
            unseen_threshold=float(self.unseen_threshold),
            threshold_source=str(self.threshold_source),
        )


def _fit_single_family(
    deltas: torch.Tensor,
    diag_cov: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N,D], got {deltas.shape}")
    if deltas.shape[0] <= 0:
        raise ValueError("deltas must be non-empty")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    n, d = int(deltas.shape[0]), int(deltas.shape[1])
    x = deltas.to(torch.float64)
    mu = x.mean(dim=0)
    if n >= 2:
        xc = x - mu.unsqueeze(0)
        cov = (xc.T @ xc) / float(max(1, n - 1))
    else:
        cov = torch.eye(d, dtype=torch.float64, device=x.device)
    if diag_cov:
        cov = torch.diag(torch.diag(cov))
    cov = cov + float(eps) * torch.eye(d, dtype=torch.float64, device=x.device)
    precision = torch.linalg.inv(cov)
    return (
        mu.to(torch.float32),
        cov.to(torch.float32),
        precision.to(torch.float32),
        n,
    )


def fit_family_md_stats(
    deltas: torch.Tensor,
    family_ids: torch.Tensor,
    diag_cov: bool = True,
    eps: float = 1e-4,
) -> FamilyMDStats:
    """Fit per-family Gaussian stats in delta space."""
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N,D], got {deltas.shape}")
    if family_ids.ndim != 1:
        raise ValueError(f"family_ids must be [N], got {family_ids.shape}")
    if deltas.shape[0] != family_ids.shape[0]:
        raise ValueError(f"N mismatch: deltas={deltas.shape[0]} family_ids={family_ids.shape[0]}")
    if int(deltas.shape[0]) <= 0:
        raise ValueError("Cannot fit family MD stats with zero samples.")

    x = deltas.detach().to(torch.float32).cpu()
    y = family_ids.detach().to(torch.long).cpu()
    uniq = sorted(int(v) for v in torch.unique(y).tolist())
    if not uniq:
        raise ValueError("No family ids found while fitting family MD stats.")

    mu_list: list[torch.Tensor] = []
    cov_list: list[torch.Tensor] = []
    prec_list: list[torch.Tensor] = []
    cnt_list: list[int] = []

    for fid in uniq:
        mask = y == int(fid)
        if int(mask.sum().item()) <= 0:
            continue
        mu_f, cov_f, prec_f, n_f = _fit_single_family(x[mask], diag_cov=bool(diag_cov), eps=float(eps))
        mu_list.append(mu_f)
        cov_list.append(cov_f)
        prec_list.append(prec_f)
        cnt_list.append(int(n_f))

    if not mu_list:
        raise ValueError("No non-empty families found while fitting family MD stats.")

    return FamilyMDStats(
        family_ids=torch.tensor(uniq, dtype=torch.long),
        mu=torch.stack(mu_list, dim=0),
        covariance=torch.stack(cov_list, dim=0),
        precision=torch.stack(prec_list, dim=0),
        counts=torch.tensor(cnt_list, dtype=torch.long),
        diag_cov=bool(diag_cov),
        eps=float(eps),
        unseen_threshold=float("nan"),
        threshold_source="unset",
    )


def mahalanobis_distances(
    deltas: torch.Tensor,
    stats: FamilyMDStats,
) -> torch.Tensor:
    """Compute MD to each family prototype/covariance."""
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N,D], got {deltas.shape}")
    if stats.mu.ndim != 2 or stats.precision.ndim != 3:
        raise ValueError("stats tensors have invalid shapes")
    if stats.mu.shape[0] <= 0:
        raise ValueError("stats has zero families")
    if deltas.shape[1] != stats.mu.shape[1]:
        raise ValueError(f"delta D={deltas.shape[1]} must match stats D={stats.mu.shape[1]}")

    x = deltas.to(stats.mu.device)
    diff = x.unsqueeze(1) - stats.mu.unsqueeze(0)  # [N,F,D]
    right = torch.einsum("nfd,fde->nfe", diff, stats.precision)  # [N,F,D]
    dist = torch.sum(right * diff, dim=-1)  # [N,F]
    return dist


def predict_family_md(
    deltas: torch.Tensor,
    stats: FamilyMDStats,
    unseen_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict closest family by MD and unseen flag."""
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N,D], got {deltas.shape}")
    if deltas.shape[0] == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=deltas.device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=deltas.device)
        empty_bool = torch.zeros((0,), dtype=torch.bool, device=deltas.device)
        all_d = torch.zeros((0, int(stats.family_ids.shape[0])), dtype=torch.float32, device=deltas.device)
        return empty_long, empty_float, empty_bool, all_d

    d_all = mahalanobis_distances(deltas, stats)
    min_d, argmin = torch.min(d_all, dim=1)
    pred_family = stats.family_ids.to(d_all.device)[argmin]

    th = float(stats.unseen_threshold) if unseen_threshold is None else float(unseen_threshold)
    if np.isfinite(th):
        is_unseen = min_d > th
    else:
        is_unseen = torch.zeros_like(min_d, dtype=torch.bool)
    return pred_family, min_d, is_unseen, d_all


def calibrate_unseen_threshold(
    min_distances_seen: torch.Tensor,
    percentile: float = 99.0,
) -> float:
    """Calibrate unseen threshold from seen-split min distances."""
    if min_distances_seen.ndim != 1:
        raise ValueError(f"min_distances_seen must be [N], got {min_distances_seen.shape}")
    if not (0.0 < float(percentile) < 100.0):
        raise ValueError(f"percentile must be in (0,100), got {percentile}")
    vals = min_distances_seen.detach().cpu().numpy().astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite values available to calibrate unseen threshold.")
    return float(np.percentile(vals, float(percentile)))


def save_family_md_stats(stats: FamilyMDStats, path: str | Path) -> None:
    """Save family MD stats artifact to .pt."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "family_ids": stats.family_ids.detach().cpu(),
        "mu": stats.mu.detach().cpu(),
        "covariance": stats.covariance.detach().cpu(),
        "precision": stats.precision.detach().cpu(),
        "counts": stats.counts.detach().cpu(),
        "diag_cov": bool(stats.diag_cov),
        "eps": float(stats.eps),
        "unseen_threshold": float(stats.unseen_threshold),
        "threshold_source": str(stats.threshold_source),
    }
    torch.save(payload, p)


def load_family_md_stats(path: str | Path, device: str | torch.device = "cpu") -> FamilyMDStats:
    """Load family MD stats artifact from .pt."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return FamilyMDStats(
        family_ids=payload["family_ids"].to(device),
        mu=payload["mu"].to(device),
        covariance=payload["covariance"].to(device),
        precision=payload["precision"].to(device),
        counts=payload["counts"].to(device),
        diag_cov=bool(payload.get("diag_cov", True)),
        eps=float(payload.get("eps", 1e-4)),
        unseen_threshold=float(payload.get("unseen_threshold", float("nan"))),
        threshold_source=str(payload.get("threshold_source", "unset")),
    )


def enroll_family_md(
    stats: FamilyMDStats,
    family_id: int,
    deltas: torch.Tensor,
    allow_update: bool = True,
    diag_cov: bool | None = None,
    eps: float | None = None,
) -> FamilyMDStats:
    """Enroll/update one family Gaussian in-place and return stats."""
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N,D], got {deltas.shape}")
    if deltas.shape[0] <= 0:
        raise ValueError("deltas must be non-empty for enrollment")
    if deltas.shape[1] != stats.mu.shape[1]:
        raise ValueError(f"delta D={deltas.shape[1]} must match stats D={stats.mu.shape[1]}")

    use_diag = bool(stats.diag_cov if diag_cov is None else diag_cov)
    use_eps = float(stats.eps if eps is None else eps)
    mu_f, cov_f, prec_f, n_f = _fit_single_family(
        deltas.detach().cpu(),
        diag_cov=use_diag,
        eps=use_eps,
    )

    fid = int(family_id)
    ids = stats.family_ids.detach().cpu().clone()
    mu = stats.mu.detach().cpu().clone()
    cov = stats.covariance.detach().cpu().clone()
    prec = stats.precision.detach().cpu().clone()
    counts = stats.counts.detach().cpu().clone()

    existing = torch.where(ids == fid)[0]
    if int(existing.numel()) > 0:
        if not allow_update:
            raise ValueError(f"family {fid} already exists and allow_update=False")
        idx = int(existing[0].item())
        mu[idx] = mu_f
        cov[idx] = cov_f
        prec[idx] = prec_f
        counts[idx] = int(n_f)
    else:
        ids = torch.cat([ids, torch.tensor([fid], dtype=torch.long)], dim=0)
        mu = torch.cat([mu, mu_f.unsqueeze(0)], dim=0)
        cov = torch.cat([cov, cov_f.unsqueeze(0)], dim=0)
        prec = torch.cat([prec, prec_f.unsqueeze(0)], dim=0)
        counts = torch.cat([counts, torch.tensor([int(n_f)], dtype=torch.long)], dim=0)

    order = torch.argsort(ids)
    stats.family_ids = ids[order].to(stats.family_ids.device)
    stats.mu = mu[order].to(stats.mu.device)
    stats.covariance = cov[order].to(stats.covariance.device)
    stats.precision = prec[order].to(stats.precision.device)
    stats.counts = counts[order].to(stats.counts.device)
    stats.diag_cov = use_diag
    stats.eps = use_eps
    return stats

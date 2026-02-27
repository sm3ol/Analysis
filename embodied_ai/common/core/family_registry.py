"""Few-shot corruption family enrollment and classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class FamilyStats:
    """Stored family representation."""

    family_id: str
    prototype: torch.Tensor
    diag_var: torch.Tensor | None = None
    num_shots: int = 0


@dataclass
class FamilyRegistry:
    """In-memory registry of few-shot family prototypes/stats."""

    distance_metric: str = "cosine"
    unseen_threshold: float = 0.2
    families: dict[str, FamilyStats] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enroll_family(
        self,
        family_id: int | str,
        deltas: torch.Tensor,
        method: str = "prototype",
        allow_update: bool = True,
    ) -> None:
        """Enroll/update a family from K delta embeddings [K,D]."""
        if method != "prototype":
            raise ValueError(f"unsupported method: {method}")
        if deltas.ndim != 2:
            raise ValueError(f"deltas must be [K,D], got {deltas.shape}")
        fid = str(family_id)
        if fid in self.families and not allow_update:
            raise ValueError(f"family {fid} already exists and allow_update=False")
        proto = deltas.mean(dim=0).detach().cpu()
        diag_var = None
        if deltas.shape[0] >= 10:
            diag_var = deltas.var(dim=0, unbiased=True).detach().cpu()
        self.families[fid] = FamilyStats(
            family_id=fid,
            prototype=proto,
            diag_var=diag_var,
            num_shots=int(deltas.shape[0]),
        )

    def classify(self, delta: torch.Tensor) -> tuple[str | None, float, bool]:
        """Classify one delta [D] against enrolled families."""
        if delta.ndim != 1:
            raise ValueError(f"delta must be [D], got {delta.shape}")
        if not self.families:
            return None, float("inf"), True
        d = delta.detach().cpu()
        best_family = None
        best_dist = float("inf")
        for fid, st in self.families.items():
            if self.distance_metric == "cosine":
                dist = float(1.0 - F.cosine_similarity(d.unsqueeze(0), st.prototype.unsqueeze(0), dim=-1).item())
            elif self.distance_metric == "euclidean":
                dist = float(torch.norm(d - st.prototype, p=2).item())
            else:
                raise ValueError(f"unsupported distance metric: {self.distance_metric}")
            if dist < best_dist:
                best_dist = dist
                best_family = fid
        is_unseen = bool(best_dist > self.unseen_threshold)
        return best_family, best_dist, is_unseen

    def save(self, path: str | Path) -> None:
        """Serialize registry to .pt."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "distance_metric": self.distance_metric,
            "unseen_threshold": self.unseen_threshold,
            "metadata": self.metadata,
            "families": {
                fid: {
                    "family_id": st.family_id,
                    "prototype": st.prototype,
                    "diag_var": st.diag_var,
                    "num_shots": st.num_shots,
                }
                for fid, st in self.families.items()
            },
        }
        torch.save(payload, p)

    @classmethod
    def load(cls, path: str | Path) -> "FamilyRegistry":
        """Load registry from .pt."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        reg = cls(
            distance_metric=str(payload["distance_metric"]),
            unseen_threshold=float(payload["unseen_threshold"]),
            metadata=dict(payload.get("metadata", {})),
        )
        fam_payload = payload.get("families", {})
        for fid, x in fam_payload.items():
            reg.families[str(fid)] = FamilyStats(
                family_id=str(x["family_id"]),
                prototype=x["prototype"].detach().cpu(),
                diag_var=None if x.get("diag_var") is None else x["diag_var"].detach().cpu(),
                num_shots=int(x.get("num_shots", 0)),
            )
        return reg


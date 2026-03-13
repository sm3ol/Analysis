"""AV training losses."""

from .supcon import SupConLoss, build_delta_embeddings

__all__ = ["SupConLoss", "build_delta_embeddings"]

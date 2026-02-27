"""Shared pooling for adapter outputs."""

from __future__ import annotations

import torch

from ..types import AdapterOutput


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Masked mean pool along a token-like dimension."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    while mask.ndim < tokens.ndim:
        mask = mask.unsqueeze(-1)
    mask_f = mask.to(dtype=tokens.dtype)
    num = (tokens * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1e-6)
    return num / den


def mean_pool_tokens(tokens: torch.Tensor, token_dim: int = -2) -> torch.Tensor:
    """Unmasked mean pool along token dimension."""
    return tokens.mean(dim=token_dim)


def pool_adapter_output(
    output: AdapterOutput,
    token_dim: int = -2,
) -> torch.Tensor:
    """Pool adapter output into a single per-sample embedding."""
    if output.token_mask is None:
        return mean_pool_tokens(output.tokens, token_dim=token_dim)
    return masked_mean_pool(output.tokens, output.token_mask, dim=token_dim)

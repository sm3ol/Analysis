"""Device resolution helpers for analysis scripts."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string and fail early on missing CUDA."""
    device_name = str(requested)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no GPU is visible in this session "
            "(torch.cuda.is_available() is False)."
        )
    return torch.device(device_name)

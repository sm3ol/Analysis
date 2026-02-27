"""Simple corruption helpers for embodied analysis sample episodes."""

from __future__ import annotations

import math

import torch


CORRUPTION_CHOICES = ("gaussian_noise", "brightness_drop", "center_occlusion", "noise_and_occlusion")


def _validate_severity(severity: float) -> float:
    return max(0.0, min(1.0, float(severity)))


def apply_corruption_batch(images: torch.Tensor, kind: str, severity: float, seed: int = 0) -> torch.Tensor:
    """Apply a deterministic corruption to [B,T,C,H,W] float images in [0,1]."""
    if images.ndim != 5:
        raise ValueError(f"expected images [B,T,C,H,W], got {tuple(images.shape)}")
    if int(images.shape[2]) != 3:
        raise ValueError(f"expected RGB images, got {tuple(images.shape)}")

    level = _validate_severity(severity)
    if level == 0.0 or kind == "none":
        return images.clone()

    x = images.clone().float().clamp(0.0, 1.0)
    if kind == "gaussian_noise":
        return _gaussian_noise(x, level, seed)
    if kind == "brightness_drop":
        return _brightness_drop(x, level)
    if kind == "center_occlusion":
        return _center_occlusion(x, level)
    if kind == "noise_and_occlusion":
        x = _gaussian_noise(x, level, seed)
        return _center_occlusion(x, min(1.0, level + 0.1))
    raise ValueError(f"unknown corruption `{kind}`; choices={CORRUPTION_CHOICES}")


def apply_corruption_frames(frames: torch.Tensor, kind: str, severity: float, seed: int = 0) -> torch.Tensor:
    """Apply a corruption to [T,H,W,C] uint8 frames and return uint8 frames."""
    if frames.ndim != 4:
        raise ValueError(f"expected frames [T,H,W,C], got {tuple(frames.shape)}")
    if int(frames.shape[-1]) != 3:
        raise ValueError(f"expected RGB frames, got {tuple(frames.shape)}")

    batch = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0
    batch = batch.unsqueeze(0)
    corrupted = apply_corruption_batch(batch, kind=kind, severity=severity, seed=seed).squeeze(0)
    out = (corrupted.permute(0, 2, 3, 1).contiguous() * 255.0).round().clamp(0.0, 255.0)
    return out.to(dtype=torch.uint8)


def _gaussian_noise(images: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    sigma = 0.03 + 0.17 * severity
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(images.shape, generator=generator, dtype=torch.float32)
    noise = noise.to(device=images.device, dtype=images.dtype)
    return (images + sigma * noise).clamp(0.0, 1.0)


def _brightness_drop(images: torch.Tensor, severity: float) -> torch.Tensor:
    factor = max(0.15, 1.0 - 0.75 * severity)
    return (images * factor).clamp(0.0, 1.0)


def _center_occlusion(images: torch.Tensor, severity: float) -> torch.Tensor:
    b, t, _c, h, w = images.shape
    frac = min(0.8, 0.18 + 0.42 * severity)
    patch_h = max(1, int(math.floor(h * frac)))
    patch_w = max(1, int(math.floor(w * frac)))
    top = max(0, (h - patch_h) // 2)
    left = max(0, (w - patch_w) // 2)
    out = images.clone()
    out[:, :, :, top : top + patch_h, left : left + patch_w] = 0.0
    return out

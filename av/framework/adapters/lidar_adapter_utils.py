"""Shared helper functions for built-in LiDAR adapter stand-ins."""

from __future__ import annotations

import torch


def normalize_xyz(points: torch.Tensor) -> torch.Tensor:
    """Normalize XYZ coordinates per sample."""
    if points.ndim != 2 or points.shape[-1] < 3:
        raise ValueError(f"expected [N,C>=3], got {points.shape}")
    out = points.clone()
    xyz = out[:, :3]
    centroid = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - centroid
    scale = xyz.norm(dim=-1).amax().clamp_min(1e-6)
    out[:, :3] = xyz / scale
    return out


def fixed_count_indices(count: int, target: int, device: torch.device) -> torch.Tensor:
    """Select exactly target indices by trimming or repeating."""
    if count <= 0:
        raise ValueError("count must be > 0")
    if target <= 0:
        raise ValueError("target must be > 0")
    if count >= target:
        return torch.linspace(0, count - 1, steps=target, device=device).round().long()
    return (torch.arange(target, device=device) % count).long()


def select_keypoints(points: torch.Tensor, num_tokens: int, descending_radius: bool = True) -> torch.Tensor:
    """Select a fixed number of keypoints ordered by radial distance."""
    if points.ndim != 2:
        raise ValueError(f"expected [N,C], got {points.shape}")
    radius = torch.norm(points[:, :3], dim=-1)
    order = torch.argsort(radius, descending=descending_radius)
    idx = fixed_count_indices(int(order.shape[0]), int(num_tokens), device=points.device)
    return points[order[idx]]


def build_grid_features(
    points: torch.Tensor,
    grid_size: int,
    max_tokens: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate point features over an XY grid."""
    if points.ndim != 2:
        raise ValueError(f"expected [N,C], got {points.shape}")
    if grid_size <= 0 or max_tokens <= 0:
        raise ValueError("grid_size and max_tokens must be > 0")

    xy = points[:, :2]
    mins = xy.amin(dim=0)
    spans = (xy.amax(dim=0) - mins).clamp_min(1e-6)
    norm_xy = ((xy - mins.unsqueeze(0)) / spans.unsqueeze(0)).clamp(0.0, 0.999999)
    gx = torch.clamp((norm_xy[:, 0] * float(grid_size)).long(), 0, grid_size - 1)
    gy = torch.clamp((norm_xy[:, 1] * float(grid_size)).long(), 0, grid_size - 1)
    cell_ids = gy * grid_size + gx
    unique_ids, inverse = torch.unique(cell_ids, sorted=True, return_inverse=True)

    counts = []
    for cell_idx in range(int(unique_ids.shape[0])):
        counts.append(int((inverse == cell_idx).sum().item()))
    order = sorted(range(len(counts)), key=lambda idx: (-counts[idx], idx))

    scene_center = points[:, :3].mean(dim=0)
    feat_list = []
    for cell_idx in order[:max_tokens]:
        mask = inverse == cell_idx
        cell_points = points[mask]
        mean = cell_points.mean(dim=0)
        if mode == "pillars":
            std = cell_points.std(dim=0, unbiased=False)
            density = torch.tensor(
                [float(cell_points.shape[0]) / float(points.shape[0])],
                device=points.device,
                dtype=points.dtype,
            )
            feat = torch.cat([mean, std, density], dim=0)
        elif mode == "centers":
            vmax = cell_points.amax(dim=0)
            center_offset = mean[:3] - scene_center
            feat = torch.cat([mean, vmax, center_offset], dim=0)
        else:
            raise ValueError(f"unsupported grid feature mode: {mode}")
        feat_list.append(feat)

    if not feat_list:
        raise RuntimeError("grid aggregation produced zero tokens")

    feature_dim = int(feat_list[0].shape[0])
    tokens = torch.zeros((max_tokens, feature_dim), device=points.device, dtype=points.dtype)
    mask = torch.zeros((max_tokens,), device=points.device, dtype=torch.bool)
    stack = torch.stack(feat_list, dim=0)
    count = int(stack.shape[0])
    tokens[:count] = stack
    mask[:count] = True
    return tokens, mask


"""Repo-backed LiDAR feature extractors for AV encoders.

The pillar blocks are adapted from OpenPCDet's pure-PyTorch modules:
- PillarVFE
- PointPillarScatter
- BaseBEVBackbone

The point-set blocks are a pure-Torch PointNet++ style fallback that keeps the
same architectural family when the CUDA pointnet2 extensions are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class EasyConfig(dict):
    """Small dict-like config object with attribute access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


class VFETemplate(nn.Module):
    """Minimal OpenPCDet VFE template."""

    def __init__(self, model_cfg, **kwargs):
        del kwargs
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError


class PFNLayer(nn.Module):
    """OpenPCDet PFN layer."""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True, last_layer: bool = False):
        super().__init__()
        self.last_vfe = bool(last_layer)
        self.use_norm = bool(use_norm)
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.part = 50000

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [
                self.linear(inputs[num_part * self.part : (num_part + 1) * self.part])
                for num_part in range(num_parts + 1)
            ]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        if self.use_norm:
            cudnn_state = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            torch.backends.cudnn.enabled = cudnn_state
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)


class PillarVFE(VFETemplate):
    """OpenPCDet PillarVFE, adapted with the same feature construction."""

    def __init__(self, model_cfg, num_point_features: int, voxel_size: tuple[float, float, float], point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = bool(self.model_cfg.USE_NORM)
        self.with_distance = bool(self.model_cfg.WITH_DISTANCE)
        self.use_absolute_xyz = bool(self.model_cfg.USE_ABSLOTE_XYZ)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = list(self.model_cfg.NUM_FILTERS)
        num_filters = [num_point_features] + list(self.num_filters)
        self.pfn_layers = nn.ModuleList(
            [
                PFNLayer(
                    num_filters[i],
                    num_filters[i + 1],
                    self.use_norm,
                    last_layer=(i >= len(num_filters) - 2),
                )
                for i in range(len(num_filters) - 1)
            ]
        )

        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num: torch.Tensor, max_num: int, axis: int = 0) -> torch.Tensor:
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num_tensor = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        return actual_num.int() > max_num_tensor

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        voxel_features = batch_dict["voxels"]
        voxel_num_points = batch_dict["voxel_num_points"]
        coords = batch_dict["voxel_coords"]

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(
            voxel_features
        ).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        )
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        )

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True))
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        features = features * mask.unsqueeze(-1).type_as(voxel_features)
        for pfn in self.pfn_layers:
            features = pfn(features)
        batch_dict["pillar_features"] = features.squeeze(1)
        return batch_dict


class PointPillarScatter(nn.Module):
    """OpenPCDet scatter to BEV pseudo-image."""

    def __init__(self, model_cfg, grid_size: tuple[int, int, int]):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = int(self.model_cfg.NUM_BEV_FEATURES)
        self.nx, self.ny, self.nz = [int(v) for v in grid_size]
        if self.nz != 1:
            raise ValueError("PointPillarScatter expects nz=1")

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pillar_features = batch_dict["pillar_features"]
        coords = batch_dict["voxel_coords"]
        batch_size = int(batch_dict["batch_size"])
        batch_spatial_features = []

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            )
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            if this_coords.numel() == 0:
                batch_spatial_features.append(spatial_feature)
                continue
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.long()
            pillars = pillar_features[batch_mask, :].t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        spatial = torch.stack(batch_spatial_features, dim=0)
        spatial = spatial.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict["spatial_features"] = spatial
        return batch_dict


class BaseBEVBackbone(nn.Module):
    """OpenPCDet BaseBEVBackbone."""

    def __init__(self, model_cfg, input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = list(self.model_cfg.get("LAYER_NUMS", []))
        layer_strides = list(self.model_cfg.get("LAYER_STRIDES", []))
        num_filters = list(self.model_cfg.get("NUM_FILTERS", []))
        upsample_strides = list(self.model_cfg.get("UPSAMPLE_STRIDES", []))
        num_upsample_filters = list(self.model_cfg.get("NUM_UPSAMPLE_FILTERS", []))
        if layer_nums and not (
            len(layer_nums) == len(layer_strides) == len(num_filters) == len(upsample_strides) == len(num_upsample_filters)
        ):
            raise ValueError("BaseBEVBackbone config lengths must match")

        num_levels = len(layer_nums)
        c_in_list = [int(input_channels), *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            stride = float(upsample_strides[idx])
            if stride >= 1.0:
                stride_i = int(round(stride))
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx],
                            num_upsample_filters[idx],
                            stride_i,
                            stride=stride_i,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                )
            else:
                stride_i = int(round(1.0 / max(stride, 1e-6)))
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx],
                            num_upsample_filters[idx],
                            stride_i,
                            stride=stride_i,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                )
        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        spatial_features = data_dict["spatial_features"]
        ups = []
        x = spatial_features
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        data_dict["spatial_features_2d"] = x
        return data_dict


@dataclass
class PillarBackboneConfig:
    """Common settings for pillar-based encoders."""

    input_dim: int
    grid_size_x: int
    grid_size_y: int
    max_pillars: int
    max_points_per_pillar: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    bev_backbone_cfg: EasyConfig
    token_pool_hw: tuple[int, int]
    pfn_filters: tuple[int, ...] = (64,)


class PillarFeatureBackbone(nn.Module):
    """Pillarization + OpenPCDet VFE/scatter/BEV backbone."""

    def __init__(self, config: PillarBackboneConfig):
        super().__init__()
        self.config = config
        voxel_x = (config.x_range[1] - config.x_range[0]) / float(config.grid_size_x)
        voxel_y = (config.y_range[1] - config.y_range[0]) / float(config.grid_size_y)
        voxel_z = max(config.z_range[1] - config.z_range[0], 1e-3)

        vfe_cfg = EasyConfig(
            USE_NORM=True,
            WITH_DISTANCE=False,
            USE_ABSLOTE_XYZ=True,
            NUM_FILTERS=list(config.pfn_filters),
        )
        self.vfe = PillarVFE(
            model_cfg=vfe_cfg,
            num_point_features=int(config.input_dim),
            voxel_size=(voxel_x, voxel_y, voxel_z),
            point_cloud_range=(
                config.x_range[0],
                config.y_range[0],
                config.z_range[0],
                config.x_range[1],
                config.y_range[1],
                config.z_range[1],
            ),
        )
        self.scatter = PointPillarScatter(EasyConfig(NUM_BEV_FEATURES=self.vfe.get_output_feature_dim()), (config.grid_size_x, config.grid_size_y, 1))
        self.backbone_2d = BaseBEVBackbone(config.bev_backbone_cfg, input_channels=self.vfe.get_output_feature_dim())

    def _sample_indices(self, count: int, target: int, device: torch.device) -> torch.Tensor:
        if count >= target:
            return torch.linspace(0, count - 1, steps=target, device=device).round().long()
        return (torch.arange(target, device=device) % count).long()

    def _pillarize_sample(self, sample: torch.Tensor, batch_idx: int) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor]]:
        x_min, x_max = self.config.x_range
        y_min, y_max = self.config.y_range
        voxel_x = (x_max - x_min) / float(self.config.grid_size_x)
        voxel_y = (y_max - y_min) / float(self.config.grid_size_y)

        x = sample[:, 0]
        y = sample[:, 1]
        valid = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        pts = sample[valid]
        if pts.numel() == 0:
            zero_voxel = sample.new_zeros((self.config.max_points_per_pillar, sample.shape[-1]))
            zero_coord = torch.tensor(
                [batch_idx, 0, self.config.grid_size_y // 2, self.config.grid_size_x // 2],
                device=sample.device,
                dtype=torch.long,
            )
            return [zero_voxel], [1], [zero_coord]

        x_idx = torch.clamp(((pts[:, 0] - x_min) / voxel_x).floor().long(), 0, self.config.grid_size_x - 1)
        y_idx = torch.clamp(((pts[:, 1] - y_min) / voxel_y).floor().long(), 0, self.config.grid_size_y - 1)
        cell_id = y_idx * self.config.grid_size_x + x_idx
        unique_ids, inverse, counts = torch.unique(cell_id, sorted=False, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True)
        selected = order[: min(int(order.shape[0]), int(self.config.max_pillars))]

        voxels: list[torch.Tensor] = []
        num_points: list[int] = []
        coords: list[torch.Tensor] = []
        for sel in selected.tolist():
            point_idx = torch.nonzero(inverse == sel, as_tuple=False).squeeze(1)
            chosen = pts[point_idx]
            if chosen.shape[0] == 0:
                continue
            keep_idx = self._sample_indices(int(chosen.shape[0]), int(self.config.max_points_per_pillar), chosen.device)
            padded = chosen[keep_idx]
            voxels.append(padded)
            num_points.append(min(int(chosen.shape[0]), int(self.config.max_points_per_pillar)))
            raw_id = int(unique_ids[sel].item())
            coord = torch.tensor(
                [batch_idx, 0, raw_id // self.config.grid_size_x, raw_id % self.config.grid_size_x],
                device=sample.device,
                dtype=torch.long,
            )
            coords.append(coord)

        if not voxels:
            zero_voxel = sample.new_zeros((self.config.max_points_per_pillar, sample.shape[-1]))
            zero_coord = torch.tensor(
                [batch_idx, 0, self.config.grid_size_y // 2, self.config.grid_size_x // 2],
                device=sample.device,
                dtype=torch.long,
            )
            return [zero_voxel], [1], [zero_coord]
        return voxels, num_points, coords

    def build_batch_dict(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        all_voxels: list[torch.Tensor] = []
        all_num_points: list[int] = []
        all_coords: list[torch.Tensor] = []
        batch_size = int(points.shape[0])
        for batch_idx in range(batch_size):
            voxels, num_points, coords = self._pillarize_sample(points[batch_idx], batch_idx)
            all_voxels.extend(voxels)
            all_num_points.extend(num_points)
            all_coords.extend(coords)
        return {
            "batch_size": batch_size,
            "voxels": torch.stack(all_voxels, dim=0),
            "voxel_num_points": torch.tensor(all_num_points, device=points.device, dtype=torch.int32),
            "voxel_coords": torch.stack(all_coords, dim=0),
        }

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        batch_dict = self.build_batch_dict(points)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        spatial = batch_dict["spatial_features_2d"]
        pooled = F.adaptive_avg_pool2d(spatial, self.config.token_pool_hw)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        shapes = {
            "voxels": tuple(batch_dict["voxels"].shape),
            "pillar_features": tuple(batch_dict["pillar_features"].shape),
            "spatial_features": tuple(batch_dict["spatial_features"].shape),
            "spatial_features_2d": tuple(spatial.shape),
            "tokens": tuple(tokens.shape),
        }
        return tokens, shapes


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Naive batched farthest point sampling in pure Torch."""
    batch_size, num_points, _ = xyz.shape
    device = xyz.device
    npoint = min(int(npoint), int(num_points))
    centroids = torch.zeros((batch_size, npoint), device=device, dtype=torch.long)
    distance = torch.full((batch_size, num_points), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, num_points, (batch_size,), device=device)
    batch_indices = torch.arange(batch_size, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1).indices
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points with batched indices."""
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class PointNetSetAbstractionMSG(nn.Module):
    """Pure-Torch PointNet++ style set abstraction with multi-scale KNN groups."""

    def __init__(self, npoint: int, nsamples: Iterable[int], in_channels: int, mlps: Iterable[Iterable[int]]):
        super().__init__()
        self.npoint = int(npoint)
        self.nsamples = [int(v) for v in nsamples]
        self.mlps = nn.ModuleList()
        for nsample, mlp_spec in zip(self.nsamples, mlps):
            del nsample
            dims = [in_channels + 3, *[int(v) for v in mlp_spec]]
            layers = []
            for idx in range(len(dims) - 1):
                layers.extend(
                    [
                        nn.Conv2d(dims[idx], dims[idx + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(dims[idx + 1]),
                        nn.ReLU(),
                    ]
                )
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        new_features = []
        for nsample, mlp in zip(self.nsamples, self.mlps):
            dists = torch.cdist(new_xyz, xyz)
            idx = dists.topk(k=min(int(nsample), int(xyz.shape[1])), dim=-1, largest=False).indices
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
            if features is not None:
                grouped_features = index_points(features, idx)
                grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped = grouped_xyz
            grouped = grouped.permute(0, 3, 1, 2).contiguous()
            encoded = mlp(grouped)
            pooled = torch.max(encoded, dim=-1).values
            new_features.append(pooled)
        fused = torch.cat(new_features, dim=1).transpose(1, 2).contiguous()
        return new_xyz, fused


class PointNet2FeatureBackbone(nn.Module):
    """PointNet++ style backbone adapted for PointRCNN/PV-RCNN embeddings."""

    def __init__(self, input_dim: int):
        super().__init__()
        feature_dim = max(0, int(input_dim) - 3)
        self.sa1 = PointNetSetAbstractionMSG(
            npoint=128,
            nsamples=(16, 32),
            in_channels=feature_dim,
            mlps=((16, 16, 32), (32, 32, 64)),
        )
        self.sa2 = PointNetSetAbstractionMSG(
            npoint=32,
            nsamples=(16, 32),
            in_channels=96,
            mlps=((64, 64, 128), (64, 96, 128)),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        xyz = points[:, :, :3].contiguous()
        features = points[:, :, 3:].contiguous() if points.shape[-1] > 3 else None
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        return l2_features, {
            "sa1_xyz": tuple(l1_xyz.shape),
            "sa1_features": tuple(l1_features.shape),
            "sa2_xyz": tuple(l2_xyz.shape),
            "sa2_features": tuple(l2_features.shape),
        }

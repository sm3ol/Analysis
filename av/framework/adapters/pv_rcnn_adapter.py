"""Faithful OpenPCDet PV-RCNN adapter for AV stage-2 LiDAR scoring."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import AdapterConfig, EncoderAdapter
from ..paths import VENDOR_OPENPCDET_ROOT
from ..types import AdapterOutput, TrainBatch


def _ensure_openpcdet_importable(openpcdet_root: Path) -> None:
    root = openpcdet_root.resolve()
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))


def _load_openpcdet_pvrcnn_cfg(openpcdet_root: Path, model_cfg_path: Path) -> Any:
    _ensure_openpcdet_importable(openpcdet_root)
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file

    tools_dir = (openpcdet_root / "tools").resolve()
    cfg_path = model_cfg_path.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"PV-RCNN config file not found: {cfg_path}")
    if not tools_dir.exists():
        raise FileNotFoundError(f"OpenPCDet tools directory not found: {tools_dir}")

    cfg = EasyDict()
    cfg.ROOT_DIR = openpcdet_root.resolve()
    cfg.LOCAL_RANK = 0

    old_cwd = os.getcwd()
    try:
        os.chdir(tools_dir)
        if cfg_path.is_relative_to(tools_dir):
            rel_cfg = str(cfg_path.relative_to(tools_dir))
        else:
            rel_cfg = str(cfg_path)
        cfg_from_yaml_file(rel_cfg, cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def _find_voxel_processor_cfg(data_cfg: Any) -> Any:
    for proc in data_cfg.DATA_PROCESSOR:
        if str(proc.NAME) == "transform_points_to_voxels":
            return proc
    raise ValueError("OpenPCDet DATA_PROCESSOR does not contain transform_points_to_voxels")


class _OpenPCDetVoxelGeneratorWrapper:
    """Official OpenPCDet voxel generator wrapper (spconv-backed)."""

    def __init__(
        self,
        vsize_xyz: list[float] | tuple[float, ...],
        coors_range_xyz: list[float] | tuple[float, ...],
        num_point_features: int,
        max_num_points_per_voxel: int,
        max_num_voxels: int,
    ) -> None:
        self.spconv_ver = 1
        self._tv = None
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except Exception:
            try:
                from spconv.utils import VoxelGenerator
            except Exception:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2
            else:
                self.spconv_ver = 1
        else:
            self.spconv_ver = 1

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels,
            )
        else:
            import cumm.tensorview as tv

            self._tv = tv
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels,
            )

    def generate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels = voxel_output["voxels"]
                coords = voxel_output["coordinates"]
                num_points = voxel_output["num_points_per_voxel"]
            else:
                voxels, coords, num_points = voxel_output
            return voxels, coords, num_points

        assert self._tv is not None
        tv_voxels, tv_coords, tv_num_points = self._voxel_generator.point_to_voxel(self._tv.from_numpy(points))
        return tv_voxels.numpy(), tv_coords.numpy(), tv_num_points.numpy()


class _OpenPCDetPVRCNNBackbone(nn.Module):
    """OpenPCDet PV-RCNN feature path: VFE -> 3D sparse backbone -> BEV -> PFE -> 2D backbone."""

    def __init__(self, pcdet_cfg: Any, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.point_cloud_range = np.asarray(pcdet_cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        voxel_proc = _find_voxel_processor_cfg(pcdet_cfg.DATA_CONFIG)
        self.voxel_size = np.asarray(voxel_proc.VOXEL_SIZE, dtype=np.float32)
        self.max_points_per_voxel = int(voxel_proc.MAX_POINTS_PER_VOXEL)
        self.max_voxels_test = int(voxel_proc.MAX_NUMBER_OF_VOXELS["test"])
        self.grid_size = np.round((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(
            np.int64
        )

        from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
        from pcdet.models.backbones_2d.map_to_bev.height_compression import HeightCompression
        from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import VoxelSetAbstraction
        from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
        from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE

        self.vfe = MeanVFE(
            model_cfg=pcdet_cfg.MODEL.VFE,
            num_point_features=self.input_dim,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            depth_downsample_factor=None,
        )
        self.backbone_3d = VoxelBackBone8x(
            model_cfg=pcdet_cfg.MODEL.BACKBONE_3D,
            input_channels=int(self.vfe.get_output_feature_dim()),
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
        )
        self.map_to_bev_module = HeightCompression(
            model_cfg=pcdet_cfg.MODEL.MAP_TO_BEV,
            grid_size=self.grid_size,
        )
        self.pfe = VoxelSetAbstraction(
            model_cfg=pcdet_cfg.MODEL.PFE,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            num_bev_features=int(self.map_to_bev_module.num_bev_features),
            num_rawpoint_features=self.input_dim,
        )
        self.backbone_2d = BaseBEVBackbone(
            model_cfg=pcdet_cfg.MODEL.BACKBONE_2D,
            input_channels=int(self.map_to_bev_module.num_bev_features),
        )
        self.num_keypoints = int(pcdet_cfg.MODEL.PFE.NUM_KEYPOINTS)
        self.output_feature_dim = int(self.pfe.num_point_features) + int(self.backbone_2d.num_bev_features)
        self._voxel_generator = _OpenPCDetVoxelGeneratorWrapper(
            vsize_xyz=self.voxel_size.tolist(),
            coors_range_xyz=self.point_cloud_range.tolist(),
            num_point_features=self.input_dim,
            max_num_points_per_voxel=self.max_points_per_voxel,
            max_num_voxels=self.max_voxels_test,
        )

    def _sanitize_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2:
            pts = pts.reshape(-1, max(1, pts.shape[-1]))
        if pts.shape[1] > self.input_dim:
            pts = pts[:, : self.input_dim]
        elif pts.shape[1] < self.input_dim:
            pad = np.zeros((pts.shape[0], self.input_dim - pts.shape[1]), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=1)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.shape[0] == 0:
            center = (self.point_cloud_range[:3] + self.point_cloud_range[3:6]) * 0.5
            fallback = np.zeros((1, self.input_dim), dtype=np.float32)
            fallback[0, :3] = center.astype(np.float32)
            pts = fallback
        return pts

    def _voxelize_batch(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if not points.is_cuda:
            raise RuntimeError("OpenPCDet PV-RCNN adapter requires CUDA tensors (pointnet2_stack CUDA ops).")

        voxels_list: list[np.ndarray] = []
        coords_list: list[np.ndarray] = []
        num_points_list: list[np.ndarray] = []
        raw_points_list: list[np.ndarray] = []

        for batch_idx in range(int(points.shape[0])):
            sample = self._sanitize_points(points[batch_idx].detach().float().cpu().numpy())
            voxels, coords, num_points = self._voxel_generator.generate(sample)
            if voxels.shape[0] == 0:
                voxels = np.zeros((1, self.max_points_per_voxel, self.input_dim), dtype=np.float32)
                voxels[0, 0, :] = sample[0]
                coords = np.zeros((1, 3), dtype=np.int32)
                num_points = np.ones((1,), dtype=np.int32)

            coords = np.asarray(coords, dtype=np.int32)
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            if coords.shape[1] > 3:
                coords = coords[:, -3:]

            coords_with_batch = np.concatenate(
                [np.full((coords.shape[0], 1), batch_idx, dtype=np.int32), coords.astype(np.int32)],
                axis=1,
            )
            points_with_batch = np.concatenate(
                [np.full((sample.shape[0], 1), float(batch_idx), dtype=np.float32), sample.astype(np.float32)],
                axis=1,
            )

            voxels_list.append(np.asarray(voxels, dtype=np.float32))
            coords_list.append(coords_with_batch)
            num_points_list.append(np.asarray(num_points, dtype=np.int32))
            raw_points_list.append(points_with_batch)

        device = points.device
        return {
            "batch_size": int(points.shape[0]),
            "voxels": torch.from_numpy(np.concatenate(voxels_list, axis=0)).to(device=device, dtype=torch.float32),
            "voxel_coords": torch.from_numpy(np.concatenate(coords_list, axis=0)).to(device=device, dtype=torch.int32),
            "voxel_num_points": torch.from_numpy(np.concatenate(num_points_list, axis=0)).to(device=device, dtype=torch.int32),
            "points": torch.from_numpy(np.concatenate(raw_points_list, axis=0)).to(device=device, dtype=torch.float32),
        }

    def _keypoint_features_per_batch(
        self,
        point_features: torch.Tensor,
        point_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        features_by_batch: list[torch.Tensor] = []
        point_batch_idx = point_coords[:, 0].long()
        for batch_idx in range(batch_size):
            batch_mask = point_batch_idx == batch_idx
            cur = point_features[batch_mask]
            if cur.shape[0] == 0:
                cur = point_features.new_zeros((1, point_features.shape[1]))
            if cur.shape[0] < self.num_keypoints:
                repeat_idx = torch.arange(self.num_keypoints, device=cur.device) % cur.shape[0]
                cur = cur[repeat_idx]
            elif cur.shape[0] > self.num_keypoints:
                cur = cur[: self.num_keypoints]
            features_by_batch.append(cur)
        return torch.stack(features_by_batch, dim=0)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        if points.is_cuda:
            torch.cuda.set_device(points.device)
        batch_dict = self._voxelize_batch(points)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        point_features = self._keypoint_features_per_batch(
            point_features=batch_dict["point_features"],
            point_coords=batch_dict["point_coords"],
            batch_size=int(points.shape[0]),
        )
        point_pooled = point_features.mean(dim=1)
        bev_pooled = F.adaptive_avg_pool2d(batch_dict["spatial_features_2d"], output_size=1).flatten(1)
        fused = torch.cat([point_pooled, bev_pooled], dim=1)
        shapes = {
            "voxels": tuple(batch_dict["voxels"].shape),
            "voxel_coords": tuple(batch_dict["voxel_coords"].shape),
            "voxel_num_points": tuple(batch_dict["voxel_num_points"].shape),
            "points": tuple(batch_dict["points"].shape),
            "point_features": tuple(batch_dict["point_features"].shape),
            "point_coords": tuple(batch_dict["point_coords"].shape),
            "spatial_features": tuple(batch_dict["spatial_features"].shape),
            "spatial_features_2d": tuple(batch_dict["spatial_features_2d"].shape),
            "point_features_batched": tuple(point_features.shape),
            "fused_features": tuple(fused.shape),
        }
        return fused, shapes


@dataclass
class PVRCNNAdapterConfig(AdapterConfig):
    """PV-RCNN backbone settings (official OpenPCDet path)."""

    encoder_name: str = "pvrcnn"
    input_dim: int = 4
    hidden_dim: int = 512
    output_dim: int = 512
    openpcdet_root: str = str(VENDOR_OPENPCDET_ROOT)
    model_cfg_path: str = str(VENDOR_OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pv_rcnn.yaml")


class PVRCNNAdapter(EncoderAdapter):
    """Adapter that exposes official OpenPCDet PV-RCNN features as a unified embedding."""

    def __init__(self, config: PVRCNNAdapterConfig):
        super().__init__(config)
        openpcdet_root = Path(config.openpcdet_root)
        model_cfg_path = Path(config.model_cfg_path)
        pcdet_cfg = _load_openpcdet_pvrcnn_cfg(openpcdet_root=openpcdet_root, model_cfg_path=model_cfg_path)

        self.backbone = _OpenPCDetPVRCNNBackbone(
            pcdet_cfg=pcdet_cfg,
            input_dim=int(config.input_dim),
        )
        self.freeze_module(self.backbone)
        self.checkpoint_status = self.load_optional_checkpoint(self.backbone)

        self.projection = nn.Sequential(
            nn.Linear(int(self.backbone.output_feature_dim), int(config.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim), int(config.output_dim)),
        )
        self.norm = nn.LayerNorm(int(config.output_dim))

    def encode(self, batch: TrainBatch) -> AdapterOutput:
        fused, shapes = self.backbone(batch.points.to(dtype=self.norm.weight.dtype))
        embedding = self.norm(self.projection(fused))
        tokens = embedding.unsqueeze(1)
        extras = {"checkpoint": self.checkpoint_status}
        if bool(self.config.return_intermediate_shapes):
            extras["intermediate_shapes"] = shapes
        return AdapterOutput(
            tokens=tokens,
            token_mask=torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool),
            belief_features=tokens,
            extras=extras,
        )

"""Official pretrained checkpoint catalog for AV LiDAR encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import CHECKPOINTS_ROOT


@dataclass(frozen=True)
class OfficialCheckpoint:
    """Source metadata for one encoder checkpoint."""

    encoder_name: str
    source_repo: str
    source_url: str
    file_id: str
    local_filename: str
    expected_point_feature_dim: int
    notes: str = ""


OFFICIAL_CHECKPOINTS: dict[str, OfficialCheckpoint] = {
    "pointpillars": OfficialCheckpoint(
        encoder_name="pointpillars",
        source_repo="OpenPCDet",
        source_url="https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing",
        file_id="1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm",
        local_filename="pointpillars_kitti_openpcdet.pth",
        expected_point_feature_dim=4,
        notes="OpenPCDet KITTI PointPillar model",
    ),
    "pointrcnn": OfficialCheckpoint(
        encoder_name="pointrcnn",
        source_repo="OpenPCDet",
        source_url="https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing",
        file_id="1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU",
        local_filename="pointrcnn_kitti_openpcdet.pth",
        expected_point_feature_dim=4,
        notes="OpenPCDet KITTI PointRCNN model",
    ),
    "pvrcnn": OfficialCheckpoint(
        encoder_name="pvrcnn",
        source_repo="OpenPCDet",
        source_url="https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing",
        file_id="1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-",
        local_filename="pvrcnn_kitti_openpcdet.pth",
        expected_point_feature_dim=4,
        notes="OpenPCDet KITTI PV-RCNN model",
    ),
    "centerpoint": OfficialCheckpoint(
        encoder_name="centerpoint",
        source_repo="OpenPCDet",
        source_url="https://drive.google.com/file/d/1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v/view?usp=sharing",
        file_id="1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v",
        local_filename="centerpoint_pp_nuscenes_openpcdet.pth",
        expected_point_feature_dim=5,
        notes="OpenPCDet nuScenes CenterPoint-PointPillar model",
    ),
}


def canonical_encoder_name(name: str) -> str:
    text = str(name).strip().lower()
    if text in ("pv_rcnn", "pvrcnn"):
        return "pvrcnn"
    if text in ("point_rcnn", "pointrcnn"):
        return "pointrcnn"
    return text


def official_checkpoint_for_encoder(name: str) -> OfficialCheckpoint:
    key = canonical_encoder_name(name)
    if key not in OFFICIAL_CHECKPOINTS:
        raise KeyError(f"no official checkpoint entry for encoder '{name}'")
    return OFFICIAL_CHECKPOINTS[key]


def default_checkpoint_path(name: str, checkpoint_root: str | Path | None = None) -> Path:
    entry = official_checkpoint_for_encoder(name)
    root = CHECKPOINTS_ROOT if checkpoint_root is None else Path(checkpoint_root)
    return root / entry.local_filename

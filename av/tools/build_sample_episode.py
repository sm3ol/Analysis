#!/usr/bin/env python3
"""Build one realistic AV LiDAR episode artifact for local inference smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_stream_name(path: Path) -> str:
    name = path.name
    if "__LIDAR_TOP__" in name:
        return name.split("__LIDAR_TOP__")[0]
    if "__" in name:
        return name.split("__")[0]
    return path.stem


def frame_sort_key(path: Path) -> tuple[str, str]:
    name = path.name
    suffix = name.split("__LIDAR_TOP__")[-1] if "__LIDAR_TOP__" in name else name
    digits = "".join(ch for ch in suffix if ch.isdigit())
    ts = int(digits) if digits else 0
    return (f"{ts:020d}", name)


def infer_feature_dim(flat: np.ndarray, preferred: int) -> int:
    if preferred > 0 and flat.size % preferred == 0:
        return preferred
    for candidate in (5, 4, 6, 3):
        if flat.size % candidate == 0:
            return candidate
    raise ValueError(f"unable to infer feature dim from flat length {flat.size}")


def sample_points(points: np.ndarray, max_points: int, min_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if points.shape[0] >= max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx].astype(np.float32, copy=False)
    if points.shape[0] < min_points:
        idx = rng.choice(points.shape[0], size=min_points, replace=True)
        points = points[idx]
    if points.shape[0] < max_points:
        pad_idx = rng.choice(points.shape[0], size=max_points - points.shape[0], replace=True)
        points = np.concatenate([points, points[pad_idx]], axis=0)
    return points.astype(np.float32, copy=False)


def build_episode(
    data_root: Path,
    output_path: Path,
    manifest_path: Path,
    episode_len: int,
    clean_prefix: int,
    point_feature_dim: int,
    max_points: int,
    min_points: int,
) -> None:
    bins = sorted(data_root.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"no .bin files found under {data_root}")

    grouped: dict[str, list[Path]] = {}
    for p in bins:
        grouped.setdefault(parse_stream_name(p), []).append(p)
    grouped = {k: sorted(v, key=frame_sort_key) for k, v in grouped.items()}

    candidates = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    stream_name, frames = candidates[0]
    if len(frames) < episode_len:
        raise RuntimeError(f"stream '{stream_name}' has only {len(frames)} frames (< {episode_len})")

    selected = frames[:episode_len]
    episode_points: list[np.ndarray] = []
    frame_names: list[str] = []
    for t, frame_path in enumerate(selected):
        flat = np.fromfile(frame_path, dtype=np.float32)
        if flat.size == 0:
            raise ValueError(f"{frame_path}: empty LiDAR file")
        feat_dim = infer_feature_dim(flat, preferred=point_feature_dim)
        points = flat.reshape(-1, feat_dim)
        if feat_dim > point_feature_dim:
            points = points[:, :point_feature_dim]
        elif feat_dim < point_feature_dim:
            pad = np.zeros((points.shape[0], point_feature_dim - feat_dim), dtype=np.float32)
            points = np.concatenate([points, pad], axis=1)
        sampled = sample_points(points, max_points=max_points, min_points=min_points, seed=104729 * (t + 1))
        episode_points.append(sampled)
        frame_names.append(frame_path.name)

    arr = np.stack(episode_points, axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=arr,
        stream_name=stream_name,
        frame_names=np.asarray(frame_names, dtype=object),
        clean_prefix=np.int64(clean_prefix),
        point_feature_dim=np.int64(point_feature_dim),
        max_points=np.int64(max_points),
    )

    root_hint = Path(__file__).resolve().parents[2]
    try:
        data_root_text = str(data_root.resolve().relative_to(root_hint.resolve()))
    except Exception:
        data_root_text = str(data_root)

    manifest = {
        "dataset": "nuScenes",
        "episode_file": output_path.name,
        "data_root": data_root_text,
        "stream_name": stream_name,
        "episode_len": int(episode_len),
        "clean_prefix": int(clean_prefix),
        "point_feature_dim": int(point_feature_dim),
        "max_points": int(max_points),
        "frame_count": int(len(frame_names)),
        "frame_names_head": frame_names[:5],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one realistic AV LiDAR episode artifact.")
    p.add_argument(
        "--data_root",
        type=str,
        default="dataset/raw/LIDAR_TOP",
    )
    p.add_argument("--episode_len", type=int, default=180)
    p.add_argument("--clean_prefix", type=int, default=50)
    p.add_argument("--point_feature_dim", type=int, default=5)
    p.add_argument("--max_points", type=int, default=1024)
    p.add_argument("--min_points", type=int, default=32)
    p.add_argument("--output_path", type=str, default="dataset/episode_000001.npz")
    p.add_argument("--manifest_path", type=str, default="dataset/sample_manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_episode(
        data_root=Path(args.data_root),
        output_path=Path(args.output_path),
        manifest_path=Path(args.manifest_path),
        episode_len=int(args.episode_len),
        clean_prefix=int(args.clean_prefix),
        point_feature_dim=int(args.point_feature_dim),
        max_points=int(args.max_points),
        min_points=int(args.min_points),
    )
    print(f"[OK] wrote {args.output_path}")
    print(f"[OK] wrote {args.manifest_path}")


if __name__ == "__main__":
    main()

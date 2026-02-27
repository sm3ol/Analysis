#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tiny synthetic nuScenes-like LiDAR tree")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--sequence_prefix", type=str, default="sceneA")
    p.add_argument("--num_frames", type=int, default=20)
    p.add_argument("--points_per_frame", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    lidar_dir = Path(args.output_root) / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_frames):
        pts = rng.standard_normal((args.points_per_frame, 5), dtype=np.float32)
        pts[:, 0] += i * 0.01
        out = lidar_dir / f"{args.sequence_prefix}__{i:04d}.bin"
        pts.astype(np.float32).tofile(out)

    print(f"Synthetic LiDAR data written to: {lidar_dir}")


if __name__ == "__main__":
    main()

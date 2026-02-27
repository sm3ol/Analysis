#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tiny synthetic MedMNIST-C style data")
    p.add_argument("--clean_root", type=str, required=True)
    p.add_argument("--corrupted_root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mini")
    p.add_argument("--corruption", type=str, default="pixelate")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    clean_root = Path(args.clean_root)
    corr_root = Path(args.corrupted_root)
    clean_root.mkdir(parents=True, exist_ok=True)
    ds_dir = corr_root / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    n = args.num_samples
    h = args.image_size
    w = args.image_size

    clean_images = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    clean_labels = rng.integers(0, 3, size=(n, 1), dtype=np.int64)

    clean_npz = clean_root / f"{args.dataset}_224.npz"
    np.savez_compressed(clean_npz, test_images=clean_images, test_labels=clean_labels)

    sev_images = []
    sev_labels = []
    for sev in range(1, 6):
        noise = rng.normal(loc=0.0, scale=sev * 3.0, size=clean_images.shape)
        corr = np.clip(clean_images.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        sev_images.append(corr)
        sev_labels.append(clean_labels.copy())

    corr_images = np.concatenate(sev_images, axis=0)
    corr_labels = np.concatenate(sev_labels, axis=0)
    corr_npz = ds_dir / f"{args.corruption}.npz"
    np.savez_compressed(corr_npz, test_images=corr_images, test_labels=corr_labels)

    print(f"Synthetic clean npz: {clean_npz}")
    print(f"Synthetic corrupted npz: {corr_npz}")


if __name__ == "__main__":
    main()

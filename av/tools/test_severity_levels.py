#!/usr/bin/env python3
"""Test whether the scorer differentiates between corruption severity levels.

Runs clean + 3 corruption intensities (low/mid/high) and reports scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

AV_ROOT = Path(__file__).resolve().parents[1]
if str(AV_ROOT) not in sys.path:
    sys.path.insert(0, str(AV_ROOT))

from framework.adapters.checkpoint_catalog import default_checkpoint_path, official_checkpoint_for_encoder
from framework.config import FrameworkConfig
from framework.core.brain_b_stats import load_clean_reference_stats
from framework.train_belief import UnifiedBeliefTrainer, build_components
from framework.types import TrainBatch


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def align_point_feature_dim(points: np.ndarray, target_dim: int) -> np.ndarray:
    if points.shape[-1] == target_dim:
        return points
    if points.shape[-1] > target_dim:
        return points[..., :target_dim].astype(np.float32, copy=False)
    pad_dim = target_dim - points.shape[-1]
    pad = np.zeros((*points.shape[:-1], pad_dim), dtype=np.float32)
    return np.concatenate([points, pad], axis=-1).astype(np.float32, copy=False)


def corrupt(points: np.ndarray, rng: np.random.Generator, sigma: float, replace_frac: float) -> np.ndarray:
    c = points.copy()
    n = c.shape[1]
    n_replace = int(n * replace_frac)
    for t in range(c.shape[0]):
        idx = rng.choice(n, size=n_replace, replace=False)
        c[t, idx] = rng.uniform(-100, 100, size=(n_replace, c.shape[2])).astype(np.float32)
    c += rng.normal(0, sigma, size=c.shape).astype(np.float32)
    return c


def make_batch(pts_1t: np.ndarray, t: int, episode_id: int, clean_prefix: int,
               is_corrupt: int, device: torch.device) -> TrainBatch:
    return TrainBatch(
        points=torch.from_numpy(pts_1t).to(device=device, dtype=torch.float32),
        episode_id=torch.tensor([episode_id], dtype=torch.long, device=device),
        timestep=torch.tensor([t], dtype=torch.long, device=device),
        stream_id=torch.tensor([0], dtype=torch.long, device=device),
        corruption_family_id=torch.tensor([-1], dtype=torch.long, device=device),
        is_corrupt=torch.tensor([is_corrupt], dtype=torch.long, device=device),
        metadata={
            "severity": torch.tensor([0], dtype=torch.long, device=device),
            "corrupt_start": torch.tensor([clean_prefix], dtype=torch.long, device=device),
            "corrupt_end": torch.tensor([clean_prefix + 30], dtype=torch.long, device=device),
            "corruption_length": torch.tensor([30], dtype=torch.long, device=device),
        },
    )


def run_episode(trainer: UnifiedBeliefTrainer, clean_points: np.ndarray, test_points: np.ndarray,
                clean_prefix: int, corrupt_len: int, episode_id: int, device: torch.device) -> list[float]:
    """Run clean warmup then test_points for corrupt_len frames. Return scores for test frames."""
    trainer.reset_stream_state()
    scores: list[float] = []
    with torch.no_grad():
        # Clean warmup
        for t in range(clean_prefix):
            b = make_batch(clean_points[t:t+1], t=t, episode_id=episode_id,
                           clean_prefix=clean_prefix, is_corrupt=0, device=device)
            trainer.score_step(b)
        # Test frames
        for t in range(clean_prefix, clean_prefix + corrupt_len):
            b = make_batch(test_points[t:t+1], t=t, episode_id=episode_id,
                           clean_prefix=clean_prefix, is_corrupt=1, device=device)
            step = trainer.score_step(b)
            scores.append(float(step.final_reliability[0].item()))
    return scores


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="pointpillars")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    os.chdir(AV_ROOT)
    device = resolve_device(args.device)
    encoder = args.encoder

    ep = np.load("dataset/episode_000001.npz", allow_pickle=True)
    raw_points = np.asarray(ep["points"], dtype=np.float32)
    clean_prefix = int(ep["clean_prefix"])

    cfg = FrameworkConfig()
    cfg.encoder_name = encoder
    cfg.model.use_pretrained = True
    cfg.model.strict_checkpoint_load = True
    ckpt_entry = official_checkpoint_for_encoder(encoder)
    cfg.data.point_feature_dim = int(ckpt_entry.expected_point_feature_dim)
    default_ckpt = AV_ROOT / default_checkpoint_path(encoder)
    cfg.model.checkpoint_path = str(default_ckpt)

    points = align_point_feature_dim(raw_points, target_dim=int(cfg.data.point_feature_dim))

    components = build_components(cfg, device=device)
    trainer = UnifiedBeliefTrainer(cfg, components, device=device)

    ckpt = torch.load(AV_ROOT / "artifacts" / f"{encoder}_stage2_checkpoint.pt",
                       map_location=str(device), weights_only=False)
    trainer.components.adapter.load_state_dict(ckpt["adapter"], strict=True)
    trainer.components.projector.load_state_dict(ckpt["projector"], strict=True)
    trainer.components.brain_a.load_state_dict(ckpt["brain_a"], strict=True)
    trainer.components.brain_b.load_state_dict(ckpt["brain_b"], strict=True)

    stats = load_clean_reference_stats(AV_ROOT / "artifacts" / "brain_b_clean_stats.npz", device=str(device))
    trainer.components.brain_b.update_stats(stats)

    trainer.components.adapter.eval()
    trainer.components.projector.eval()
    trainer.components.brain_a.eval()

    rng = np.random.default_rng(42)
    corrupt_len = 30

    levels = {
        "clean (no corruption)":        points,
        "low  (sigma=2,  replace=10%)": corrupt(points, rng, sigma=2.0,  replace_frac=0.1),
        "mid  (sigma=10, replace=40%)": corrupt(points, rng, sigma=10.0, replace_frac=0.4),
        "high (sigma=30, replace=90%)": corrupt(points, rng, sigma=30.0, replace_frac=0.9),
    }

    print(f"\nEncoder: {encoder}")
    print(f"{'Level':<35} | {'Mean':>8} | {'Min':>8} | {'Max':>8} | First 5 scores")
    print(f"{'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")

    eid = 0
    for label, test_pts in levels.items():
        scores = run_episode(trainer, clean_points=points, test_points=test_pts,
                             clean_prefix=clean_prefix, corrupt_len=corrupt_len,
                             episode_id=eid, device=device)
        eid += 1
        head = [round(s, 3) for s in scores[:5]]
        print(f"{label:<35} | {np.mean(scores):>8.4f} | {np.min(scores):>8.4f} | {np.max(scores):>8.4f} | {head}")

    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate that the two-brain scorer assigns high scores to clean frames
and low scores to corrupted frames, using trained stage-2 weights.

Pass criteria:
  - Mean clean-prefix score >= 0.9
  - Mean corrupted-frame score < 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

AV_ROOT = Path(__file__).resolve().parents[1]
if str(AV_ROOT) not in sys.path:
    sys.path.insert(0, str(AV_ROOT))

from framework.adapters.checkpoint_catalog import (
    default_checkpoint_path,
    official_checkpoint_for_encoder,
)
from framework.config import FrameworkConfig
from framework.core.brain_b_stats import (
    fit_clean_reference_stats,
    load_clean_reference_stats,
)
from framework.core.pooling import pool_adapter_output
from framework.train_belief import UnifiedBeliefTrainer, build_components
from framework.types import TrainBatch


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_episode(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        clean_prefix = int(data["clean_prefix"]) if "clean_prefix" in data else 50
    return {"points": points, "clean_prefix": clean_prefix}


def align_point_feature_dim(points: np.ndarray, target_dim: int) -> np.ndarray:
    if points.shape[-1] == target_dim:
        return points
    if points.shape[-1] > target_dim:
        return points[..., :target_dim].astype(np.float32, copy=False)
    pad_dim = target_dim - points.shape[-1]
    pad = np.zeros((*points.shape[:-1], pad_dim), dtype=np.float32)
    return np.concatenate([points, pad], axis=-1).astype(np.float32, copy=False)


def corrupt_frames(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create heavily corrupted point cloud frames.

    Replaces point geometry with uniform random values in [-100, 100] and adds
    large Gaussian noise. This destroys spatial structure that even robust
    architectures (e.g., PointRCNN with PointNet++ set abstraction) rely on.
    """
    corrupted = rng.uniform(-100.0, 100.0, size=points.shape).astype(np.float32)
    noise = rng.normal(0, 30.0, size=corrupted.shape).astype(np.float32)
    corrupted += noise
    return corrupted


def make_batch(
    points_1t: np.ndarray,
    t: int,
    episode_id: int,
    clean_prefix: int,
    is_corrupt: int,
    device: torch.device,
) -> TrainBatch:
    return TrainBatch(
        points=torch.from_numpy(points_1t).to(device=device, dtype=torch.float32),
        episode_id=torch.tensor([episode_id], dtype=torch.long, device=device),
        timestep=torch.tensor([t], dtype=torch.long, device=device),
        stream_id=torch.tensor([0], dtype=torch.long, device=device),
        corruption_family_id=torch.tensor([-1], dtype=torch.long, device=device),
        is_corrupt=torch.tensor([is_corrupt], dtype=torch.long, device=device),
        metadata={
            "severity": torch.tensor([5 if is_corrupt else 0], dtype=torch.long, device=device),
            "corrupt_start": torch.tensor([clean_prefix], dtype=torch.long, device=device),
            "corrupt_end": torch.tensor([clean_prefix + 30], dtype=torch.long, device=device),
            "corruption_length": torch.tensor([30 if is_corrupt else 0], dtype=torch.long, device=device),
        },
    )


def fit_brain_b_from_prefix(
    trainer: UnifiedBeliefTrainer,
    points: np.ndarray,
    clean_prefix: int,
    device: torch.device,
) -> None:
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for t in range(min(clean_prefix, points.shape[0])):
            b = make_batch(points[t:t+1], t=t, episode_id=999, clean_prefix=clean_prefix, is_corrupt=0, device=device)
            out = trainer.components.adapter(b)
            pooled = pool_adapter_output(out)
            z = trainer.components.projector(pooled)
            zs.append(z.detach())
    emb = torch.cat(zs, dim=0)
    stats = fit_clean_reference_stats(
        emb,
        shrinkage=float(trainer.config.brain_b.covariance_shrinkage),
        eps=float(trainer.config.brain_b.covariance_eps),
    )
    trainer.components.brain_b.update_stats(stats)


def run_frames(
    trainer: UnifiedBeliefTrainer,
    points: np.ndarray,
    start: int,
    end: int,
    episode_id: int,
    clean_prefix: int,
    is_corrupt: int,
    device: torch.device,
) -> list[float]:
    """Run frames [start, end) and return per-frame final_reliability scores."""
    scores: list[float] = []
    with torch.no_grad():
        for t in range(start, min(end, points.shape[0])):
            batch = make_batch(
                points[t:t+1], t=t, episode_id=episode_id,
                clean_prefix=clean_prefix, is_corrupt=is_corrupt, device=device,
            )
            step = trainer.score_step(batch)
            scores.append(float(step.final_reliability[0].item()))
    return scores


def test_encoder(
    encoder: str,
    episode_path: Path,
    device: torch.device,
    stage2_checkpoint: Path | None,
    brain_b_stats_path: Path | None,
) -> dict[str, Any]:
    """Test one encoder: clean scores >= 0.9, corrupt scores < 0.8."""
    episode = load_episode(episode_path)
    points = episode["points"]
    clean_prefix = int(episode["clean_prefix"])

    cfg = FrameworkConfig()
    cfg.encoder_name = encoder
    cfg.model.use_pretrained = True
    cfg.model.strict_checkpoint_load = True

    ckpt_entry = official_checkpoint_for_encoder(encoder)
    cfg.data.point_feature_dim = int(ckpt_entry.expected_point_feature_dim)
    default_ckpt = default_checkpoint_path(encoder)
    if not default_ckpt.is_absolute():
        default_ckpt = AV_ROOT / default_ckpt
    cfg.model.checkpoint_path = str(default_ckpt)

    points = align_point_feature_dim(points, target_dim=int(cfg.data.point_feature_dim))

    components = build_components(cfg, device=device)
    trainer = UnifiedBeliefTrainer(cfg, components, device=device)

    loaded_trained = False
    if stage2_checkpoint and stage2_checkpoint.exists():
        payload = torch.load(stage2_checkpoint, map_location=str(device), weights_only=False)
        trainer.components.adapter.load_state_dict(payload["adapter"], strict=True)
        trainer.components.projector.load_state_dict(payload["projector"], strict=True)
        trainer.components.brain_a.load_state_dict(payload["brain_a"], strict=True)
        trainer.components.brain_b.load_state_dict(payload["brain_b"], strict=True)
        loaded_trained = True

    if brain_b_stats_path and brain_b_stats_path.exists():
        stats = load_clean_reference_stats(brain_b_stats_path, device=str(device))
        trainer.components.brain_b.update_stats(stats)
    else:
        fit_brain_b_from_prefix(trainer, points, clean_prefix, device)

    trainer.components.adapter.eval()
    trainer.components.projector.eval()
    trainer.components.brain_a.eval()

    # --- Phase 1: Run clean frames (episode 0) ---
    trainer.reset_stream_state()
    clean_scores = run_frames(
        trainer, points,
        start=0, end=clean_prefix, episode_id=0,
        clean_prefix=clean_prefix, is_corrupt=0, device=device,
    )

    # --- Phase 2: Create corrupted frames and run them (episode 1, fresh state) ---
    trainer.reset_stream_state()
    rng = np.random.default_rng(42)
    corrupt_points = corrupt_frames(points, rng)

    # Run a short clean prefix first so the temporal model has a baseline
    warmup_scores = run_frames(
        trainer, points,
        start=0, end=clean_prefix, episode_id=1,
        clean_prefix=clean_prefix, is_corrupt=0, device=device,
    )

    # Now feed heavily corrupted frames
    corrupt_start = clean_prefix
    corrupt_end = min(clean_prefix + 30, points.shape[0])
    corrupt_scores = run_frames(
        trainer, corrupt_points,
        start=corrupt_start, end=corrupt_end, episode_id=1,
        clean_prefix=clean_prefix, is_corrupt=1, device=device,
    )

    # --- Evaluate ---
    # Skip the first few clean frames (calibrator warmup)
    eval_clean = clean_scores[10:] if len(clean_scores) > 10 else clean_scores
    mean_clean = float(np.mean(eval_clean)) if eval_clean else float("nan")
    mean_corrupt = float(np.mean(corrupt_scores)) if corrupt_scores else float("nan")

    clean_pass = mean_clean >= 0.9
    corrupt_pass = mean_corrupt < 0.8

    result = {
        "encoder": encoder,
        "trained_checkpoint_loaded": loaded_trained,
        "clean_prefix_len": len(clean_scores),
        "corrupt_len": len(corrupt_scores),
        "mean_clean_score": round(mean_clean, 4),
        "mean_corrupt_score": round(mean_corrupt, 4),
        "clean_pass": clean_pass,
        "corrupt_pass": corrupt_pass,
        "overall_pass": clean_pass and corrupt_pass,
        "clean_scores_head": [round(s, 4) for s in clean_scores[:10]],
        "corrupt_scores_head": [round(s, 4) for s in corrupt_scores[:10]],
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score threshold validation test.")
    p.add_argument("--encoder", type=str, default="all",
                   help="Encoder name or 'all' for all four.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--episode_path", type=str, default="dataset/episode_000001.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(AV_ROOT)
    device = resolve_device(args.device)
    episode_path = Path(args.episode_path)
    if not episode_path.is_absolute():
        episode_path = AV_ROOT / episode_path

    if args.encoder == "all":
        encoders = ["pointpillars", "pointrcnn", "pvrcnn", "centerpoint"]
    else:
        encoders = [args.encoder]

    artifacts = AV_ROOT / "artifacts"
    brain_b_stats = artifacts / "brain_b_clean_stats.npz"

    all_results = []
    overall_pass = True
    for enc in encoders:
        stage2_ckpt = artifacts / f"{enc}_stage2_checkpoint.pt"
        print(f"\n{'='*60}")
        print(f"  Testing: {enc}")
        print(f"  Stage-2 checkpoint: {stage2_ckpt.name} (exists={stage2_ckpt.exists()})")
        print(f"{'='*60}")

        result = test_encoder(
            encoder=enc,
            episode_path=episode_path,
            device=device,
            stage2_checkpoint=stage2_ckpt if stage2_ckpt.exists() else None,
            brain_b_stats_path=brain_b_stats if brain_b_stats.exists() else None,
        )
        all_results.append(result)

        status = "PASS" if result["overall_pass"] else "FAIL"
        clean_status = "PASS" if result["clean_pass"] else "FAIL"
        corrupt_status = "PASS" if result["corrupt_pass"] else "FAIL"
        print(f"  Clean  mean={result['mean_clean_score']:.4f}  (>= 0.9 ? {clean_status})")
        print(f"  Corrupt mean={result['mean_corrupt_score']:.4f}  (< 0.8 ? {corrupt_status})")
        print(f"  Overall: {status}")
        if not result["overall_pass"]:
            overall_pass = False

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Encoder':<14} | {'Clean':>8} | {'Corrupt':>8} | {'Status':>6}")
    print(f"{'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for r in all_results:
        status = "PASS" if r["overall_pass"] else "FAIL"
        print(f"{r['encoder']:<14} | {r['mean_clean_score']:>8.4f} | {r['mean_corrupt_score']:>8.4f} | {status:>6}")

    # Write results JSON
    out_path = AV_ROOT / "outputs" / "score_threshold_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out_path}")

    if overall_pass:
        print("\n[PASS] All encoders passed score threshold validation.")
    else:
        print("\n[FAIL] One or more encoders failed score threshold validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()

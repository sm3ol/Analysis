#!/usr/bin/env python3
"""Minimal realtime-style AV runner for centerpoint with stage timing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

def _resolve_av_root() -> Path:
    return Path(__file__).resolve().parents[1]

AV_ROOT = _resolve_av_root()
if str(AV_ROOT) not in sys.path:
    sys.path.insert(0, str(AV_ROOT))

from framework.config import FrameworkConfig
from framework.train_belief import build_components
from framework.core.brain_b_stats import load_clean_reference_stats
from framework.core.pooling import pool_adapter_output
from framework.core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine
from framework.types import TrainBatch


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device: torch.device) -> float:
    sync_device(device)
    return perf_counter()


def load_episode(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"episode file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        clean_prefix = int(data["clean_prefix"]) if "clean_prefix" in data else 50
        stream_name = str(data["stream_name"]) if "stream_name" in data else "unknown"
    if points.ndim != 3:
        raise ValueError(f"episode points must be [T,N,D], got {points.shape}")
    return {
        "points": points,
        "clean_prefix": clean_prefix,
        "stream_name": stream_name,
    }


def align_point_feature_dim(points: np.ndarray, target_dim: int) -> np.ndarray:
    if points.shape[-1] == target_dim:
        return points
    if points.shape[-1] > target_dim:
        return points[..., :target_dim].astype(np.float32, copy=False)
    pad_dim = target_dim - points.shape[-1]
    pad = np.zeros((*points.shape[:-1], pad_dim), dtype=np.float32)
    return np.concatenate([points, pad], axis=-1).astype(np.float32, copy=False)


def make_batch(points_1t: np.ndarray, clean_prefix: int, t: int, device: torch.device) -> TrainBatch:
    return TrainBatch(
        points=torch.from_numpy(points_1t).to(device=device, dtype=torch.float32),
        episode_id=torch.tensor([0], dtype=torch.long, device=device),
        timestep=torch.tensor([t], dtype=torch.long, device=device),
        stream_id=torch.tensor([0], dtype=torch.long, device=device),
        corruption_family_id=torch.tensor([-1], dtype=torch.long, device=device),
        is_corrupt=torch.tensor([0], dtype=torch.long, device=device),
        metadata={
            "severity": torch.tensor([0], dtype=torch.long, device=device),
            "corrupt_start": torch.tensor([clean_prefix], dtype=torch.long, device=device),
            "corrupt_end": torch.tensor([clean_prefix], dtype=torch.long, device=device),
            "corruption_length": torch.tensor([0], dtype=torch.long, device=device),
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal realtime-style runner for AV centerpoint.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--episode_path", type=str, default="dataset/episode_000001.npz")
    p.add_argument("--point_feature_dim", type=int, default=5)
    p.add_argument("--brain_b_stats", type=str, default="artifacts/brain_b_clean_stats.npz")
    p.add_argument("--warmup_steps", type=int, default=5)
    return p.parse_args()


def scorer_step_from_z(
    components,
    state_machine: ReliabilityStateMachine,
    state: ReliabilityState,
    z: torch.Tensor,
    timestep: int,
    clean_prefix: int,
    ema_alpha: float,
):
    z_i = z[0]

    if state.belief_ema is None:
        state.belief_ema = z_i.detach()

    belief_i = state.belief_ema.to(z_i.device)
    a_out = components.brain_a(belief_i.unsqueeze(0), z_i.unsqueeze(0))
    b_out = components.brain_b(z_i.unsqueeze(0))

    r_a = a_out.reliability.squeeze(0)
    r_b = b_out.reliability.squeeze(0)

    d_clean = torch.norm(z_i.detach() - components.brain_b.mu_clean.to(z_i.device), p=2)
    d_bad = None
    if state.mu_bad is not None:
        d_bad = torch.norm(z_i.detach() - state.mu_bad.to(z_i.device), p=2)

    warmup_known_clean = timestep < int(clean_prefix)

    if state.mode == ReliabilityMode.PERSISTENT:
        threshold_b = getattr(state_machine.config, "persistent_enter_threshold_b", None)
        if threshold_b is None:
            threshold_b = state_machine.config.clean_like_threshold_b
        raw_suspicious = bool(r_b.item() < threshold_b)
    elif state.mode == ReliabilityMode.RECOVERING:
        raw_suspicious = False
    else:
        raw_suspicious = bool(r_a.item() < state_machine.config.suspicious_threshold_a)

    suspicious = False if warmup_known_clean else raw_suspicious

    step_result = state_machine.step(
        state=state,
        z_t=z_i.detach(),
        r_a=r_a,
        r_b=r_b,
        suspicious=suspicious,
        d_clean=d_clean,
        d_bad=d_bad,
        r_b_recover=r_b,
    )

    if step_result.update_belief and not suspicious:
        state.belief_ema = ema_alpha * belief_i.detach() + (1.0 - ema_alpha) * z_i.detach()

    return step_result.state


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    episode_path = Path(args.episode_path)
    if not episode_path.is_absolute():
        episode_path = (Path.cwd() / episode_path).resolve()

    brain_b_stats_path = Path(args.brain_b_stats)
    if not brain_b_stats_path.is_absolute():
        brain_b_stats_path = (Path.cwd() / brain_b_stats_path).resolve()

    episode = load_episode(episode_path)
    points = align_point_feature_dim(episode["points"], target_dim=int(args.point_feature_dim))

    cfg = FrameworkConfig()
    cfg.encoder_name = "centerpoint"
    cfg.data.point_feature_dim = int(args.point_feature_dim)

    components = build_components(cfg, device=device)
    stats = load_clean_reference_stats(brain_b_stats_path, device=str(device))
    components.brain_b.update_stats(stats)

    components.adapter.eval()
    components.projector.eval()
    components.brain_a.eval()
    components.brain_b.eval()

    state_machine = ReliabilityStateMachine(cfg.temporal)

    ema_window = max(1, int(cfg.temporal.belief_ema_window))
    ema_alpha = float(ema_window - 1) / float(ema_window)

    warmup_steps = min(int(args.warmup_steps), int(points.shape[0]))
    state = ReliabilityState()

    with torch.no_grad():
        for t in range(warmup_steps):
            batch = make_batch(points[t:t+1], clean_prefix=episode["clean_prefix"], t=t, device=device)
            out = components.adapter(batch)
            pooled = pool_adapter_output(out)
            z = components.projector(pooled)
            state = scorer_step_from_z(
                components=components,
                state_machine=state_machine,
                state=state,
                z=z,
                timestep=t,
                clean_prefix=episode["clean_prefix"],
                ema_alpha=ema_alpha,
            )

    state = ReliabilityState()

    timings = {
        "input_s": 0.0,
        "adapter_s": 0.0,
        "pool_s": 0.0,
        "projector_s": 0.0,
        "scorer_state_s": 0.0,
    }

    sync_device(device)
    t0 = perf_counter()

    with torch.no_grad():
        for t in range(points.shape[0]):
            s0 = now(device)
            batch = make_batch(points[t:t+1], clean_prefix=episode["clean_prefix"], t=t, device=device)
            s1 = now(device)
            timings["input_s"] += (s1 - s0)

            out = components.adapter(batch)
            s2 = now(device)
            timings["adapter_s"] += (s2 - s1)

            pooled = pool_adapter_output(out)
            s3 = now(device)
            timings["pool_s"] += (s3 - s2)

            z = components.projector(pooled)
            s4 = now(device)
            timings["projector_s"] += (s4 - s3)

            state = scorer_step_from_z(
                components=components,
                state_machine=state_machine,
                state=state,
                z=z,
                timestep=t,
                clean_prefix=episode["clean_prefix"],
                ema_alpha=ema_alpha,
            )
            s5 = now(device)
            timings["scorer_state_s"] += (s5 - s4)

    sync_device(device)
    elapsed_s = perf_counter() - t0

    print("Minimal realtime-style run complete")
    print(f"device: {device}")
    print(f"stream_name: {episode['stream_name']}")
    print(f"num_frames: {points.shape[0]}")
    print(f"total_runtime_s: {elapsed_s:.6f}")
    print(f"mean_runtime_ms_per_frame: {(elapsed_s / points.shape[0]) * 1000.0:.6f}")
    print("Stage breakdown:")
    for k, v in timings.items():
        print(f"{k}: {v:.6f} s total, {(v / points.shape[0]) * 1000.0:.6f} ms/frame")
    print(f"sum_of_stages_s: {sum(timings.values()):.6f}")


if __name__ == "__main__":
    main()

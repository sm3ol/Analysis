#!/usr/bin/env python3
"""Run AV encoder + scorer inference on one staged LiDAR episode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _resolve_av_root() -> Path:
    # .../Analysis/av/tools -> .../Analysis/av
    return Path(__file__).resolve().parents[1]


AV_ROOT = _resolve_av_root()
if str(AV_ROOT) not in sys.path:
    sys.path.insert(0, str(AV_ROOT))

from framework.adapters.checkpoint_catalog import (  # noqa: E402
    default_checkpoint_path,
    official_checkpoint_for_encoder,
)
from framework.config import FrameworkConfig  # noqa: E402
from framework.core.brain_b_stats import (  # noqa: E402
    fit_clean_reference_stats,
    load_clean_reference_stats,
)
from framework.core.pooling import pool_adapter_output  # noqa: E402
from framework.train_belief import (  # noqa: E402
    UnifiedBeliefTrainer,
    build_components,
)
from framework.types import TrainBatch  # noqa: E402


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _iter_checkpoint_statuses(status: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    if isinstance(status, dict) and "requested" in status and "loaded" in status:
        return [(prefix or "checkpoint", status)]
    if isinstance(status, dict):
        rows: list[tuple[str, dict[str, Any]]] = []
        for key, value in status.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_iter_checkpoint_statuses(value, child))
        return rows
    return []


def _print_checkpoint_status(status: Any) -> None:
    rows = _iter_checkpoint_statuses(status)
    if not rows:
        print(json.dumps({"checkpoint_status": "missing"}))
        return
    for name, payload in rows:
        print(
            json.dumps(
                {
                    "module": name,
                    "checkpoint_path": payload.get("path"),
                    "loaded": bool(payload.get("loaded", False)),
                    "missing_keys_count": int(payload.get("missing_keys_count", 0)),
                    "unexpected_keys_count": int(payload.get("unexpected_keys_count", 0)),
                    "error": payload.get("error"),
                }
            )
        )


def load_episode(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"episode file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        clean_prefix = int(data["clean_prefix"]) if "clean_prefix" in data else 50
        stream_name = str(data["stream_name"]) if "stream_name" in data else "unknown"
        frame_names = [str(x) for x in data["frame_names"]] if "frame_names" in data else []
    if points.ndim != 3:
        raise ValueError(f"episode points must be [T,N,D], got {points.shape}")
    return {
        "points": points,
        "clean_prefix": clean_prefix,
        "stream_name": stream_name,
        "frame_names": frame_names,
    }


def align_point_feature_dim(points: np.ndarray, target_dim: int) -> np.ndarray:
    if points.shape[-1] == target_dim:
        return points
    if points.shape[-1] > target_dim:
        return points[..., :target_dim].astype(np.float32, copy=False)
    pad_dim = target_dim - points.shape[-1]
    pad = np.zeros((*points.shape[:-1], pad_dim), dtype=np.float32)
    return np.concatenate([points, pad], axis=-1).astype(np.float32, copy=False)


def fit_brain_b_from_clean_prefix(
    trainer: UnifiedBeliefTrainer,
    points: np.ndarray,
    clean_prefix: int,
    device: torch.device,
) -> None:
    prefix = max(1, min(int(clean_prefix), int(points.shape[0])))
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for t in range(prefix):
            b = TrainBatch(
                points=torch.from_numpy(points[t : t + 1]).to(device=device, dtype=torch.float32),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AV inference on one staged episode.")
    p.add_argument("--encoder", type=str, default="centerpoint")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--episode_path", type=str, default="dataset/episode_000001.npz")
    p.add_argument("--output_json", type=str, default="outputs/inference_episode_result.json")
    p.add_argument("--use_pretrained_backbone", type=int, choices=[0, 1], default=1)
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--brain_b_stats", type=str, default="")
    p.add_argument("--allow_dummy_weights", type=int, choices=[0, 1], default=1)
    p.add_argument("--recover_required_steps", type=int, default=25)
    p.add_argument("--recover_rewarm_steps", type=int, default=25)
    p.add_argument("--recover_anchor_mode", type=str, default="strict")
    p.add_argument("--clean_like_threshold_b", type=float, default=0.95)
    p.add_argument("--suspicious_threshold_a", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    invocation_cwd = Path.cwd()
    episode_path = Path(args.episode_path)
    if not episode_path.is_absolute():
        episode_path = (invocation_cwd / episode_path).resolve()
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = (invocation_cwd / output_json).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve() if str(args.checkpoint_path).strip() else None
    brain_b_stats_path = Path(args.brain_b_stats).resolve() if str(args.brain_b_stats).strip() else None

    os.chdir(AV_ROOT)
    device = resolve_device(args.device)
    episode = load_episode(episode_path)
    points = episode["points"]
    clean_prefix = int(episode["clean_prefix"])

    cfg = FrameworkConfig()
    cfg.encoder_name = str(args.encoder)
    cfg.model.use_pretrained = bool(int(args.use_pretrained_backbone))
    cfg.model.strict_checkpoint_load = True
    if cfg.model.use_pretrained:
        ckpt_entry = official_checkpoint_for_encoder(cfg.encoder_name)
        cfg.data.point_feature_dim = int(ckpt_entry.expected_point_feature_dim)
        default_ckpt = default_checkpoint_path(cfg.encoder_name)
        if not default_ckpt.is_absolute():
            default_ckpt = AV_ROOT / default_ckpt
        cfg.model.checkpoint_path = str(default_ckpt)
    if checkpoint_path is not None:
        cfg.model.checkpoint_path = str(checkpoint_path)
    cfg.temporal.recover_required_steps = int(args.recover_required_steps)
    cfg.temporal.recover_rewarm_steps = int(args.recover_rewarm_steps)
    cfg.temporal.recover_anchor_mode = str(args.recover_anchor_mode)
    cfg.temporal.clean_like_threshold_b = float(args.clean_like_threshold_b)
    cfg.temporal.suspicious_threshold_a = float(args.suspicious_threshold_a)
    points = align_point_feature_dim(points, target_dim=int(cfg.data.point_feature_dim))

    components = build_components(cfg, device=device)
    _print_checkpoint_status(getattr(components.adapter, "checkpoint_status", {}))
    trainer = UnifiedBeliefTrainer(cfg, components, device=device)

    model_checkpoint = checkpoint_path
    loaded_trained = False
    if model_checkpoint and model_checkpoint.exists():
        payload = torch.load(model_checkpoint, map_location=str(device), weights_only=False)
        trainer.components.adapter.load_state_dict(payload["adapter"], strict=True)
        trainer.components.projector.load_state_dict(payload["projector"], strict=True)
        trainer.components.brain_a.load_state_dict(payload["brain_a"], strict=True)
        trainer.components.brain_b.load_state_dict(payload["brain_b"], strict=True)
        loaded_trained = True
    elif not bool(int(args.allow_dummy_weights)):
        raise RuntimeError("trained checkpoint was not provided and allow_dummy_weights=0")

    brain_b_stats = brain_b_stats_path
    if brain_b_stats and brain_b_stats.exists():
        stats = load_clean_reference_stats(brain_b_stats, device=str(device))
        trainer.components.brain_b.update_stats(stats)
    else:
        fit_brain_b_from_clean_prefix(trainer, points=points, clean_prefix=clean_prefix, device=device)

    trainer.components.adapter.eval()
    trainer.components.projector.eval()
    trainer.components.brain_a.eval()
    trainer.reset_stream_state()

    out_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for t in range(points.shape[0]):
            batch = TrainBatch(
                points=torch.from_numpy(points[t : t + 1]).to(device=device, dtype=torch.float32),
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
            step = trainer.score_step(batch)
            mode_idx = int(step.diagnostics["mode_id"][0].item())
            out_rows.append(
                {
                    "timestep": int(t),
                    "r_a": float(step.r_a[0].item()),
                    "r_b": float(step.r_b[0].item()),
                    "final_reliability": float(step.final_reliability[0].item()),
                    "suspicious": int(step.suspicious[0].item() > 0.5),
                    "alarm": int(step.alarm[0].item() > 0.5),
                    "mode_id": mode_idx,
                    "mode_name": str(step.mode_name),
                }
            )

    mean_final = float(np.mean([r["final_reliability"] for r in out_rows])) if out_rows else float("nan")
    alarm_rate = float(np.mean([r["alarm"] for r in out_rows])) if out_rows else float("nan")
    mode_hist: dict[str, int] = {}
    for row in out_rows:
        mode_hist[row["mode_name"]] = mode_hist.get(row["mode_name"], 0) + 1

    result = {
        "encoder": cfg.encoder_name,
        "device": str(device),
        "episode_path": str(episode_path),
        "stream_name": episode["stream_name"],
        "episode_len": int(points.shape[0]),
        "point_shape": [int(v) for v in points.shape],
        "trained_checkpoint_loaded": bool(loaded_trained),
        "brain_b_stats_loaded": bool(brain_b_stats and brain_b_stats.exists()),
        "metrics": {
            "mean_final_reliability": mean_final,
            "alarm_rate": alarm_rate,
            "mode_histogram": mode_hist,
        },
        "trace_head": out_rows[:10],
    }

    out_path = output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

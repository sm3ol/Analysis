#!/usr/bin/env python3
"""Run encoder-only inference on one staged LiDAR episode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _resolve_project_root() -> Path:
    # .../Analysis/av/tools -> .../<project-root>
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.AV.framework.adapters.checkpoint_catalog import (  # noqa: E402
    default_checkpoint_path,
    official_checkpoint_for_encoder,
)
from training.AV.framework.config import FrameworkConfig  # noqa: E402
from training.AV.framework.core.pooling import pool_adapter_output  # noqa: E402
from training.AV.framework.train_belief import build_components  # noqa: E402
from training.AV.framework.types import TrainBatch  # noqa: E402


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
    if points.ndim != 3:
        raise ValueError(f"episode points must be [T,N,D], got {points.shape}")
    return {"points": points, "clean_prefix": clean_prefix, "stream_name": stream_name}


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
    p = argparse.ArgumentParser(description="Run encoder-only AV inference on one staged episode.")
    p.add_argument("--encoder", type=str, default="centerpoint")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--episode_path", type=str, default="dataset/episode_000001.npz")
    p.add_argument("--output_json", type=str, default="outputs/encoder_only_result.json")
    p.add_argument("--output_npz", type=str, default="")
    p.add_argument("--use_pretrained_backbone", type=int, choices=[0, 1], default=1)
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--allow_dummy_weights", type=int, choices=[0, 1], default=1)
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
    output_npz = Path(args.output_npz).resolve() if str(args.output_npz).strip() else None
    checkpoint_path = Path(args.checkpoint_path).resolve() if str(args.checkpoint_path).strip() else None

    os.chdir(PROJECT_ROOT)
    device = resolve_device(args.device)
    episode = load_episode(episode_path)
    points = episode["points"]

    cfg = FrameworkConfig()
    cfg.encoder_name = str(args.encoder)
    cfg.model.use_pretrained = bool(int(args.use_pretrained_backbone))
    cfg.model.strict_checkpoint_load = True
    if cfg.model.use_pretrained:
        ckpt_entry = official_checkpoint_for_encoder(cfg.encoder_name)
        cfg.data.point_feature_dim = int(ckpt_entry.expected_point_feature_dim)
        default_ckpt = default_checkpoint_path(cfg.encoder_name)
        if not default_ckpt.is_absolute():
            default_ckpt = PROJECT_ROOT / default_ckpt
        cfg.model.checkpoint_path = str(default_ckpt)
    if checkpoint_path is not None:
        cfg.model.checkpoint_path = str(checkpoint_path)

    points = align_point_feature_dim(points, target_dim=int(cfg.data.point_feature_dim))
    components = build_components(cfg, device=device)
    _print_checkpoint_status(getattr(components.adapter, "checkpoint_status", {}))
    components.adapter.eval()
    components.projector.eval()

    loaded_trained = False
    model_checkpoint = checkpoint_path
    if model_checkpoint and model_checkpoint.exists():
        payload = torch.load(model_checkpoint, map_location=str(device), weights_only=False)
        components.adapter.load_state_dict(payload["adapter"], strict=True)
        components.projector.load_state_dict(payload["projector"], strict=True)
        loaded_trained = True
    elif not bool(int(args.allow_dummy_weights)):
        raise RuntimeError("trained checkpoint was not provided and allow_dummy_weights=0")

    embeddings = []
    with torch.no_grad():
        for t in range(points.shape[0]):
            b = make_batch(points[t : t + 1], clean_prefix=int(episode["clean_prefix"]), t=t, device=device)
            out = components.adapter(b)
            pooled = pool_adapter_output(out)
            z = components.projector(pooled)
            embeddings.append(z.squeeze(0).detach().float().cpu().numpy())

    emb = np.stack(embeddings, axis=0)  # [T, D]
    norms = np.linalg.norm(emb, axis=1)

    result = {
        "encoder": cfg.encoder_name,
        "device": str(device),
        "episode_path": str(episode_path),
        "stream_name": episode["stream_name"],
        "episode_len": int(points.shape[0]),
        "embedding_shape": [int(v) for v in emb.shape],
        "trained_checkpoint_loaded": bool(loaded_trained),
        "embedding_stats": {
            "mean": float(emb.mean()),
            "std": float(emb.std()),
            "min": float(emb.min()),
            "max": float(emb.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        },
        "trace_head": [
            {
                "timestep": int(i),
                "norm": float(norms[i]),
                "mean": float(emb[i].mean()),
                "std": float(emb[i].std()),
            }
            for i in range(min(10, emb.shape[0]))
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[OK] wrote {output_json}")

    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, embeddings=emb.astype(np.float32))
        print(f"[OK] wrote {output_npz}")


if __name__ == "__main__":
    main()


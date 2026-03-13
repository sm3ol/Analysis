#!/usr/bin/env python3
"""Strict preflight checks for AV inference stack readiness."""

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
from framework.core.pooling import pool_adapter_output  # noqa: E402
from framework.train_belief import build_components  # noqa: E402
from framework.types import TrainBatch  # noqa: E402

DEFAULT_ENCODERS = ["pointpillars", "pointrcnn", "pvrcnn", "centerpoint"]


def parse_encoder_list(raw: str) -> list[str]:
    text = str(raw).strip().lower()
    if not text or text == "all":
        return list(DEFAULT_ENCODERS)
    out = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("encoders list is empty")
    return out


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


def load_episode(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"episode file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        clean_prefix = int(data["clean_prefix"]) if "clean_prefix" in data else 50
    if points.ndim != 3:
        raise ValueError(f"episode points must be [T,N,D], got {points.shape}")
    return {"points": points, "clean_prefix": clean_prefix}


def align_point_feature_dim(points: np.ndarray, target_dim: int) -> np.ndarray:
    if points.shape[-1] == target_dim:
        return points
    if points.shape[-1] > target_dim:
        return points[..., :target_dim].astype(np.float32, copy=False)
    pad_dim = target_dim - points.shape[-1]
    pad = np.zeros((*points.shape[:-1], pad_dim), dtype=np.float32)
    return np.concatenate([points, pad], axis=-1).astype(np.float32, copy=False)


def make_one_step_batch(points_1t: np.ndarray, clean_prefix: int, device: torch.device) -> TrainBatch:
    return TrainBatch(
        points=torch.from_numpy(points_1t).to(device=device, dtype=torch.float32),
        episode_id=torch.tensor([0], dtype=torch.long, device=device),
        timestep=torch.tensor([0], dtype=torch.long, device=device),
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


def _checkpoint_rows_for_result(status: Any) -> list[dict[str, Any]]:
    rows = []
    for module_name, payload in _iter_checkpoint_statuses(status):
        rows.append(
            {
                "module": module_name,
                "path": payload.get("path"),
                "requested": bool(payload.get("requested", False)),
                "loaded": bool(payload.get("loaded", False)),
                "missing_keys_count": int(payload.get("missing_keys_count", 0)),
                "unexpected_keys_count": int(payload.get("unexpected_keys_count", 0)),
                "error": payload.get("error"),
            }
        )
    return rows


def run_encoder_check(
    encoder_name: str,
    episode: dict[str, Any],
    device: torch.device,
    require_pretrained: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "encoder": encoder_name,
        "ready": False,
        "reason": "",
        "checkpoint": [],
    }
    try:
        if encoder_name == "pvrcnn" and device.type != "cuda":
            raise RuntimeError("pvrcnn requires CUDA (OpenPCDet pointnet2_stack CUDA ops)")

        cfg = FrameworkConfig()
        cfg.encoder_name = str(encoder_name)
        cfg.model.use_pretrained = bool(require_pretrained)
        cfg.model.strict_checkpoint_load = True

        ckpt_entry = official_checkpoint_for_encoder(cfg.encoder_name)
        cfg.data.point_feature_dim = int(ckpt_entry.expected_point_feature_dim)
        ckpt_path = default_checkpoint_path(cfg.encoder_name)
        if not ckpt_path.is_absolute():
            ckpt_path = AV_ROOT / ckpt_path
        cfg.model.checkpoint_path = str(ckpt_path)
        if require_pretrained and not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

        components = build_components(cfg, device=device)
        result["checkpoint"] = _checkpoint_rows_for_result(getattr(components.adapter, "checkpoint_status", {}))

        if require_pretrained:
            for row in result["checkpoint"]:
                if row["requested"] and not row["loaded"]:
                    raise RuntimeError(
                        f"checkpoint load failed ({row['module']}): "
                        f"missing={row['missing_keys_count']} unexpected={row['unexpected_keys_count']} "
                        f"error={row['error']}"
                    )

        aligned = align_point_feature_dim(episode["points"], target_dim=int(cfg.data.point_feature_dim))
        batch = make_one_step_batch(
            points_1t=aligned[0:1],
            clean_prefix=int(episode["clean_prefix"]),
            device=device,
        )

        components.adapter.eval()
        components.projector.eval()
        components.brain_a.eval()
        with torch.no_grad():
            out = components.adapter(batch)
            pooled = pool_adapter_output(out)
            z = components.projector(pooled)
            a = components.brain_a(z, z)
            b = components.brain_b(z)

        result["embedding_shape"] = [int(v) for v in z.shape]
        result["embedding_stats"] = {
            "mean": float(z.mean().item()),
            "std": float(z.std(unbiased=False).item()),
            "min": float(z.min().item()),
            "max": float(z.max().item()),
        }
        result["scorer_stats"] = {
            "r_a_mean": float(a.reliability.mean().item()),
            "r_b_mean": float(b.reliability.mean().item()),
        }
        result["ready"] = True
        result["reason"] = "ok"
    except Exception as exc:
        result["ready"] = False
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AV inference runtime for all real encoders.")
    parser.add_argument("--encoders", type=str, default="all")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--episode_path", type=str, default="dataset/episode_000001.npz")
    parser.add_argument("--require_pretrained", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invocation_cwd = Path.cwd()
    episode_path = Path(args.episode_path)
    if not episode_path.is_absolute():
        episode_path = (invocation_cwd / episode_path).resolve()
    output_json = Path(args.output_json).resolve() if str(args.output_json).strip() else None

    os.chdir(AV_ROOT)
    device = resolve_device(args.device)
    episode = load_episode(episode_path)
    encoders = parse_encoder_list(args.encoders)

    summary = {
        "project_root": str(AV_ROOT),
        "device": str(device),
        "episode_path": str(episode_path),
        "encoders": encoders,
        "results": [],
    }
    for encoder in encoders:
        res = run_encoder_check(
            encoder_name=encoder,
            episode=episode,
            device=device,
            require_pretrained=bool(int(args.require_pretrained)),
        )
        summary["results"].append(res)

    summary["ready_all"] = bool(all(bool(r.get("ready", False)) for r in summary["results"]))
    print(json.dumps(summary, indent=2))

    if output_json is not None:
        out_path = output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_path}")

    if not summary["ready_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from embodied_ai.common.config import FrameworkConfig
from embodied_ai.common.device import resolve_device
from embodied_ai.common.core.brain_b_stats import fit_clean_reference_stats, save_clean_reference_stats
from embodied_ai.common.runtime import build_runtime
from embodied_ai.common.types import TrainBatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic encoder + scorer runtime self-test.")
    p.add_argument("--module", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--local_repo_root", type=str, default="")
    return p.parse_args()


def _make_batch(device: torch.device, batch_size: int, image_size: int, step: int) -> TrainBatch:
    return TrainBatch(
        images=torch.rand((batch_size, 3, image_size, image_size), device=device),
        episode_id=torch.arange(batch_size, dtype=torch.long, device=device),
        timestep=torch.full((batch_size,), step, dtype=torch.long, device=device),
        corruption_family_id=torch.zeros((batch_size,), dtype=torch.long, device=device),
        is_corrupt=torch.zeros((batch_size,), dtype=torch.long, device=device),
    )


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(args.module)
    device = resolve_device(args.device)
    cfg = FrameworkConfig(encoder_name=mod.ENCODER_NAME)
    latent_dim = int(cfg.model.common_latent_dim)
    stats = fit_clean_reference_stats(torch.randn((max(8, int(args.batch_size) * 2), latent_dim), dtype=torch.float32))
    stats_path = Path(f"/tmp/{mod.ENCODER_NAME}_analysis_brainb_stats.npz")
    save_clean_reference_stats(stats, stats_path)
    cfg.brain_b.stats_artifact_path = str(stats_path)

    out: dict[str, Any] = {"module": args.module, "device": str(device), "result": {}}
    try:
        adapter = mod.build_adapter(local_repo_root=args.local_repo_root or None)
        runtime = build_runtime(cfg, adapter=adapter, device=device)
        for m in (runtime.components.adapter, runtime.components.projector, runtime.components.brain_a, runtime.components.brain_b):
            m.eval()
        step0 = runtime.score_step(_make_batch(device, int(args.batch_size), int(args.image_size), 0))
        step1 = runtime.score_step(_make_batch(device, int(args.batch_size), int(args.image_size), 1))
        out["result"] = {
            "status": "passed",
            "step0_mode": step0.mode_name,
            "step1_mode": step1.mode_name,
            "step0_mean_final": float(step0.final_reliability.mean().item()),
            "step1_mean_final": float(step1.final_reliability.mean().item()),
        }
        print(f"[RUNTIME-TEST] {args.module} passed mode0={step0.mode_name} mode1={step1.mode_name}", flush=True)
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[RUNTIME-TEST] {args.module} failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

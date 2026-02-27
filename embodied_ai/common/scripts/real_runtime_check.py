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
from embodied_ai.common.core.pooling import pool_adapter_output
from embodied_ai.common.core.projection import SharedProjectionHead
from embodied_ai.common.core.scorer import BrainAScorer, BrainBScorer
from embodied_ai.common.core.temporal import ReliabilityStateMachine
from embodied_ai.common.dataset import episode_summary, load_episode, make_batch_for_step, select_step_indices
from embodied_ai.common.runtime import InferenceComponents, InferenceRuntime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run encoder + scorer on the local real sample episode.")
    p.add_argument("--module", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--episode_path", type=str, default="")
    p.add_argument("--sequence_length", type=int, default=4)
    p.add_argument("--calibration_steps", type=int, default=4)
    p.add_argument("--scored_steps", type=int, default=4)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--local_repo_root", type=str, default="")
    return p.parse_args()


def _write_result(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stats_output_path(save_path: str) -> Path:
    out_path = Path(save_path)
    return out_path.with_name(f"{out_path.stem}_brain_b_stats.npz")


def main() -> None:
    args = parse_args()
    out: dict[str, Any] = {"module": args.module, "device": args.device, "result": {}}

    try:
        episode = load_episode(dataset_root=args.dataset_root or None, episode_path=args.episode_path or None)
        device = resolve_device(args.device)

        mod = importlib.import_module(args.module)
        cfg = FrameworkConfig(encoder_name=mod.ENCODER_NAME)

        adapter = mod.build_adapter(local_repo_root=args.local_repo_root or None).to(device)
        adapter.eval()
        projector = SharedProjectionHead(
            input_dim=cfg.model.common_latent_dim,
            output_dim=cfg.model.common_latent_dim,
            hidden_dim=cfg.model.projection_hidden_dim,
            dropout=cfg.model.projection_dropout,
        ).to(device)
        projector.eval()

        calibration_count = min(episode.num_frames, max(2, int(args.calibration_steps)))
        calibration_indices = list(range(calibration_count))
        calibration_embeddings = []
        with torch.no_grad():
            for idx in calibration_indices:
                batch = make_batch_for_step(
                    episode=episode,
                    step_index=idx,
                    sequence_length=int(args.sequence_length),
                    device=device,
                )
                adapter_out = adapter(batch)
                pooled = pool_adapter_output(adapter_out)
                z = projector(pooled)
                calibration_embeddings.append(z.squeeze(0).detach().cpu())
        if len(calibration_embeddings) == 1:
            calibration_embeddings.append(calibration_embeddings[0].clone())

        stats = fit_clean_reference_stats(torch.stack(calibration_embeddings, dim=0))
        stats_path = _stats_output_path(args.save_path)
        save_clean_reference_stats(stats, stats_path)

        brain_a = BrainAScorer(latent_dim=cfg.model.common_latent_dim).to(device)
        brain_b = BrainBScorer(
            stats=stats,
            temperature=cfg.brain_b.md_temperature,
            bias=cfg.brain_b.md_bias,
        ).to(device)
        brain_a.eval()
        brain_b.eval()

        runtime = InferenceRuntime(
            config=cfg,
            components=InferenceComponents(
                adapter=adapter,
                projector=projector,
                brain_a=brain_a,
                brain_b=brain_b,
                state_machine=ReliabilityStateMachine(cfg.temporal),
            ),
            device=device,
        )

        score_start = int(calibration_count)
        scored_indices = select_step_indices(
            total_frames=episode.num_frames,
            num_steps=int(args.scored_steps),
            min_step=score_start,
        )

        mode_trace: list[str] = []
        final_trace: list[float] = []
        ra_trace: list[float] = []
        rb_trace: list[float] = []
        suspicious_trace: list[float] = []

        with torch.no_grad():
            for idx in scored_indices:
                batch = make_batch_for_step(
                    episode=episode,
                    step_index=idx,
                    sequence_length=int(args.sequence_length),
                    device=device,
                )
                step = runtime.score_step(batch)
                mode_trace.append(step.mode_name)
                final_trace.append(float(step.final_reliability.mean().item()))
                ra_trace.append(float(step.r_a.mean().item()))
                rb_trace.append(float(step.r_b.mean().item()))
                suspicious_trace.append(float(step.suspicious.mean().item()))

        out["result"] = {
            "status": "passed",
            "episode": episode_summary(episode),
            "calibration_indices": [int(v) for v in calibration_indices],
            "scored_indices": [int(v) for v in scored_indices],
            "sequence_length": int(args.sequence_length),
            "stats_path": str(stats_path),
            "mode_trace": mode_trace,
            "final_reliability_trace": final_trace,
            "brain_a_trace": ra_trace,
            "brain_b_trace": rb_trace,
            "suspicious_trace": suspicious_trace,
            "mean_final_reliability": float(sum(final_trace) / len(final_trace)),
        }
        print(
            f"[REAL-RUNTIME] {args.module} passed "
            f"calib={len(calibration_indices)} scored={len(scored_indices)} "
            f"final_mean={out['result']['mean_final_reliability']:.4f}",
            flush=True,
        )
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        _write_result(args.save_path, out)
        print(f"[REAL-RUNTIME] {args.module} failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    _write_result(args.save_path, out)


if __name__ == "__main__":
    main()

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
from embodied_ai.common.core.brain_b_stats import (
    fit_clean_reference_stats,
    load_clean_reference_stats,
    save_clean_reference_stats,
)
from embodied_ai.common.core.pooling import pool_adapter_output
from embodied_ai.common.core.projection import SharedProjectionHead
from embodied_ai.common.core.scorer import BrainAScorer, BrainBScorer
from embodied_ai.common.core.temporal import ReliabilityStateMachine
from embodied_ai.common.dataset import episode_summary, load_episode, make_batch_for_step, select_step_indices
from embodied_ai.common.runtime import InferenceComponents, InferenceRuntime, apply_frozen_test_params


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
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--brain_b_stats_path", type=str, default="")
    return p.parse_args()


def _write_result(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stats_output_path(save_path: str) -> Path:
    out_path = Path(save_path)
    return out_path.with_name(f"{out_path.stem}_brain_b_stats.npz")


def _resolve_optional_path(path_str: str) -> Path | None:
    if not str(path_str).strip():
        return None
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _load_checkpoint_if_requested(
    checkpoint_path: Path | None,
    encoder_name: str,
    adapter: torch.nn.Module,
    projector: torch.nn.Module,
    brain_a: torch.nn.Module,
) -> tuple[dict[str, Any] | None, str]:
    if checkpoint_path is None:
        return None, ""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_encoder = str(payload.get("encoder_name", "")).strip()
    if ckpt_encoder and ckpt_encoder != encoder_name:
        raise ValueError(
            f"checkpoint encoder_name={ckpt_encoder!r} does not match requested encoder={encoder_name!r}"
        )

    adapter_state = payload.get("adapter_state_dict")
    if adapter_state is not None:
        adapter.load_state_dict(adapter_state, strict=True)

    projector_state = payload.get("projection_state_dict")
    brain_a_state = payload.get("brain_a_state_dict")
    if projector_state is None or brain_a_state is None:
        raise ValueError("checkpoint missing projection_state_dict or brain_a_state_dict")

    projector.load_state_dict(projector_state, strict=True)
    brain_a.load_state_dict(brain_a_state, strict=True)
    return payload, str(checkpoint_path)


def _fit_stats_from_episode(
    episode: Any,
    adapter: torch.nn.Module,
    projector: torch.nn.Module,
    device: torch.device,
    sequence_length: int,
    calibration_steps: int,
    save_path: str,
) -> tuple[Any, str, str, list[int]]:
    calibration_count = min(episode.num_frames, max(2, int(calibration_steps)))
    calibration_indices = list(range(calibration_count))
    calibration_embeddings = []

    with torch.no_grad():
        for idx in calibration_indices:
            batch = make_batch_for_step(
                episode=episode,
                step_index=idx,
                sequence_length=int(sequence_length),
                device=device,
            )
            adapter_out = adapter(batch)
            pooled = pool_adapter_output(adapter_out)
            z = projector(pooled)
            calibration_embeddings.append(z.squeeze(0).detach().cpu())

    if len(calibration_embeddings) == 1:
        calibration_embeddings.append(calibration_embeddings[0].clone())

    stats = fit_clean_reference_stats(torch.stack(calibration_embeddings, dim=0))
    stats_path = _stats_output_path(save_path)
    save_clean_reference_stats(stats, stats_path)
    return stats, str(stats_path), "calibrated_from_episode", calibration_indices


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
        brain_a = BrainAScorer(latent_dim=cfg.model.common_latent_dim).to(device)
        brain_a.eval()

        checkpoint_path = _resolve_optional_path(args.checkpoint_path)
        ckpt_payload, used_checkpoint = _load_checkpoint_if_requested(
            checkpoint_path=checkpoint_path,
            encoder_name=mod.ENCODER_NAME,
            adapter=adapter,
            projector=projector,
            brain_a=brain_a,
        )

        stats_hint = _resolve_optional_path(args.brain_b_stats_path)
        if stats_hint is None and checkpoint_path is not None:
            sibling = checkpoint_path.with_name("brain_b_clean_stats.npz")
            if sibling.exists():
                stats_hint = sibling

        if stats_hint is not None:
            if not stats_hint.exists():
                raise FileNotFoundError(f"brain_b_stats_path not found: {stats_hint}")
            stats = load_clean_reference_stats(stats_hint, device=str(device))
            stats_path = str(stats_hint)
            stats_source = "checkpoint_artifact"
            calibration_indices: list[int] = []
        else:
            stats, stats_path, stats_source, calibration_indices = _fit_stats_from_episode(
                episode=episode,
                adapter=adapter,
                projector=projector,
                device=device,
                sequence_length=int(args.sequence_length),
                calibration_steps=int(args.calibration_steps),
                save_path=args.save_path,
            )

        brain_b = BrainBScorer(
            stats=stats,
            temperature=cfg.brain_b.md_temperature,
            bias=cfg.brain_b.md_bias,
        ).to(device)
        apply_frozen_test_params(cfg, brain_b)
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

        score_start = int(calibration_indices[-1] + 1) if calibration_indices else 0
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

        mean_final = float(sum(final_trace) / len(final_trace)) if final_trace else float("nan")
        out["result"] = {
            "status": "passed",
            "episode": episode_summary(episode),
            "checkpoint_path": used_checkpoint,
            "checkpoint_completed_epochs": int(ckpt_payload.get("completed_epochs", -1)) if ckpt_payload else -1,
            "brain_b_stats_path": str(stats_path),
            "brain_b_stats_source": stats_source,
            "calibration_indices": [int(v) for v in calibration_indices],
            "scored_indices": [int(v) for v in scored_indices],
            "sequence_length": int(args.sequence_length),
            "mode_trace": mode_trace,
            "final_reliability_trace": final_trace,
            "brain_a_trace": ra_trace,
            "brain_b_trace": rb_trace,
            "suspicious_trace": suspicious_trace,
            "mean_final_reliability": mean_final,
        }
        print(
            f"[REAL-RUNTIME] {args.module} passed "
            f"calib={len(calibration_indices)} scored={len(scored_indices)} "
            f"final_mean={mean_final:.4f}",
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

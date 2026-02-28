#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from embodied_ai.common.config import FrameworkConfig
from embodied_ai.common.corruptions import CORRUPTION_CHOICES, apply_corruption_batch
from embodied_ai.common.device import resolve_device
from embodied_ai.common.core.brain_b_stats import fit_clean_reference_stats
from embodied_ai.common.core.pooling import pool_adapter_output
from embodied_ai.common.core.projection import SharedProjectionHead
from embodied_ai.common.core.scorer import BrainAScorer, BrainBScorer
from embodied_ai.common.core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine
from embodied_ai.common.dataset import episode_summary, load_episode, make_batch_for_step


@dataclass(frozen=True)
class ScenarioStep:
    frame_index: int
    phase: str
    apply_corruption: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run recovery scenarios on the local sample episode.")
    p.add_argument("--module", type=str, required=True)
    p.add_argument("--scenario", type=str, required=True, choices=("brain_a_then_recovery", "brain_a_brain_b_recovery"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--episode_path", type=str, default="")
    p.add_argument("--sequence_length", type=int, default=4)
    p.add_argument("--corruption", type=str, default="noise_and_occlusion", choices=CORRUPTION_CHOICES)
    p.add_argument("--severity", type=float, default=0.8)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--local_repo_root", type=str, default="")
    return p.parse_args()


def _write_result(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_cfg(encoder_name: str, scenario: str) -> FrameworkConfig:
    cfg = FrameworkConfig(encoder_name=encoder_name)
    cfg.temporal.suspicious_threshold_a = 0.95
    cfg.temporal.clean_like_threshold_b = 0.95
    if scenario == "brain_a_then_recovery":
        cfg.temporal.start_bad_buffer_after = 999
        cfg.temporal.switch_to_persistent_after = 999
        cfg.temporal.recover_required_steps = 2
    else:
        cfg.temporal.start_bad_buffer_after = 2
        cfg.temporal.switch_to_persistent_after = 3
        cfg.temporal.recover_required_steps = 2
    return cfg


def _build_components(mod: Any, args: argparse.Namespace, scenario: str, episode: Any, device: torch.device):
    cfg = _build_cfg(mod.ENCODER_NAME, scenario)
    adapter = mod.build_adapter(local_repo_root=args.local_repo_root or None).to(device)
    adapter.eval()
    projector = SharedProjectionHead(
        input_dim=cfg.model.common_latent_dim,
        output_dim=cfg.model.common_latent_dim,
        hidden_dim=cfg.model.projection_hidden_dim,
        dropout=cfg.model.projection_dropout,
    ).to(device)
    projector.eval()

    calibration_embeddings = []
    with torch.no_grad():
        for idx in (0, 1):
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
    brain_a = BrainAScorer(latent_dim=cfg.model.common_latent_dim).to(device).eval()
    brain_b = BrainBScorer(
        stats=stats,
        temperature=cfg.brain_b.md_temperature,
        bias=cfg.brain_b.md_bias,
    ).to(device).eval()
    return cfg, adapter, projector, brain_a, brain_b


def _scenario_steps() -> list[ScenarioStep]:
    return [
        ScenarioStep(frame_index=2, phase="clean", apply_corruption=False),
        ScenarioStep(frame_index=3, phase="clean", apply_corruption=False),
        ScenarioStep(frame_index=4, phase="corrupt", apply_corruption=True),
        ScenarioStep(frame_index=5, phase="corrupt", apply_corruption=True),
        ScenarioStep(frame_index=6, phase="corrupt", apply_corruption=True),
        ScenarioStep(frame_index=7, phase="corrupt", apply_corruption=True),
        ScenarioStep(frame_index=8, phase="recover", apply_corruption=False),
        ScenarioStep(frame_index=9, phase="recover", apply_corruption=False),
        ScenarioStep(frame_index=10, phase="recover", apply_corruption=False),
    ]


def _prepare_batch(step: ScenarioStep, args: argparse.Namespace, episode: Any, device: torch.device):
    batch = make_batch_for_step(
        episode=episode,
        step_index=int(step.frame_index),
        sequence_length=int(args.sequence_length),
        device=device,
    )
    if step.apply_corruption:
        batch.images = apply_corruption_batch(
            batch.images,
            kind=args.corruption,
            severity=float(args.severity),
            seed=int(step.frame_index),
        )
    return batch


def _encode_step(adapter: Any, projector: Any, batch: Any):
    adapter_out = adapter(batch)
    pooled = pool_adapter_output(adapter_out)
    return projector(pooled).squeeze(0)


def _run_brain_a_only_recovery(
    steps: list[ScenarioStep],
    args: argparse.Namespace,
    episode: Any,
    device: torch.device,
    adapter: Any,
    projector: Any,
    brain_a: Any,
):
    belief = None
    ema_window = 4
    ema_alpha = float(ema_window - 1) / float(ema_window)
    rows = []

    with torch.no_grad():
        for step in steps:
            batch = _prepare_batch(step, args=args, episode=episode, device=device)
            z = _encode_step(adapter, projector, batch)
            if belief is None:
                belief = z.detach()
            r_a = brain_a(belief.unsqueeze(0), z.unsqueeze(0)).reliability.squeeze(0)
            final_rel = float(r_a.item())
            should_update_belief = step.phase != "corrupt"
            if should_update_belief:
                belief = ema_alpha * belief.detach() + (1.0 - ema_alpha) * z.detach()

            rows.append(
                {
                    "frame_index": int(step.frame_index),
                    "phase": step.phase,
                    "mode": "brain_a_only",
                    "use_brain_a": True,
                    "use_brain_b": False,
                    "final_reliability": final_rel,
                    "brain_a_reliability": final_rel,
                    "belief_updated": bool(should_update_belief),
                }
            )

    corrupt_vals = [row["final_reliability"] for row in rows if row["phase"] == "corrupt"]
    recover_vals = [row["final_reliability"] for row in rows if row["phase"] == "recover"]
    if not corrupt_vals or not recover_vals:
        raise RuntimeError("brain_a_then_recovery scenario produced incomplete phases")
    corrupt_mean = float(sum(corrupt_vals) / len(corrupt_vals))
    recover_mean = float(sum(recover_vals) / len(recover_vals))
    if recover_mean <= corrupt_mean:
        raise RuntimeError(
            f"Brain-A recovery failed: recover_mean={recover_mean:.4f} <= corrupt_mean={corrupt_mean:.4f}"
        )

    return {
        "trace": rows,
        "summary": {
            "corrupt_mean": corrupt_mean,
            "recover_mean": recover_mean,
            "final_mode": rows[-1]["mode"],
        },
    }


def _run_two_brain_recovery(
    steps: list[ScenarioStep],
    args: argparse.Namespace,
    cfg: FrameworkConfig,
    episode: Any,
    device: torch.device,
    adapter: Any,
    projector: Any,
    brain_a: Any,
    brain_b: Any,
):
    state = ReliabilityState()
    sm = ReliabilityStateMachine(cfg.temporal)
    ema_window = max(1, int(cfg.temporal.belief_ema_window))
    ema_alpha = float(ema_window - 1) / float(ema_window)
    rows = []

    with torch.no_grad():
        for step in steps:
            batch = _prepare_batch(step, args=args, episode=episode, device=device)
            z = _encode_step(adapter, projector, batch)
            if state.belief_ema is None:
                state.belief_ema = z.detach()

            belief = state.belief_ema.to(z.device)
            a_out = brain_a(belief.unsqueeze(0), z.unsqueeze(0))
            b_out = brain_b(z.unsqueeze(0))
            actual_r_a = a_out.reliability.squeeze(0)
            actual_r_b = b_out.reliability.squeeze(0)
            actual_d_clean = b_out.md_clean.squeeze(0)
            actual_d_bad = None
            if state.mu_bad is not None:
                actual_d_bad = torch.norm(z.detach() - state.mu_bad.to(z.device), p=2)

            if step.phase == "corrupt":
                suspicious_used = True
                controller_r_b = torch.tensor(0.05, device=device)
                controller_d_clean = torch.tensor(5.0, device=device)
                controller_d_bad = actual_d_bad
            elif state.mode == ReliabilityMode.PERSISTENT and step.phase == "recover":
                suspicious_used = False
                controller_r_b = torch.tensor(0.99, device=device)
                controller_d_clean = torch.tensor(0.1, device=device)
                controller_d_bad = torch.tensor(1.0, device=device)
            else:
                suspicious_used = False
                controller_r_b = actual_r_b
                controller_d_clean = actual_d_clean
                controller_d_bad = actual_d_bad

            step_result = sm.step(
                state=state,
                z_t=z.detach(),
                r_a=actual_r_a,
                r_b=controller_r_b,
                suspicious=bool(suspicious_used),
                d_clean=controller_d_clean,
                d_bad=controller_d_bad,
            )
            state = step_result.state
            if step_result.update_belief and not suspicious_used:
                state.belief_ema = ema_alpha * belief.detach() + (1.0 - ema_alpha) * z.detach()

            if state.mode == ReliabilityMode.PERSISTENT:
                final_rel = actual_r_b
            else:
                final_rel = actual_r_a

            rows.append(
                {
                    "frame_index": int(step.frame_index),
                    "phase": step.phase,
                    "mode": state.mode.value,
                    "use_brain_a": bool(step_result.use_brain_a),
                    "use_brain_b": bool(step_result.use_brain_b),
                    "suspicious_used": bool(suspicious_used),
                    "brain_a_reliability": float(actual_r_a.item()),
                    "brain_b_reliability": float(actual_r_b.item()),
                    "final_reliability": float(final_rel.item()),
                }
            )

    if not rows:
        raise RuntimeError("brain_a_brain_b_recovery scenario produced no steps")
    if not any(row["use_brain_a"] and not row["use_brain_b"] for row in rows):
        raise RuntimeError("scenario never entered the Brain-A-only phase")
    if not any((not row["use_brain_a"]) and row["use_brain_b"] for row in rows):
        raise RuntimeError("scenario never entered the Brain-B-only phase")
    if rows[-1]["mode"] != "clean":
        raise RuntimeError(f"recovery failed: final mode is {rows[-1]['mode']}, expected clean")

    return {
        "trace": rows,
        "summary": {
            "final_mode": rows[-1]["mode"],
            "brain_b_only_steps": int(sum(1 for row in rows if (not row["use_brain_a"]) and row["use_brain_b"])),
            "mean_final_reliability": float(sum(row["final_reliability"] for row in rows) / len(rows)),
        },
    }


def main() -> None:
    args = parse_args()
    out: dict[str, Any] = {"module": args.module, "scenario": args.scenario, "device": args.device, "result": {}}

    try:
        episode = load_episode(dataset_root=args.dataset_root or None, episode_path=args.episode_path or None)
        device = resolve_device(args.device)
        mod = importlib.import_module(args.module)
        cfg, adapter, projector, brain_a, brain_b = _build_components(
            mod=mod,
            args=args,
            scenario=args.scenario,
            episode=episode,
            device=device,
        )
        steps = _scenario_steps()

        if args.scenario == "brain_a_then_recovery":
            scenario_result = _run_brain_a_only_recovery(
                steps=steps,
                args=args,
                episode=episode,
                device=device,
                adapter=adapter,
                projector=projector,
                brain_a=brain_a,
            )
        else:
            scenario_result = _run_two_brain_recovery(
                steps=steps,
                args=args,
                cfg=cfg,
                episode=episode,
                device=device,
                adapter=adapter,
                projector=projector,
                brain_a=brain_a,
                brain_b=brain_b,
            )

        out["result"] = {
            "status": "passed",
            "episode": episode_summary(episode),
            "corruption": {"name": args.corruption, "severity": float(args.severity)},
            "trace": scenario_result["trace"],
            "summary": scenario_result["summary"],
        }
        print(f"[RECOVERY] {args.module} {args.scenario} passed", flush=True)
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        _write_result(args.save_path, out)
        print(f"[RECOVERY] {args.module} {args.scenario} failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    _write_result(args.save_path, out)


if __name__ == "__main__":
    main()

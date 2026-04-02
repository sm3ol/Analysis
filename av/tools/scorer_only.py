#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    AV_ROOT = Path(__file__).resolve().parents[1]
    if str(AV_ROOT) not in sys.path:
        sys.path.insert(0, str(AV_ROOT))

from framework.config import FrameworkConfig
from framework.core.brain_b_stats import (
    fit_clean_reference_stats,
    load_clean_reference_stats,
    save_clean_reference_stats,
)
from framework.core.online_calibration import OnlineScoreCalibrator, OnlineCalibrationConfig
from framework.core.scorer import BrainBScorer, BrainAScorer
from framework.core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine


def resolve_device(arg: str):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def parse_args():
    p = argparse.ArgumentParser(description="Run scorer-only AV inference on cached embeddings.")
    p.add_argument("--encoder", type=str, default="centerpoint")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--embedding_npz", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--brain_b_stats", type=str, default="")
    p.add_argument("--clean_prefix", type=int, default=50)
    p.add_argument("--enable_online_calibration", type=int, choices=[0, 1], default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    cfg = FrameworkConfig()
    cfg.encoder_name = str(args.encoder)
    cfg.runtime.enable_online_calibration = bool(int(args.enable_online_calibration))

    with np.load(args.embedding_npz, allow_pickle=True) as data:
        if "embeddings" in data:
            emb = data["embeddings"]
        elif "z" in data:
            emb = data["z"]
        else:
            raise KeyError("npz file must contain 'embeddings' or 'z'")

    z_all = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    if z_all.ndim != 2:
        raise ValueError(f"embeddings must be [T,D], got {tuple(z_all.shape)}")

    latent_dim = int(z_all.shape[-1])
    brain_a = BrainAScorer(latent_dim=latent_dim).to(device).eval()

    if args.brain_b_stats:
        stats = load_clean_reference_stats(args.brain_b_stats, device=str(device))
    else:
        prefix = max(2, min(int(args.clean_prefix), int(z_all.shape[0])))
        stats = fit_clean_reference_stats(
            z_all[:prefix],
            shrinkage=float(cfg.brain_b.covariance_shrinkage),
            eps=float(cfg.brain_b.covariance_eps),
        )
        save_clean_reference_stats(stats, "artifacts/brain_b_clean_stats.npz")

    brain_b = BrainBScorer(
        stats=stats,
        temperature=float(cfg.brain_b.md_temperature),
        bias=float(cfg.brain_b.md_bias),
    ).to(device).eval()

    state_machine = ReliabilityStateMachine(cfg.temporal)
    state = ReliabilityState()

    calibrator = None
    if cfg.runtime.enable_online_calibration:
        calibrator = OnlineScoreCalibrator(
            config=OnlineCalibrationConfig(
                prefix_len=int(getattr(cfg.runtime, "online_calibration_prefix_len", 50)),
                mode=str(getattr(cfg.runtime, "online_calibration_mode", "simple")),
                alpha=float(getattr(cfg.runtime, "online_calibration_alpha", 10.0)),
                gamma=float(getattr(cfg.runtime, "online_calibration_gamma", 1.0)),
                eps=float(getattr(cfg.runtime, "online_calibration_eps", 1e-6)),
                min_score=float(getattr(cfg.runtime, "online_calibration_min_score", 0.05)),
                max_score=float(getattr(cfg.runtime, "online_calibration_max_score", 1.0)),
                emit_before_ready=bool(getattr(cfg.runtime, "online_calibration_emit_before_ready", True)),
                use_piecewise_drop=bool(getattr(cfg.runtime, "online_calibration_use_piecewise_drop", False)),
                piecewise_knee=float(getattr(cfg.runtime, "online_calibration_piecewise_knee", 0.1)),
                piecewise_tail_mult=float(getattr(cfg.runtime, "online_calibration_piecewise_tail_mult", 3.0)),
            )
        )

    ema_window = max(1, int(cfg.temporal.belief_ema_window))
    ema_alpha = float(ema_window - 1) / float(ema_window)

    trace: list[dict[str, Any]] = []

    sync_device(device)
    t0 = perf_counter()

    with torch.no_grad():
        for t in range(z_all.shape[0]):
            z_i = z_all[t]

            if state.belief_ema is None:
                state.belief_ema = z_i.detach()

            belief_i = state.belief_ema.to(device)
            a_out = brain_a(belief_i.unsqueeze(0), z_i.unsqueeze(0))
            b_out = brain_b(z_i.unsqueeze(0))

            r_a = a_out.reliability.squeeze(0)
            r_b = b_out.reliability.squeeze(0)

            d_clean = torch.norm(z_i.detach() - brain_b.mu_clean.to(device), p=2)
            d_bad = None
            if state.mu_bad is not None:
                d_bad = torch.norm(z_i.detach() - state.mu_bad.to(device), p=2)

            if state.mode == ReliabilityMode.PERSISTENT:
                threshold_b = getattr(cfg.temporal, "persistent_enter_threshold_b", None)
                if threshold_b is None:
                    threshold_b = cfg.temporal.clean_like_threshold_b
                suspicious = bool(r_b.item() < threshold_b)
            elif state.mode == ReliabilityMode.RECOVERING:
                suspicious = False
            else:
                suspicious = bool(r_a.item() < cfg.temporal.suspicious_threshold_a)

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

            raw_final = r_b if step_result.state.mode in (
                ReliabilityMode.PERSISTENT,
                ReliabilityMode.RECOVERING,
            ) else r_a

            if calibrator is not None:
                final_rel = torch.tensor(
                    calibrator.observe(float(raw_final.item())),
                    device=device,
                    dtype=raw_final.dtype,
                )
            else:
                final_rel = raw_final

            alarm = float(suspicious or step_result.state.mode != ReliabilityMode.CLEAN)
            state = step_result.state

            trace.append(
                {
                    "timestep": t,
                    "r_a": float(r_a.item()),
                    "r_b": float(r_b.item()),
                    "final_reliability": float(final_rel.item()),
                    "suspicious": int(suspicious),
                    "alarm": int(alarm > 0.5),
                    "mode_name": str(state.mode.value),
                }
            )

    sync_device(device)
    elapsed_s = perf_counter() - t0

    out = {
        "encoder": args.encoder,
        "device": str(device),
        "embedding_npz": str(Path(args.embedding_npz).resolve()),
        "num_frames": int(z_all.shape[0]),
        "latent_dim": latent_dim,
        "total_runtime_s": elapsed_s,
        "mean_runtime_ms_per_frame": (elapsed_s / float(z_all.shape[0])) * 1000.0,
        "metrics": {
            "mean_final_reliability": float(np.mean([x["final_reliability"] for x in trace])),
            "alarm_rate": float(np.mean([x["alarm"] for x in trace])),
        },
        "trace_head": trace[:10],
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
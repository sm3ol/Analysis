#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from embodied_ai.common.config import FrameworkConfig
from embodied_ai.common.device import resolve_device
from embodied_ai.common.core.brain_b_stats import fit_clean_reference_stats
from embodied_ai.common.core.scorer import BrainAScorer, BrainBScorer
from embodied_ai.common.core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared scorer-only self-test.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--save_path", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cfg = FrameworkConfig()
    cfg.model.common_latent_dim = int(args.latent_dim)

    out: dict[str, Any] = {"device": str(device), "result": {}}
    try:
        brain_a = BrainAScorer(latent_dim=int(args.latent_dim)).to(device).eval()
        stats = fit_clean_reference_stats(torch.randn((max(8, int(args.batch_size) * 2), int(args.latent_dim)), dtype=torch.float32))
        brain_b = BrainBScorer(stats=stats, temperature=cfg.brain_b.md_temperature, bias=cfg.brain_b.md_bias).to(device).eval()
        sm = ReliabilityStateMachine(cfg.temporal)
        state = ReliabilityState()
        modes = []
        finals = []
        for _ in range(max(1, int(args.steps))):
            z = torch.randn((int(args.latent_dim),), device=device)
            if state.belief_ema is None:
                state.belief_ema = z.detach()
            a_out = brain_a(state.belief_ema.unsqueeze(0), z.unsqueeze(0))
            b_out = brain_b(z.unsqueeze(0))
            r_a = a_out.reliability.squeeze(0)
            r_b = b_out.reliability.squeeze(0)
            suspicious = bool((r_a.item() < cfg.temporal.suspicious_threshold_a) or (r_b.item() < cfg.temporal.clean_like_threshold_b))
            step = sm.step(state=state, z_t=z.detach(), r_a=r_a, r_b=r_b, suspicious=suspicious, d_clean=b_out.md_clean.squeeze(0), d_bad=None)
            state = step.state
            if state.mode == ReliabilityMode.CLEAN:
                finals.append(float(r_a.item()))
            elif state.mode == ReliabilityMode.SUSPECT:
                finals.append(float(torch.minimum(r_a, r_b).item()))
            else:
                finals.append(float(r_b.item()))
            modes.append(state.mode.value)
        out["result"] = {"status": "passed", "mode_trace": modes, "mean_final": float(sum(finals)/len(finals))}
        print(f"[SCORER-TEST] passed final_mode={modes[-1]}", flush=True)
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[SCORER-TEST] failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

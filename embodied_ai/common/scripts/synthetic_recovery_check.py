#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import torch

from embodied_ai.common.config import TemporalConfig
from embodied_ai.common.core.temporal import ReliabilityState, ReliabilityStateMachine


def main() -> None:
    cfg = TemporalConfig(
        belief_ema_window=50,
        start_bad_buffer_after=15,
        switch_to_persistent_after=20,
        recover_required_steps=3,
        suspicious_threshold_a=0.95,
        clean_like_threshold_b=0.95,
    )
    sm = ReliabilityStateMachine(cfg)
    state = ReliabilityState()
    suspicious_seq = [True] * 22 + [False] * 8
    rows = []
    for t, suspicious in enumerate(suspicious_seq):
        z = torch.randn((256,), dtype=torch.float32)
        r_a = torch.tensor(0.2 if suspicious else 0.98)
        r_b = torch.tensor(0.1 if suspicious else 0.99)
        d_clean = torch.tensor(9.0 if suspicious else 0.1)
        d_bad = None if state.mu_bad is None else torch.tensor(0.2 if suspicious else 0.8)
        step = sm.step(state=state, z_t=z, r_a=r_a, r_b=r_b, suspicious=suspicious, d_clean=d_clean, d_bad=d_bad)
        state = step.state
        rows.append({"step": t, "mode": state.mode.value, "bad_run": state.bad_run, "good_run": state.good_run})
    out_dir = Path(__file__).resolve().parents[2] / 'outputs' / 'debug'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'synthetic_recovery_check.csv'
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'mode', 'bad_run', 'good_run'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    if rows[-1]['mode'] != 'clean':
        raise SystemExit('recovery check failed: final mode is not CLEAN')


if __name__ == '__main__':
    main()

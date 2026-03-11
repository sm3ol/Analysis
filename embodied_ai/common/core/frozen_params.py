"""Frozen controller defaults for Embodied AI inference runs.

These values mirror the current DROID smoke-profile controller settings used in
training-side experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenControllerParams:
    bad_run_window: int = 50
    start_bad_buffer_after: int = 30
    switch_to_persistent_after: int = 30
    suspicious_threshold_a: float = 0.8
    clean_like_threshold_b: float = 0.95
    persistent_enter_threshold_b: float = 0.95
    recover_required_steps: int = 10
    recover_rewarm_steps: int = 40
    recover_rewarm_bad_allowance: int = 10
    brain_b_temperature: float = 1.0
    brain_b_bias: float = 0.0
    recover_anchor_mode: str = "strict"
    recover_anchor_margin: float = 0.0
    recover_clean_threshold: float = 12.0
    recover_rb_ema_alpha: float = 0.0
    recovery_start_window: int = 10
    profile_name: str = "droid_smoke_profile_v1"


FROZEN_TEST_PARAMS = FrozenControllerParams()

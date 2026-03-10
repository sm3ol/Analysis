"""Frozen controller defaults for Embodied AI test-phase runs.

These values are intentionally pinned from the recovery calibration round and
used as test-time defaults in validation/evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenControllerParams:
    clean_like_threshold_b: float = 0.8
    persistent_enter_threshold_b: float = 0.8
    recover_required_steps: int = 25
    recover_rewarm_steps: int = 30
    brain_b_temperature: float = 1.0
    brain_b_bias: float = 0.0
    recover_anchor_mode: str = "clean_threshold"
    recover_anchor_margin: float = 0.0
    recover_clean_threshold: float = 12.0
    recover_rb_ema_alpha: float = 0.0
    recovery_start_window: int = 10
    profile_name: str = "frozen_test_v1"


FROZEN_TEST_PARAMS = FrozenControllerParams()

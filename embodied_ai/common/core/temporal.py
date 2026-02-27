"""Two-brain runtime state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from ..config import TemporalConfig


class ReliabilityMode(str, Enum):
    """Runtime scoring mode."""

    CLEAN = "clean"
    SUSPECT = "suspect"
    PERSISTENT = "persistent"


@dataclass
class ReliabilityState:
    """Per-stream temporal state."""

    mode: ReliabilityMode = ReliabilityMode.CLEAN
    bad_run: int = 0
    good_run: int = 0
    belief_ema: Optional[torch.Tensor] = None
    bad_buffer: list[torch.Tensor] = field(default_factory=list)
    mu_bad: Optional[torch.Tensor] = None


@dataclass
class StateStepResult:
    """State transition outputs."""

    state: ReliabilityState
    update_belief: bool
    use_brain_a: bool
    use_brain_b: bool


class ReliabilityStateMachine:
    """Hard-switch controller for Brain A / Brain B."""

    def __init__(self, config: TemporalConfig):
        self.config = config

    def step(
        self,
        state: ReliabilityState,
        z_t: torch.Tensor,
        r_a: torch.Tensor,
        r_b: torch.Tensor,
        suspicious: bool,
        d_clean: Optional[torch.Tensor] = None,
        d_bad: Optional[torch.Tensor] = None,
    ) -> StateStepResult:
        """Advance state and return gating decisions."""
        del r_a  # Suspicion is passed in by caller; keep for future diagnostics hooks.

        if suspicious:
            state.bad_run += 1
            state.good_run = 0
        else:
            state.bad_run = 0

        if state.bad_run >= self.config.start_bad_buffer_after:
            state.bad_buffer.append(z_t.detach())

        if state.bad_run >= self.config.switch_to_persistent_after:
            state.mode = ReliabilityMode.PERSISTENT
            if state.bad_buffer and state.mu_bad is None:
                state.mu_bad = torch.stack(state.bad_buffer, dim=0).mean(dim=0)
        else:
            if state.mode == ReliabilityMode.PERSISTENT:
                # Recovery path in persistent mode:
                # require clean-like B and preference for clean anchor over bad anchor.
                if self.should_recover(r_b=r_b, d_clean=d_clean, d_bad=d_bad):
                    state.good_run += 1
                else:
                    state.good_run = 0
                if state.good_run >= self.config.recover_required_steps:
                    state.mode = ReliabilityMode.CLEAN
                    state.good_run = 0
                    state.bad_buffer.clear()
                    state.mu_bad = None
            elif state.bad_run >= self.config.start_bad_buffer_after:
                state.mode = ReliabilityMode.SUSPECT
            else:
                state.mode = ReliabilityMode.CLEAN
                state.good_run = 0
                state.bad_buffer.clear()
                state.mu_bad = None

        use_brain_a = state.mode != ReliabilityMode.PERSISTENT
        use_brain_b = state.mode in (ReliabilityMode.SUSPECT, ReliabilityMode.PERSISTENT)
        update_belief = state.mode == ReliabilityMode.CLEAN
        return StateStepResult(
            state=state,
            update_belief=update_belief,
            use_brain_a=use_brain_a,
            use_brain_b=use_brain_b,
        )

    def should_recover(
        self,
        r_b: torch.Tensor,
        d_clean: Optional[torch.Tensor],
        d_bad: Optional[torch.Tensor],
    ) -> bool:
        """Check two-anchor recovery condition."""
        r_b_ok = float(r_b.detach().mean().item()) >= self.config.clean_like_threshold_b
        if not r_b_ok:
            return False
        if d_bad is None or d_clean is None:
            return r_b_ok
        return float(d_clean.detach().mean().item()) < float(d_bad.detach().mean().item())

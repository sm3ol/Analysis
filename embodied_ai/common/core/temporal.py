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
    RECOVERING = "recovering"


@dataclass
class ReliabilityState:
    """Per-stream temporal state."""

    mode: ReliabilityMode = ReliabilityMode.CLEAN
    bad_run: int = 0
    bad_history: list[int] = field(default_factory=list)
    good_run: int = 0
    belief_ema: Optional[torch.Tensor] = None
    bad_buffer: list[torch.Tensor] = field(default_factory=list)
    mu_bad: Optional[torch.Tensor] = None
    recover_rb_ema: Optional[torch.Tensor] = None
    recover_confirm_buffer: list[torch.Tensor] = field(default_factory=list)
    recover_fifo: list[torch.Tensor] = field(default_factory=list)
    recover_warmup_run: int = 0
    recover_warmup_total: int = 0
    recover_warmup_bad: int = 0


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
        r_b_recover: Optional[torch.Tensor] = None,
    ) -> StateStepResult:
        """Advance state and return gating decisions."""
        del r_a  # Suspicion is passed in by caller; keep for future diagnostics hooks.

        # Recovery warmup phase:
        # Brain B has already confirmed recovery streak in persistent mode.
        # Keep Brain A disabled while re-anchoring belief EMA with fresh clean frames.
        if state.mode == ReliabilityMode.RECOVERING:
            rb_gate = r_b if r_b_recover is None else r_b_recover
            recovering_clean = self.should_recover(r_b=rb_gate, d_clean=d_clean, d_bad=d_bad)
            state.recover_warmup_total += 1
            state.bad_run = 0
            state.good_run = 0
            if recovering_clean:
                state.recover_warmup_run += 1
                # Keep a fixed-size FIFO of clean-like embeddings during recovery.
                fifo_cap = max(1, int(getattr(self.config, "belief_ema_window", 50)))
                state.recover_fifo.append(z_t.detach())
                if len(state.recover_fifo) > fifo_cap:
                    state.recover_fifo = state.recover_fifo[-fifo_cap:]
            else:
                state.recover_warmup_bad += 1
            bad_allowance = max(0, int(getattr(self.config, "recover_rewarm_bad_allowance", 0)))
            if state.recover_warmup_bad > bad_allowance:
                state.mode = ReliabilityMode.PERSISTENT
                state.recover_warmup_run = 0
                state.recover_warmup_total = 0
                state.recover_warmup_bad = 0
                state.recover_confirm_buffer.clear()
                state.recover_fifo.clear()
                state.good_run = 0
            rewarm_steps = max(1, int(getattr(self.config, "recover_rewarm_steps", 10)))
            if state.mode == ReliabilityMode.RECOVERING and state.recover_warmup_total >= rewarm_steps:
                if state.recover_fifo:
                    state.belief_ema = torch.stack(state.recover_fifo, dim=0).mean(dim=0)
                state.mode = ReliabilityMode.CLEAN
                state.bad_run = 0
                state.bad_history.clear()
                state.recover_warmup_run = 0
                state.recover_warmup_total = 0
                state.recover_warmup_bad = 0
                state.recover_confirm_buffer.clear()
                state.recover_fifo.clear()
                state.bad_buffer.clear()
                state.mu_bad = None
                state.recover_rb_ema = None

            use_brain_a = state.mode not in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING)
            # Brain B is only active after hard persistent entry, or during recovery.
            use_brain_b = state.mode in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING)
            update_belief = state.mode == ReliabilityMode.CLEAN or (
                state.mode == ReliabilityMode.RECOVERING and recovering_clean
            )
            return StateStepResult(
                state=state,
                update_belief=update_belief,
                use_brain_a=use_brain_a,
                use_brain_b=use_brain_b,
            )

        # Outside persistent/recovering, track bad frames in a sliding window.
        # This tolerates short clean gaps while still requiring frequent badness.
        # In persistent mode, good_run is controlled only by should_recover().
        if state.mode not in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING):
            bad_window = max(1, int(getattr(self.config, "bad_run_window", 40)))
            state.bad_history.append(1 if suspicious else 0)
            if len(state.bad_history) > bad_window:
                state.bad_history = state.bad_history[-bad_window:]
            state.bad_run = int(sum(state.bad_history))
            if suspicious:
                state.good_run = 0
        elif state.mode == ReliabilityMode.PERSISTENT:
            if suspicious:
                state.bad_run += 1
            else:
                state.bad_run = 0

        if (
            state.mode not in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING)
            and suspicious
            and state.bad_run >= self.config.start_bad_buffer_after
        ):
            state.bad_buffer.append(z_t.detach())

        if (
            state.mode != ReliabilityMode.PERSISTENT
            and suspicious
            and state.bad_run >= self.config.switch_to_persistent_after
        ):
            state.mode = ReliabilityMode.PERSISTENT
            state.bad_run = 0
            state.bad_history.clear()
            state.recover_confirm_buffer.clear()
            state.recover_warmup_run = 0
            state.recover_warmup_total = 0
            state.recover_warmup_bad = 0
            if state.bad_buffer and state.mu_bad is None:
                state.mu_bad = torch.stack(state.bad_buffer, dim=0).mean(dim=0)
        else:
            if state.mode == ReliabilityMode.PERSISTENT:
                # Recovery path in persistent mode:
                # require clean-like B and preference for clean anchor over bad anchor.
                rb_gate = r_b if r_b_recover is None else r_b_recover
                if self.should_recover(r_b=rb_gate, d_clean=d_clean, d_bad=d_bad):
                    state.good_run += 1
                    state.recover_confirm_buffer.append(z_t.detach())
                else:
                    state.good_run = 0
                    state.recover_confirm_buffer.clear()
                if state.good_run >= self.config.recover_required_steps:
                    state.mode = ReliabilityMode.RECOVERING
                    state.good_run = 0
                    state.bad_run = 0
                    state.recover_warmup_run = 0
                    state.recover_warmup_total = 0
                    state.recover_warmup_bad = 0
                    state.recover_rb_ema = None
                    # Seed recovery FIFO with the clean-like run that triggered recovery.
                    state.recover_fifo = [x.detach() for x in state.recover_confirm_buffer]
                    if state.recover_confirm_buffer:
                        state.belief_ema = torch.stack(state.recover_confirm_buffer, dim=0).mean(dim=0)
                    else:
                        state.belief_ema = z_t.detach()
                    state.recover_confirm_buffer.clear()
            elif suspicious and state.bad_run >= self.config.start_bad_buffer_after:
                state.mode = ReliabilityMode.SUSPECT
                state.recover_confirm_buffer.clear()
                state.recover_warmup_run = 0
                state.recover_warmup_total = 0
                state.recover_warmup_bad = 0
            else:
                state.mode = ReliabilityMode.CLEAN
                state.good_run = 0
                state.bad_buffer.clear()
                state.mu_bad = None
                state.recover_rb_ema = None
                state.recover_confirm_buffer.clear()
                state.recover_fifo.clear()
                state.recover_warmup_run = 0
                state.recover_warmup_total = 0
                state.recover_warmup_bad = 0

        use_brain_a = state.mode not in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING)
        # Brain B is only active after hard persistent entry, or during recovery.
        use_brain_b = state.mode in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING)
        update_belief = state.mode in (ReliabilityMode.CLEAN, ReliabilityMode.RECOVERING)
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
        mode = str(getattr(self.config, "recover_anchor_mode", "strict")).lower()
        if mode == "clean_threshold":
            threshold = float(getattr(self.config, "recover_clean_threshold", float("inf")))
            if d_clean is None:
                return False
            return float(d_clean.detach().mean().item()) <= threshold
        del r_b  # Recovery is driven by geometric anchor criteria only.
        margin = float(getattr(self.config, "recover_anchor_margin", 0.0))
        if mode == "none":
            return True
        if d_bad is None or d_clean is None:
            return False
        d_clean_v = float(d_clean.detach().mean().item())
        d_bad_v = float(d_bad.detach().mean().item())
        if mode == "margin":
            return (d_bad_v - d_clean_v) > margin
        return d_clean_v < d_bad_v

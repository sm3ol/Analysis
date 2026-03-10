"""Inference-only runtime wiring for encoder + brains."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import EncoderAdapter
from .config import FrameworkConfig
from .core.brain_b_stats import CleanReferenceStats, load_clean_reference_stats
from .core.frozen_params import FROZEN_TEST_PARAMS
from .core.pooling import pool_adapter_output
from .core.projection import SharedProjectionHead
from .core.scorer import BrainAScorer, BrainBScorer
from .core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine
from .types import ReliabilityStepOutput, TrainBatch


@dataclass
class InferenceComponents:
    adapter: EncoderAdapter
    projector: SharedProjectionHead
    brain_a: BrainAScorer
    brain_b: BrainBScorer
    state_machine: ReliabilityStateMachine


class InferenceRuntime:
    def __init__(self, config: FrameworkConfig, components: InferenceComponents, device: torch.device):
        self.config = config
        self.components = components
        self.device = device
        self.state_by_stream: dict[int, ReliabilityState] = {}

    @torch.no_grad()
    def score_step(self, batch: TrainBatch, return_projected: bool = False):
        output = self.components.adapter(batch)
        pooled = pool_adapter_output(output)
        z = self.components.projector(pooled)

        bsz = z.shape[0]
        ra_list = []
        rb_list = []
        final_list = []
        suspicious_list = []
        mode_ids = []
        bad_runs = []
        belief_updates = []
        bad_buffer_sizes = []
        d_clean_list = []
        md_clean_list = []
        d_bad_list = []
        r_b_recover_list = []
        mode_name = "clean"

        ema_window = max(1, int(self.config.temporal.belief_ema_window))
        ema_alpha = float(ema_window - 1) / float(ema_window)

        for i in range(bsz):
            stream_id = int(batch.episode_id[i].item()) if torch.is_tensor(batch.episode_id) else int(i)
            state = self.state_by_stream.get(stream_id, ReliabilityState())
            z_i = z[i]
            if state.belief_ema is None:
                state.belief_ema = z_i.detach()

            belief_i = state.belief_ema.to(z_i.device)
            a_out = self.components.brain_a(belief_i.unsqueeze(0), z_i.unsqueeze(0))
            b_out = self.components.brain_b(z_i.unsqueeze(0))
            r_a = a_out.reliability.squeeze(0)
            r_b = b_out.reliability.squeeze(0)
            md_clean = b_out.md_clean.squeeze(0)
            d_clean = torch.norm(
                z_i.detach() - self.components.brain_b.mu_clean.to(z_i.device),
                p=2,
            )
            recover_rb_ema_alpha = float(getattr(self.config.temporal, "recover_rb_ema_alpha", 0.0))
            r_b_recover = r_b
            if recover_rb_ema_alpha > 0.0 and state.mode == ReliabilityMode.PERSISTENT:
                prev_rb = r_b.detach() if state.recover_rb_ema is None else state.recover_rb_ema.to(z_i.device)
                r_b_recover = recover_rb_ema_alpha * prev_rb + (1.0 - recover_rb_ema_alpha) * r_b.detach()
                state.recover_rb_ema = r_b_recover.detach()
            elif state.mode != ReliabilityMode.PERSISTENT:
                state.recover_rb_ema = None
            d_bad = None
            if state.mu_bad is not None:
                d_bad = torch.norm(z_i.detach() - state.mu_bad.to(z_i.device), p=2)
            # During the known-clean warmup prefix, calibrate belief/state only.
            # This prevents early false alarms from pushing the stream into
            # SUSPECT/PERSISTENT before any corruption can occur.
            warmup_known_clean = False
            if batch.timestep is not None and "corrupt_start" in batch.metadata:
                c_start = int(batch.metadata["corrupt_start"][i].item())
                t_i = int(batch.timestep[i].item())
                warmup_known_clean = t_i < c_start
            if state.mode == ReliabilityMode.PERSISTENT:
                persistent_enter_threshold_b = float(
                    getattr(
                        self.config.temporal,
                        "persistent_enter_threshold_b",
                        self.config.temporal.clean_like_threshold_b,
                    )
                )
                raw_suspicious = bool(r_b.item() < persistent_enter_threshold_b)
            elif state.mode == ReliabilityMode.RECOVERING:
                # Recovery warmup disables Brain A alarms while EMA is rebuilt.
                raw_suspicious = False
            else:
                raw_suspicious = bool(r_a.item() < self.config.temporal.suspicious_threshold_a)
            suspicious = False if warmup_known_clean else raw_suspicious

            step_result = self.components.state_machine.step(
                state=state,
                z_t=z_i.detach(),
                r_a=r_a,
                r_b=r_b,
                suspicious=suspicious,
                d_clean=d_clean,
                d_bad=d_bad,
                r_b_recover=r_b_recover,
            )

            if step_result.update_belief and not suspicious:
                state.belief_ema = ema_alpha * belief_i.detach() + (1.0 - ema_alpha) * z_i.detach()

            if step_result.state.mode in (ReliabilityMode.PERSISTENT, ReliabilityMode.RECOVERING):
                final_rel = r_b
            else:
                final_rel = r_a

            self.state_by_stream[stream_id] = step_result.state
            if i == 0:
                mode_name = step_result.state.mode.value
            ra_list.append(r_a)
            rb_list.append(r_b)
            final_list.append(final_rel)
            suspicious_list.append(torch.tensor(float(suspicious), device=z.device))
            mode_ids.append(torch.tensor(float(list(ReliabilityMode).index(step_result.state.mode)), device=z.device))
            bad_runs.append(torch.tensor(float(step_result.state.bad_run), device=z.device))
            belief_updates.append(torch.tensor(float(step_result.update_belief), device=z.device))
            bad_buffer_sizes.append(torch.tensor(float(len(step_result.state.bad_buffer)), device=z.device))
            d_clean_list.append(d_clean)
            md_clean_list.append(md_clean)
            r_b_recover_list.append(r_b_recover)
            d_bad_list.append(torch.tensor(float("nan"), device=z.device) if d_bad is None else d_bad)

        step_out = ReliabilityStepOutput(
            r_a=torch.stack(ra_list, dim=0),
            r_b=torch.stack(rb_list, dim=0),
            final_reliability=torch.stack(final_list, dim=0),
            mode_name=mode_name,
            suspicious=torch.stack(suspicious_list, dim=0),
            diagnostics={
                "mode_id": torch.stack(mode_ids, dim=0),
                "bad_run": torch.stack(bad_runs, dim=0),
                "belief_update_enabled": torch.stack(belief_updates, dim=0),
                "bad_buffer_size": torch.stack(bad_buffer_sizes, dim=0),
                "d_clean": torch.stack(d_clean_list, dim=0),
                "md_clean": torch.stack(md_clean_list, dim=0),
                "r_b_recover": torch.stack(r_b_recover_list, dim=0),
                "d_bad": torch.stack(d_bad_list, dim=0),
            },
        )
        if return_projected:
            return step_out, z
        return step_out


def apply_frozen_test_params(config: FrameworkConfig, brain_b: BrainBScorer | None = None) -> None:
    """Apply frozen test-time controller defaults from training calibration."""
    config.temporal.clean_like_threshold_b = float(FROZEN_TEST_PARAMS.clean_like_threshold_b)
    config.temporal.recover_required_steps = int(FROZEN_TEST_PARAMS.recover_required_steps)
    config.temporal.recover_rewarm_steps = int(FROZEN_TEST_PARAMS.recover_rewarm_steps)
    config.temporal.persistent_enter_threshold_b = float(FROZEN_TEST_PARAMS.persistent_enter_threshold_b)
    config.temporal.recover_anchor_mode = str(FROZEN_TEST_PARAMS.recover_anchor_mode)
    config.temporal.recover_anchor_margin = float(FROZEN_TEST_PARAMS.recover_anchor_margin)
    config.temporal.recover_clean_threshold = float(FROZEN_TEST_PARAMS.recover_clean_threshold)
    config.temporal.recover_rb_ema_alpha = float(FROZEN_TEST_PARAMS.recover_rb_ema_alpha)
    config.temporal.recovery_start_window = int(FROZEN_TEST_PARAMS.recovery_start_window)
    config.temporal.controller_profile = str(FROZEN_TEST_PARAMS.profile_name)
    if brain_b is not None:
        brain_b.set_calibration(
            temperature=float(FROZEN_TEST_PARAMS.brain_b_temperature),
            bias=float(FROZEN_TEST_PARAMS.brain_b_bias),
        )


def build_runtime(config: FrameworkConfig, adapter: EncoderAdapter, device: torch.device) -> InferenceRuntime:
    adapter = adapter.to(device)
    projector = SharedProjectionHead(
        input_dim=config.model.common_latent_dim,
        output_dim=config.model.common_latent_dim,
        hidden_dim=config.model.projection_hidden_dim,
        dropout=config.model.projection_dropout,
    ).to(device)
    brain_a = BrainAScorer(latent_dim=config.model.common_latent_dim).to(device)
    stats: CleanReferenceStats = load_clean_reference_stats(config.brain_b.stats_artifact_path, device=str(device))
    brain_b = BrainBScorer(
        stats=stats,
        temperature=config.brain_b.md_temperature,
        bias=config.brain_b.md_bias,
    ).to(device)
    apply_frozen_test_params(config, brain_b)
    state_machine = ReliabilityStateMachine(config.temporal)
    return InferenceRuntime(
        config=config,
        components=InferenceComponents(
            adapter=adapter,
            projector=projector,
            brain_a=brain_a,
            brain_b=brain_b,
            state_machine=state_machine,
        ),
        device=device,
    )

"""Inference-only runtime wiring for encoder + brains."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import EncoderAdapter
from .config import FrameworkConfig
from .core.brain_b_stats import CleanReferenceStats, load_clean_reference_stats
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
        d_bad_list = []
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
            d_clean = b_out.md_clean.squeeze(0)
            d_bad = None
            if state.mu_bad is not None:
                d_bad = torch.norm(z_i.detach() - state.mu_bad.to(z_i.device), p=2)
            suspicious = bool(
                (r_a.item() < self.config.temporal.suspicious_threshold_a)
                or (r_b.item() < self.config.temporal.clean_like_threshold_b)
            )

            step_result = self.components.state_machine.step(
                state=state,
                z_t=z_i.detach(),
                r_a=r_a,
                r_b=r_b,
                suspicious=suspicious,
                d_clean=d_clean,
                d_bad=d_bad,
            )

            if step_result.update_belief:
                state.belief_ema = ema_alpha * belief_i.detach() + (1.0 - ema_alpha) * z_i.detach()

            if step_result.state.mode == ReliabilityMode.CLEAN:
                final_rel = r_a
            elif step_result.state.mode == ReliabilityMode.SUSPECT:
                final_rel = torch.minimum(r_a, r_b)
            else:
                final_rel = r_b

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
                "d_bad": torch.stack(d_bad_list, dim=0),
            },
        )
        if return_projected:
            return step_out, z
        return step_out


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

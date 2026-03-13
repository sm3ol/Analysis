"""Unified LiDAR-only trainer for the AV stage-2 framework."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    FRAMEWORK_PARENT = Path(__file__).resolve().parents[1]
    if str(FRAMEWORK_PARENT) not in sys.path:
        sys.path.insert(0, str(FRAMEWORK_PARENT))
    from framework.adapters import (
        CenterPointAdapter,
        CenterPointAdapterConfig,
        LidarPointNetAdapter,
        LidarPointNetAdapterConfig,
        PointPillarsAdapter,
        PointPillarsAdapterConfig,
        PointRCNNAdapter,
        PointRCNNAdapterConfig,
        PVRCNNAdapter,
        PVRCNNAdapterConfig,
    )
    from framework.adapters.base import EncoderAdapter
    from framework.config import FrameworkConfig
    from framework.core.brain_b_stats import (
        CleanReferenceStats,
        fit_clean_reference_stats,
        placeholder_clean_reference_stats,
    )
    from framework.core.pooling import pool_adapter_output
    from framework.core.projection import SharedProjectionHead
    from framework.core.scorer import BrainAScorer, BrainBScorer
    from framework.core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine
    from framework.losses import SupConLoss, build_delta_embeddings
    from framework.types import ReliabilityStepOutput, TrainBatch
    from framework.validation.metrics import EpisodeTrace
else:
    from .adapters import (
        CenterPointAdapter,
        CenterPointAdapterConfig,
        LidarPointNetAdapter,
        LidarPointNetAdapterConfig,
        PointPillarsAdapter,
        PointPillarsAdapterConfig,
        PointRCNNAdapter,
        PointRCNNAdapterConfig,
        PVRCNNAdapter,
        PVRCNNAdapterConfig,
    )
    from .adapters.base import EncoderAdapter
    from .config import FrameworkConfig
    from .core.brain_b_stats import (
        CleanReferenceStats,
        fit_clean_reference_stats,
        placeholder_clean_reference_stats,
    )
    from .core.pooling import pool_adapter_output
    from .core.projection import SharedProjectionHead
    from .core.scorer import BrainAScorer, BrainBScorer
    from .core.temporal import ReliabilityMode, ReliabilityState, ReliabilityStateMachine
    from .losses import SupConLoss, build_delta_embeddings
    from .types import ReliabilityStepOutput, TrainBatch
    from .validation.metrics import EpisodeTrace


@dataclass
class TrainerComponents:
    """Container for runtime modules."""

    adapter: EncoderAdapter
    projector: SharedProjectionHead
    brain_a: BrainAScorer
    brain_b: BrainBScorer
    state_machine: ReliabilityStateMachine
    supcon_loss: Optional[SupConLoss]


class UnifiedBeliefTrainer:
    """Single-process LiDAR trainer with the embodied-style two-brain API."""

    def __init__(self, config: FrameworkConfig, components: TrainerComponents, device: torch.device):
        self.config = config
        self.components = components
        self.device = device
        self.state_by_episode: dict[int, ReliabilityState] = {}

        params = [
            p
            for p in (
                list(self.components.adapter.parameters())
                + list(self.components.projector.parameters())
                + list(self.components.brain_a.parameters())
            )
            if p.requires_grad
        ]
        if not params:
            raise RuntimeError("UnifiedBeliefTrainer: no trainable parameters found.")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.optim.learning_rate),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def reset_stream_state(self) -> None:
        """Reset per-episode temporal state."""
        self.state_by_episode.clear()

    def _move_batch_to_device(self, batch: TrainBatch) -> TrainBatch:
        metadata: dict[str, Any] = {}
        for key, value in batch.metadata.items():
            metadata[key] = value.to(self.device) if torch.is_tensor(value) else value
        return TrainBatch(
            points=batch.points.to(self.device),
            episode_id=batch.episode_id.to(self.device),
            timestep=batch.timestep.to(self.device),
            stream_id=batch.stream_id.to(self.device),
            corruption_family_id=None
            if batch.corruption_family_id is None
            else batch.corruption_family_id.to(self.device),
            is_corrupt=None if batch.is_corrupt is None else batch.is_corrupt.to(self.device),
            metadata=metadata,
        )

    def _range_penalty_loss(
        self,
        scores: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        below = torch.relu(low - scores)
        above = torch.relu(scores - high)
        return (below.square() + above.square()).mean()

    def _corrupt_target_band(self, severity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config.score_targets
        sev = severity.to(torch.float32)
        low = torch.empty_like(sev)
        high = torch.empty_like(sev)

        mask_s1 = sev <= 1.0
        mask_s5 = sev >= 5.0
        mask_mid = ~(mask_s1 | mask_s5)
        mask_s3 = mask_mid & (sev <= 3.0)
        mask_s35 = mask_mid & (sev > 3.0)

        low[mask_s1] = float(cfg.severity1_low)
        high[mask_s1] = float(cfg.severity1_high)
        low[mask_s5] = float(cfg.severity5_low)
        high[mask_s5] = float(cfg.severity5_high)

        if bool(mask_s3.any().item()):
            alpha = ((sev[mask_s3] - 1.0) / 2.0).clamp(0.0, 1.0)
            low[mask_s3] = (1.0 - alpha) * float(cfg.severity1_low) + alpha * float(cfg.severity3_low)
            high[mask_s3] = (1.0 - alpha) * float(cfg.severity1_high) + alpha * float(cfg.severity3_high)
        if bool(mask_s35.any().item()):
            alpha = ((sev[mask_s35] - 3.0) / 2.0).clamp(0.0, 1.0)
            low[mask_s35] = (1.0 - alpha) * float(cfg.severity3_low) + alpha * float(cfg.severity5_low)
            high[mask_s35] = (1.0 - alpha) * float(cfg.severity3_high) + alpha * float(cfg.severity5_high)
        return low, high

    def _target_score_bands(self, batch: TrainBatch) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config.score_targets
        is_corrupt = batch.is_corrupt.to(self.device)
        severity = batch.metadata["severity"].to(self.device)
        timestep = batch.timestep.to(self.device)
        corrupt_start = batch.metadata["corrupt_start"].to(self.device)

        low = torch.full_like(severity, float(cfg.clean_low), dtype=torch.float32)
        high = torch.full_like(severity, float(cfg.clean_high), dtype=torch.float32)

        warmup_mask = timestep < corrupt_start
        if bool(warmup_mask.any().item()):
            low[warmup_mask] = float(cfg.warmup_clean_low)
            high[warmup_mask] = float(cfg.warmup_clean_high)

        corrupt_mask = is_corrupt > 0
        if bool(corrupt_mask.any().item()):
            c_low, c_high = self._corrupt_target_band(severity[corrupt_mask])
            low[corrupt_mask] = c_low
            high[corrupt_mask] = c_high
        return low, high

    def _compute_score_supervision_loss(self, batch: TrainBatch, out: ReliabilityStepOutput) -> torch.Tensor:
        low, high = self._target_score_bands(batch)
        cfg = self.config.score_targets
        final_loss = self._range_penalty_loss(out.final_reliability, low, high)
        a_loss = self._range_penalty_loss(out.r_a, low, high)
        b_loss = self._range_penalty_loss(out.r_b, low, high)
        return (
            float(cfg.final_loss_weight) * final_loss
            + float(cfg.brain_a_loss_weight) * a_loss
            + float(cfg.brain_b_loss_weight) * b_loss
        )

    def train_step(self, batch: TrainBatch) -> dict[str, torch.Tensor]:
        self.components.adapter.train()
        self.components.projector.train()
        self.components.brain_a.train()
        self.optimizer.zero_grad(set_to_none=True)
        out, z = self.score_step(batch, return_projected=True)
        main_loss = self._compute_score_supervision_loss(batch, out)
        supcon_loss = self._compute_supcon_from_batch(z, batch)
        loss = main_loss + float(self.config.supcon.lambda_weight) * supcon_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.components.adapter.parameters())
            + list(self.components.projector.parameters())
            + list(self.components.brain_a.parameters()),
            float(self.config.optim.grad_clip_norm),
        )
        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "main_loss": main_loss.detach(),
            "supcon_loss": supcon_loss.detach(),
            "mean_reliability": out.final_reliability.mean(),
            "mean_r_a": out.r_a.mean(),
            "mean_r_b": out.r_b.mean(),
            "alarm_rate": out.alarm.mean(),
        }

    @torch.no_grad()
    def eval_step(self, batch: TrainBatch) -> dict[str, torch.Tensor]:
        self.components.adapter.eval()
        self.components.projector.eval()
        self.components.brain_a.eval()
        out = self.score_step(batch)
        return {
            "mean_reliability": out.final_reliability.mean(),
            "mean_r_a": out.r_a.mean(),
            "mean_r_b": out.r_b.mean(),
            "alarm_rate": out.alarm.mean(),
        }

    def score_step(
        self,
        batch: TrainBatch,
        return_projected: bool = False,
    ) -> ReliabilityStepOutput | tuple[ReliabilityStepOutput, torch.Tensor]:
        output = self.components.adapter(batch)
        pooled = pool_adapter_output(output)
        z = self.components.projector(pooled)

        bsz = z.shape[0]
        ra_list = []
        rb_list = []
        final_list = []
        suspicious_list = []
        alarm_list = []
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
            episode_key = int(batch.episode_id[i].item())
            state = self.state_by_episode.get(episode_key, ReliabilityState())
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

            corrupt_start = int(batch.metadata["corrupt_start"][i].item())
            timestep = int(batch.timestep[i].item())
            warmup_known_clean = timestep < corrupt_start

            if state.mode == ReliabilityMode.PERSISTENT:
                persistent_enter_threshold_b = getattr(
                    self.config.temporal,
                    "persistent_enter_threshold_b",
                    None,
                )
                if persistent_enter_threshold_b is None:
                    persistent_enter_threshold_b = self.config.temporal.clean_like_threshold_b
                raw_suspicious = bool(r_b.item() < persistent_enter_threshold_b)
            elif state.mode == ReliabilityMode.RECOVERING:
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
            alarm = float(suspicious or step_result.state.mode != ReliabilityMode.CLEAN)

            self.state_by_episode[episode_key] = step_result.state
            if i == 0:
                mode_name = step_result.state.mode.value

            ra_list.append(r_a)
            rb_list.append(r_b)
            final_list.append(final_rel)
            suspicious_list.append(torch.tensor(float(suspicious), device=z.device))
            alarm_list.append(torch.tensor(alarm, device=z.device))
            mode_ids.append(torch.tensor(float(list(ReliabilityMode).index(step_result.state.mode)), device=z.device))
            bad_runs.append(torch.tensor(float(step_result.state.bad_run), device=z.device))
            belief_updates.append(torch.tensor(float(step_result.update_belief), device=z.device))
            bad_buffer_sizes.append(torch.tensor(float(len(step_result.state.bad_buffer)), device=z.device))
            md_clean_list.append(md_clean)
            d_clean_list.append(d_clean)
            r_b_recover_list.append(r_b_recover)
            if d_bad is None:
                d_bad_list.append(torch.tensor(float("nan"), device=z.device))
            else:
                d_bad_list.append(d_bad)

        step_out = ReliabilityStepOutput(
            r_a=torch.stack(ra_list, dim=0),
            r_b=torch.stack(rb_list, dim=0),
            final_reliability=torch.stack(final_list, dim=0),
            alarm=torch.stack(alarm_list, dim=0),
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

    def _aggregate_tensor_metrics(self, values: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if not values:
            return {}
        keys = values[0].keys()
        out: dict[str, float] = {}
        for key in keys:
            stacked = torch.stack([row[key].detach().float().cpu() for row in values], dim=0)
            out[key] = float(stacked.mean().item())
        return out

    def _compute_supcon_from_batch(self, z: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        """Optional supervised contrastive loss on shared latent embeddings."""
        if self.components.supcon_loss is None:
            return torch.zeros((), device=self.device)
        if batch.corruption_family_id is None or batch.is_corrupt is None:
            raise RuntimeError("SupCon enabled but batch missing corruption labels.")

        family = batch.corruption_family_id.to(self.device)
        is_corrupt = batch.is_corrupt.to(self.device)
        corr_mask = (is_corrupt > 0) & (family >= 0)
        clean_mask = is_corrupt == 0

        if int(corr_mask.sum().item()) == 0:
            raise RuntimeError("SupCon enabled but batch has zero corrupt samples.")
        corr_family = family[corr_mask]
        uniq, counts = torch.unique(corr_family, return_counts=True)
        min_pos = int(self.config.supcon.min_positives_per_family)
        valid_families = uniq[counts >= min_pos]
        if valid_families.numel() == 0:
            comp = {int(k.item()): int(v.item()) for k, v in zip(uniq, counts)}
            raise RuntimeError(
                f"SupCon batch composition insufficient: need >= {min_pos} positives/family, got {comp}"
            )

        valid_corr = corr_mask & torch.isin(family, valid_families)
        corr_idx = torch.where(valid_corr)[0]
        corr_labels = family[valid_corr]
        if corr_idx.numel() == 0:
            raise RuntimeError("SupCon: no valid corrupt samples after family filtering.")

        if self.config.supcon.use_delta_embeddings:
            clean_idx = torch.where(clean_mask)[0]
            if clean_idx.numel() == 0:
                raise RuntimeError("SupCon(delta) requires at least one clean sample in batch.")
            clean_pick = clean_idx[torch.arange(corr_idx.numel(), device=z.device) % clean_idx.numel()]
            emb = build_delta_embeddings(z[clean_pick], z[corr_idx])
        else:
            emb = z[corr_idx]

        return self.components.supcon_loss(emb, corr_labels)

    def build_supcon_loss(
        self,
        z_clean: torch.Tensor,
        z_corrupt: torch.Tensor,
        family_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optional SupCon term in common latent space."""
        if self.components.supcon_loss is None:
            return torch.zeros((), device=self.device)
        if self.config.supcon.use_delta_embeddings:
            emb = build_delta_embeddings(z_clean, z_corrupt)
        else:
            emb = z_corrupt
        return self.components.supcon_loss(emb, family_labels)

    def train_epoch(self, train_loader: DataLoader[TrainBatch]) -> dict[str, float]:
        """Run one full training epoch."""
        self.reset_stream_state()
        step_metrics: list[dict[str, torch.Tensor]] = []
        max_steps = int(self.config.optim.max_steps_per_epoch)
        for step_idx, batch in enumerate(train_loader):
            if max_steps > 0 and step_idx >= max_steps:
                break
            batch = self._move_batch_to_device(batch)
            step_metrics.append(self.train_step(batch))
        return self._aggregate_tensor_metrics(step_metrics)

    @torch.no_grad()
    def refit_brain_b_from_loader(self, loader: DataLoader[TrainBatch]) -> CleanReferenceStats:
        """Refit Brain-B clean-reference stats from clean training frames."""
        self.components.adapter.eval()
        self.components.projector.eval()
        self.reset_stream_state()
        zs: list[torch.Tensor] = []
        for batch in loader:
            batch = self._move_batch_to_device(batch)
            clean_mask = batch.is_corrupt == 0
            if int(clean_mask.sum().item()) == 0:
                continue
            sub = TrainBatch(
                points=batch.points[clean_mask],
                episode_id=batch.episode_id[clean_mask],
                timestep=batch.timestep[clean_mask],
                stream_id=batch.stream_id[clean_mask],
                corruption_family_id=None
                if batch.corruption_family_id is None
                else batch.corruption_family_id[clean_mask],
                is_corrupt=batch.is_corrupt[clean_mask],
                metadata={
                    key: value[clean_mask] if torch.is_tensor(value) else value
                    for key, value in batch.metadata.items()
                },
            )
            output = self.components.adapter(sub)
            pooled = pool_adapter_output(output)
            z = self.components.projector(pooled)
            zs.append(z.detach())
        if not zs:
            raise RuntimeError("no clean samples available for Brain-B fit")
        embeddings = torch.cat(zs, dim=0)
        stats = fit_clean_reference_stats(
            embeddings,
            shrinkage=float(self.config.brain_b.covariance_shrinkage),
            eps=float(self.config.brain_b.covariance_eps),
        )
        self.components.brain_b.update_stats(stats)
        return stats

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader[TrainBatch]) -> tuple[dict[str, float], list[EpisodeTrace]]:
        """Run evaluation and collect episode traces."""
        self.components.adapter.eval()
        self.components.projector.eval()
        self.components.brain_a.eval()
        self.reset_stream_state()
        step_metrics: list[dict[str, torch.Tensor]] = []
        traces: dict[int, EpisodeTrace] = {}

        for batch in loader:
            batch = self._move_batch_to_device(batch)
            out = self.score_step(batch)
            step_metrics.append(
                {
                    "mean_reliability": out.final_reliability.mean(),
                    "mean_r_a": out.r_a.mean(),
                    "mean_r_b": out.r_b.mean(),
                    "alarm_rate": out.alarm.mean(),
                }
            )
            for i in range(batch.points.shape[0]):
                episode_id = int(batch.episode_id[i].item())
                trace = traces.get(episode_id)
                if trace is None:
                    trace = EpisodeTrace(
                        episode_id=episode_id,
                        split=str(batch.metadata.get("split", [""])[i]) if "split" in batch.metadata else "",
                        corruption_length=int(batch.metadata["corruption_length"][i].item()),
                        corrupt_start=int(batch.metadata["corrupt_start"][i].item()),
                        corrupt_end=int(batch.metadata["corrupt_end"][i].item()),
                        alarm=[],
                        mode_name=[],
                        final_reliability=[],
                        is_corrupt=[],
                    )
                    traces[episode_id] = trace
                mode_index = int(out.diagnostics["mode_id"][i].item())
                trace.alarm.append(int(out.alarm[i].item() > 0.5))
                trace.mode_name.append(list(ReliabilityMode)[mode_index].value)
                trace.final_reliability.append(float(out.final_reliability[i].item()))
                trace.is_corrupt.append(int(batch.is_corrupt[i].item()))
        return self._aggregate_tensor_metrics(step_metrics), [traces[key] for key in sorted(traces)]


def build_adapter(config: FrameworkConfig) -> EncoderAdapter:
    """Factory for LiDAR adapters."""
    name = config.encoder_name.lower()
    hidden_dim = max(int(config.model.token_hidden_dim), int(config.model.embedding_dim))
    common = {
        "input_dim": int(config.data.point_feature_dim),
        "hidden_dim": hidden_dim,
        "output_dim": int(config.model.common_latent_dim),
        "freeze_backbone": bool(config.model.freeze_backbone),
        "use_pretrained": bool(config.model.use_pretrained),
        "checkpoint_path": config.model.checkpoint_path,
        "strict_checkpoint_load": bool(config.model.strict_checkpoint_load),
        "return_intermediate_shapes": bool(config.model.return_intermediate_shapes),
    }
    if name in ("pointpillars", "pillars"):
        return PointPillarsAdapter(PointPillarsAdapterConfig(**common))
    if name in ("pointrcnn", "point_rcnn"):
        return PointRCNNAdapter(PointRCNNAdapterConfig(**common))
    if name in ("pv_rcnn", "pvrcnn"):
        return PVRCNNAdapter(PVRCNNAdapterConfig(**common))
    if name in ("centerpoint", "center_point"):
        return CenterPointAdapter(CenterPointAdapterConfig(**common))
    if name in ("pointnet_lidar", "pointnet", "lidar"):
        return LidarPointNetAdapter(LidarPointNetAdapterConfig(**common))
    raise ValueError(f"unsupported AV LiDAR encoder: {config.encoder_name}")


def build_components(
    config: FrameworkConfig,
    device: torch.device,
    stats: CleanReferenceStats | None = None,
) -> TrainerComponents:
    """Create the adapter, shared modules, and Brain-B scorer."""
    adapter = build_adapter(config).to(device)
    projector = SharedProjectionHead(
        input_dim=int(config.model.common_latent_dim),
        output_dim=int(config.model.common_latent_dim),
        hidden_dim=int(config.model.projection_hidden_dim),
        dropout=float(config.model.projection_dropout),
    ).to(device)
    brain_a = BrainAScorer(latent_dim=int(config.model.common_latent_dim)).to(device)
    stats = stats or placeholder_clean_reference_stats(int(config.model.common_latent_dim), device)
    brain_b = BrainBScorer(
        stats=stats,
        temperature=float(config.brain_b.md_temperature),
        bias=float(config.brain_b.md_bias),
    ).to(device)
    state_machine = ReliabilityStateMachine(config.temporal)
    supcon = SupConLoss(temperature=float(config.supcon.temperature)) if config.supcon.enabled else None
    return TrainerComponents(
        adapter=adapter,
        projector=projector,
        brain_a=brain_a,
        brain_b=brain_b,
        state_machine=state_machine,
        supcon_loss=supcon,
    )


def parse_args() -> argparse.Namespace:
    """CLI for the trainer module."""
    parser = argparse.ArgumentParser(description="AV unified belief trainer module.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """Dry-run trainer initialization."""
    args = parse_args()
    cfg = FrameworkConfig()
    cfg.model.common_latent_dim = int(args.latent_dim)
    cfg.model.embedding_dim = int(args.embedding_dim)
    device = torch.device(args.device)
    _ = build_components(cfg, device=device)
    print("AV train_belief initialized.")


if __name__ == "__main__":
    main()

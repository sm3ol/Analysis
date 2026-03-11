"""Framework configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemporalConfig:
    """Runtime temporal behavior for two-brain reliability."""

    belief_ema_window: int = 50
    bad_run_window: int = 50
    start_bad_buffer_after: int = 30
    switch_to_persistent_after: int = 30
    recover_required_steps: int = 10
    recover_rewarm_steps: int = 40
    recover_rewarm_bad_allowance: int = 10
    suspicious_threshold_a: float = 0.80
    clean_like_threshold_b: float = 0.95


@dataclass
class BrainBConfig:
    """Offline clean-reference and MD scoring configuration."""

    covariance_shrinkage: float = 0.01
    covariance_eps: float = 1e-5
    md_temperature: float = 1.0
    md_bias: float = 0.0
    stats_artifact_path: str = "artifacts/brain_b_clean_stats.npz"


@dataclass
class SupConConfig:
    """Supervised contrastive training settings."""

    enabled: bool = False
    temperature: float = 0.07
    lambda_weight: float = 0.1
    use_delta_embeddings: bool = True
    min_positives_per_family: int = 2


@dataclass
class ScoreTargetConfig:
    """Target score bands for clean and corrupted frames."""

    clean_low: float = 0.95
    clean_high: float = 1.0
    warmup_clean_low: float = 0.99
    warmup_clean_high: float = 1.0
    severity1_low: float = 0.6
    severity1_high: float = 0.7
    severity3_low: float = 0.3
    severity3_high: float = 0.5
    severity5_low: float = 0.05
    severity5_high: float = 0.2
    final_loss_weight: float = 1.0
    brain_a_loss_weight: float = 0.5
    brain_b_loss_weight: float = 0.25


@dataclass
class DataSplitConfig:
    """Episode-level split counts."""

    train_episodes: int = 800
    val_episodes: int = 100
    test_seen_episodes: int = 100
    test_unseen_episodes: int = 100
    unseen_holdout_families: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Shared model dimensions."""

    common_latent_dim: int = 256
    projection_hidden_dim: int = 512
    projection_dropout: float = 0.0


@dataclass
class OptimConfig:
    """Optimizer and loop settings."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_steps: int = 10000
    grad_clip_norm: float = 1.0


@dataclass
class FrameworkConfig:
    """Top-level training and evaluation configuration."""

    encoder_name: str = "dinov2"
    seed: int = 0
    output_dir: str = "outputs"
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    brain_b: BrainBConfig = field(default_factory=BrainBConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)
    score_targets: ScoreTargetConfig = field(default_factory=ScoreTargetConfig)
    splits: DataSplitConfig = field(default_factory=DataSplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

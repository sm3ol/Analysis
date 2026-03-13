"""Framework configuration dataclasses for standalone AV inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from .paths import ARTIFACTS_ROOT, AV_ROOT, OUTPUTS_ROOT


@dataclass
class TemporalConfig:
    """Runtime temporal behavior for the two-brain controller."""

    clean_prefix: int = 50
    belief_ema_window: int = 50
    start_bad_buffer_after: int = 15
    switch_to_persistent_after: int = 20
    persistent_switch_length: int = 20
    recover_required_steps: int = 25
    recover_rewarm_steps: int = 25
    suspicious_threshold_a: float = 0.5
    clean_like_threshold_b: float = 0.7
    recover_anchor_mode: str = "strict"
    recover_anchor_margin: float = 0.0
    recover_clean_threshold: float = float("inf")
    recover_rb_ema_alpha: float = 0.0
    persistent_enter_threshold_b: float | None = None


@dataclass
class BrainBConfig:
    """Brain-B clean-reference fitting and scoring configuration."""

    covariance_shrinkage: float = 0.01
    covariance_eps: float = 1e-5
    md_temperature: float = 1.0
    md_bias: float = 0.0
    stats_artifact_path: str = str(ARTIFACTS_ROOT / "brain_b_clean_stats.npz")


@dataclass
class ScoreTargetConfig:
    """Target score bands for clean and corrupted timesteps."""

    clean_low: float = 0.9
    clean_high: float = 1.0
    warmup_clean_low: float = 0.99
    warmup_clean_high: float = 1.0
    severity1_low: float = 0.6
    severity1_high: float = 0.75
    severity3_low: float = 0.3
    severity3_high: float = 0.5
    severity5_low: float = 0.05
    severity5_high: float = 0.2
    final_loss_weight: float = 1.0
    brain_a_loss_weight: float = 0.5
    brain_b_loss_weight: float = 0.25


@dataclass
class DataSplitConfig:
    """Leak-free stream-level split configuration."""

    train_ratio: float = 0.6
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    test_unseen_ratio: float = 0.1
    unseen_holdout_families: list[str] = field(default_factory=list)
    test_unseen_use_all_members: bool = True


@dataclass
class DataConfig:
    """AV LiDAR dataset and episode construction settings."""

    data_root: str = str(AV_ROOT / 'dataset' / 'raw' / 'LIDAR_TOP')
    episode_len: int = 120
    stride: int = 120
    clean_prefix: int = 50
    min_corruption_len: int = 10
    max_corruption_len: int = 30
    severity_set: list[int] = field(default_factory=lambda: [1, 3, 5])
    point_feature_dim: int = 5
    max_points: int = 4096
    min_points: int = 32
    frame_cache_size: int = 512
    use_family_balanced_sampler: bool = True
    representative_overrides: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model dimensions for the LiDAR adapter and shared latent space."""

    token_hidden_dim: int = 128
    embedding_dim: int = 512
    common_latent_dim: int = 256
    projection_hidden_dim: int = 512
    projection_dropout: float = 0.0
    freeze_backbone: bool = False
    use_pretrained: bool = True
    checkpoint_path: str | None = None
    strict_checkpoint_load: bool = True
    return_intermediate_shapes: bool = False


@dataclass
class SupConConfig:
    """Supervised contrastive auxiliary loss settings."""

    enabled: bool = False
    temperature: float = 0.07
    lambda_weight: float = 0.1
    use_delta_embeddings: bool = True
    min_positives_per_family: int = 2
    use_supcon: bool = False
    supcon_temperature: float = 0.07
    lambda_supcon: float = 0.1
    min_positives_per_label: int = 2


@dataclass
class OptimConfig:
    """Optimizer and loop settings."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 3
    max_steps_per_epoch: int = 0
    grad_clip_norm: float = 1.0


@dataclass
class ValidationConfig:
    """Validation reporting and temporal stratification settings."""

    include_auroc: bool = True
    short_max_len: int = 16
    medium_max_len: int = 23


@dataclass
class RuntimeConfig:
    """Single-process runtime configuration."""

    device: str = "auto"
    preferred_gpu_index: int = 1
    num_workers: int = 0


@dataclass
class FrameworkConfig:
    """Top-level AV runtime configuration."""

    encoder_name: str = "pointpillars"
    seed: int = 0
    output_dir: str = str(OUTPUTS_ROOT)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    brain_b: BrainBConfig = field(default_factory=BrainBConfig)
    score_targets: ScoreTargetConfig = field(default_factory=ScoreTargetConfig)
    splits: DataSplitConfig = field(default_factory=DataSplitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

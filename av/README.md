# AV Inference Pack (Standalone LiDAR)

Standalone inference pack for the AV two-brain reliability scorer.
Supports four LiDAR encoders — **PointPillars**, **PointRCNN**, **PV-RCNN**, **CenterPoint** —
with a learned Brain A (temporal anomaly) + frozen Brain B (Mahalanobis distance) scoring pipeline
and streaming online calibration.

## Folder Structure

```
Analysis/av/
├── framework/                  # Core AV framework
│   ├── adapters/               # Per-encoder LiDAR adapters
│   ├── core/                   # Scorers, temporal state, online calibration
│   ├── losses/                 # Supervised contrastive loss
│   ├── validation/             # Metrics (AUROC, episode traces)
│   ├── config.py               # All configuration dataclasses
│   ├── paths.py                # Path constants (all paths resolve here)
│   ├── train_belief.py         # Trainer / inference entry point
│   └── types.py                # Data types (TrainBatch, StepOutput)
├── tools/                      # Inference & utility scripts
│   ├── preflight_inference.py  # Strict readiness check
│   ├── run_encoder_only_episode.py
│   ├── run_inference_episode.py
│   ├── build_sample_episode.py
│   └── download_official_checkpoints.py
├── vendor/openpcdet/           # Vendored OpenPCDet backbone
├── checkpoints/                # Pretrained encoder weights (.pth)
├── artifacts/                  # Stage-2 checkpoints, Brain-B stats, family MD
├── dataset/                    # Sample episodes
├── configs/                    # Taxonomy manifest
├── outputs/                    # Inference results
├── requirements.txt
├── setup_env.sh
├── preflight_inference.sh
├── run_encoder_only.sh         # Per-encoder: encoder-only inference
├── run_*_encoder_only.sh       # Encoder-specific wrappers
├── run_*_with_scorer.sh        # Encoder-specific full pipeline
├── run_real_with_scorer.sh     # Full pipeline with scorer
├── build_sample_episode.sh
├── verify_all_encoders.sh
└── selected_encoders.txt
```

## Quick Start

```bash
cd Analysis/av

# 1. Set up Python environment
bash setup_env.sh

# 2. Preflight check (validates all encoders)
DEVICE=cpu bash preflight_inference.sh

# 3. Run encoder-only inference
DEVICE=cpu bash run_encoder_only.sh pointpillars

# 4. Run full pipeline with scorer
DEVICE=cpu bash run_real_with_scorer.sh pointpillars
```

## Environment Setup

```bash
cd Analysis/av
bash setup_env.sh
```

This installs Python dependencies from `requirements.txt`. For PV-RCNN, you also need
CUDA-compatible `spconv` and `pointnet2` builds on the machine — these are not pip-installable
and must match your CUDA version.

## Running Encoder-Only Inference

Per-encoder scripts run just the LiDAR encoder (no scorer), useful for hardware profiling:

```bash
DEVICE=cuda:0 bash run_encoder_only.sh <encoder>
```

where `<encoder>` is one of: `pointpillars`, `pointrcnn`, `pvrcnn`, `centerpoint`.

Encoder-specific wrappers also exist:

```bash
DEVICE=cuda:0 bash run_pointpillars_encoder_only.sh
DEVICE=cuda:0 bash run_pvrcnn_encoder_only.sh
```

Outputs are saved to `outputs/` as JSON files with per-frame latency and embedding statistics.

## Running with Scorer

Full pipeline: encoder + projection + Brain A + Brain B + temporal state machine:

```bash
DEVICE=cuda:0 bash run_real_with_scorer.sh <encoder>
```

With trained stage-2 weights:

```bash
CHECKPOINT_PATH="artifacts/pointpillars_stage2_checkpoint.pt" \
BRAIN_B_STATS="artifacts/brain_b_clean_stats.npz" \
DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars
```

## Online Calibration

Online calibration normalizes raw reliability scores using a streaming baseline computed from
the first N frames (prefix). It maps score drops to calibrated values via an exponential function.

Key config knobs (in `RuntimeConfig`):

| Field | Default | Description |
|-------|---------|-------------|
| `enable_online_calibration` | `True` | Enable/disable calibration |
| `online_calibration_apply_in_train` | `False` | Apply during training (usually off) |
| `online_calibration_prefix_len` | `50` | Frames to compute baseline |
| `online_calibration_mode` | `"simple"` | `"simple"` or `"normalized"` |
| `online_calibration_alpha` | `10.0` | Exponential mapping steepness |
| `online_calibration_gamma` | `1.0` | Drop shaping exponent |
| `online_calibration_min_score` | `0.05` | Floor for calibrated scores |
| `online_calibration_max_score` | `1.0` | Ceiling for calibrated scores |
| `online_calibration_emit_before_ready` | `True` | Emit values before baseline frozen |
| `online_calibration_use_piecewise_drop` | `False` | Two-slope drop shaping |

## Using Trained Stage-2 Weights

Per-encoder trained artifacts are in `artifacts/`:

```
artifacts/
├── brain_b_clean_stats.npz                    # Shared Brain-B reference stats
├── {encoder}_stage2_checkpoint.pt             # Trained stage-2 weights
├── {encoder}_family_md_enrolled.pt            # Family Mahalanobis distance stats
└── {encoder}_family_md_calibration.json       # Family MD calibration thresholds
```

Encoders: `pointpillars`, `pointrcnn`, `pvrcnn`, `centerpoint`.

To use a specific encoder's trained weights, set `CHECKPOINT_PATH` to the corresponding
`{encoder}_stage2_checkpoint.pt` file.

## Building Sample Episodes

A bundled sample episode is included at `dataset/episode_000001.npz`.
To rebuild from raw LiDAR frames:

```bash
DATA_ROOT=/path/to/LIDAR_TOP bash build_sample_episode.sh
```

## Hardware Profiling

- **Encoder-only** (`run_encoder_only.sh`): Profiles pure encoder forward pass — use this
  for per-encoder latency/throughput measurements.
- **With scorer** (`run_real_with_scorer.sh`): Full pipeline including projection, scoring,
  and temporal state — use this for end-to-end latency.

Both modes save per-frame timing to `outputs/`.

## Troubleshooting

**PV-RCNN CUDA errors**: PV-RCNN requires `spconv` and CUDA pointnet2 ops compiled for your
GPU architecture. Ensure `CUDA_HOME` is set and matches your PyTorch CUDA version.

**Checkpoint not found**: Run `python3 tools/download_official_checkpoints.py --encoders all`
to download pretrained encoder weights.

**Device selection**: Set `DEVICE=cpu` to run without GPU. For multi-GPU machines, use
`DEVICE=cuda:N` to select a specific GPU.

**Import errors**: All imports must resolve within `Analysis/av/`. If you see import errors
referencing `training/`, something is wrong — this pack should be fully self-contained.

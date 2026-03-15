# AV Inference Pack (Standalone LiDAR)

Standalone inference pack for the AV two-brain reliability scorer.
Supports four LiDAR encoders — **PointPillars**, **PointRCNN**, **PV-RCNN**, **CenterPoint** —
with a learned Brain A (temporal anomaly) + frozen Brain B (Mahalanobis distance) scoring pipeline
and streaming online calibration.

This pack is fully self-contained — all imports resolve within `Analysis/av/`, with no
dependencies on the sibling training repo.

## Folder Structure

```
Analysis/av/
├── framework/                         # Core AV framework
│   ├── adapters/                      # Per-encoder LiDAR adapters
│   ├── core/                          # Scorers, temporal state, online calibration
│   │   ├── online_calibration.py      # Streaming score normalization
│   │   ├── scorer.py                  # Brain A + Brain B scorers
│   │   ├── temporal.py                # State machine (CLEAN→SUSPICIOUS→BAD→...)
│   │   ├── brain_b_stats.py           # Clean reference statistics
│   │   ├── family_md.py               # Family Mahalanobis distance
│   │   └── ...
│   ├── losses/                        # Supervised contrastive loss
│   ├── validation/                    # Metrics (AUROC, episode traces)
│   ├── config.py                      # All configuration dataclasses
│   ├── paths.py                       # Path constants (all paths resolve here)
│   ├── train_belief.py                # Trainer / inference entry point
│   └── types.py                       # Data types (TrainBatch, StepOutput)
├── tools/                             # Inference, testing & utility scripts
│   ├── preflight_inference.py         # Strict readiness check
│   ├── run_encoder_only_episode.py    # Encoder-only inference
│   ├── run_inference_episode.py       # Full pipeline inference
│   ├── test_score_thresholds.py       # Validates clean>=0.9, corrupt<0.8
│   ├── test_severity_levels.py        # Validates low/mid/high score separation
│   ├── build_sample_episode.py        # Build episode from raw LiDAR frames
│   └── download_official_checkpoints.py
├── vendor/openpcdet/                  # Vendored OpenPCDet backbone
├── checkpoints/                       # Pretrained encoder weights (.pth)
│   ├── pointpillars_kitti_openpcdet.pth
│   ├── pointrcnn_kitti_openpcdet.pth
│   ├── pvrcnn_kitti_openpcdet.pth
│   └── centerpoint_pp_nuscenes_openpcdet.pth
├── artifacts/                         # Trained stage-2 weights + Brain-B stats
│   ├── brain_b_clean_stats.npz
│   ├── {encoder}_stage2_checkpoint.pt
│   ├── {encoder}_family_md_enrolled.pt
│   └── {encoder}_family_md_calibration.json
├── dataset/                           # Sample episodes
│   ├── episode_000001.npz
│   └── sample_manifest.json
├── configs/                           # Taxonomy manifest
├── outputs/                           # Inference results and logs
├── requirements.txt
├── setup_env.sh                       # Create venv and install deps
├── preflight_inference.sh             # Validate all encoders load + run
├── run_encoder_only.sh                # Generic encoder-only runner
├── run_real_with_scorer.sh            # Generic full-pipeline runner
├── run_{encoder}_encoder_only.sh      # Per-encoder shortcuts (x4)
├── run_{encoder}_with_scorer.sh       # Per-encoder shortcuts (x4)
├── smoke_inference.sh                 # Preflight + full inference all encoders
├── verify_all_encoders.sh             # Run encoder-only + with-scorer for all 4
├── build_sample_episode.sh
└── selected_encoders.txt
```

## Quick Start

```bash
cd "Analysis/av"

# 1. Activate the AV virtual environment
source /path/to/.venv_av/bin/activate

# 2. Preflight — validates all 4 encoders load and forward-pass
DEVICE=cuda:0 bash preflight_inference.sh

# 3. Encoder-only inference (one encoder)
DEVICE=cuda:0 bash run_encoder_only.sh pointpillars

# 4. Full pipeline with scorer (one encoder)
DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars

# 5. Verify all encoders (encoder-only + with-scorer for all 4)
DEVICE=cuda:0 bash verify_all_encoders.sh

# 6. Run score validation tests
python3 tools/test_score_thresholds.py --encoder all --device cuda:0
python3 tools/test_severity_levels.py --encoder all --device cuda:0
```

## Environment Setup

The AV pack requires Python 3.10+ with PyTorch, NumPy, and scikit-learn. Install
dependencies into a virtual environment:

```bash
bash setup_env.sh
```

Or use an existing environment that has the packages listed in `requirements.txt`:

```
numpy>=1.24
scikit-learn>=1.3
torch==2.5.1
PyYAML>=6.0
easydict>=1.13
gdown>=5.2
```

**PV-RCNN additional requirements**: `spconv` and CUDA-compiled `pointnet2` ops must be
installed separately and must match your CUDA version. These are not pip-installable.

## Preflight Check

Validates that every encoder can load its pretrained checkpoint and run a one-step forward
pass through adapter + projector + scorers:

```bash
# All encoders
DEVICE=cuda:0 bash preflight_inference.sh

# Specific encoders
ENCODERS=pointpillars,centerpoint DEVICE=cuda:0 bash preflight_inference.sh
```

Output is written to `outputs/preflight_inference_*.json`.

## Running Encoder-Only Inference

Runs just the LiDAR encoder and projection head — no scorer, no temporal state. Useful for
hardware profiling of pure encoder latency/throughput.

**Generic script** (pass encoder name as argument):

```bash
DEVICE=cuda:0 bash run_encoder_only.sh pointpillars
DEVICE=cuda:0 bash run_encoder_only.sh pointrcnn
DEVICE=cuda:0 bash run_encoder_only.sh pvrcnn
DEVICE=cuda:0 bash run_encoder_only.sh centerpoint
```

**Per-encoder shortcuts**:

```bash
DEVICE=cuda:0 bash run_pointpillars_encoder_only.sh
DEVICE=cuda:0 bash run_pointrcnn_encoder_only.sh
DEVICE=cuda:0 bash run_pvrcnn_encoder_only.sh
DEVICE=cuda:0 bash run_centerpoint_encoder_only.sh
```

**Custom episode or checkpoint**:

```bash
EPISODE_PATH=/path/to/episode.npz \
CHECKPOINT_PATH=/path/to/custom_checkpoint.pt \
DEVICE=cuda:0 bash run_encoder_only.sh pointpillars
```

**Output**: JSON file in `outputs/encoder_only_{encoder}.json` with per-frame embedding
statistics (mean, std, norm) and an NPZ file with the raw embeddings.

## Running with Scorer (Full Pipeline)

Full inference pipeline: encoder + projection + Brain A + Brain B + temporal state machine +
online calibration. Produces per-frame reliability scores, alarm flags, and mode transitions.

**Generic script**:

```bash
DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars
DEVICE=cuda:0 bash run_real_with_scorer.sh centerpoint
```

**Per-encoder shortcuts**:

```bash
DEVICE=cuda:0 bash run_pointpillars_with_scorer.sh
DEVICE=cuda:0 bash run_pointrcnn_with_scorer.sh
DEVICE=cuda:0 bash run_pvrcnn_with_scorer.sh
DEVICE=cuda:0 bash run_centerpoint_with_scorer.sh
```

**With trained stage-2 weights** (loads adapter, projector, Brain A, Brain B from checkpoint):

```bash
CHECKPOINT_PATH="artifacts/pointpillars_stage2_checkpoint.pt" \
BRAIN_B_STATS="artifacts/brain_b_clean_stats.npz" \
DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars
```

**With dummy weights** (default — uses pretrained backbone, random scorer weights):

```bash
ALLOW_DUMMY_WEIGHTS=1 DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars
```

**Output**: JSON file in `outputs/inference_{encoder}.json` with per-frame `r_a`, `r_b`,
`final_reliability`, `alarm`, `mode_name`, and summary statistics.

## Verify All Encoders

Runs both encoder-only and with-scorer inference for all 4 encoders and prints a pass/fail
summary table:

```bash
DEVICE=cuda:0 bash verify_all_encoders.sh
```

Output:

```
encoder        | encoder_only | with_scorer
---------------+--------------+-------------
pointpillars   | PASS         | PASS
pointrcnn      | PASS         | PASS
pvrcnn         | PASS         | PASS
centerpoint    | PASS         | PASS
```

Detailed logs are saved to `outputs/verify_all/*.log`.

## Smoke Inference

Runs preflight followed by full inference on all encoders:

```bash
DEVICE=cuda:0 bash smoke_inference.sh
```

To run specific encoders only:

```bash
ENCODERS=pointpillars,centerpoint DEVICE=cuda:0 bash smoke_inference.sh
```

## Score Validation Tests

### Threshold Test (`test_score_thresholds.py`)

Validates that clean frames score >= 0.9 and heavily corrupted frames score < 0.8:

```bash
# All encoders
python3 tools/test_score_thresholds.py --encoder all --device cuda:0

# Single encoder
python3 tools/test_score_thresholds.py --encoder pointpillars --device cuda:0
```

Expected output:

```
Encoder        |    Clean |  Corrupt | Status
---------------+----------+----------+-------
pointpillars   |   0.9967 |   0.0782 |   PASS
pointrcnn      |   0.9995 |   0.4333 |   PASS
pvrcnn         |   0.9822 |   0.1140 |   PASS
centerpoint    |   0.9987 |   0.0873 |   PASS
```

### Severity Level Test (`test_severity_levels.py`)

Validates that the scorer differentiates between low, mid, and high corruption intensities:

```bash
# All encoders
for enc in pointpillars pointrcnn pvrcnn centerpoint; do
  python3 tools/test_severity_levels.py --encoder "$enc" --device cuda:0
done

# Single encoder
python3 tools/test_severity_levels.py --encoder pointpillars --device cuda:0
```

Expected output (scores decrease monotonically with corruption severity):

```
Level                               |     Mean |      Min |      Max
------------------------------------+----------+----------+---------
clean (no corruption)               |   0.988  |   0.944  |   1.000
low  (sigma=2,  replace=10%)        |   0.550  |   0.055  |   0.788
mid  (sigma=10, replace=40%)        |   0.171  |   0.055  |   0.355
high (sigma=30, replace=90%)        |   0.088  |   0.055  |   0.142
```

## Online Calibration

Online calibration normalizes raw reliability scores using a streaming baseline computed from
the first N frames (prefix). It maps score drops to calibrated values via an exponential
function. This runs automatically during inference when enabled.

The calibration pipeline:
1. Accumulates raw scores from the first `prefix_len` clean frames
2. Freezes a baseline mean (and optionally std) from the prefix
3. For each subsequent frame, computes `drop = max(0, baseline - raw_score)`
4. Maps the drop to a calibrated score: `calibrated = min + (max-min) * exp(-alpha * drop^gamma)`

Key config knobs (in `RuntimeConfig` within `framework/config.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `enable_online_calibration` | `True` | Enable/disable calibration |
| `online_calibration_apply_in_train` | `False` | Apply during training (usually off) |
| `online_calibration_prefix_len` | `50` | Number of frames to compute baseline |
| `online_calibration_mode` | `"simple"` | `"simple"` or `"normalized"` (divides by std) |
| `online_calibration_alpha` | `10.0` | Exponential mapping steepness |
| `online_calibration_gamma` | `1.0` | Drop shaping exponent |
| `online_calibration_eps` | `1e-6` | Epsilon for normalized mode |
| `online_calibration_min_score` | `0.05` | Floor for calibrated scores |
| `online_calibration_max_score` | `1.0` | Ceiling for calibrated scores |
| `online_calibration_emit_before_ready` | `True` | Emit values before baseline is frozen |
| `online_calibration_use_piecewise_drop` | `False` | Two-slope drop shaping |
| `online_calibration_piecewise_knee` | `0.1` | Knee point for piecewise shaping |
| `online_calibration_piecewise_tail_mult` | `3.0` | Tail multiplier for piecewise shaping |

## Using Trained Stage-2 Weights

Per-encoder trained artifacts are in `artifacts/`:

```
artifacts/
├── brain_b_clean_stats.npz                    # Shared Brain-B clean reference stats
├── pointpillars_stage2_checkpoint.pt          # PointPillars trained weights
├── pointpillars_family_md_enrolled.pt         # PointPillars family MD stats
├── pointpillars_family_md_calibration.json    # PointPillars family MD thresholds
├── pointrcnn_stage2_checkpoint.pt
├── pointrcnn_family_md_enrolled.pt
├── pointrcnn_family_md_calibration.json
├── pvrcnn_stage2_checkpoint.pt
├── pvrcnn_family_md_enrolled.pt
├── pvrcnn_family_md_calibration.json
├── centerpoint_stage2_checkpoint.pt
├── centerpoint_family_md_enrolled.pt
└── centerpoint_family_md_calibration.json
```

Each `*_stage2_checkpoint.pt` contains state dicts for `adapter`, `projector`, `brain_a`, and
`brain_b`. To load trained weights for a specific encoder:

```bash
CHECKPOINT_PATH="artifacts/pointpillars_stage2_checkpoint.pt" \
BRAIN_B_STATS="artifacts/brain_b_clean_stats.npz" \
DEVICE=cuda:0 bash run_real_with_scorer.sh pointpillars
```

If no `CHECKPOINT_PATH` is provided and `ALLOW_DUMMY_WEIGHTS=1` (the default), the pipeline
runs with the pretrained backbone but random scorer weights. Set `ALLOW_DUMMY_WEIGHTS=0` to
require a trained checkpoint.

## Building Sample Episodes

A bundled sample episode is included at `dataset/episode_000001.npz` (180 frames, 1024 points
per frame, 5 features per point).

To rebuild from raw LiDAR frames:

```bash
DATA_ROOT=/path/to/LIDAR_TOP bash build_sample_episode.sh
```

Or with custom parameters:

```bash
python3 tools/build_sample_episode.py \
  --data_root /path/to/LIDAR_TOP \
  --episode_len 180 \
  --clean_prefix 50 \
  --point_feature_dim 5 \
  --max_points 1024 \
  --output_path dataset/episode_000001.npz
```

## Hardware Profiling

- **Encoder-only** (`run_encoder_only.sh`): Profiles pure encoder forward pass — use this
  for per-encoder latency/throughput measurements without scoring overhead.
- **With scorer** (`run_real_with_scorer.sh`): Full pipeline including projection, scoring,
  temporal state, and online calibration — use this for end-to-end latency.

Both modes save per-frame results to `outputs/`. Use `DEVICE=cuda:N` to target a specific GPU.

## Downloading Pretrained Checkpoints

If checkpoints are missing from `checkpoints/`:

```bash
python3 tools/download_official_checkpoints.py --encoders all
```

Or for specific encoders:

```bash
python3 tools/download_official_checkpoints.py --encoders pointpillars,centerpoint
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | PyTorch device (`cpu`, `cuda:0`, `cuda:1`, `auto`) |
| `CHECKPOINT_PATH` | (none) | Path to trained stage-2 checkpoint |
| `BRAIN_B_STATS` | (none) | Path to Brain-B clean reference stats |
| `EPISODE_PATH` | `dataset/episode_000001.npz` | Input episode file |
| `ALLOW_DUMMY_WEIGHTS` | `1` | Allow running without trained checkpoint |
| `ENCODERS` | `all` | Comma-separated encoder list (for preflight/smoke) |
| `OUTPUT_JSON` | (auto) | Override output JSON path |
| `OUTPUT_NPZ` | (auto) | Override output NPZ path (encoder-only) |
| `DATA_ROOT` | `dataset/raw/LIDAR_TOP` | Raw LiDAR frames directory |

## Troubleshooting

**PV-RCNN CUDA errors**: PV-RCNN requires `spconv` and CUDA `pointnet2` ops compiled for your
GPU architecture. Ensure `CUDA_HOME` is set and matches your PyTorch CUDA version.

**Checkpoint not found**: Run `python3 tools/download_official_checkpoints.py --encoders all`
to download pretrained encoder weights to `checkpoints/`.

**Device selection**: Set `DEVICE=cpu` to run without GPU. For multi-GPU machines, use
`DEVICE=cuda:N` to select a specific GPU.

**Import errors**: All imports must resolve within `Analysis/av/`. If you see import errors
referencing `training/`, something is wrong — this pack is fully self-contained.

**Shape mismatch errors**: The encoder expects a specific point feature dimension (typically
4 or 5). The inference scripts auto-pad or truncate via `align_point_feature_dim`. If you see
shape errors, check that your episode was built with the correct `--point_feature_dim`.

**Missing Python packages**: Activate the correct virtual environment (`.venv_av`) or run
`bash setup_env.sh` to create one with all dependencies.

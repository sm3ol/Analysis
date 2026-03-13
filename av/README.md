# AV Inference Pack (Standalone LiDAR)

This folder is self-contained for AV stage-2 inference. The AV tools import only from `Analysis/av`, not from the sibling training repo.

## Included

Supported real encoder names:
- `pointpillars`
- `pointrcnn`
- `pvrcnn`
- `centerpoint`

Included:
- local AV framework: `framework/`
- vendored PV-RCNN backend: `vendor/openpcdet/`
- local checkpoints: `checkpoints/`
- local Brain-B stats artifact: `artifacts/brain_b_clean_stats.npz`
- staged sample episode: `dataset/episode_000001.npz`
- tools:
  - `tools/run_inference_episode.py`
  - `tools/run_encoder_only_episode.py`
  - `tools/preflight_inference.py`
  - `tools/download_official_checkpoints.py`

## Environment Setup

```bash
cd Analysis/av
bash setup_env.sh
```

For `pvrcnn`, `spconv` and CUDA-compatible pointnet2 builds still need to work on the machine. The repo is standalone from `training`, but `pvrcnn` still depends on those third-party CUDA packages.

## Local Assets

Default local paths:
- checkpoints: `Analysis/av/checkpoints/*.pth`
- Brain-B stats: `Analysis/av/artifacts/brain_b_clean_stats.npz`
- vendored OpenPCDet: `Analysis/av/vendor/openpcdet`

If checkpoints are missing, download them with:

```bash
cd Analysis/av
python3 tools/download_official_checkpoints.py --encoders all
```

## Strict Readiness Check

```bash
cd Analysis/av
DEVICE=cuda:0 bash preflight_inference.sh
```

This validates, per encoder:
- checkpoint presence and strict load status
- one-step forward pass through adapter + projector + scorers
- embedding/scorer stats output

## Run Inference

Encoder only:

```bash
cd Analysis/av
DEVICE=cuda:0 bash run_encoder_only.sh centerpoint
```

Encoder + scorer:

```bash
cd Analysis/av
DEVICE=cuda:0 bash run_real_with_scorer.sh centerpoint
```

With trained AV stage-2 weights:

```bash
cd Analysis/av
CHECKPOINT_PATH="/path/to/av_stage2_checkpoint.pt" BRAIN_B_STATS="/path/to/brain_b_clean_stats.npz" DEVICE="cuda:0" bash run_real_with_scorer.sh centerpoint
```

## Sample Episode

A bundled staged sample is already included under `dataset/episode_000001.npz`.
To rebuild it from raw LiDAR frames, pass an explicit root or set `DATA_ROOT`:

```bash
cd Analysis/av
DATA_ROOT=/path/to/LIDAR_TOP bash build_sample_episode.sh
```

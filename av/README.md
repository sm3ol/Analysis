# AV Inference Pack (LiDAR Only)

This folder is synced to the AV stage-2 stack in `training/AV/framework`.

## Scope

Supported real encoder names:
- `pointpillars`
- `pointrcnn`
- `pvrcnn`
- `centerpoint`

Included:
- staged realistic sample episode: `dataset/episode_000001.npz`
- one-episode inference runner: `tools/run_inference_episode.py`
- encoder-only runner: `tools/run_encoder_only_episode.py`
- strict runtime preflight checker: `tools/preflight_inference.py`
- wrappers:
  - `preflight_inference.sh`
  - `run_encoder_only.sh`
  - `run_real_with_scorer.sh`
  - `run_<encoder>_encoder_only.sh` for each of 4 encoders
  - `run_<encoder>_with_scorer.sh` for each of 4 encoders
  - `verify_all_encoders.sh` (single pass/fail wrapper)

## Prerequisites

This pack depends on the parent project layout and code:
- `training/AV/framework`
- `training/playground/OpenPCDet`
- `training/AV/checkpoints/*.pth`

If you clone only `Analysis`, inference will not run.

## Environment Setup

Recommended: use the project AV env (`.venv_av`) from the workspace root.

If creating a local env under `Analysis/av`:

```bash
cd Analysis/av
bash setup_env.sh
```

For `pvrcnn`, OpenPCDet CUDA ops must be available (`spconv`, pointnet2 stack). After activating your env, run:

```bash
pip install -e ../../training/playground/OpenPCDet
```

If this build step fails, fix CUDA/compiler compatibility first, then rerun preflight.

## Strict Readiness Check (Required)

Build or refresh the staged episode:

```bash
cd Analysis/av
bash build_sample_episode.sh
```

Run strict preflight for all 4 encoders:

```bash
cd Analysis/av
DEVICE=cuda:1 bash preflight_inference.sh
```

Preflight validates, per encoder:
- checkpoint file presence and strict load status
- one-step forward pass through adapter + projector + scorers
- embedding shape/stats output

If preflight exits with success and reports `"ready_all": true`, the machine is ready for inference with all listed encoders.

## Run Inference

Encoder only (no scorer state machine):

```bash
cd Analysis/av
DEVICE=cuda:1 bash run_encoder_only.sh centerpoint
```

Encoder + scorer:

```bash
cd Analysis/av
bash run_real_with_scorer.sh centerpoint
```

With trained AV stage-2 weights:

```bash
cd Analysis/av
CHECKPOINT_PATH="/path/to/av_stage2_checkpoint.pt" \
BRAIN_B_STATS="/path/to/brain_b_clean_stats.npz" \
DEVICE="cuda:1" \
bash run_real_with_scorer.sh centerpoint
```

## Per-Encoder Scripts

Encoder-only files:
- `run_pointpillars_encoder_only.sh`
- `run_pointrcnn_encoder_only.sh`
- `run_pvrcnn_encoder_only.sh`
- `run_centerpoint_encoder_only.sh`

Encoder+scorer files:
- `run_pointpillars_with_scorer.sh`
- `run_pointrcnn_with_scorer.sh`
- `run_pvrcnn_with_scorer.sh`
- `run_centerpoint_with_scorer.sh`

Examples:

```bash
cd Analysis/av
DEVICE=cuda:1 bash run_pointpillars_encoder_only.sh
DEVICE=cuda:1 bash run_pointpillars_with_scorer.sh
```

## One-Shot Verification

Run all 4 encoders in both modes and report PASS/FAIL:

```bash
cd Analysis/av
DEVICE=cuda:1 bash verify_all_encoders.sh
```

Outputs:
- `outputs/preflight_inference_*.json`
- `outputs/encoder_only_<encoder>.json`
- `outputs/encoder_only_<encoder>.npz`
- `outputs/inference_<encoder>.json`
- `outputs/verify_all/*.log`

## Current Controller Defaults

- `recover_required_steps = 25`
- `recover_rewarm_steps = 25`
- `recover_anchor_mode = strict`
- `suspicious_threshold_a = 0.95`
- `clean_like_threshold_b = 0.95`

## Notes

- LiDAR-only for current AV stage-2.
- `allow_dummy_weights=1` is intended only for temporary smoke checks while full training checkpoints are not available.

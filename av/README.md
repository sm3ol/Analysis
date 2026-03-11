# AV Inference Pack (LiDAR Only)

This folder is now aligned to the current AV stage-2 training stack under:
- `training/AV/framework`

It is intended for inference-time checks while training is running, with either:
- trained weights (preferred), or
- dummy scorer/projector weights (temporary smoke mode).

## What Is Included

- Real encoder names aligned with training:
  - `pointpillars`
  - `pointrcnn`
  - `pvrcnn`
  - `centerpoint`
- Current taxonomy/config snapshot:
  - `configs/av_taxonomy_manifest.json`
- Realistic staged episode artifacts:
  - `dataset/episode_000001.npz`
  - `dataset/sample_manifest.json`
- Inference tooling:
  - `tools/run_inference_episode.py`
  - `run_real_with_scorer.sh`
- Inference smoke:
  - `smoke_inference.sh`
- Episode builder:
  - `tools/build_sample_episode.py`
  - `build_sample_episode.sh`

Legacy taxonomy-metrics scripts remain in this folder for compatibility.

## Environment

```bash
cd Analysis/av
bash setup_env.sh
```

If you already use the project AV environment from the parent workspace, you can
reuse it instead of creating `.venv`.

## Build/Refresh the Staged Episode

Default source uses real nuScenes sweeps:

```bash
cd Analysis/av
bash build_sample_episode.sh
```

Custom source:

```bash
bash build_sample_episode.sh /path/to/nuscenes/sweeps/LIDAR_TOP
```

## Run Inference on One Episode

Default (dummy scorer/projector if no trained checkpoint is provided):

```bash
cd Analysis/av
bash run_real_with_scorer.sh centerpoint
```

With trained weights from AV training output:

```bash
cd Analysis/av
CHECKPOINT_PATH="/path/to/av_stage2_checkpoint.pt" \
BRAIN_B_STATS="/path/to/brain_b_clean_stats.npz" \
DEVICE="cuda:1" \
bash run_real_with_scorer.sh centerpoint
```

Output JSON is written under:
- `outputs/inference_<encoder>.json`

Quick multi-encoder smoke:

```bash
cd Analysis/av
bash smoke_inference.sh
```

## Current Controller Defaults (Synced)

This pack is synced to current AV controller defaults:
- `recover_required_steps = 25`
- `recover_rewarm_steps = 25`
- `recover_anchor_mode = strict`
- `suspicious_threshold_a = 0.95`
- `clean_like_threshold_b = 0.95`

## Notes

- This is LiDAR-only in current AV stage-2 scope.
- `pvrcnn` relies on OpenPCDet/spconv availability in the active environment.
- Until training finishes, dummy scorer/projector behavior is expected to be
  unstable; use it only for pipeline sanity, not quality evaluation.

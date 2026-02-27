# Embodied AI Inference Pack

This folder is organized by encoder.

Folders:
- `common/`: shared scorer/runtime code used by all encoders
- `dinov2/`
- `openvla/`
- `siglip/`
- `rt1/`
- `assets/`: local model assets used by standalone runs
- `vendor/`: local third-party code snapshots used by standalone runs
- `dataset/`: local real sample episodes used by real-data checks
- `outputs/`: local outputs and profiling artifacts

Each encoder folder contains:
- the real adapter file for that encoder
- an encoder-only test
- an encoder+scorer test
- a local README

## Shared Scorer Checks

Shared scorer-only smoke:
```bash
bash smoke_test.sh
```

The smoke flow now defaults to `cuda`. If no GPU is visible, it fails instead
of silently falling back to CPU. You can override the device explicitly with:

```bash
EMBODIED_DEVICE=cpu bash smoke_test.sh
```

This runs:
- scorer-only check (no encoder weights)
- temporal recovery check
- corrupted sample build from the staged episode

## Quick Start

From this folder:

```bash
bash setup_env.sh
source .venv/bin/activate
```

If you only want the shared scorer smoke without encoder backbones, you can use
the smaller environment:

```bash
bash setup_env.sh smoke
source .venv/bin/activate
```

Verify that CUDA is visible before running the default smoke flow:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Then run:

```bash
bash smoke_test.sh
bash dinov2/smoke_encoder_only.sh
bash dinov2/smoke_with_scorer.sh
bash openvla/smoke_encoder_only.sh
bash openvla/smoke_with_scorer.sh
bash siglip/smoke_encoder_only.sh
bash siglip/smoke_with_scorer.sh
bash rt1/smoke_encoder_only.sh
bash rt1/smoke_with_scorer.sh
```

If you need to force CPU for debugging, override the default:

```bash
EMBODIED_DEVICE=cpu bash dinov2/smoke_with_scorer.sh
```

## Per-Encoder Checks

Examples:

```bash
cd dinov2 && bash run_encoder_only.sh
cd dinov2 && bash run_with_scorer.sh
```

```bash
cd openvla && bash run_encoder_only.sh
cd openvla && bash run_with_scorer.sh
```

Real-data examples using the local sample episode in `dataset/`:

```bash
cd dinov2 && bash run_real_encoder_only.sh
cd dinov2 && bash run_real_with_scorer.sh
```

```bash
bash run_real_encoder_matrix.sh
bash run_real_runtime_matrix.sh
```

Build a corrupted copy of the staged sample episode:

```bash
bash build_corrupted_sample.sh
```

Recovery scenarios using real data:

```bash
cd dinov2 && bash run_brain_a_recovery.sh
cd dinov2 && bash run_brain_a_brain_b_recovery.sh
```

Scenario meanings:
- `run_brain_a_recovery.sh`: corrupt the staged episode, activate Brain A, and
  verify that the temporal logic returns to clean mode
- `run_brain_a_brain_b_recovery.sh`: corrupt the staged episode, activate Brain
  A, then Brain B, and verify recovery back to clean mode

The per-encoder smoke scripts now include:
- synthetic encoder/scorer self-tests
- real encoder-only and encoder+scorer checks on `dataset/`
- Brain-A-only recovery run on corrupted real data
- Brain-A to Brain-B to recovery run on corrupted real data

## Dataset

You do not need real data for the synthetic checks.
The real-data checks in this sub-repo now use the local `dataset/` folder.
One sample episode is already staged there so every encoder can run against the
same input path. Additional sample episodes can be added later as more real
profiling coverage is needed.

The currently staged sample is:
- `dataset/episode_000001.pt`

The corruption helper writes a derived artifact to:
- `outputs/corrupted/episode_000001_noise_and_occlusion.pt`

## Standalone Dependencies

This sub-repo now carries its own runtime dependencies that were previously
looked up through the parent ASPLOS repo:

- `assets/dinov2/facebook--dinov2-base/`: local DINOv2 checkpoint
- `assets/siglip/google--siglip-base-patch16-224/`: local SigLIP checkpoint
- `vendor/rt1_pytorch/`: local RT-1 package snapshot containing `maruya24_rt1`

That means the real-data smoke scripts run from paths inside
`Analysis/embodied_ai` by default. They no longer require
`training/Embodied_AI/...` paths at runtime.

## Outputs

The smoke and real-data runs write structured artifacts under `outputs/`.

Common output groups:
- `outputs/*_adapter_self_test.json`: synthetic encoder-only checks
- `outputs/*_runtime_self_test.json`: synthetic encoder + scorer checks
- `outputs/*_real_adapter_check.json`: real episode encoder-only checks
- `outputs/*_real_runtime_check.json`: real episode encoder + scorer checks
- `outputs/*_brain_a_recovery.json`: Brain-A-only recovery scenario results
- `outputs/*_brain_a_brain_b_recovery.json`: Brain-A to Brain-B recovery results
- `outputs/corrupted/`: corrupted episode copies built from the staged sample
- `outputs/debug/`: shared debugging outputs such as synthetic temporal traces

These JSON files are intended to be easy to diff and inspect before the
benchmarking layer is added.

## Validated Status

The current embodied smoke flow has been validated end to end for:
- `dinov2`
- `openvla`
- `siglip`
- `rt1`

That validation includes:
- synthetic encoder-only checks
- synthetic encoder + scorer checks
- real encoder-only checks on `dataset/episode_000001.pt`
- real encoder + scorer checks on the same episode
- corrupted real-data recovery with Brain A only
- corrupted real-data recovery with Brain A, Brain B, and return to clean mode

In other words, the sub-repo is already past the "wiring only" stage. It now
has a repeatable smoke-test path for the real local sample episode.

## Pass Criteria

Treat a run as successful when:
- the script ends with a `[DONE] ... passed` line
- the matching JSON file appears under `outputs/`
- recovery scripts end with `[RECOVERY] ... passed`

Useful artifacts to check after a run:
- `outputs/*_real_adapter_check.json`
- `outputs/*_real_runtime_check.json`
- `outputs/*_brain_a_recovery.json`
- `outputs/*_brain_a_brain_b_recovery.json`

## Notes

`siglip` and `openvla` may print a `SiglipVisionModel LOAD REPORT` with many
`UNEXPECTED` keys during startup. In this setup, those warnings are expected
because the loader is using the vision tower while the checkpoint also contains
text-tower parameters. The smoke tests should still be treated as successful if
the scripts continue and end with `passed`.

The real-data checks use only the staged sample under `dataset/` and the
derived corrupted copy under `outputs/corrupted/`, so the embodied analysis
sub-repo can be run as a standalone folder without reading from the larger
ASPLOS repo at runtime.

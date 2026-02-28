# Embodied AI Inference Pack

This folder is the current reference workflow for students using the Analysis
workspace.

It is organized around three separate validation targets:
- scorer-only
- encoder-only
- encoder + scorer

Students should run those in that order. That order is intentional:
- if scorer-only fails, the shared scorer/runtime path is broken
- if scorer-only passes but encoder-only fails, the issue is in the encoder path
- if both pass but encoder + scorer fails, the integration path is where to debug

## Folder Layout

- `common/`: shared scorer, temporal logic, config, and utility code
- `dinov2/`: DINOv2 adapter and scripts
- `openvla/`: OpenVLA fused-vision adapter and scripts
- `siglip/`: SigLIP adapter and scripts
- `rt1/`: RT-1 adapter and scripts
- `assets/`: local model assets used by standalone encoder runs
- `vendor/`: vendored third-party code needed by standalone runs
- `dataset/`: local sample episodes used by the real-data checks
- `outputs/`: JSON results, corrupted samples, and debug artifacts

## Before You Start

Students should complete this checklist once per fresh clone.

1. Pull Git LFS assets.
2. Use Python 3.10+.
3. Create a local virtual environment.
4. Install the correct dependency set.
5. Confirm the staged sample episode exists.

Recommended full setup:

```bash
cd Analysis
git lfs install
git lfs pull
cd embodied_ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Smoke-only setup (shared scorer path only, no encoder backbones):

```bash
cd Analysis/embodied_ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-smoke.txt
```

Important:
- `setup_env.sh` uses the machine's current `python3`
- if the default `python3` is older than 3.10, create the venv manually as
  shown above instead of relying on `setup_env.sh`
- the dependency files intentionally pin `torch==2.5.1` because newer PyTorch
  releases can break the sample-episode loader in the original codebase

## Required External Tools

These are not Python packages and must be installed separately if missing:
- Git LFS

After `git lfs pull`, verify the staged sample exists as a real file:

```bash
ls -lh dataset/episode_000001.pt
```

It should be a real multi-megabyte `.pt` file, not a tiny Git LFS pointer.

## Validated Dependency Stack

The current student workflow is validated against:
- `torch==2.5.1`
- `torchvision==0.20.1`
- `transformers>=4.45,<5`
- `huggingface_hub>=0.24,<1`
- `av>=12.0`
- `numpy>=1.24`

Why these matter:
- `torch==2.5.1` avoids the newer `torch.load` behavior change that can break
  the original sample-episode loader
- `torchvision` is required for the vendored RT-1 implementation
- `av` prevents the `transformers` import path from failing because this repo
  also has a sibling `Analysis/av` directory

## Device Matrix

Use the profile that matches the student machine.

### Profile A: NVIDIA GPU (preferred)

Use this on machines with a working CUDA setup.

How to verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected:
- first value should be `True`
- second value should be `>= 1`

Commands should use:
- `EMBODIED_DEVICE=cuda`
- `--device cuda`

### Profile B: Apple Silicon native MPS

Use this on Apple Silicon Macs when running:
- `siglip`
- `rt1`

Commands should use:
- `EMBODIED_DEVICE=mps`
- `--device mps`

These paths were validated successfully on Apple Silicon:
- `siglip` encoder-only
- `siglip` encoder + scorer
- `rt1` encoder-only
- `rt1` encoder + scorer

### Profile C: Apple Silicon MPS with CPU fallback

Use this on Apple Silicon Macs when running:
- `dinov2`
- `openvla`

Why this profile exists:
- those two encoder paths can hit the MPS backend limitation for
  `aten::upsample_bicubic2d.out`
- when that happens, native MPS execution stops with `NotImplementedError`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` lets PyTorch run only the unsupported ops on
  CPU while the rest of the graph continues on `mps`

Commands should use:
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `EMBODIED_DEVICE=mps`
- `--device mps`

These paths were validated successfully on Apple Silicon with fallback enabled:
- `dinov2` encoder-only
- `dinov2` encoder + scorer
- `openvla` encoder-only
- `openvla` encoder + scorer

### Profile D: CPU-only

Use this when:
- there is no usable GPU
- CUDA is unavailable
- MPS is unavailable
- you want the simplest debugging path

Commands should use:
- `EMBODIED_DEVICE=cpu`
- `--device cpu`

This is the slowest path, but it avoids GPU backend-specific issues.

## Workflow 1: Scorer-Only

Run the shared scorer path first. This validates the shared projection, Brain A,
Brain B, temporal state machine, and the corruption helper.

What it runs:
- `common/scripts/scorer_self_test.py`
- `common/scripts/synthetic_recovery_check.py`
- `build_corrupted_sample.sh`

### Scorer-only on NVIDIA / CUDA

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=cuda bash smoke_test.sh
```

### Scorer-only on Apple Silicon

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=mps bash smoke_test.sh
```

### Scorer-only on CPU

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=cpu bash smoke_test.sh
```

What success looks like:
- `[SCORER-TEST] passed ...`
- `outputs/scorer_self_test.json` exists
- `outputs/debug/synthetic_recovery_check.csv` exists
- `outputs/corrupted/episode_000001_noise_and_occlusion.pt` exists
- final line: `[DONE] Embodied AI shared smoke tests passed`

Use this mode when the student wants to test the scorer stack by itself,
without loading any encoder backbone.

## Workflow 2: Encoder-Only

Use encoder-only mode when the student wants to validate only the encoder
adapter path.

This mode checks:
- model asset loading
- image preprocessing
- token shape and adapter output
- real sample-episode loading for the encoder path

### Encoder-only on NVIDIA / CUDA

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash dinov2/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_encoder_only.sh --device cuda
```

### Encoder-only on Apple Silicon (native MPS where supported)

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash siglip/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_encoder_only.sh --device mps
```

### Encoder-only on Apple Silicon (MPS with fallback for unsupported ops)

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_encoder_only.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash openvla/run_encoder_only.sh --device mps
```

### Encoder-only on CPU

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cpu bash dinov2/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash openvla/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash siglip/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash rt1/run_encoder_only.sh --device cpu
```

What success looks like:
- `[ADAPTER-TEST] ... passed ...`
- the matching `outputs/*_adapter_self_test.json` file exists

Use this mode when the student wants encoder behavior without the scorer logic.

## Workflow 3: Encoder + Scorer

Use this mode when the student wants the integrated inference path.

This mode checks:
- encoder adapter output
- projection into the shared latent space
- scorer execution
- temporal reliability logic
- real sample-episode runtime path

### Encoder + scorer on NVIDIA / CUDA

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash dinov2/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_with_scorer.sh --device cuda
```

### Encoder + scorer on Apple Silicon (native MPS where supported)

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash siglip/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_with_scorer.sh --device mps
```

### Encoder + scorer on Apple Silicon (MPS with fallback for unsupported ops)

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_with_scorer.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash openvla/run_with_scorer.sh --device mps
```

### Encoder + scorer on CPU

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cpu bash dinov2/run_with_scorer.sh --device cpu
EMBODIED_DEVICE=cpu bash openvla/run_with_scorer.sh --device cpu
EMBODIED_DEVICE=cpu bash siglip/run_with_scorer.sh --device cpu
EMBODIED_DEVICE=cpu bash rt1/run_with_scorer.sh --device cpu
```

What success looks like:
- `[RUNTIME-TEST] ... passed ...`
- the matching `outputs/*_runtime_self_test.json` file exists

Use this mode when the student wants the full implemented pipeline with both the
encoder and scorer together.

## Real-Data Checks

Each encoder also has real-data versions of the two main checks. These read the
staged sample episode from `dataset/`.

### Real-data checks on NVIDIA / CUDA

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash dinov2/run_real_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_real_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash openvla/run_real_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_real_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash siglip/run_real_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_real_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash rt1/run_real_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_real_with_scorer.sh --device cuda
```

### Real-data checks on Apple Silicon

Use the same device rule as the synthetic checks:
- `siglip` and `rt1`: native `mps`
- `dinov2` and `openvla`: `mps` with fallback enabled

Examples:

```bash
EMBODIED_DEVICE=mps bash siglip/run_real_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_real_with_scorer.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_real_with_scorer.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash openvla/run_real_with_scorer.sh --device mps
```

### Real-data checks on CPU

```bash
EMBODIED_DEVICE=cpu bash dinov2/run_real_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash dinov2/run_real_with_scorer.sh --device cpu
```

Repeat the same pattern for `openvla`, `siglip`, and `rt1`.

Matrix wrappers from the `embodied_ai` root:

```bash
bash run_real_encoder_matrix.sh
bash run_real_runtime_matrix.sh
```

Those wrappers inherit the same device considerations, so students should use
per-encoder scripts when they need explicit Apple fallback control.

## Recovery Scenarios

These scripts use the staged real sample, build a corrupted copy, and validate
that the temporal logic recovers correctly.

Per-encoder commands exist in all four encoder folders:
- `run_brain_a_recovery.sh`
- `run_brain_a_brain_b_recovery.sh`

Example on NVIDIA / CUDA:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=cuda bash dinov2/run_brain_a_recovery.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_brain_a_brain_b_recovery.sh --device cuda
```

Example on Apple Silicon for DINOv2:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_brain_a_recovery.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_brain_a_brain_b_recovery.sh --device mps
```

Interpretation:
- `run_brain_a_recovery.sh`: Brain A reacts to corruption and the state machine
  returns to clean mode
- `run_brain_a_brain_b_recovery.sh`: Brain A triggers, Brain B participates,
  and the system still returns to clean mode

## Copy-Paste Device-Specific Run Sequences

Use the section below that matches the student's machine.

### Full sequence: NVIDIA / CUDA machine

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash smoke_test.sh

EMBODIED_DEVICE=cuda bash dinov2/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash openvla/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash siglip/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_with_scorer.sh --device cuda

EMBODIED_DEVICE=cuda bash rt1/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_with_scorer.sh --device cuda
```

### Full sequence: Apple Silicon Mac

This is the exact pattern validated on Apple Silicon:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash smoke_test.sh

EMBODIED_DEVICE=mps bash siglip/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash rt1/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_with_scorer.sh --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_encoder_only.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash dinov2/run_with_scorer.sh --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash openvla/run_encoder_only.sh --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 EMBODIED_DEVICE=mps bash openvla/run_with_scorer.sh --device mps
```

### Full sequence: CPU-only machine

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cpu bash smoke_test.sh

EMBODIED_DEVICE=cpu bash dinov2/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash dinov2/run_with_scorer.sh --device cpu

EMBODIED_DEVICE=cpu bash openvla/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash openvla/run_with_scorer.sh --device cpu

EMBODIED_DEVICE=cpu bash siglip/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash siglip/run_with_scorer.sh --device cpu

EMBODIED_DEVICE=cpu bash rt1/run_encoder_only.sh --device cpu
EMBODIED_DEVICE=cpu bash rt1/run_with_scorer.sh --device cpu
```

## Output Files

Common artifacts written under `outputs/`:
- `*_adapter_self_test.json`: synthetic encoder-only results
- `*_runtime_self_test.json`: synthetic encoder + scorer results
- `*_real_adapter_check.json`: real-data encoder-only results
- `*_real_runtime_check.json`: real-data encoder + scorer results
- `*_brain_a_recovery.json`: Brain-A-only recovery results
- `*_brain_a_brain_b_recovery.json`: Brain-A to Brain-B recovery results
- `outputs/corrupted/`: corrupted copies of the staged sample episode
- `outputs/debug/`: debug traces such as synthetic temporal transitions

Students should inspect these files after each run instead of relying only on
terminal output.

## Known Failure Modes

1. `git lfs pull` was not run.
- symptom: `dataset/episode_000001.pt` is tiny and `torch.load` fails

2. Python is older than 3.10.
- symptom: syntax errors or type-annotation import failures in encoder modules

3. `av` is missing.
- symptom: `transformers` import fails with `av.__spec__ is None`

4. `torchvision` is missing.
- symptom: RT-1 import fails when loading the vendored tokenizer

5. Apple `mps` backend lacks an operator.
- symptom: `NotImplementedError` for an `aten::...` op
- fix: rerun with `PYTORCH_ENABLE_MPS_FALLBACK=1`

6. Script defaults to `cuda` on a non-CUDA machine.
- symptom: `CUDA was requested, but no GPU is visible in this session`
- fix: explicitly set `EMBODIED_DEVICE=mps` or `EMBODIED_DEVICE=cpu`, and pass
  the matching `--device` flag to encoder scripts

## Pass Criteria

Treat a run as successful when:
- the script prints `passed`
- the expected JSON artifact appears under `outputs/`
- recovery runs also write the expected recovery result file

If students are unsure where a failure belongs, use this rule:
- scorer-only fails: shared scorer/runtime problem
- encoder-only fails: encoder or model-asset problem
- encoder + scorer fails after encoder-only passes: integration problem

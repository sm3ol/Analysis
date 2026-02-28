# Embodied AI Inference Pack

This folder is the current reference workflow for students using the Analysis
workspace.

It is organized around three separate validation targets:
- scorer-only
- encoder-only
- encoder + scorer

Students should run those in that order:
- if scorer-only fails, the shared scorer/runtime path is broken
- if scorer-only passes but encoder-only fails, the issue is in the encoder path
- if both pass but encoder + scorer fails, the integration path is where to debug

Important:
- normal scoring and recovery testing are not the same thing
- if the goal is simply to score data, students should use the normal scoring
  scripts and ignore the recovery scripts unless they explicitly want to test
  corruption handling
- the dedicated scoring guide is in `SCORING_MANUAL.md`
- the normal student workflow below is documented as strict GPU-only
- if a command cannot run on the selected GPU, it should fail with an explicit
  error

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

## Strict GPU Device Matrix

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

### Profile B: Apple Silicon strict MPS

Use this on Apple Silicon Macs when you want strict GPU-only validation.

Commands should use:
- `EMBODIED_DEVICE=mps`
- `--device mps`

Validated strict MPS results on Apple Silicon:
- `siglip` encoder-only: passes
- `siglip` encoder + scorer: passes
- `siglip` real encoder + scorer: passes
- `siglip` recovery scenarios: pass
- `rt1` encoder-only: passes
- `rt1` encoder + scorer: passes
- `rt1` real encoder + scorer: passes
- `rt1` recovery scenarios: pass
- `dinov2`: fails on strict MPS because `aten::upsample_bicubic2d.out` is not
  implemented on MPS
- `openvla`: fails on strict MPS for the same reason

This is expected behavior. In the strict GPU workflow, those failures should be
left visible.

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

### Scorer-only on Apple Silicon / MPS

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=mps bash smoke_test.sh
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

### Encoder-only on Apple Silicon / MPS

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash dinov2/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_encoder_only.sh --device mps
```

Expected Apple strict-MPS result:
- `siglip` and `rt1` should pass
- `dinov2` and `openvla` should fail with the MPS backend error

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

### Encoder + scorer on Apple Silicon / MPS

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash dinov2/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_with_scorer.sh --device mps
```

Expected Apple strict-MPS result:
- `siglip` and `rt1` should pass
- `dinov2` and `openvla` should fail with the MPS backend error

What success looks like:
- `[RUNTIME-TEST] ... passed ...`
- the matching `outputs/*_runtime_self_test.json` file exists

Use this mode when the student wants the full implemented pipeline with both the
encoder and scorer together.

## Scoring vs Recovery

Students often confuse the recovery scripts with the normal scorer path. They
serve different purposes.

Normal scoring:
- this is the regular inference path
- use:
  - `run_with_scorer.sh`
  - `run_real_with_scorer.sh`
- this path does not manually force Brain A or Brain B
- the runtime state machine decides behavior internally

Recovery testing:
- this is an optional stress-test path
- it intentionally injects corruption and then checks whether the temporal logic
  responds and later stabilizes again
- use:
  - `run_brain_a_recovery.sh`
  - `run_brain_a_brain_b_recovery.sh`

If the student is only trying to score an input and collect scorer outputs, the
recovery scripts are not required.

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

### Real-data checks on Apple Silicon / MPS

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash dinov2/run_real_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash dinov2/run_real_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash openvla/run_real_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_real_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash siglip/run_real_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_real_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash rt1/run_real_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_real_with_scorer.sh --device mps
```

Expected Apple strict-MPS result:
- `siglip` and `rt1` should pass
- `dinov2` and `openvla` should fail with the MPS backend error

Matrix wrappers from the `embodied_ai` root:

```bash
bash run_real_encoder_matrix.sh
bash run_real_runtime_matrix.sh
```

Those wrappers do not override the default device, so on Apple Silicon students
should prefer the explicit per-encoder commands above.

## Recovery Scenarios

These scripts use the staged real sample, build a corrupted copy, and validate
that the temporal logic recovers correctly.

This section is optional. It is not part of the normal scoring flow.

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

Example on Apple Silicon / MPS:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
EMBODIED_DEVICE=mps bash dinov2/run_brain_a_recovery.sh --device mps
EMBODIED_DEVICE=mps bash dinov2/run_brain_a_brain_b_recovery.sh --device mps
```

Expected Apple strict-MPS result:
- `siglip` and `rt1` should pass
- `dinov2` and `openvla` should fail with the MPS backend error

Interpretation:
- `run_brain_a_recovery.sh`: Brain A reacts to corruption and the state machine
  returns to clean mode
- `run_brain_a_brain_b_recovery.sh`: Brain A triggers, Brain B participates,
  and the system still returns to clean mode

## Copy-Paste Strict GPU Validation Sequences

Use the block that matches the student machine.

### Full sequence: NVIDIA / CUDA machine

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash smoke_test.sh

EMBODIED_DEVICE=cuda bash dinov2/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_real_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_brain_a_recovery.sh --device cuda
EMBODIED_DEVICE=cuda bash dinov2/run_brain_a_brain_b_recovery.sh --device cuda

EMBODIED_DEVICE=cuda bash openvla/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_real_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_brain_a_recovery.sh --device cuda
EMBODIED_DEVICE=cuda bash openvla/run_brain_a_brain_b_recovery.sh --device cuda

EMBODIED_DEVICE=cuda bash siglip/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_real_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_brain_a_recovery.sh --device cuda
EMBODIED_DEVICE=cuda bash siglip/run_brain_a_brain_b_recovery.sh --device cuda

EMBODIED_DEVICE=cuda bash rt1/run_encoder_only.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_real_with_scorer.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_brain_a_recovery.sh --device cuda
EMBODIED_DEVICE=cuda bash rt1/run_brain_a_brain_b_recovery.sh --device cuda
```

### Full sequence: Apple Silicon strict MPS

This is the exact strict GPU-only pattern validated on Apple Silicon:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=mps bash smoke_test.sh

EMBODIED_DEVICE=mps bash dinov2/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_encoder_only.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_encoder_only.sh --device mps

EMBODIED_DEVICE=mps bash dinov2/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash dinov2/run_real_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_real_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_real_with_scorer.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_real_with_scorer.sh --device mps

EMBODIED_DEVICE=mps bash dinov2/run_brain_a_recovery.sh --device mps
EMBODIED_DEVICE=mps bash dinov2/run_brain_a_brain_b_recovery.sh --device mps

EMBODIED_DEVICE=mps bash openvla/run_brain_a_recovery.sh --device mps
EMBODIED_DEVICE=mps bash openvla/run_brain_a_brain_b_recovery.sh --device mps

EMBODIED_DEVICE=mps bash siglip/run_brain_a_recovery.sh --device mps
EMBODIED_DEVICE=mps bash siglip/run_brain_a_brain_b_recovery.sh --device mps

EMBODIED_DEVICE=mps bash rt1/run_brain_a_recovery.sh --device mps
EMBODIED_DEVICE=mps bash rt1/run_brain_a_brain_b_recovery.sh --device mps
```

Expected Apple strict-MPS result for that sequence:
- `siglip`: passes
- `rt1`: passes
- `dinov2`: fails with the explicit MPS backend error
- `openvla`: fails with the explicit MPS backend error

That outcome is the correct strict GPU compatibility result on Apple Silicon.

## Looping One Episode

There is a dedicated loop runner for repeated scoring on the staged sample
episode:

- `run_real_runtime_loop.sh`

Purpose:
- repeatedly run the real-data encoder + scorer path on the same local episode
- save one JSON file per iteration
- make repeated scoring easy without writing a shell loop by hand

Usage:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate
bash run_real_runtime_loop.sh <encoder> [iterations]
```

GPU-only examples:

```bash
EMBODIED_DEVICE=cuda bash run_real_runtime_loop.sh siglip 10 --device cuda
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh siglip 10 --device mps
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh rt1 10 --device mps
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh dinov2 10 --device mps
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh openvla 10 --device mps
```

Expected Apple strict-MPS loop result:
- `siglip` and `rt1` should run successfully
- `dinov2` and `openvla` should fail with the explicit MPS backend error

Behavior:
- the script reuses the staged episode in `dataset/`
- it runs the real encoder + scorer path multiple times
- it writes outputs under `outputs/loops/<encoder>/`

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
- `outputs/loops/<encoder>/`: repeated real runtime results from the loop script

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
- in the strict GPU workflow, treat this as a real compatibility failure on
  that machine

6. Script defaults to `cuda` on a non-CUDA machine.
- symptom: `CUDA was requested, but no GPU is visible in this session`
- fix: explicitly set `EMBODIED_DEVICE=mps` on Apple Silicon, and pass the
  matching `--device mps` flag to encoder scripts

## Pass Criteria

Treat a run as successful when:
- the script prints `passed`
- the expected JSON artifact appears under `outputs/`
- recovery runs also write the expected recovery result file

If students are unsure where a failure belongs, use this rule:
- scorer-only fails: shared scorer/runtime problem
- encoder-only fails: encoder or model-asset problem
- encoder + scorer fails after encoder-only passes: integration problem

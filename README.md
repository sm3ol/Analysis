# Analysis Workspace

This repository is the student-facing profiling and validation workspace for the
encoder and inference stacks used in this project.

The workspace is organized by domain:
- `embodied_ai/`
- `av/`
- `medical/`

Right now, `embodied_ai/` is the most complete path and should be treated as
the primary student entry point.

## What Students Are Expected To Do

The intended workflow is split into three validation layers:

1. Scorer-only
- validate the shared projector, Brain A, Brain B, and temporal logic without
  loading encoder backbones

2. Encoder-only
- validate the encoder adapter, preprocessing path, and token outputs without
  the shared scorer stack

3. Encoder + scorer
- validate the full implemented inference path through encoder, projection,
  scorer, and temporal logic

This split lets students isolate failures cleanly:
- scorer-only fails: shared scorer/runtime issue
- encoder-only fails: encoder/model-asset issue
- encoder + scorer fails after encoder-only passes: integration issue

## Current Domain Status

- `embodied_ai/`
  - fully usable for student smoke checks and repeated validation
  - includes shared scorer checks, per-encoder checks, one staged sample
    episode, optional recovery scenarios, and a loop runner
  - includes local model assets for DINOv2 and SigLIP plus a vendored RT-1
    snapshot
- `av/`
  - copied encoder workspace with smoke tooling
  - not yet documented at the same level as `embodied_ai/`
- `medical/`
  - copied encoder workspace with smoke tooling
  - not yet documented at the same level as `embodied_ai/`

## Setup Checklist

Students should verify all of the following before running anything:

1. Git LFS is installed.
2. LFS assets have been pulled for the clone.
3. Python 3.10+ is available. Python 3.11 is recommended.
4. A local virtual environment exists in the relevant domain folder.
5. Dependencies were installed from the domain-local requirements file.

For `embodied_ai/`, the recommended setup is:

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

If a student only wants the scorer-only shared smoke path and will not load any
encoder backbones, they can use the smaller environment:

```bash
cd Analysis/embodied_ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-smoke.txt
```

## Dependency Notes

The `embodied_ai` workflow is validated against the pinned dependency set in:
- `embodied_ai/requirements.txt`
- `embodied_ai/requirements-smoke.txt`

These files already exist in the repo and are the exact files students should install from.

Important details:
- `torch==2.5.1` is intentionally pinned because newer PyTorch versions can
  break the staged sample-episode loader in the original repo code
- `torchvision==0.20.1` is required by the vendored RT-1 path
- `transformers` and `huggingface_hub` are required by DINOv2, OpenVLA, and
  SigLIP
- `av` is required so the `transformers` import path coexists cleanly with the
  sibling `Analysis/av` directory

## Strict GPU Policy

The student instructions in this repo are now written for strict GPU-only runs.
That means:
- no CPU fallback is part of the normal documented workflow
- if a command cannot run on the selected GPU backend, it should fail with an
  explicit error
- that failure should be treated as a real compatibility result, not something
  to hide

### NVIDIA / CUDA

This is the preferred environment for full GPU coverage.

How to verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected:
- first value should be `True`
- second value should be `>= 1`

### Apple Silicon / MPS

This is supported as a strict GPU-only validation path, but it does not support
all four embodied encoders equally.

Validated strict MPS results on Apple Silicon:
- `siglip`: passes
- `rt1`: passes
- `dinov2`: fails with an MPS backend limitation
- `openvla`: fails with the same MPS backend limitation

That behavior is expected and should remain visible in the student workflow.

## Student Guides

The detailed device-by-device runbook is in:
- `embodied_ai/README.md`

The scoring-only quick guide is in:
- `embodied_ai/SCORING_MANUAL.md`

Students should use those two files as the operating guides for the current
analysis workspace.

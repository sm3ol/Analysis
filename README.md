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

## System Overview

The embodied analysis path is organized so that encoder backbones remain
adapter-isolated. Each encoder is integrated through its own adapter boundary,
which keeps encoder-specific loading, preprocessing, and tokenization separate
from the shared inference path.

After the adapter boundary, the projection layer and BBC scorer path are shared
across encoders. This keeps the downstream scoring and runtime logic
encoder-agnostic and makes cross-encoder comparisons easier to evaluate under a
consistent systems interface.

## Performance Evaluation Protocol

The profiling workflow should report the following required metrics:
- encoder-only latency (ms)
- scorer-only latency (ms)
- end-to-end latency (ms)
- overhead percentage, defined as `(end-to-end - encoder-only) / encoder-only`
- peak GPU memory
- GPU model
- CUDA version
- PyTorch version

Optional supporting metrics:
- kernel launch count
- Nsight Systems trace

Measurement rules:
- use a fixed batch size for each reported comparison
- use a fixed precision mode for each reported comparison
- run warmup iterations before any timed measurement
- report at least median latency and p95 latency
- cluster profiling may be used for exploratory analysis
- final reported numbers must be collected on the reference GPU (`Ada6000`)


### Metric Definitions

Encoder-only latency:
- measure the forward pass through the adapter and encoder only
- exclude projection and scorer execution
- use CUDA-synchronized timing

Scorer-only latency:
- measure projection, Brain A, Brain B, and temporal logic
- run on cached embeddings to isolate scorer compute

End-to-end latency:
- measure the full path from preprocessing through encoder, projection,
  scorer, and temporal logic

Overhead %:
- compute `(end-to-end - encoder-only) / encoder-only`
- report the result as a percentage

Peak GPU memory:
- measure with `torch.cuda.max_memory_allocated()`

Throughput (optional but recommended):
- report samples per second
- use this to characterize steady-state execution

Latency stability:
- report median (`p50`) and `p95`
- note visible jitter when `p95` deviates materially from the median

### Optional Advanced Metrics

Optional but encouraged metrics:
- kernel launch count
- GPU utilization percentage
- SM occupancy, when available through Nsight Compute
- memory bandwidth utilization
- host-to-device transfer time
- synchronization stall detection
- CUDA Graph effectiveness, measured as latency before and after capture

These metrics are useful for deeper bottleneck analysis, but they are not
mandatory for initial reporting.

### Tooling Guidance

Required baseline tools:
- `torch.profiler`
- `torch.cuda.synchronize()` for correct timing
- `torch.cuda.max_memory_allocated()`
- `nvidia-smi` for sanity checks

Advanced tools (optional):
- Nsight Systems (`nsys`) for timeline-level profiling
- Nsight Compute (`ncu`) for kernel-level metrics
- `torch.compile` (PyTorch 2.x) for graph-level optimization experiments
- CUDA Graph APIs for steady-state loop optimization

All final reported numbers must be collected without profiler overhead enabled.
Profilers are for diagnosis, not final timing.

### Experimental Discipline Rules

- always report GPU model and CUDA version
- do not mix numbers from different GPUs in the same comparison table
- fix batch size and precision before collecting numbers
- warm up the GPU before timing
- run multiple iterations and report the median

### Separation of Concerns

- measurement is owned by Anu
- compute optimization is owned by Patrick
- pipeline optimization is owned by Parth
- tool selection must match the optimization goal; do not use Nsight Compute
  when the issue is pipeline-level

## Responsibilities

Anu:
- build a reproducible benchmark harness
- collect metrics
- produce CSV reports
- identify bottlenecks

Patrick:
- optimize BBC scorer compute
- reduce scorer-only latency
- maintain numerical correctness
- provide before/after comparison

Parth:
- optimize end-to-end execution
- investigate CUDA Graph capture
- reduce sync and allocation overhead
- improve pipeline efficiency

## Reporting Requirements

- results must be submitted as CSV
- each report must include environment metadata
- each report must include a short written summary explaining the primary
  bottlenecks and the changes made

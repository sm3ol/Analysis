# Analysis Workspace

This folder is the profiling and benchmarking workspace for the encoder and
inference stacks used in this repository.

The goal is not just to keep copied code. The goal is to give students a clean,
domain-organized place where they can:

- run encoder-only checks
- run encoder + scorer / brain inference checks
- benchmark end-to-end inference behavior
- compare latency, memory, and execution overhead across encoders
- prepare for deeper profiling with PyTorch Profiler, Nsight Systems, and
  related tools

## Big Picture

This workspace is being prepared for three profiling tracks:

- `embodied_ai`
- `av`
- `medical`

Each domain should eventually support the same high-level workflow:

1. Verify the encoder path works by itself
2. Verify the scorer / brain path works by itself
3. Verify the encoder + scorer path works together
4. Run benchmark harnesses with standardized settings
5. Collect outputs in a consistent format for comparison across runs

The target use case is student-owned performance work:

- measurement / benchmarking
- scorer optimization
- pipeline / execution optimization

So this folder is meant to become the stable place where students run the same
commands, produce the same artifacts, and compare changes without touching the
main project structure.

## Current Domain Layout

- `embodied_ai`
  - now structured in a per-encoder layout
  - each encoder has its own folder
  - shared scorer / brain code lives in `common/`
  - this is the most actively prepared domain right now
- `av`
  - currently contains the copied AV encoder path and smoke tooling
  - not yet refactored to match the new per-encoder embodied layout
- `medical`
  - currently contains the copied medical encoder path and smoke tooling
  - not yet refactored to match the new per-encoder embodied layout

## Embodied AI Design

`Analysis/embodied_ai` is the reference structure we are moving toward.

It is focused on the newer modular inference path, not the older taxonomy
runner.

That means:

- one folder per encoder:
  - `dinov2/`
  - `openvla/`
  - `siglip/`
  - `rt1/`
- one shared area:
  - `common/`

The shared area contains code that should not be duplicated across encoders:

- config
- shared datatypes
- pooling
- projection
- Brain A scorer
- Brain B scorer
- temporal state machine
- runtime wiring for encoder + scorer inference
- shared smoke / self-test utilities

Each encoder folder contains:

- the real adapter for that encoder
- a local README
- an encoder-only test
- an encoder + scorer test
- simple smoke wrappers for both

This is the structure students should use when profiling a single encoder.

## What We Are Profiling

There are three layers of interest:

1. Encoder-only
- adapter cost
- model forward latency
- memory use
- launch behavior

2. Scorer-only
- projector
- Brain A
- Brain B
- temporal state logic

3. Encoder + scorer
- actual inference path through adapter, projector, scorer, and temporal logic
- synchronization and overhead across the full path

This split is intentional because it lets students isolate where time is going.

## Smoke Tests vs Real Profiling

There are two classes of checks in this workspace:

1. Synthetic smoke tests
- use synthetic tensors or synthetic toy data
- useful for wiring validation
- useful before real data is available
- good for CI-style sanity checks

2. Real-data inference benchmarks
- use actual input samples
- useful for meaningful profiling numbers
- needed for production-like latency and memory analysis

Right now, the workspace is set up so the synthetic checks can be prepared first,
and real sample data can be added later without redesigning the structure.

## Data Status

The original tracked repo does not currently include the full input datasets
needed for real profiling:

- no Bridge TFRecords input tree
- no nuScenes LiDAR dataset tree
- no MedMNIST / MedMNIST-C input tree

Because of that:

- `embodied_ai` currently relies on synthetic state/runtime checks and synthetic
  tensors unless real sample episodes are added
- `av` smoke tests generate tiny synthetic LiDAR data locally
- `medical` smoke tests generate tiny synthetic MedMNIST-C style data locally

For `embodied_ai`, the next important milestone is to add a small real sample
set (for example, 10 episodes copied from the server) under:

- `Analysis/embodied_ai/dataset/`

Once that sample exists, this workspace can be extended with:

- real inference loaders
- benchmark harnesses with stable settings
- profiler wrappers for the real input path

There is now a local starter sample episode staged under:

- `Analysis/embodied_ai/dataset/episode_000001.pt`

This makes the embodied analysis sub-repo self-contained enough to run shared
real-data encoder checks without reaching back into the larger training dataset
tree at runtime.

## Dependency Status

Some smoke tests can run without real data, but that does not mean every encoder
is runnable immediately on a fresh machine.

Current practical constraints:

- Python dependencies must still be installed in each domain with the provided
  `setup_env.sh`
- some encoders rely on local model assets or cached checkpoints
- `rt1` also requires the local `rt1_pytorch` dependency and will not run until
  that code is available

So the current workspace is best understood as:

- structurally ready
- synthetically testable in parts
- waiting for real data and some local dependencies for full profiling

## Student Workflow

The intended workflow for students is:

1. Enter the relevant domain folder
2. Create the local environment with `setup_env.sh`
3. Run smoke checks first
4. Run encoder-only and encoder+scorer tests
5. Run benchmark/profiler commands once real data is available
6. Save outputs in the domain-local `outputs/` area

For example:

```bash
cd Analysis/embodied_ai
bash setup_env.sh smoke
bash smoke_test.sh
```

Then move into a specific encoder:

```bash
cd dinov2
bash run_encoder_only.sh
bash run_with_scorer.sh
bash run_real_encoder_only.sh
bash run_real_with_scorer.sh
bash run_brain_a_recovery.sh
bash run_brain_a_brain_b_recovery.sh
```

Or run the full real-data matrix from the embodied sub-repo root:

```bash
cd Analysis/embodied_ai
bash run_real_encoder_matrix.sh
bash run_real_runtime_matrix.sh
```

The embodied sub-repo also now supports:

- building a corrupted copy of the staged sample episode
- running a Brain-A-only recovery path on corrupted real data
- running a Brain-A to Brain-B to recovery path on corrupted real data

## Current Embodied Run Matrix

As of February 27, 2026, `Analysis/embodied_ai` is the most complete analysis
sub-repo in this workspace.

The intended GPU-first smoke flow is:

```bash
cd Analysis/embodied_ai
bash setup_env.sh
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
bash smoke_test.sh
```

Then run the per-encoder checks:

```bash
bash dinov2/smoke_encoder_only.sh
bash dinov2/smoke_with_scorer.sh
bash openvla/smoke_encoder_only.sh
bash openvla/smoke_with_scorer.sh
bash siglip/smoke_encoder_only.sh
bash siglip/smoke_with_scorer.sh
bash rt1/smoke_encoder_only.sh
bash rt1/smoke_with_scorer.sh
```

What these commands cover:

- encoder-only adapter validation on synthetic inputs
- encoder-only validation on the staged real episode
- encoder + scorer validation on synthetic inputs
- encoder + scorer validation on the staged real episode
- corrupted-input recovery using Brain A only
- corrupted-input recovery using Brain A, then Brain B, then return to clean

Expected success signal:

- each script ends with a `[DONE] ... passed` line
- per-run JSON artifacts are written under `Analysis/embodied_ai/outputs/`

Non-blocking warning to expect:

- `siglip` and `openvla` may print `SiglipVisionModel LOAD REPORT` entries with
  many `UNEXPECTED` keys because the loader uses the vision tower while the
  local checkpoint also contains text-tower weights

Standalone note:

- the embodied sub-repo now carries its own staged dataset, local DINOv2 and
  SigLIP model files under `embodied_ai/assets/`, and a local RT-1 package
  snapshot under `embodied_ai/vendor/`
- the runtime no longer depends on `training/Embodied_AI/...` paths inside the
  larger ASPLOS repo

## Near-Term Plan

The current priority is:

1. Finish making `embodied_ai` the reference clean structure
2. Add a small real embodied sample set for inference profiling
3. Build benchmark harnesses around real embodied inference
4. Later, refactor `av` and `medical` to match the same per-encoder structure

## Summary

This folder is the analysis and profiling layer for the project.

It is meant to become:

- clean enough for students to use directly
- structured enough to isolate encoder vs scorer vs full-pipeline costs
- flexible enough to support synthetic smoke checks now and real profiling once
  sample data is added

For `Analysis/embodied_ai`, that structure is now already concrete:

- one real sample episode is staged locally
- all four current encoders can be smoke-tested against that sample
- the scorer stack can be exercised in both normal and corrupted/recovery flows
- outputs are written in a stable, per-run JSON format under `outputs/`

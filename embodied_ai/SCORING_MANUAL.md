# Embodied AI Scoring Manual

This manual is the short operational guide for students who want to run the
embodied scoring stack without getting confused by the optional recovery tools.

## The Three Normal Scoring Paths

These are the standard scoring paths students should use day to day.

1. Scorer-only
- tests the shared scoring stack by itself
- does not load an encoder backbone
- use this first

2. Encoder-only
- tests the encoder adapter without the scorer stack
- use this to validate model loading and token generation

3. Encoder + scorer
- tests the full scoring pipeline
- this is the normal integrated inference path

If the goal is simply "run scoring," these are the commands that matter.

## What To Install First

The repo already includes the requirement files students should use:
- `requirements.txt` for the full encoder workflow
- `requirements-smoke.txt` for scorer-only smoke

Use:

```bash
pip install -r requirements.txt
```

## What Recovery Is

Recovery is not the normal scoring path.

Recovery scripts are separate stress tests that intentionally inject corruption
into the sample episode and then check whether the temporal logic moves through
its states and returns to a clean mode.

Recovery is useful only when you want to validate:
- that corruption is detected
- that the temporal state machine changes behavior
- that the scorer logic stabilizes again after the corrupted segment

If you are not explicitly testing recovery behavior, you can ignore the
recovery scripts entirely.

## Which Scripts Are Normal Scoring

Use these for normal scoring:

- Shared scorer-only:
  - `bash smoke_test.sh`
- Per-encoder encoder-only:
  - `bash <encoder>/run_encoder_only.sh`
- Per-encoder encoder + scorer:
  - `bash <encoder>/run_with_scorer.sh`
- Per-encoder real-data encoder-only:
  - `bash <encoder>/run_real_encoder_only.sh`
- Per-encoder real-data encoder + scorer:
  - `bash <encoder>/run_real_with_scorer.sh`

## Which Scripts Are Recovery-Only

Use these only when you explicitly want corruption and recovery behavior:

- `bash <encoder>/run_brain_a_recovery.sh`
- `bash <encoder>/run_brain_a_brain_b_recovery.sh`

These are not the standard "score one episode" scripts.

## Brain A vs Brain B

For normal scoring:
- you do not manually toggle Brain A or Brain B
- the runtime state machine decides which logic is active based on the current
  state and the observed reliability values

For explicit validation:
- `run_brain_a_recovery.sh` is the Brain-A-only recovery scenario
- `run_brain_a_brain_b_recovery.sh` is the scenario that must pass through a
  Brain-A phase and later a Brain-B-only phase

So:
- if you only want normal scoring, use `run_with_scorer.sh`
- if you want proof that the system entered specific brain phases, use the
  recovery scripts

## Strict GPU-Only Rule

The student workflow in this repo is documented as strict GPU-only.

That means:
- do not use CPU fallback as part of the normal instructions
- if an encoder cannot run on the selected GPU backend, it should fail with an
  explicit error
- that failure is the compatibility result for that machine

## Apple Silicon Reality Check

Strict MPS validation on Apple Silicon currently gives this result:
- `siglip`: passes
- `rt1`: passes
- `dinov2`: fails with the MPS `upsample_bicubic2d` limitation
- `openvla`: fails with the same MPS limitation

So on an Apple Mac:
- strict GPU-only scoring is fully usable for `siglip` and `rt1`
- `dinov2` and `openvla` will correctly fail instead of silently using CPU

## Looping One Episode

The repository includes a dedicated loop runner:

- `bash run_real_runtime_loop.sh <encoder> [iterations]`

This repeatedly runs the real-data encoder + scorer path on the local sample
episode and writes one JSON output per iteration under:

- `outputs/loops/<encoder>/`

GPU-only examples:

```bash
cd Analysis/embodied_ai
source .venv/bin/activate

EMBODIED_DEVICE=cuda bash run_real_runtime_loop.sh siglip 10 --device cuda
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh siglip 10 --device mps
EMBODIED_DEVICE=mps bash run_real_runtime_loop.sh rt1 10 --device mps
```

On Apple strict MPS:
- `siglip` and `rt1` should run
- `dinov2` and `openvla` should fail with the explicit MPS backend error

## Decision Table

If the student says:
- "I only want to test the scorer": use `bash smoke_test.sh`
- "I want to test the encoder alone": use `bash <encoder>/run_encoder_only.sh`
- "I want the normal full scoring path": use `bash <encoder>/run_with_scorer.sh`
- "I want to score the staged real sample": use `bash <encoder>/run_real_with_scorer.sh`
- "I want repeated scoring on the same episode": use `bash run_real_runtime_loop.sh <encoder> [iterations]`
- "I want to test corruption handling": use the recovery scripts

## Practical Rule

If the student is only trying to "run scoring," they should do this:

1. `bash smoke_test.sh`
2. `bash <encoder>/run_encoder_only.sh`
3. `bash <encoder>/run_with_scorer.sh`
4. `bash <encoder>/run_real_with_scorer.sh`
5. `bash run_real_runtime_loop.sh <encoder> <iterations>` if repeated scoring
   is needed

They should not use the recovery scripts unless the explicit goal is to test
corruption handling and temporal recovery behavior.

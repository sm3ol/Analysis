#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENCODER="${1:-centerpoint}"
shift || true

EPISODE_PATH="${EPISODE_PATH:-$SCRIPT_DIR/dataset/episode_000001.npz}"
OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/outputs/encoder_only_${ENCODER}.json}"
OUTPUT_NPZ="${OUTPUT_NPZ:-$SCRIPT_DIR/outputs/encoder_only_${ENCODER}.npz}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DEVICE="${DEVICE:-auto}"
ALLOW_DUMMY_WEIGHTS="${ALLOW_DUMMY_WEIGHTS:-1}"

python3 "$SCRIPT_DIR/tools/run_encoder_only_episode.py" \
  --encoder "$ENCODER" \
  --device "$DEVICE" \
  --episode_path "$EPISODE_PATH" \
  --output_json "$OUTPUT_JSON" \
  --output_npz "$OUTPUT_NPZ" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --allow_dummy_weights "$ALLOW_DUMMY_WEIGHTS" \
  "$@"


#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENCODER="${1:-centerpoint}"
shift || true

EPISODE_PATH="${EPISODE_PATH:-$SCRIPT_DIR/dataset/episode_000001.npz}"
OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/outputs/inference_${ENCODER}.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
BRAIN_B_STATS="${BRAIN_B_STATS:-}"
DEVICE="${DEVICE:-auto}"

python3 "$SCRIPT_DIR/tools/run_inference_episode.py" \
  --encoder "$ENCODER" \
  --device "$DEVICE" \
  --episode_path "$EPISODE_PATH" \
  --output_json "$OUTPUT_JSON" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --brain_b_stats "$BRAIN_B_STATS" \
  --allow_dummy_weights 1 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER="${1:-centerpoint}"
shift || true

EMBEDDING_NPZ="${EMBEDDING_NPZ:-$SCRIPT_DIR/outputs/encoder_only_${ENCODER}.npz}"
OUTPUT_JSON="${OUTPUT_JSON:-$SCRIPT_DIR/outputs/scorer_only_${ENCODER}.json}"
DEVICE="${DEVICE:-auto}"
BRAIN_B_STATS="${BRAIN_B_STATS:-}"
CLEAN_PREFIX="${CLEAN_PREFIX:-50}"

python3 "$SCRIPT_DIR/tools/scorer_only.py" \
  --encoder "$ENCODER" \
  --device "$DEVICE" \
  --embedding_npz "$EMBEDDING_NPZ" \
  --output_json "$OUTPUT_JSON" \
  --brain_b_stats "$BRAIN_B_STATS" \
  --clean_prefix "$CLEAN_PREFIX" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

EPISODE_PATH="${EPISODE_PATH:-$SCRIPT_DIR/dataset/episode_000001.npz}"
DEVICE="${DEVICE:-auto}"
ENCODERS="${ENCODERS:-all}"

if [[ ! -f "$EPISODE_PATH" ]]; then
  echo "[INFO] staged episode not found at $EPISODE_PATH; generating now"
  bash "$SCRIPT_DIR/build_sample_episode.sh"
fi

python3 "$SCRIPT_DIR/tools/preflight_inference.py" \
  --encoders "$ENCODERS" \
  --device "$DEVICE" \
  --episode_path "$EPISODE_PATH" \
  --require_pretrained 1 \
  --output_json "$OUT_DIR/preflight_inference_${ENCODERS//,/__}.json" \
  "$@"

echo "[DONE] preflight passed"


#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

if [[ ! -f "$SCRIPT_DIR/dataset/episode_000001.npz" ]]; then
  echo "[INFO] staged episode not found; generating now"
  bash "$SCRIPT_DIR/build_sample_episode.sh"
fi

for enc in pointpillars pointrcnn centerpoint; do
  echo "[SMOKE] encoder=${enc}"
  python3 "$SCRIPT_DIR/tools/run_inference_episode.py" \
    --encoder "$enc" \
    --episode_path "$SCRIPT_DIR/dataset/episode_000001.npz" \
    --output_json "$OUT_DIR/inference_${enc}.json" \
    --allow_dummy_weights 1 \
    --device auto
done

echo "[DONE] AV inference smoke passed"

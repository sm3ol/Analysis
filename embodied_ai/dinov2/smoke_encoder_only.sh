#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${EMBODIED_DEVICE:-cuda}"
bash "$SCRIPT_DIR/run_encoder_only.sh" --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/run_real_encoder_only.sh" --device "$DEVICE" "$@"
echo "[DONE] dinov2 encoder smoke tests passed"

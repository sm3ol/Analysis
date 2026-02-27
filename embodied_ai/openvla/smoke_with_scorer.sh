#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${EMBODIED_DEVICE:-cuda}"
bash "$SCRIPT_DIR/run_with_scorer.sh" --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/run_real_with_scorer.sh" --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/run_brain_a_recovery.sh" --device "$DEVICE" "$@"
bash "$SCRIPT_DIR/run_brain_a_brain_b_recovery.sh" --device "$DEVICE" "$@"
echo "[DONE] openvla scorer smoke tests passed"

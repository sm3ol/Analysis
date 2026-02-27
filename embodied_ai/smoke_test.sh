#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${EMBODIED_DEVICE:-cuda}"
PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/common/scripts/scorer_self_test.py" \
  --device "$DEVICE" \
  --save_path "$SCRIPT_DIR/outputs/scorer_self_test.json"
PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/common/scripts/synthetic_recovery_check.py"
bash "$SCRIPT_DIR/build_corrupted_sample.sh"
echo "[DONE] Embodied AI shared smoke tests passed"

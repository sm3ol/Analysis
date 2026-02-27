#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/../common/scripts/recovery_scenario_check.py" \
  --module embodied_ai.rt1 \
  --scenario brain_a_brain_b_recovery \
  --dataset_root "$SCRIPT_DIR/../dataset" \
  --save_path "$SCRIPT_DIR/../outputs/rt1_brain_a_brain_b_recovery.json" \
  "$@"

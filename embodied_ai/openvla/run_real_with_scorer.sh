#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/../common/scripts/real_runtime_check.py" \
  --module embodied_ai.openvla \
  --dataset_root "$SCRIPT_DIR/../dataset" \
  --save_path "$SCRIPT_DIR/../outputs/openvla_real_runtime_check.json" \
  "$@"

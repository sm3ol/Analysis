#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/../common/scripts/runtime_self_test.py"   --module embodied_ai.dinov2   --save_path "$SCRIPT_DIR/../outputs/dinov2_runtime_self_test.json"   "$@"

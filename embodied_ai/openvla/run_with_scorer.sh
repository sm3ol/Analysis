#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/../common/scripts/runtime_self_test.py"   --module embodied_ai.openvla   --save_path "$SCRIPT_DIR/../outputs/openvla_runtime_self_test.json"   "$@"

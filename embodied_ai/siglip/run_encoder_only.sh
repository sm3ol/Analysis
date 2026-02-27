#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/../common/scripts/adapter_self_test.py"   --module embodied_ai.siglip   --save_path "$SCRIPT_DIR/../outputs/siglip_adapter_self_test.json"   "$@"

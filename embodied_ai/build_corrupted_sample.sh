#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" python3 "$SCRIPT_DIR/common/scripts/build_corrupted_episode.py" \
  --dataset_root "$SCRIPT_DIR/dataset" \
  "$@"

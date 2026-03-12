#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Encoder-only: adapter backbone is frozen and not saved in checkpoints.
# No checkpoint root needed — adapter uses its original pretrained weights.
bash "$SCRIPT_DIR/dinov2/run_real_encoder_only.sh" "$@"

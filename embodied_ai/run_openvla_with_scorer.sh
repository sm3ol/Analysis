#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_OUTPUTS="$SCRIPT_DIR/../../training/Embodied_AI/outputs"
EMBODIED_CHECKPOINT_ROOT="${EMBODIED_CHECKPOINT_ROOT:-$TRAINING_OUTPUTS/droid_subset_min200_smoke_openvla_20260311_023410}"
export EMBODIED_CHECKPOINT_ROOT
bash "$SCRIPT_DIR/openvla/run_real_with_scorer.sh" "$@"

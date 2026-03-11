#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="${EMBODIED_CHECKPOINT_ROOT:-}"
EXTRA_ARGS=()
if [[ -n "${CHECKPOINT_ROOT}" ]]; then
  CKPT_PATH="${CHECKPOINT_ROOT}/siglip/checkpoint.pt"
  STATS_PATH="${CHECKPOINT_ROOT}/siglip/brain_b_clean_stats.npz"
  if [[ -f "${CKPT_PATH}" ]]; then
    EXTRA_ARGS+=(--checkpoint_path "${CKPT_PATH}")
  else
    echo "[WARN] checkpoint not found at ${CKPT_PATH}; running without trained checkpoint" >&2
  fi
  if [[ -f "${STATS_PATH}" ]]; then
    EXTRA_ARGS+=(--brain_b_stats_path "${STATS_PATH}")
  fi
fi

CMD=(
  python3 "$SCRIPT_DIR/../common/scripts/real_runtime_check.py"
  --module embodied_ai.siglip
  --dataset_root "$SCRIPT_DIR/../dataset"
  --save_path "$SCRIPT_DIR/../outputs/siglip_real_runtime_check.json"
)
CMD+=("${EXTRA_ARGS[@]}")
CMD+=("$@")
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" "${CMD[@]}"

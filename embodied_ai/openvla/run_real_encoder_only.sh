#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="${EMBODIED_CHECKPOINT_ROOT:-}"
EXTRA_ARGS=()
if [[ -n "${CHECKPOINT_ROOT}" ]]; then
  CKPT_PATH="${CHECKPOINT_ROOT}/openvla/checkpoint.pt"
  if [[ -f "${CKPT_PATH}" ]]; then
    EXTRA_ARGS+=(--checkpoint_path "${CKPT_PATH}")
  else
    echo "[WARN] checkpoint not found at ${CKPT_PATH}; running without trained checkpoint" >&2
  fi
fi

CMD=(
  python3 "$SCRIPT_DIR/../common/scripts/real_adapter_check.py"
  --module embodied_ai.openvla
  --dataset_root "$SCRIPT_DIR/../dataset"
  --save_path "$SCRIPT_DIR/../outputs/openvla_real_adapter_check.json"
)
CMD+=("${EXTRA_ARGS[@]}")
CMD+=("$@")
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" "${CMD[@]}"

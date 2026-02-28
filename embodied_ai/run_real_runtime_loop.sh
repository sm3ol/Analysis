#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: bash run_real_runtime_loop.sh <encoder> [iterations] [extra real_runtime_check args...]"
  echo "example: bash run_real_runtime_loop.sh siglip 10 --device cpu"
  exit 1
fi

ENCODER="$1"
shift

ITERATIONS="${LOOP_ITERATIONS:-5}"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  ITERATIONS="$1"
  shift
fi

DEFAULT_DEVICE="${EMBODIED_DEVICE:-cuda}"
EFFECTIVE_DEVICE="$DEFAULT_DEVICE"
USE_DEFAULT_DEVICE=1

EXTRA_ARGS=("$@")
for ((j = 0; j < ${#EXTRA_ARGS[@]}; j++)); do
  if [[ "${EXTRA_ARGS[$j]}" == "--device" && $((j + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
    EFFECTIVE_DEVICE="${EXTRA_ARGS[$((j + 1))]}"
    USE_DEFAULT_DEVICE=0
    break
  fi
done

case "$ENCODER" in
  dinov2|openvla|siglip|rt1)
    MODULE="embodied_ai.$ENCODER"
    ;;
  *)
    echo "unknown encoder: $ENCODER"
    echo "choices: dinov2 openvla siglip rt1"
    exit 1
    ;;
esac

OUT_DIR="$SCRIPT_DIR/outputs/loops/$ENCODER"
mkdir -p "$OUT_DIR"

STATUS=0

for ((i = 1; i <= ITERATIONS; i++)); do
  SAVE_PATH="$OUT_DIR/${ENCODER}_real_runtime_loop_${i}.json"
  CMD=(python3 "$SCRIPT_DIR/common/scripts/real_runtime_check.py"     --module "$MODULE"     --dataset_root "$SCRIPT_DIR/dataset"     --save_path "$SAVE_PATH")
  if [[ "$USE_DEFAULT_DEVICE" == "1" ]]; then
    CMD+=(--device "$DEFAULT_DEVICE")
  fi
  CMD+=("$@")

  echo "[LOOP] iteration ${i}/${ITERATIONS} encoder=${ENCODER} device=${EFFECTIVE_DEVICE}"
  if ! PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" "${CMD[@]}"; then
    STATUS=1
  fi
done

exit "$STATUS"

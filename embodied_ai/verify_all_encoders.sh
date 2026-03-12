#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/verify_all"
mkdir -p "$OUT_DIR"

DEVICE="${DEVICE:-auto}"
EPISODE_PATH="${EPISODE_PATH:-$SCRIPT_DIR/dataset/episode_000001.pt}"

encoders=(dinov2 openvla siglip rt1)
overall_rc=0

printf "%-10s | %-14s | %-14s\n" "encoder" "encoder_only" "with_scorer"
printf "%-10s-+-%-14s-+-%-14s\n" "----------" "--------------" "--------------"

for enc in "${encoders[@]}"; do
  eo_status="PASS"
  ws_status="PASS"

  if ! DEVICE="$DEVICE" EPISODE_PATH="$EPISODE_PATH" \
    bash "$SCRIPT_DIR/run_${enc}_encoder_only.sh" \
    >"$OUT_DIR/${enc}_encoder_only.log" 2>&1; then
    eo_status="FAIL"
    overall_rc=1
  fi

  if ! DEVICE="$DEVICE" EPISODE_PATH="$EPISODE_PATH" \
    bash "$SCRIPT_DIR/run_${enc}_with_scorer.sh" \
    >"$OUT_DIR/${enc}_with_scorer.log" 2>&1; then
    ws_status="FAIL"
    overall_rc=1
  fi

  printf "%-10s | %-14s | %-14s\n" "$enc" "$eo_status" "$ws_status"
done

echo ""
if [[ "$overall_rc" -eq 0 ]]; then
  echo "[DONE] all encoder checks passed"
else
  echo "[DONE] one or more checks failed — see $OUT_DIR/*.log"
fi

exit "$overall_rc"

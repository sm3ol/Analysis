#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/verify_all"
mkdir -p "$OUT_DIR"

DEVICE="${DEVICE:-auto}"
ALLOW_DUMMY_WEIGHTS="${ALLOW_DUMMY_WEIGHTS:-1}"
EPISODE_PATH="${EPISODE_PATH:-$SCRIPT_DIR/dataset/episode_000001.npz}"

encoders=(pointpillars pointrcnn pvrcnn centerpoint)
overall_rc=0

if [[ ! -f "$EPISODE_PATH" ]]; then
  echo "[INFO] staged episode not found; generating now"
  bash "$SCRIPT_DIR/build_sample_episode.sh"
fi

echo "[STEP] strict preflight check"
if DEVICE="$DEVICE" ENCODERS="all" EPISODE_PATH="$EPISODE_PATH" bash "$SCRIPT_DIR/preflight_inference.sh" >"$OUT_DIR/preflight.log" 2>&1; then
  preflight_status="PASS"
else
  preflight_status="FAIL"
  overall_rc=1
fi
echo "[PRECHECK] $preflight_status"

printf "%-14s | %-12s | %-12s\n" "encoder" "encoder_only" "with_scorer"
printf "%-14s-+-%-12s-+-%-12s\n" "--------------" "------------" "------------"
for enc in "${encoders[@]}"; do
  eo_status="PASS"
  ws_status="PASS"
  if ! DEVICE="$DEVICE" ALLOW_DUMMY_WEIGHTS="$ALLOW_DUMMY_WEIGHTS" EPISODE_PATH="$EPISODE_PATH" \
    bash "$SCRIPT_DIR/run_${enc}_encoder_only.sh" >"$OUT_DIR/${enc}_encoder_only.log" 2>&1; then
    eo_status="FAIL"
    overall_rc=1
  fi
  if ! DEVICE="$DEVICE" ALLOW_DUMMY_WEIGHTS="$ALLOW_DUMMY_WEIGHTS" EPISODE_PATH="$EPISODE_PATH" \
    bash "$SCRIPT_DIR/run_${enc}_with_scorer.sh" >"$OUT_DIR/${enc}_with_scorer.log" 2>&1; then
    ws_status="FAIL"
    overall_rc=1
  fi
  printf "%-14s | %-12s | %-12s\n" "$enc" "$eo_status" "$ws_status"
done

if [[ "$overall_rc" -eq 0 ]]; then
  echo "[DONE] all encoder checks passed"
else
  echo "[DONE] one or more checks failed (see $OUT_DIR/*.log)"
fi

exit "$overall_rc"


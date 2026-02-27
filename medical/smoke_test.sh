#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_ROOT="$SCRIPT_DIR/smoke_data/clean"
CORR_ROOT="$SCRIPT_DIR/smoke_data/corrupted"

python3 "$SCRIPT_DIR/tools/generate_synthetic_medmnistc.py" \
  --clean_root "$CLEAN_ROOT" \
  --corrupted_root "$CORR_ROOT" \
  --dataset mini \
  --corruption pixelate \
  --num_samples 16 \
  --image_size 32

encoders=(clip_vit clip_resnet biomedclip_vit medclip_resnet)
for enc in "${encoders[@]}"; do
  echo "[SMOKE] medical encoder=${enc}"
  OUT_DIR="$SCRIPT_DIR/outputs/smoke_${enc}"
  python3 "$SCRIPT_DIR/taxonomy_metrics_medmnistc.py" \
    --clean_root "$CLEAN_ROOT" \
    --corrupted_root "$CORR_ROOT" \
    --datasets mini \
    --corruptions pixelate \
    --severities 1 \
    --encoders "$enc" \
    --max_samples_per_dataset 16 \
    --batch_size 8 \
    --input_size 32 \
    --output_dir "$OUT_DIR" \
    --mock_encoder_preflight

  latest_run="$(ls -dt "$OUT_DIR"/run_* | head -n 1)"
  test -f "$latest_run/metrics_table.csv"
  echo "[OK] ${enc} -> $latest_run/metrics_table.csv"
done

echo "[DONE] Medical smoke tests passed for: ${encoders[*]}"

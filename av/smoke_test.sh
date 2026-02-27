#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/smoke_data/nuscenes_mock"

python3 "$SCRIPT_DIR/tools/generate_synthetic_lidar_data.py" --output_root "$DATA_ROOT"

encoders=(pointpillars pointrcnn pv_rcnn centerpoint)
for enc in "${encoders[@]}"; do
  echo "[SMOKE] AV encoder=${enc}"
  OUT_DIR="$SCRIPT_DIR/outputs/smoke_${enc}"
  python3 "$SCRIPT_DIR/taxonomy_metrics_av_lidar.py" \
    --data_root "$DATA_ROOT" \
    --encoders "$enc" \
    --severities 1 \
    --validation_severities 1 \
    --distance cosine \
    --episode_len 5 \
    --episode_stride 2 \
    --num_episodes 8 \
    --max_samples_per_scope 16 \
    --rep_mode pooled \
    --output_dir "$OUT_DIR"

  latest_run="$(ls -dt "$OUT_DIR"/run_* | head -n 1)"
  test -f "$latest_run/metrics_table.csv"
  echo "[OK] ${enc} -> $latest_run/metrics_table.csv"
done

echo "[DONE] AV smoke tests passed for: ${encoders[*]}"

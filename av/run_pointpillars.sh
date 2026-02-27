#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <nuscenes_data_root> [extra args]"
  exit 1
fi
DATA_ROOT="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/taxonomy_metrics_av_lidar.py" \
  --data_root "$DATA_ROOT" \
  --encoders pointpillars \
  --severities 1 \
  --distance cosine \
  --max_samples_per_scope 64 \
  --output_dir "$SCRIPT_DIR/outputs/pointpillars" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <medmnist_clean_root> <medmnist_corrupted_root> [extra args]"
  exit 1
fi
CLEAN_ROOT="$1"
CORRUPTED_ROOT="$2"
shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/taxonomy_metrics_medmnistc.py" \
  --clean_root "$CLEAN_ROOT" \
  --corrupted_root "$CORRUPTED_ROOT" \
  --encoders biomedclip_vit \
  --severities 1 \
  --max_samples_per_dataset 128 \
  --output_dir "$SCRIPT_DIR/outputs/biomedclip_vit" \
  "$@"

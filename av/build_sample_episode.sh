#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${1:-$PROJECT_ROOT/Taxonomy_experiment/AV_experiment/data/nuscenes/sweeps/LIDAR_TOP}"

python3 "$SCRIPT_DIR/tools/build_sample_episode.py" \
  --data_root "$DATA_ROOT" \
  --output_path "$SCRIPT_DIR/dataset/episode_000001.npz" \
  --manifest_path "$SCRIPT_DIR/dataset/sample_manifest.json"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${1:-${DATA_ROOT:-$SCRIPT_DIR/dataset/raw/LIDAR_TOP}}"

python3 "$SCRIPT_DIR/tools/build_sample_episode.py"   --data_root "$DATA_ROOT"   --output_path "$SCRIPT_DIR/dataset/episode_000001.npz"   --manifest_path "$SCRIPT_DIR/dataset/sample_manifest.json"

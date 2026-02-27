#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT_DIR/embodied_ai/smoke_test.sh"
bash "$ROOT_DIR/av/smoke_test.sh"
bash "$ROOT_DIR/medical/smoke_test.sh"

echo "[DONE] All Analysis subrepo smoke tests passed"

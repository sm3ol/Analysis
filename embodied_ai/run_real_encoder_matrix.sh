#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS=0
for encoder in dinov2 openvla siglip rt1; do
  if ! bash "$SCRIPT_DIR/$encoder/run_real_encoder_only.sh" "$@"; then
    STATUS=1
  fi
done
exit "$STATUS"

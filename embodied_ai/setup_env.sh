#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
if [[ "$MODE" == "smoke" ]]; then
  pip install -r requirements-smoke.txt
else
  pip install -r requirements.txt
fi

#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
if [[ "$MODE" == "smoke" ]]; then
  pip install "numpy>=1.24"
else
  pip install -r requirements.txt
  if [[ -f "./vendor/openpcdet/requirements.txt" ]]; then
    pip install -r ./vendor/openpcdet/requirements.txt
  fi
  if [[ -d "./vendor/openpcdet" ]]; then
    pip install -e ./vendor/openpcdet
  fi
fi

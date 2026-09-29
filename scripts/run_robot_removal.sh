#!/usr/bin/env bash
# Batch runner wrapper using the ComfyUI venv.
#   bash scripts/run_robot_removal.sh --manifest M.jsonl --output-root DIR [--force] [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${COMFYUI_VENV:-/venv/main}
exec "$VENV/bin/python" -m robot_removal.batch \
  --config robot_removal/config.yaml "$@"

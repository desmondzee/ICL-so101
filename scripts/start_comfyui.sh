#!/usr/bin/env bash
# Start ComfyUI bound to localhost only.
# Usage: bash scripts/start_comfyui.sh [--listen 127.0.0.1] [--port 18188]
set -euo pipefail

LISTEN=127.0.0.1
PORT=18188
COMFY=${COMFYUI_DIR:-/workspace/ComfyUI}
VENV=${COMFYUI_VENV:-/venv/main}
while [ $# -gt 0 ]; do
  case "$1" in
    --listen) LISTEN=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

if curl -sf "http://$LISTEN:$PORT/system_stats" >/dev/null 2>&1; then
  echo "ComfyUI already listening on $LISTEN:$PORT"
  exit 0
fi

cd "$COMFY"
exec "$VENV/bin/python" main.py --disable-auto-launch --disable-xformers --listen "$LISTEN" --port "$PORT" --enable-cors-header

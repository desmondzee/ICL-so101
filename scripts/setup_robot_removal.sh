#!/usr/bin/env bash
# Idempotent setup for robot removal: verifies the ComfyUI install, Python env,
# and pinned model files. Usage: bash scripts/setup_robot_removal.sh
set -euo pipefail
cd "$(dirname "$0")/.."

COMFY=${COMFYUI_DIR:-/workspace/ComfyUI}
VENV=${COMFYUI_VENV:-/venv/main}

echo "== environment =="
"$VENV/bin/python" - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "== comfyui =="
test -f "$COMFY/main.py" || { echo "missing $COMFY/main.py"; exit 1; }
git -C "$COMFY" rev-parse HEAD
git -C "$COMFY" describe --tags 2>/dev/null || true

echo "== model files (sha256 checked against data/robot_removal/model_manifest.json) =="
"$VENV/bin/python" - <<'PY'
import hashlib, json, os
from pathlib import Path
man = json.load(open("data/robot_removal/model_manifest.json"))
comfy = Path(os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI"))
bad = []
for rel, want in man["files"].items():
    p = Path(rel) if rel.startswith("/") else comfy / rel
    if not p.exists():
        bad.append((rel, "missing")); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != want:
        bad.append((rel, f"sha256 {h[:12]} != {want[:12]}"))
    else:
        print("ok", rel)
if bad:
    for r, e in bad: print("BAD", r, e)
    raise SystemExit(1)
PY

echo "== deps =="
"$VENV/bin/python" -c "import yaml, scipy, numpy, PIL, requests; print('deps ok')"
"$VENV/bin/python" -m robot_removal.test_transform

echo "setup ok"

#!/usr/bin/env python3
"""v3 rerun: mask-free Qwen edit for suspected failures + clean controls.

Feeds the ORIGINAL image to Qwen (no LaMa reference) with the removal
positive / robot+shadow negative at cfg 4, then composites inside the
stored expanded mask. Output goes to <out_dir>/<key>/ next to per-episode
artifacts; the production run dir is untouched.

Usage: /venv/main/bin/python scripts/v3_rerun.py <run_dir> <out_dir> [--controls N]
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_removal import postprocess
from robot_removal import workflow as wf
from robot_removal.batch import config_hash, load_config
from robot_removal.comfy import ComfyClient
from robot_removal.masks import mask_file_to_bool


def process(client, cfg, run_dir: Path, out_dir: Path, key: str) -> dict:
    ep = run_dir / key
    dest = out_dir / key
    dest.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(ep / "original_rgb.png").convert("RGB"))
    mask = mask_file_to_bool(ep / "expanded_robot_mask.png")

    up = client.upload_image(ep / "original_rgb.png", name=f"v3_{wf.safe_id(key)}.png", overwrite=True)
    pid = client.submit(wf.build_edit_prompt(up, f"v3_{wf.safe_id(key)}", cfg))
    hist = client.wait(pid, timeout=cfg.get("job_timeout_s", 1800))
    qf = [f for f in client.output_files(hist) if "_qwen_" in f["filename"]]
    if not qf:
        raise RuntimeError(f"{key}: no qwen output")
    qwen = np.asarray(Image.open(io.BytesIO(client.view(qf[0]["filename"], qf[0].get("subfolder", "")))).convert("RGB"))
    if qwen.shape != original.shape:
        raise RuntimeError(f"{key}: qwen size {qwen.shape[:2]} != original {original.shape[:2]}")

    composite = postprocess.composite(original, qwen, mask, cfg["mask"].get("soften_px", 0))
    postprocess.save_png_atomic(Image.fromarray(qwen), dest / "inpaint_native.png")
    postprocess.save_png_atomic(Image.fromarray(composite), dest / "composite_native.png")
    stats = postprocess.assert_preserved(dest / "composite_native.png", original, mask)
    final, crop = postprocess.zerowam_transform(Image.fromarray(composite), cfg["output"]["width"], cfg["output"]["height"])
    postprocess.save_png_atomic(final, dest / "zerowam_480x320.png")

    flags = [f for f in (postprocess.flat_fill_flag(original, composite, mask), postprocess.seam_flag(composite, mask)) if f]
    meta = {
        "key": key,
        "variant": "v3_maskfree",
        "config_hash": config_hash(cfg),
        "mask_coverage": float(mask.mean()),
        "flags": flags,
        "preserved": stats,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_controls = 40
    if "--controls" in sys.argv:
        n_controls = int(sys.argv[sys.argv.index("--controls") + 1])

    quarantined = sorted(json.loads((run_dir / "quarantine.json").read_text()))
    accepted = set(json.loads((run_dir / "accepted.json").read_text()))
    clean = sorted(accepted - set(quarantined))
    rng = np.random.default_rng(11)
    controls = [clean[i] for i in rng.choice(len(clean), min(n_controls, len(clean)), replace=False)]

    cfg = load_config(Path("robot_removal/config.yaml"))
    client = ComfyClient(cfg.get("comfy_url", "http://127.0.0.1:18188"))
    results, failed = [], []
    keys = quarantined + controls
    for i, key in enumerate(keys):
        t0 = time.time()
        try:
            meta = process(client, cfg, run_dir, out_dir, key)
            results.append(meta)
            print(f"[{i + 1}/{len(keys)}] {key} {time.time() - t0:.1f}s flags={meta['flags']}", flush=True)
        except Exception as e:
            failed.append({"key": key, "error": str(e)})
            print(f"[{i + 1}/{len(keys)}] {key} FAILED {e}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps({"suspected": quarantined, "controls": controls}, indent=1))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "failed.json").write_text(json.dumps(failed, indent=1))
    print(json.dumps({"done": len(results), "failed": len(failed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

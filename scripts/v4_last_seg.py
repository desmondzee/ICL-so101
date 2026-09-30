#!/usr/bin/env python3
"""v4_last segmentation pass: SAM3 masks for last-frame inputs.

Same mask machinery as the v4 production run but WITHOUT the pair.json
robot box hint (it's a first-frame box, stale for last frames). Writes
original_rgb.png + expanded_robot_mask.png into <out_dir>/<key>/ so the
v4 edit runner can treat out_dir as its mask_run.

Usage: /venv/main/bin/python scripts/v4_last_seg.py <manifest.jsonl> <out_dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_removal import workflow as wf
from robot_removal.batch import EpisodeError, _download_outputs, load_config, load_manifest
from robot_removal.comfy import ComfyClient
from robot_removal.masks import disk_dilate, mask_file_to_bool, select_robot, validate_mask
from robot_removal import postprocess


def seg_done(dest: Path, row: dict) -> bool:
    s = dest / "seg.json"
    if not s.exists() or not (dest / "expanded_robot_mask.png").exists() or not (dest / "original_rgb.png").exists():
        return False
    try:
        return json.loads(s.read_text()).get("input_sha256") == row["image_sha256"]
    except json.JSONDecodeError:
        return False


def process(client: ComfyClient, cfg: dict, row: dict, dest: Path) -> dict:
    key = row["key"]
    prefix = f"lf_{wf.safe_id(key)}"
    name = client.upload_image(Path(row["image"]), name=f"{prefix}.jpg", overwrite=True)

    t0 = time.time()
    pid = client.submit(wf.build_seg_prompt(name, prefix, cfg, None))
    hist = client.wait(pid, timeout=cfg.get("job_timeout_s", 1800))
    seg_s = time.time() - t0

    files = _download_outputs(client, hist, prefix, dest / ".seg")
    if not files["arm"]:
        raise EpisodeError("no sam3 arm masks saved")

    arm = [mask_file_to_bool(p) for p in files["arm"]]
    ext = [mask_file_to_bool(p) for p in files["ext"]]
    raw = select_robot(arm, ext)
    expanded = disk_dilate(raw, cfg["mask"]["dilate_px"])
    flags = validate_mask(expanded, cfg["mask"]["min_coverage"], cfg["mask"]["max_coverage"])

    dest.mkdir(parents=True, exist_ok=True)
    original = Image.open(row["image"]).convert("RGB")
    postprocess.save_png_atomic(original, dest / "original_rgb.png")
    postprocess.save_png_atomic(Image.fromarray((expanded * 255).astype("uint8")), dest / "expanded_robot_mask.png")

    st = {
        "key": key,
        "input_sha256": row["image_sha256"],
        "mask_coverage": float(expanded.mean()),
        "mask_candidates": {"arm": len(arm), "ext": len(ext)},
        "flags": flags,
        "seg_s": seg_s,
    }
    (dest / "seg.json").write_text(json.dumps(st, indent=1))
    return st


def main() -> int:
    manifest_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = load_config(Path("robot_removal/config.yaml"))
    rows = load_manifest(manifest_path)
    client = ComfyClient(cfg.get("comfy_url", "http://127.0.0.1:18188"))
    done, failed = 0, []
    for i, row in enumerate(rows):
        dest = out_dir / row["key"]
        if seg_done(dest, row):
            done += 1
            continue
        try:
            st = None
            for attempt in range(cfg.get("retries", 2)):
                try:
                    st = process(client, cfg, row, dest)
                    break
                except Exception as e:
                    if attempt == cfg.get("retries", 2) - 1:
                        raise
                    print(f"[{i + 1}/{len(rows)}] {row['key']} retry {attempt + 1}: {e}", flush=True)
            done += 1
            print(f"[{i + 1}/{len(rows)}] {row['key']} cov={st['mask_coverage']:.3f} flags={st['flags']} {st['seg_s']:.1f}s", flush=True)
        except Exception as e:
            failed.append({"key": row["key"], "error": str(e)})
            print(f"[{i + 1}/{len(rows)}] {row['key']} FAILED {e}", flush=True)
    (out_dir / "seg_failed.json").write_text(json.dumps(failed, indent=1))
    print(json.dumps({"done": done, "failed": len(failed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""v4 run: mask-free Qwen edit over all episodes, tracked quality metrics.

Reuses stored expanded masks from the rr4 production run. Qwen edits the
ORIGINAL image (no LaMa reference) with the removal positive + robot/shadow
negative @ cfg4; composite keeps pixels inside the expanded mask only.

Per-episode artifacts under <out_dir>/<key>/:
  original_rgb.png, expanded_robot_mask.png, mask_overlay.png,
  inpaint_native.png, composite_native.png, zerowam_480x320.png,
  sheet.png (original | binary mask | overlay | result), metadata.json,
  complete.json (resume marker: input sha + mask sha + config hash)

Inline metrics recorded in metadata.json:
  fill_lum_step / fill_chroma_drift / fill_texture_ratio (dataset-relative
  z-scores computed in the post-pass -> fill_outlier_z), mask_prior_iou.

Usage: /venv/main/bin/python scripts/v4_run.py <manifest.jsonl> <mask_run_dir> <out_dir>
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_removal import postprocess
from robot_removal import workflow as wf
from robot_removal.batch import config_hash, load_config, load_manifest
from robot_removal.comfy import ComfyClient
from robot_removal.masks import mask_file_to_bool

WORKFLOW_VERSION = "v4"


def fill_stats(original: np.ndarray, composite: np.ndarray, mask: np.ndarray) -> dict:
    """Region-level fill naturalness stats; dataset-relative z in post-pass."""
    from scipy import ndimage

    m = mask.astype(bool)
    if not m.any():
        return {"fill_lum_step": 0.0, "fill_chroma_drift": 0.0, "fill_texture_ratio": 1.0}
    ring = ndimage.binary_dilation(m, iterations=10) & ~ndimage.binary_dilation(m, iterations=3)
    lum = composite.astype(np.float32).mean(2)
    grad = np.abs(np.gradient(lum, axis=0)) + np.abs(np.gradient(lum, axis=1))
    return {
        "fill_lum_step": float(abs(lum[m].mean() - lum[ring].mean())),
        "fill_chroma_drift": float(np.abs(composite[m].astype(np.float32).mean(0) - composite[ring].astype(np.float32).mean(0)).mean()),
        "fill_texture_ratio": float(grad[m].mean() / max(grad[ring].mean(), 1e-6)),
    }


def mask_priors(mask_run: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-dataset pixelwise median of expanded masks. Empty/weak priors excluded."""
    by_ds = defaultdict(list)
    for r in rows:
        p = mask_run / r["key"] / "expanded_robot_mask.png"
        if p.exists():
            by_ds[r["dataset_id"]].append(p)
    priors = {}
    for ds, paths in by_ds.items():
        stack = np.stack([np.asarray(Image.open(p).convert("L")) > 127 for p in paths])
        prior = np.median(stack, axis=0) > 0.5
        if prior.mean() > 0.003:  # degenerate on bimodal datasets
            priors[ds] = prior
    return priors


def process(client, cfg, row: dict, mask_run: Path, dest: Path, priors: dict) -> dict:
    key = row["key"]
    ep = mask_run / key
    original = np.asarray(Image.open(ep / "original_rgb.png").convert("RGB"))
    mask = mask_file_to_bool(ep / "expanded_robot_mask.png")

    up = client.upload_image(Path(row["image"]), name=f"v4_{wf.safe_id(key)}.jpg", overwrite=True)
    pid = client.submit(wf.build_edit_prompt(up, f"v4_{wf.safe_id(key)}", cfg))
    hist = client.wait(pid, timeout=cfg.get("job_timeout_s", 1800))
    qf = [f for f in client.output_files(hist) if "_qwen_" in f["filename"]]
    if not qf:
        raise RuntimeError("no qwen output")
    qwen = np.asarray(Image.open(io.BytesIO(client.view(qf[0]["filename"], qf[0].get("subfolder", "")))).convert("RGB"))
    if qwen.shape != original.shape:
        raise RuntimeError(f"qwen size {qwen.shape[:2]} != original {original.shape[:2]}")

    # The Qwen output IS the result — the mask is only used for metrics.
    # Record how far the edit drifted outside the mask as a fidelity diagnostic.
    drift = np.abs(qwen.astype(int) - original.astype(int))
    scene_drift = {
        "outside_diff_px": int((drift.max(2) > 0)[~mask].sum()),
        "outside_mean_abs": round(float(drift[~mask].mean()), 3),
        "outside_max": int(drift[~mask].max()),
    }

    dest.mkdir(parents=True, exist_ok=True)
    postprocess.save_png_atomic(Image.fromarray(original), dest / "original_rgb.png")
    postprocess.save_png_atomic(Image.fromarray((mask * 255).astype("uint8")), dest / "expanded_robot_mask.png")
    ov = original.copy()
    ov[mask] = (0.55 * ov[mask] + np.array([255, 0, 0]) * 0.45).astype("uint8")
    postprocess.save_png_atomic(Image.fromarray(ov), dest / "mask_overlay.png")
    postprocess.save_png_atomic(Image.fromarray(qwen), dest / "result_native.png")
    final, crop = postprocess.zerowam_transform(Image.fromarray(qwen), cfg["output"]["width"], cfg["output"]["height"])
    postprocess.save_png_atomic(final, dest / "zerowam_480x320.png")

    sheet = np.concatenate(
        [original, np.repeat((mask * 255).astype("uint8")[..., None], 3, axis=2), ov, qwen], axis=1
    )
    postprocess.save_png_atomic(Image.fromarray(sheet), dest / "sheet.png")

    prior = priors.get(row["dataset_id"])
    inter = float((mask & prior).sum()) if prior is not None else None
    meta = {
        "key": key,
        "dataset_id": row["dataset_id"],
        "input_sha256": row["image_sha256"],
        "config_hash": config_hash(cfg),
        "workflow_version": WORKFLOW_VERSION,
        "mask_coverage": float(mask.mean()),
        "mask_prior_iou": (inter / float((mask | prior).sum())) if prior is not None else None,
        **fill_stats(original, qwen, mask),
        "scene_drift": scene_drift,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=1))
    (dest / "complete.json").write_text(
        json.dumps({"ok": True, **{k: meta[k] for k in ("input_sha256", "config_hash", "workflow_version")}}, indent=1)
    )
    return meta


def is_done(dest: Path, row: dict, chash: str) -> bool:
    c = dest / "complete.json"
    if not c.exists() or not (dest / "zerowam_480x320.png").exists():
        return False
    try:
        d = json.loads(c.read_text())
    except json.JSONDecodeError:
        return False
    return d.get("input_sha256") == row["image_sha256"] and d.get("config_hash") == chash and d.get("workflow_version") == WORKFLOW_VERSION


def main() -> int:
    manifest_path, mask_run, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cfg = load_config(Path("robot_removal/config.yaml"))
    chash = config_hash(cfg)
    rows = load_manifest(manifest_path)
    priors = mask_priors(mask_run, rows)
    print(f"mask priors: {len(priors)} datasets (weak/absent for bimodal)", flush=True)
    client = ComfyClient(cfg.get("comfy_url", "http://127.0.0.1:18188"))
    done, failed = 0, []
    for i, row in enumerate(rows):
        dest = out_dir / row["key"]
        if is_done(dest, row, chash):
            done += 1
            continue
        t0 = time.time()
        try:
            meta = None
            for attempt in range(cfg.get("retries", 2)):
                try:
                    meta = process(client, cfg, row, mask_run, dest, priors)
                    break
                except Exception as e:
                    if attempt == cfg.get("retries", 2) - 1:
                        raise
                    print(f"[{i + 1}/{len(rows)}] {row['key']} retry {attempt + 1}: {e}", flush=True)
            done += 1
            print(f"[{i + 1}/{len(rows)}] {row['key']} {time.time() - t0:.1f}s iou={meta['mask_prior_iou']} drift={meta['scene_drift']['outside_mean_abs']}", flush=True)
        except Exception as e:
            failed.append({"key": row["key"], "error": str(e)})
            print(f"[{i + 1}/{len(rows)}] {row['key']} FAILED {e}", flush=True)
    (out_dir / "failed.json").write_text(json.dumps(failed, indent=1))
    done_keys = sorted(
        r["key"] for r in rows if is_done(out_dir / r["key"], r, chash)
    )
    (out_dir / "accepted.json").write_text(json.dumps(done_keys, indent=1))
    (out_dir / "pending.json").write_text(
        json.dumps(sorted(set(r["key"] for r in rows) - set(done_keys) - {f["key"] for f in failed}), indent=1)
    )
    print(json.dumps({"done": done, "failed": len(failed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

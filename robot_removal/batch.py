"""Resumable two-pass batch runner: segment all -> edit all.

Two passes keep each model family resident in VRAM (SAM3 hot during pass A,
Qwen hot during pass B) instead of swapping per image.

Per input:
  outputs/<run>/<dataset>/<episode>/
    original.jpg  original_rgb.png  raw_robot_mask.png  expanded_robot_mask.png
    mask_overlay.png  reference_lama.png  inpaint_native.png  composite_native.png
    zerowam_480x320.png  metadata.json  metrics.json  complete.json  attempts/

complete.json is written last after validation; hash-aware skip on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import postprocess, reference, workflow as wf
from .comfy import ComfyClient
from .masks import disk_dilate, mask_file_to_bool, select_robot, validate_mask

WORKFLOW_VERSION = "rr4"


def config_hash(cfg: dict) -> str:
    keep = {k: cfg[k] for k in ("sam3", "qwen", "models", "mask", "output", "reference") if k in cfg}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


class EpisodeError(RuntimeError):
    pass


def _is_complete(ep_dir: Path, row: dict, chash: str) -> bool:
    c = ep_dir / "complete.json"
    if not c.exists():
        return False
    try:
        d = json.loads(c.read_text())
    except json.JSONDecodeError:
        return False
    return (
        d.get("input_sha256") == row["image_sha256"]
        and d.get("config_hash") == chash
        and d.get("workflow_version") == WORKFLOW_VERSION
        and (ep_dir / "zerowam_480x320.png").exists()
    )


def _download_outputs(client: ComfyClient, hist: dict, prefix: str, dest: Path) -> dict[str, list[Path]]:
    got = {"arm": [], "ext": [], "qwen": []}
    for f in client.output_files(hist):
        name = f["filename"]
        if not name.startswith(prefix):
            continue
        kind = "arm" if "_arm_" in name else "ext" if "_ext_" in name else "qwen" if "_qwen_" in name else None
        if kind is None:
            continue
        data = client.view(name, f.get("subfolder", ""), f.get("type", "output"))
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        got[kind].append(p)
    return {k: sorted(v) for k, v in got.items()}


def phase_a(client: ComfyClient, row: dict, cfg: dict, ep_dir: Path, attempt_no: int) -> dict:
    """SAM3 masks -> selection -> dilate -> LaMa reference. Writes .stage/state.json."""
    mask_cfg = cfg["mask"]
    key = row["key"]
    prefix = f"rr_{wf.safe_id(key)}_{attempt_no}"
    src = Path(row["image"])
    stage = ep_dir / ".stage"

    name = client.upload_image(src, name=f"{prefix}_{src.name}", overwrite=True)
    box_px = wf.box_2d_to_px(row.get("robot_box_2d"), row["width"], row["height"])

    t0 = time.time()
    seg_pid = client.submit(wf.build_seg_prompt(name, prefix, cfg, box_px))
    hist = client.wait(seg_pid, timeout=cfg.get("job_timeout_s", 1800))
    seg_s = time.time() - t0
    files = _download_outputs(client, hist, prefix, stage / f"attempt_{attempt_no}")
    if not files["arm"]:
        raise EpisodeError("no sam3 arm masks saved")

    arm = [mask_file_to_bool(p) for p in files["arm"]]
    ext = [mask_file_to_bool(p) for p in files["ext"]]
    raw = select_robot(arm, ext, box_px)
    expanded = disk_dilate(raw, mask_cfg["dilate_px"])
    flags = validate_mask(expanded, mask_cfg["min_coverage"], mask_cfg["max_coverage"])

    original_pil = Image.open(src).convert("RGB")
    t0 = time.time()
    ref_rgb = reference.build_reference(
        original_pil, expanded,
        cfg.get("reference", {}).get("method", "lama"),
        cfg.get("reference", {}).get("device", "cpu"),
    )
    ref_s = time.time() - t0
    Image.fromarray(ref_rgb).save(stage / "reference.png")
    Image.fromarray((raw * 255).astype("uint8")).save(stage / "raw_mask.png")
    Image.fromarray((expanded * 255).astype("uint8")).save(stage / "expanded_mask.png")

    state = {
        "key": key,
        "attempt": attempt_no,
        "input_sha256": row["image_sha256"],
        "seg_pid": seg_pid,
        "seg_s": seg_s,
        "ref_s": ref_s,
        "mask_coverage": float(expanded.mean()),
        "raw_coverage": float(raw.mean()),
        "flags": flags,
        "mask_candidates": {"arm": len(arm), "ext": len(ext)},
    }
    (stage / "state.json.tmp").write_text(json.dumps(state, indent=1))
    (stage / "state.json.tmp").replace(stage / "state.json")
    return state


def phase_b(client: ComfyClient, row: dict, cfg: dict, ep_dir: Path, attempt_no: int) -> dict:
    """Qwen edit on the reference -> composite -> assert -> 480x320 -> artifacts."""
    out_cfg = cfg["output"]
    key = row["key"]
    prefix = f"rr_{wf.safe_id(key)}_{attempt_no}"
    src = Path(row["image"])
    stage = ep_dir / ".stage"
    state = json.loads((stage / "state.json").read_text())

    ref_name = client.upload_image(stage / "reference.png", name=f"{prefix}_ref.png", overwrite=True)
    t0 = time.time()
    edit_pid = client.submit(wf.build_edit_prompt(ref_name, prefix, cfg))
    hist = client.wait(edit_pid, timeout=cfg.get("job_timeout_s", 1800))
    edit_s = time.time() - t0
    qwen_files = []
    for f in client.output_files(hist):
        if f["filename"].startswith(prefix) and "_qwen_" in f["filename"]:
            p = stage / f["filename"]
            p.write_bytes(client.view(f["filename"], f.get("subfolder", "")))
            qwen_files.append(p)
    if not qwen_files:
        raise EpisodeError("no qwen output saved")

    t0 = time.time()
    original_pil = Image.open(src).convert("RGB")
    original = np.asarray(original_pil)
    expanded = mask_file_to_bool(stage / "expanded_mask.png")
    raw = mask_file_to_bool(stage / "raw_mask.png")
    qwen = np.asarray(Image.open(qwen_files[0]).convert("RGB"))
    if qwen.shape != original.shape:
        raise EpisodeError(f"qwen size {qwen.shape[:2]} != original {original.shape[:2]}")
    composite = postprocess.composite(original, qwen, expanded, cfg["mask"].get("soften_px", 0))
    ref_rgb = np.asarray(Image.open(stage / "reference.png").convert("RGB"))

    ep_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ep_dir / "original.jpg")
    postprocess.save_png_atomic(Image.fromarray(original), ep_dir / "original_rgb.png")
    postprocess.save_png_atomic(Image.fromarray((raw * 255).astype("uint8")), ep_dir / "raw_robot_mask.png")
    postprocess.save_png_atomic(Image.fromarray((expanded * 255).astype("uint8")), ep_dir / "expanded_robot_mask.png")
    ov = original.copy()
    ov[expanded] = (0.55 * ov[expanded] + np.array([255, 0, 0]) * 0.45).astype("uint8")
    postprocess.save_png_atomic(Image.fromarray(ov), ep_dir / "mask_overlay.png")
    postprocess.save_png_atomic(Image.fromarray(ref_rgb), ep_dir / "reference_lama.png")
    postprocess.save_png_atomic(Image.fromarray(qwen), ep_dir / "inpaint_native.png")
    postprocess.save_png_atomic(Image.fromarray(composite), ep_dir / "composite_native.png")
    stats = postprocess.assert_preserved(ep_dir / "composite_native.png", original, expanded)

    final, crop = postprocess.zerowam_transform(Image.fromarray(composite), out_cfg["width"], out_cfg["height"])
    postprocess.save_png_atomic(final, ep_dir / "zerowam_480x320.png")
    flags = list(state["flags"])
    flags += postprocess.crop_risk(crop["crop"], crop["resized"], row.get("object_boxes", []), row["width"], row["height"])
    for flag in (postprocess.flat_fill_flag(original, composite, expanded), postprocess.seam_flag(composite, expanded)):
        if flag:
            flags.append(flag)
    post_s = time.time() - t0

    meta = {
        "key": key,
        "input_sha256": row["image_sha256"],
        "config_hash": config_hash(cfg),
        "workflow_version": WORKFLOW_VERSION,
        "robot_box_2d": row.get("robot_box_2d"),
        "mask_coverage": state["mask_coverage"],
        "flags": flags,
        "crop": crop,
        "preservation": stats,
        "prompt": cfg["qwen"]["prompt"],
        "attempt": attempt_no,
        "comfy_prompt_ids": {"segment": state["seg_pid"], "edit": edit_pid},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics = {
        "segment_s": state["seg_s"],
        "reference_s": state["ref_s"],
        "edit_s": edit_s,
        "postprocess_s": post_s,
        "mask_candidates": state["mask_candidates"],
    }
    (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
    (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (ep_dir / "complete.json").write_text(json.dumps({"ok": True, **{k: meta[k] for k in ("input_sha256", "config_hash", "workflow_version", "flags")}}, indent=1))
    shutil.rmtree(stage, ignore_errors=True)
    return {"meta": meta, "metrics": metrics}


def _locked(ep_dir: Path):
    """Context manager: acquire .lock dir, reclaiming stale (dead-pid) locks."""
    lock = ep_dir / ".lock"
    ep_dir.mkdir(parents=True, exist_ok=True)

    class _L:
        ok = False

        def __enter__(self):
            try:
                os.mkdir(lock)
            except FileExistsError:
                pid = (lock / "pid").read_text().strip() if (lock / "pid").exists() else ""
                if pid.isdigit() and Path(f"/proc/{pid}").exists():
                    return self
                shutil.rmtree(lock, ignore_errors=True)
                os.mkdir(lock)
            (lock / "pid").write_text(str(os.getpid()))
            self.ok = True
            return self

        def __exit__(self, *a):
            if self.ok:
                shutil.rmtree(lock, ignore_errors=True)

    return _L()


def _attempt(ep_dir: Path) -> int:
    f = ep_dir / ".attempt"
    n = int(f.read_text()) if f.exists() else 0
    f.write_text(str(n + 1))
    return n


def run(manifest_path: Path, cfg_path: Path, output_root: Path, force: bool = False, limit: int | None = None) -> int:
    cfg = load_config(cfg_path)
    chash = config_hash(cfg)
    rows = load_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    if force:
        for row in rows:
            ep = output_root / row["dataset_id"] / row["episode_id"]
            shutil.rmtree(ep / ".stage", ignore_errors=True)
            (ep / "complete.json").unlink(missing_ok=True)
    client = ComfyClient(cfg["comfy_url"])
    output_root.mkdir(parents=True, exist_ok=True)
    retries = cfg.get("retries", 2)
    t_run = time.time()

    def work(row, phase_fn, tag):
        key = row["key"]
        ep_dir = output_root / row["dataset_id"] / row["episode_id"]
        with _locked(ep_dir) as lk:
            if not lk.ok:
                return ("locked", key)
            err = None
            start = _attempt(ep_dir)
            for a in range(start, start + retries + 1):
                try:
                    out = phase_fn(client, row, cfg, ep_dir, a)
                    return ("ok", key, out)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
            (ep_dir / "failed.json").write_text(json.dumps({"key": key, "phase": tag, "error": err}, indent=1))
            return ("failed", key, err)

    # pass A: segmentation + references
    todo = [r for r in rows if not _is_complete(output_root / r["dataset_id"] / r["episode_id"], r, chash)]
    print(f"pass A: {len(todo)} to segment ({len(rows) - len(todo)} already complete)")
    for i, row in enumerate(todo):
        ep_dir = output_root / row["dataset_id"] / row["episode_id"]
        if (ep_dir / ".stage" / "state.json").exists():
            continue
        res = work(row, phase_a, "segment")
        if res[0] == "ok":
            s = res[2]
            print(f"A[{i + 1}/{len(todo)}] {row['key']}: cov={s['mask_coverage']:.3f} flags={s['flags']} {s['seg_s']:.0f}s")
        else:
            print(f"A[{i + 1}/{len(todo)}] {row['key']}: {res[0]} {res[-1]}", file=sys.stderr)

    # pass B: Qwen edits + composite
    todo2 = [r for r in todo if (output_root / r["dataset_id"] / r["episode_id"] / ".stage/state.json").exists()]
    print(f"pass B: {len(todo2)} to edit")
    for i, row in enumerate(todo2):
        res = work(row, phase_b, "edit")
        if res[0] == "ok":
            m = res[2]["metrics"]
            print(f"B[{i + 1}/{len(todo2)}] {row['key']}: ok flags={res[2]['meta']['flags']} edit={m['edit_s']:.0f}s")
        else:
            print(f"B[{i + 1}/{len(todo2)}] {row['key']}: {res[0]} {res[-1]}", file=sys.stderr)

    results = {"accepted": [], "failed": [], "skipped": []}
    for row in rows:
        ep_dir = output_root / row["dataset_id"] / row["episode_id"]
        if _is_complete(ep_dir, row, chash):
            (results["skipped"] if row not in todo else results["accepted"]).append(row["key"])
        elif (ep_dir / "failed.json").exists():
            results["failed"].append({"key": row["key"], **json.loads((ep_dir / "failed.json").read_text())})
    done = set(results["accepted"]) | set(results["skipped"])
    failed_keys = {f["key"] for f in results["failed"]}
    pending = [r["key"] for r in rows if r["key"] not in done and r["key"] not in failed_keys]
    (output_root / "accepted.json").write_text(json.dumps(results["accepted"], indent=1))
    (output_root / "failed.json").write_text(json.dumps(results["failed"], indent=1))
    (output_root / "pending.json").write_text(json.dumps(pending, indent=1))
    wall = time.time() - t_run
    print(f"done: {len(results['accepted'])} completed this run, {len(results['skipped'])} already done, {len(results['failed'])} failed, {len(pending)} pending; wall={wall:.0f}s")
    return 1 if results["failed"] else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    return run(args.manifest, args.config, args.output_root, args.force, args.limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

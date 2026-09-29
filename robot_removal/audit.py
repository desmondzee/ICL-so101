"""Post-run audit: independent SAM3 pass flagging likely problems.

Metrics (one scalar each, see plan §9):
- residual_robot: fraction of the COMPOSITE covered by robot-like detections.
  Robot pixels inside the mask = regenerated fill; outside = mask missed them.
- object_preservation: for non-actor pair.json entities whose box overlaps the
  mask, re-detect by name on original and composite; metric = min pre/post
  detection IoU. Low = the fill destroyed the object.

Also logs uncovered_px diagnostics (original detections outside mask).

Flags are for review, not proof. Usage:
  python -m robot_removal.audit <run_dir> --manifest <manifest.jsonl> [--limit N] [--sample 0-1]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .comfy import ComfyClient, ComfyError
from .masks import mask_file_to_bool
from .workflow import box_2d_to_px

ROBOT_PROMPT = "robot arm:4, robotic arm:4, robot gripper:4"


def detect_multi(client: ComfyClient, image_path: Path, cfg: dict, texts: list[str], threshold: float = 0.45) -> dict[str, list[np.ndarray]]:
    """One ComfyUI job: per-text SAM3 detections. Returns {text: [bool masks]}.

    Empty detections produce an empty image batch which crashes SaveImage
    (ComfyUI IndexError). On that failure we fall back to one job per text so
    a single empty result doesn't lose the others; a crashed text is empty.
    """
    run = uuid.uuid4().hex[:8]
    name = client.upload_image(image_path, name=f"audit_{run}_{image_path.name}", overwrite=True)
    out: dict[str, list[np.ndarray]] = {t: [] for t in texts}
    try:
        hist = _detect_job(client, name, cfg, texts, threshold, run)
    except ComfyError as e:
        if "index 0 is out of bounds" not in str(e):
            raise
        for i, t in enumerate(texts):
            try:
                hist = _detect_job(client, name, cfg, [t], threshold, f"{run}s{i}")
            except ComfyError as e2:
                if "index 0 is out of bounds" not in str(e2):
                    raise
                continue
            _collect(client, hist, out, run, texts, f"{run}s{i}", {0: i})
        return out
    _collect(client, hist, out, run, texts, run)
    return out


def _detect_job(client: ComfyClient, name: str, cfg: dict, texts: list[str], threshold: float, run: str) -> dict:
    prompt: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cfg["models"]["sam3"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
    }
    for i, text in enumerate(texts):
        prompt[f"3{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": text}}
        prompt[f"5{i}"] = {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["1", 0],
                "image": ["2", 0],
                "conditioning": [f"3{i}", 0],
                "threshold": threshold,
                "refine_iterations": 0,
                "individual_masks": True,
            },
        }
        prompt[f"8{i}"] = {"class_type": "MaskToImage", "inputs": {"mask": [f"5{i}", 0]}}
        prompt[f"9{i}"] = {"class_type": "SaveImage", "inputs": {"images": [f"8{i}", 0], "filename_prefix": f"am{run}_{i}"}}
    return client.wait(client.submit(prompt), timeout=600)


def _collect(client, hist, out, run, texts, prefix_run, idx_map=None):
    for f in client.output_files(hist):
        fn = f["filename"]  # am{run}_{i}_00001_.png
        if not fn.startswith(f"am{prefix_run}_"):
            continue
        local = int(fn[len(f"am{prefix_run}_"):].split("_")[0])
        idx = idx_map.get(local, local) if idx_map else local
        tmp = Path(f"/tmp/audit_{uuid.uuid4().hex}.png")
        tmp.write_bytes(client.view(fn, f.get("subfolder", "")))
        out[texts[idx]].append(mask_file_to_bool(tmp))
        tmp.unlink()


def box_overlap(box_px, mask: np.ndarray) -> float:
    x0, y0, x1, y1 = box_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].sum() / ((y1 - y0) * (x1 - x0)))


def qualifying_entities(row: dict, mask: np.ndarray) -> list[dict]:
    """Non-actor entities with a box substantially inside the mask and smaller
    than ~40% of the frame (excludes background entities like 'table')."""
    w, h = row["width"], row["height"]
    out = []
    for e in row.get("object_boxes", []):
        if not e.get("box_2d"):
            continue
        b = box_2d_to_px(e["box_2d"], w, h)
        if not b:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area > 0.4 * w * h or area < 100:
            continue
        if box_overlap(b, mask) > 0.2:
            out.append({"id": e["id"], "name": e["name"], "box": b})
    return out


def audit_episode(client: ComfyClient, ep: Path, cfg: dict, row: dict) -> dict:
    meta = json.loads((ep / "metadata.json").read_text())
    mask = mask_file_to_bool(ep / "expanded_robot_mask.png")
    entities = qualifying_entities(row, mask)
    texts = [ROBOT_PROMPT] + [e["name"] for e in entities]
    comp_path = ep / "composite_native.png"
    if not comp_path.exists():
        comp_path = ep / "result_native.png"
    orig = detect_multi(client, ep / "original_rgb.png", cfg, texts)
    comp = detect_multi(client, comp_path, cfg, texts)

    orig_robot = np.logical_or.reduce(orig[ROBOT_PROMPT]) if orig[ROBOT_PROMPT] else np.zeros_like(mask)
    comp_robot = np.logical_or.reduce(comp[ROBOT_PROMPT]) if comp[ROBOT_PROMPT] else np.zeros_like(mask)
    uncovered = int((orig_robot & ~ndimage.binary_dilation(mask, iterations=4)).sum())

    per_entity = {}
    for e in entities:
        o = np.logical_or.reduce(orig[e["name"]]) if orig[e["name"]] else None
        c = np.logical_or.reduce(comp[e["name"]]) if comp[e["name"]] else None
        if o is None or not o.any():
            continue  # entity not detectable in original — cannot verify
        iou = float((o & c).sum() / (o | c).sum()) if c is not None and c.any() else 0.0
        per_entity[e["id"]] = round(iou, 3)

    return {
        "key": meta["key"],
        "mask_coverage": round(float(mask.mean()), 4),
        "residual_robot": round(float(comp_robot.mean()), 4),
        "uncovered_px": uncovered,
        "object_preservation": min(per_entity.values()) if per_entity else None,
        "object_ious": per_entity,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path("robot_removal/config.yaml").read_text())
    rows = {r["key"]: r for r in (json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip())}
    client = ComfyClient(cfg["comfy_url"])
    eps = sorted(args.run_dir.glob("*/*/composite_native.png")) or sorted(args.run_dir.glob("*/*/result_native.png"))
    random.seed(args.seed)
    if args.sample < 1.0:
        eps = random.sample(eps, max(1, int(len(eps) * args.sample)))
    if args.limit:
        eps = eps[: args.limit]

    findings = []
    for i, comp in enumerate(eps):
        ep = comp.parent
        key = f"{ep.parent.name}/{ep.name}"
        try:
            entry = audit_episode(client, ep, cfg, rows[key])
        except Exception as e:  # noqa: BLE001
            entry = {"key": key, "error": str(e)}
        findings.append(entry)
        marks = []
        if entry.get("residual_robot", 0) > 0.04:
            marks.append(f"residual_robot:{entry['residual_robot']}")
        if entry.get("uncovered_px", 0) > 1500:
            marks.append(f"uncovered:{entry['uncovered_px']}")
        if entry.get("object_preservation") is not None and entry["object_preservation"] < 0.4:
            marks.append(f"object_preservation:{entry['object_preservation']}")
        if marks:
            print(f"[{i + 1}/{len(eps)}] {key}: {marks}", flush=True)
        if i % 50 == 0:
            print(f"[{i + 1}/{len(eps)}] {key}", flush=True)
    out = args.run_dir / "audit.json"
    out.write_text(json.dumps(findings, indent=1))
    n = sum(1 for f in findings if f.get("residual_robot", 0) > 0.04 or f.get("uncovered_px", 0) > 1500
            or (f.get("object_preservation") is not None and f["object_preservation"] < 0.4))
    print(f"audited {len(findings)}; flagged: {n} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

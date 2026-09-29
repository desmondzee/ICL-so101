"""Post-run audit: independent SAM3 pass flagging likely problems.

- coverage_miss: robot detections on the ORIGINAL that the expanded mask
  left uncovered (>1500 px) — robot pixels the removal mask missed.
- robot_fill: robot detections on the COMPOSITE overlapping the mask
  (>30% of mask area) — the regenerated fill still reads as a robot.

Flags are for review, not proof. Usage:
  python -m robot_removal.audit <run_dir> [--limit N] [--sample 0-1]
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

from .comfy import ComfyClient
from .masks import mask_file_to_bool

ROBOT_PROMPT = "robot arm:4, robotic arm:4, robot gripper:4"


def detect_masks(client: ComfyClient, image_path: Path, cfg: dict, threshold: float = 0.3) -> list[np.ndarray]:
    name = client.upload_image(image_path, name=f"audit_{uuid.uuid4().hex}_{image_path.name}", overwrite=True)
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cfg["models"]["sam3"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ROBOT_PROMPT}},
        "5": {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["1", 0],
                "image": ["2", 0],
                "conditioning": ["3", 0],
                "threshold": threshold,
                "refine_iterations": 0,
                "individual_masks": True,
            },
        },
        "8": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"audit_{uuid.uuid4().hex[:8]}"}},
    }
    hist = client.wait(client.submit(prompt), timeout=600)
    masks = []
    for f in client.output_files(hist):
        if f["filename"].startswith("audit"):
            p = Path(f"/tmp/audit_{uuid.uuid4().hex}.png")
            p.write_bytes(client.view(f["filename"], f.get("subfolder", "")))
            masks.append(mask_file_to_bool(p))
            p.unlink()
    return masks


def audit_episode(client: ComfyClient, ep: Path, cfg: dict) -> dict:
    meta = json.loads((ep / "metadata.json").read_text())
    mask = mask_file_to_bool(ep / "expanded_robot_mask.png")
    orig_ms = detect_masks(client, ep / "original.jpg", cfg)
    comp_ms = detect_masks(client, ep / "composite_native.png", cfg)

    orig_union = np.logical_or.reduce(orig_ms) if orig_ms else np.zeros_like(mask)
    uncovered = int((orig_union & ~ndimage.binary_dilation(mask, iterations=4)).sum())
    comp_overlap = np.logical_or.reduce([m & mask for m in comp_ms]).sum() if comp_ms else 0
    fill_frac = comp_overlap / max(1, int(mask.sum()))

    return {
        "key": meta["key"],
        "orig_detections": len(orig_ms),
        "comp_detections": len(comp_ms),
        "uncovered_px": uncovered,
        "coverage_miss": bool(uncovered > 1500),
        "robot_fill_frac": round(float(fill_frac), 3),
        "robot_fill": bool(fill_frac > 0.3 and mask.sum() > 2000),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path("robot_removal/config.yaml").read_text())
    client = ComfyClient(cfg["comfy_url"])
    eps = sorted(args.run_dir.glob("*/*/composite_native.png"))
    random.seed(args.seed)
    if args.sample < 1.0:
        eps = random.sample(eps, max(1, int(len(eps) * args.sample)))
    if args.limit:
        eps = eps[: args.limit]

    findings = []
    for i, comp in enumerate(eps):
        ep = comp.parent
        try:
            entry = audit_episode(client, ep, cfg)
        except Exception as e:  # noqa: BLE001
            entry = {"key": ep.parent.name + "/" + ep.name, "error": str(e)}
        findings.append(entry)
        marks = [k for k in ("coverage_miss", "robot_fill") if entry.get(k)]
        if marks:
            print(f"[{i + 1}/{len(eps)}] {entry['key']}: {marks} uncovered={entry.get('uncovered_px')} fill={entry.get('robot_fill_frac')}")
    out = args.run_dir / "audit.json"
    out.write_text(json.dumps(findings, indent=1))
    n = sum(1 for f in findings if f.get("coverage_miss") or f.get("robot_fill"))
    print(f"audited {len(findings)}; flagged: {n} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

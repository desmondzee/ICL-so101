#!/usr/bin/env python3
"""v3 experiment: light Gaussian noise on the Qwen reference image.

Diagnoses under test (from full-run sanity inspection):
  - luminance ghosts: LaMa fill keeps a robot-shaped brightness bump that
    cfg=1 Qwen copies pixel-exact -> faint silhouette in the composite
      lerobot__svla_so101_pickplace/episode_037, episode_038
      aiden-li__so101-grabtissue/episode_137
  - regenerated secondary robot: Qwen paints a follower arm back into the
    masked region
      aiden-li__so101-grabtissue/episode_138
  - blotch/seam fills
      aiden-li__so101-open-upper-drawer/episode_003
  - controls: mask-side failures noise cannot fix (floods, misses) and one
    clean episode to check noise does not degrade good outputs
      pouring-liquid/episode_021, pbvr__so101_test002/episode_083,
      Cornito__so101_tea2/episode_094, sattgle__clean2-test/episode_013

Hypothesis: cfg=1 reproduces the reference almost verbatim, so residual
LaMa ghosts/shadows survive. Light noise on the reference marks it as
"imperfect input to be edited", letting the model clean silhouettes and
shadows instead of copying them. Variants perturb the whole reference or
only outside the mask; composites always keep pixels inside the mask only.

Usage: /venv/main/bin/python scripts/v3_test.py [--variants n4,n8,n8out]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_removal import postprocess
from robot_removal import workflow as wf
from robot_removal.batch import load_config
from robot_removal.comfy import ComfyClient
from robot_removal.masks import mask_file_to_bool

RUN = Path("outputs/robot_removal/full")
OUT = Path("outputs/robot_removal/v3test")
EPISODES = [
    "lerobot__svla_so101_pickplace/episode_037",
    "lerobot__svla_so101_pickplace/episode_038",
    "aiden-li__so101-grabtissue/episode_137",
    "aiden-li__so101-grabtissue/episode_138",
    "aiden-li__so101-open-upper-drawer/episode_003",
    "LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_021",
    "pbvr__so101_test002/episode_083",
    "Cornito__so101_tea2/episode_094",
    "sattgle__clean2-test/episode_013",
]
ALL_VARIANTS = ["base", "n4", "n8", "n8out", "n16"]


def variant_ref(ref: np.ndarray, mask: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    sigma = {"base": 0, "n4": 4, "n8": 8, "n8out": 8, "n16": 16}[name]
    noise = rng.normal(0, sigma, ref.shape)
    if name == "n8out":
        noise = noise * (~mask)[..., None]
    return np.clip(ref.astype(np.float32) + noise, 0, 255).astype("uint8")


def run_variant(client, cfg, ep: Path, name: str, ref_img: np.ndarray, original: np.ndarray, mask: np.ndarray, dest: Path):
    ref_path = dest / f"ref_{name}.png"
    Image.fromarray(ref_img).save(ref_path)
    up = client.upload_image(ref_path, name=f"v3_{ep.name}_{name}.png", overwrite=True)
    pid = client.submit(wf.build_edit_prompt(up, f"v3_{ep.name}_{name}", cfg))
    hist = client.wait(pid, timeout=1800)
    qwen_files = [f for f in client.output_files(hist) if "_qwen_" in f["filename"]]
    if not qwen_files:
        raise RuntimeError(f"{ep.name}/{name}: no qwen output")
    qwen = np.asarray(Image.open(__import__("io").BytesIO(client.view(qwen_files[0]["filename"], qwen_files[0].get("subfolder", "")))).convert("RGB"))
    comp = postprocess.composite(original, qwen, mask, cfg["mask"].get("soften_px", 0))
    Image.fromarray(qwen).save(dest / f"qwen_{name}.png")
    Image.fromarray(comp).save(dest / f"comp_{name}.png")
    return qwen, comp


def main() -> int:
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL_VARIANTS
    cfg = load_config(Path("robot_removal/config.yaml"))
    client = ComfyClient()
    OUT.mkdir(parents=True, exist_ok=True)
    for key in EPISODES:
        ep = RUN / key
        dest = OUT / key
        dest.mkdir(parents=True, exist_ok=True)
        original = np.asarray(Image.open(ep / "original_rgb.png").convert("RGB"))
        ref = np.asarray(Image.open(ep / "reference_lama.png").convert("RGB"))
        mask = mask_file_to_bool(ep / "expanded_robot_mask.png")
        rows = []
        for v in variants:
            t0 = time.time()
            qwen, comp = run_variant(client, cfg, ep, v, variant_ref(ref, mask, v, np.random.default_rng(7)), original, mask, dest)
            print(f"{key} {v}: {time.time() - t0:.1f}s", flush=True)
            rows.append(np.concatenate([ref, qwen, comp], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(dest / "sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

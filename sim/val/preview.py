"""Preview images for a recorded validation task: per-episode contact sheets and robot-free first/last pairs.

    uv run python -m sim.val.preview --task sort_blocks
    uv run python -m sim.val.preview --compare   # sim front frame next to real SO-101 front frames

Writes to data/so101_sim_val/previews/.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from sim.val.record import OUT

EXPORT = OUT.parents[0] / "so101_export" / "lerobot"
REAL = ("chirag1701__can", "pbvr__so101_test002", "b3rnd__record-50-episodes", "lerobot__svla_so101_pickplace", "tenkau__so101_color_block")


def video_frame(mp4: Path, t: float) -> Image.Image:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return Image.fromarray(np.frombuffer(raw, np.uint8).reshape(480, 640, 3))


def _label(im, text):
    ImageDraw.Draw(im).text((6, 4), text, fill=(255, 255, 0))
    return im


def contact_sheet(root: Path, episode: int, n: int = 6) -> Image.Image:
    eps = pd.concat(pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet")))
    row = eps[eps.episode_index == episode].iloc[0]
    rows = []
    for cam in ("front", "wrist"):
        key = f"observation.images.{cam}"
        mp4 = root / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
        t0, t1 = row[f"videos/{key}/from_timestamp"], row[f"videos/{key}/to_timestamp"]
        ts = np.linspace(t0, t1 - 0.05, n)
        rows.append([_label(video_frame(mp4, t), f"{cam} t={t - t0:.1f}s") for t in ts])
    sheet = Image.new("RGB", (640 * n, 480 * 2))
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            sheet.paste(im, (640 * c, 480 * r))
    return sheet.resize((320 * n, 480))


def first_last(task: str) -> Image.Image:
    folders = sorted((OUT / "frames" / task).glob("episode_*"))
    sheet = Image.new("RGB", (1280, 480 * len(folders)))
    for i, f in enumerate(folders):
        sheet.paste(_label(Image.open(f / "first.png"), f"{f.name} first (robot hidden)"), (0, 480 * i))
        sheet.paste(_label(Image.open(f / "last.png"), f"{f.name} last (robot hidden)"), (640, 480 * i))
    return sheet.resize((960, 360 * len(folders)))


def compare(sim_front: Image.Image) -> Image.Image:
    ims = [_label(sim_front, "SIM front")]
    for ds in REAL:
        mp4 = EXPORT / ds / "videos/observation.images.front/chunk-000/file-000.mp4"
        ims.append(_label(video_frame(mp4, 3.0), ds))
    sheet = Image.new("RGB", (640 * 3, 480 * 2))
    for i, im in enumerate(ims):
        sheet.paste(im, (640 * (i % 3), 480 * (i // 3)))
    return sheet.resize((1440, 720))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    out = OUT / "previews"
    out.mkdir(parents=True, exist_ok=True)
    if a.task:
        root = OUT / "lerobot" / a.task
        n_eps = len(sorted((OUT / "frames" / a.task).glob("episode_*")))
        for ep in range(n_eps):
            contact_sheet(root, ep).save(out / f"{a.task}_episode_{ep:03d}_sheet.jpg", quality=88)
        first_last(a.task).save(out / f"{a.task}_first_last.jpg", quality=88)
    if a.compare:
        root = OUT / "lerobot" / (a.task or "sort_blocks")
        compare(video_frame(next((root / "videos/observation.images.front").rglob("*.mp4")), 8.0)).save(out / "front_sim_vs_real.jpg", quality=88)
    print(out)


if __name__ == "__main__":
    main()

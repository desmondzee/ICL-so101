"""Contact sheets for visual review: one row per camera and episode, evenly spaced frames."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from curate.io import load

FRAMES = 8
CELL = (240, 180)


def row(video: Path, start: float, length_s: float) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        times = [start + length_s * (i + 0.5) / FRAMES for i in range(FRAMES)]
        for i, t in enumerate(times):
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", f"{tmp}/{i}.jpg"], check=True, stdin=subprocess.DEVNULL)
        return [Image.open(f"{tmp}/{i}.jpg").convert("RGB").resize(CELL) for i in range(FRAMES)]


def sheet(root: Path, out: Path, episodes: list[int] | None = None) -> Path:
    src = load(root)
    fps = src.info["fps"]
    lengths = dict(zip(src.episodes.episode_index, src.episodes.length))
    episodes = episodes or [0, src.info["total_episodes"] // 2]
    rows = []
    for ep in episodes:
        for key, name in src.cameras.items():
            video, start = src.video(key, ep)
            if video.exists():
                rows.append((f"ep {ep} | {name}", row(video, start, lengths[ep] / fps)))
    header = 40
    img = Image.new("RGB", (CELL[0] * FRAMES, header + len(rows) * (CELL[1] + 16)), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{root.name} | {'; '.join(sorted(set(src.tasks.values())))[:200]}", fill="black")
    for r, (label, cells) in enumerate(rows):
        y = header + r * (CELL[1] + 16)
        d.text((6, y), label, fill="black")
        for c, cell in enumerate(cells):
            img.paste(cell, (c * CELL[0], y + 14))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=85)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/qc"))
    args = p.parse_args(argv)
    for root in args.roots:
        name = "__".join(root.parts[-2:])
        try:
            print(sheet(root, args.out / f"{name}.jpg"))
        except Exception as e:
            print(f"{root}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

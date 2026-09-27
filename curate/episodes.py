"""Per-episode checks on a selected dataset: camera swaps by image motion, and review sheets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curate.io import Source, load

SELECTION = Path(__file__).with_name("selection.json")
PER_SHEET = 4
FRONT_FRAMES, WRIST_FRAMES = 6, 3
CELL = (200, 150)


def _frames(video: Path, start: float, length_s: float, n: int, size: tuple[int, int]) -> np.ndarray:
    """n evenly spaced RGB frames, [n, h, w, 3] uint8."""
    w, h = size
    out = []
    for i in range(n):
        t = start + length_s * (i + 0.5) / n
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    return np.stack(out)


def motion(frames: np.ndarray) -> float:
    """Median over pixels of the temporal std of grey level; high for a camera that moves."""
    return float(np.median(frames.mean(-1).std(0)))


def _cams(src: Source, entry: dict) -> tuple[str, str | None]:
    by_name = {v: k for k, v in src.cameras.items()}
    return by_name[entry["front"]], by_name.get(entry["wrist"]) if entry["wrist"] else None


def check(entry: dict, sheets: Path) -> dict:
    src = load(entry["root"])
    front, wrist = _cams(src, entry)
    fps = src.info["fps"]
    tasks = src.tasks
    rows, report = [], {}
    eps = src.episodes.sort_values("episode_index")
    for ep, length in zip(eps.episode_index, eps.length):
        ep, secs = int(ep), length / fps
        v, s = src.video(front, ep)
        f = _frames(v, s, secs, FRONT_FRAMES, CELL)
        rec = {"front_motion": round(motion(f), 1)}
        w = None
        if wrist:
            v, s = src.video(wrist, ep)
            w = _frames(v, s, secs, WRIST_FRAMES, CELL)
            rec["wrist_motion"] = round(motion(w), 1)
            rec["swapped"] = rec["front_motion"] > rec["wrist_motion"]
        report[ep] = rec
        rows.append((ep, f, w))
    task = "; ".join(sorted(set(tasks.values())))[:160]
    sheets.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), PER_SHEET):
        chunk = rows[i : i + PER_SHEET]
        width = CELL[0] * (FRONT_FRAMES + WRIST_FRAMES) + 8
        img = Image.new("RGB", (width, 24 + len(chunk) * (CELL[1] + 18)), "white")
        d = ImageDraw.Draw(img)
        d.text((4, 4), f"{entry['name']} | {task}", fill="black")
        for r, (ep, f, w) in enumerate(chunk):
            y = 24 + r * (CELL[1] + 18)
            d.text((4, y), f"episode {ep}  |  front x{FRONT_FRAMES}   ||   wrist x{WRIST_FRAMES if w is not None else 0}", fill="black")
            for c, fr in enumerate(f):
                img.paste(Image.fromarray(fr), (c * CELL[0], y + 16))
            if w is not None:
                for c, fr in enumerate(w):
                    img.paste(Image.fromarray(fr), ((FRONT_FRAMES + c) * CELL[0] + 8, y + 16))
        img.save(sheets / f"{entry['name']}__{i // PER_SHEET:03d}.jpg", quality=80)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/so101_curated/_episodes"))
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    entries = [e for e in json.loads(SELECTION.read_text()) if not args.only or e["name"] in args.only]

    def run(entry):
        path = args.out / "motion" / f"{entry['name']}.json"
        if path.exists():
            return entry["name"], "cached"
        report = check(entry, args.out / "sheets")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1))
        swapped = sum(r.get("swapped", False) for r in report.values())
        return entry["name"], f"{len(report)} episodes, {swapped} swapped"

    with ThreadPoolExecutor(args.workers) as ex:
        for name, msg in ex.map(run, entries):
            print(f"{name}: {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

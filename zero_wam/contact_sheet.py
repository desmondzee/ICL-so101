"""Tile frames from saved rollout MP4s into contact-sheet PNGs for review.

Usage:
    python -m zero_wam.contact_sheet [paths...]

Each path may be an MP4 or a directory scanned recursively for MP4s.
With no paths, scans outputs/zero_wam. Writes "<stem>.contact.png" next to
each video.

The upstream eval client stacks "Real Observation" on top of a 340 px
"Imagined Video Stream" section that is always an empty placeholder, so by
default only the real-observation strip is tiled (--full to keep it).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

FRAMES_PER_SHEET = 24
CELL_WIDTH = 384
IMAGINED_SECTION = 340  # 40 px title bar + 300 px "Coming soon" row
MIN_CROP_HEIGHT = 120


def probe(video: Path) -> tuple[int, int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "csv=p=0", str(video),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height, frames = (int(v) for v in out.split(","))
    return width, height, frames


def contact_grid(count: int) -> tuple[int, int]:
    """Column count from 4 to 6 that leaves the fewest empty cells."""
    if count <= 6:
        return count, 1
    best_cols, best_pad = 5, math.ceil(count / 5) * 5 - count
    for cols in range(4, min(6, count) + 1):
        pad = math.ceil(count / cols) * cols - count
        if pad < best_pad or (pad == best_pad and cols > best_cols):
            best_cols, best_pad = cols, pad
    return best_cols, math.ceil(count / best_cols)


def write_sheet(video: Path, frames: int, full: bool, output: Path) -> None:
    width, height, total = probe(video)
    picks = min(frames, total)
    # Evenly spaced indices, always including the first and last frame.
    idx = sorted({round(i * (total - 1) / (picks - 1)) for i in range(picks)})
    select = "+".join(f"eq(n\\,{i})" for i in idx)
    crop_h = height - IMAGINED_SECTION
    filters = [f"select='{select}'"]
    if not full and crop_h >= MIN_CROP_HEIGHT:
        filters.append(f"crop={width}:{crop_h}:0:0")
    filters.append(f"scale={CELL_WIDTH}:-1")
    cols, rows = contact_grid(len(idx))
    filters.append(f"tile={cols}x{rows}:padding=4:margin=4:color=white")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video),
            "-vf", ",".join(filters),
            "-frames:v", "1", str(output),
        ],
        check=True,
    )


def iter_videos(paths: list[Path]) -> list[Path]:
    videos = []
    for path in paths:
        if path.is_dir():
            videos.extend(sorted(path.rglob("*.mp4")))
        elif path.is_file():
            videos.append(path)
    return [v for v in videos if not v.stem.endswith(".contact")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("outputs/zero_wam")],
        help="MP4s or directories to scan (default: outputs/zero_wam)",
    )
    parser.add_argument(
        "--frames", type=int, default=FRAMES_PER_SHEET,
        help=f"frames per sheet (default {FRAMES_PER_SHEET})",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="keep the 'Imagined Video Stream' placeholder section",
    )
    args = parser.parse_args()

    videos = iter_videos(args.paths)
    if not videos:
        print("no MP4s found", file=sys.stderr)
        return 1
    for video in videos:
        output = video.with_suffix("").with_name(video.stem + ".contact.png")
        write_sheet(video, args.frames, args.full, output)
        print(f"{video} -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

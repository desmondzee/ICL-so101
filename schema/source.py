"""LeRobot v3 episode access: metadata, gripper-based step segments, frame extraction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

CLOSED = 5.0
HOLD_GAP = 3.0
MIN_HOLD_S = 0.3


@dataclass
class Episode:
    root: Path
    repo: str
    revision: str
    robot: str
    index: int
    fps: float
    length: int
    task: str
    gripper_action: np.ndarray
    gripper_state: np.ndarray
    video_from: dict[str, float]


def fetch(repo: str, root: Path) -> str:
    revision = HfApi().dataset_info(repo).sha
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=root)
    return revision


def load_episode(root: Path, repo: str, revision: str, index: int) -> Episode:
    info = json.loads((root / "meta/info.json").read_text())
    eps = pq.read_table(next((root / "meta/episodes").rglob("*.parquet"))).to_pylist()
    ep = next(e for e in eps if e["episode_index"] == index)
    data = pq.read_table(root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
    rows = data.filter(data["episode_index"].to_numpy() == index)
    grip = info["features"]["action"]["names"].index("gripper.pos")
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    return Episode(
        root=root,
        repo=repo,
        revision=revision,
        robot=info["robot_type"],
        index=index,
        fps=float(info["fps"]),
        length=int(ep["length"]),
        task=ep["tasks"][0],
        gripper_action=np.stack(rows["action"].to_numpy(zero_copy_only=False))[:, grip],
        gripper_state=np.stack(rows["observation.state"].to_numpy(zero_copy_only=False))[:, grip],
        video_from={c: float(ep[f"videos/{c}/from_timestamp"]) for c in cams},
    )


def holds(ep: Episode) -> list[tuple[int, int]]:
    """[grasp, release) runs where the jaws are blocked by an object."""
    held = (ep.gripper_state - ep.gripper_action > HOLD_GAP) & (ep.gripper_action < CLOSED)
    min_run = round(MIN_HOLD_S * ep.fps)
    runs, start = [], None
    for i, h in enumerate(np.append(held, False)):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if runs and start - runs[-1][1] < min_run:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((start, i))
            start = None
    return [r for r in runs if r[1] - r[0] >= min_run]


def segments(ep: Episode, steps: list[str]) -> list[dict]:
    runs = holds(ep)
    if len(runs) != len(steps):
        raise ValueError(f"episode {ep.index}: {len(runs)} grasps for {len(steps)} steps")
    bounds = [0] + [(r[1] + n[0]) // 2 for r, n in zip(runs, runs[1:])] + [ep.length]
    return [
        {"step": s, "start": bounds[i], "end": bounds[i + 1], "grasp": g, "release": r, "source": "gripper"}
        for i, (s, (g, r)) in enumerate(zip(steps, runs))
    ]


def video_path(ep: Episode, camera: str) -> Path:
    return next((ep.root / "videos" / camera).rglob("*.mp4"))


def frame(ep: Episode, camera: str, index: int, out: Path) -> Path:
    t = ep.video_from[camera] + min(index, ep.length - 1) / ep.fps
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", str(video_path(ep, camera)), "-frames:v", "1", "-q:v", "2", str(out)],
        check=True,
    )
    return out


def ref(ep: Episode, camera: str, index: int) -> str:
    return f"{ep.repo}@{ep.revision[:7]}:{camera}#episode={ep.index}&frame={index}"

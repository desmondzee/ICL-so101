"""LeRobot v3 episode access: metadata, gripper holds, step segments from holds, frame extraction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

HOLD_GAP = 1.5
HOLD_GAP_FRAC = 0.05
HOLD_OPEN_FRAC = 0.08
MIN_HOLD_S = 0.5
CYCLE_FRAC = 0.25


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
    videos: dict[str, tuple[Path, float]]


def fetch(repo: str, root: Path) -> str:
    revision = HfApi().dataset_info(repo).sha
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=root)
    return revision


def local_source(root: Path) -> tuple[str, str]:
    """Repo label and revision for a curated dataset, from its curation.json."""
    cur = json.loads((root / "meta/curation.json").read_text())
    src = cur["source"]
    return (f"{src['repo']}/{src['path']}" if "path" in src else src["repo"]), src.get("revision", "local")


def episodes(root: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True)


def load_episode(root: Path, repo: str, revision: str, index: int) -> Episode:
    info = json.loads((root / "meta/info.json").read_text())
    ep = episodes(root).set_index("episode_index").loc[index]
    data = pd.read_parquet(root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
    rows = data[data["episode_index"] == index]
    grip = info["features"]["action"]["names"].index("gripper.pos")
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    videos = {
        c: (root / info["video_path"].format(video_key=c, chunk_index=ep[f"videos/{c}/chunk_index"], file_index=ep[f"videos/{c}/file_index"]),
            float(ep[f"videos/{c}/from_timestamp"]))
        for c in cams
    }
    return Episode(
        root=root, repo=repo, revision=revision, robot=info["robot_type"], index=index, fps=float(info["fps"]),
        length=int(ep["length"]), task=ep["tasks"][0],
        gripper_action=np.stack(rows["action"].to_numpy())[:, grip],
        gripper_state=np.stack(rows["observation.state"].to_numpy())[:, grip],
        videos=videos,
    )


def holds(ep: Episode) -> list[tuple[int, int]]:
    """[grasp, release) runs where the jaws are blocked by an object: the measured gripper stays more open than
    commanded and more open than its own fully closed position. Thresholds scale with the episode's gripper range.
    """
    a, s = ep.gripper_action, ep.gripper_state
    closed, span = float(np.percentile(s, 2)), float(np.percentile(s, 98) - np.percentile(s, 2))
    gap = max(HOLD_GAP, HOLD_GAP_FRAC * span)
    held = (s - a > gap) & (s - closed > max(HOLD_GAP, HOLD_OPEN_FRAC * span))
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


def cycles(ep: Episode) -> list[tuple[int, int]]:
    """[close, open) runs of the commanded gripper: closes by at least CYCLE_FRAC of its range below the last open level.

    Catches grasps where the operator closes only to the object's width, which leave no blocked-jaw signal.
    """
    a = np.convolve(ep.gripper_action, np.ones(5) / 5, mode="same")
    span = float(np.percentile(a, 98) - np.percentile(a, 2))
    if span <= 0:
        return []
    min_run = round(MIN_HOLD_S * ep.fps)
    runs, start, peak = [], None, a[0]
    for i, x in enumerate(np.append(a, np.inf)):
        if start is None:
            peak = max(peak, x)
            if peak - x > CYCLE_FRAC * span:
                start = i
        elif x > a[start] + CYCLE_FRAC * span or i == len(a):
            if i - start >= min_run:
                runs.append((start, i))
            start, peak = None, x
    return runs


def uniform(ep: Episode, n: int) -> list[int]:
    return [round((ep.length - 1) * i / (n - 1)) for i in range(n)]


def segments(ep: Episode, steps: list[str]) -> list[dict]:
    runs = holds(ep)
    if len(runs) != len(steps):
        raise ValueError(f"episode {ep.index}: {len(runs)} grasps for {len(steps)} steps")
    bounds = [0] + [(r[1] + n[0]) // 2 for r, n in zip(runs, runs[1:])] + [ep.length]
    return [
        {"step": s, "start": bounds[i], "end": bounds[i + 1], "grasp": g, "release": r, "source": "gripper"}
        for i, (s, (g, r)) in enumerate(zip(steps, runs))
    ]


def frame(ep: Episode, camera: str, index: int, out: Path) -> Path:
    video, start = ep.videos[camera]
    out.parent.mkdir(parents=True, exist_ok=True)
    for back in range(3):
        t = max(0.0, start + (max(0, min(index, ep.length - 1) - back) - 0.25) / ep.fps)
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)],
            stdin=subprocess.DEVNULL, capture_output=True,
        )
        if r.returncode == 0 and out.exists():
            return out
    raise RuntimeError(f"episode {ep.index}: cannot read frame {index} of {video}")


def ref(ep: Episode, camera: str, index: int) -> str:
    return f"{ep.repo}@{ep.revision[:7]}:{camera}#episode={ep.index}&frame={index}"

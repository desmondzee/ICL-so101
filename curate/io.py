"""Read local LeRobot v2.1 and v3.0 datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Source:
    root: Path
    info: dict
    original: dict
    tasks: dict[int, str]
    episodes: pd.DataFrame

    @property
    def version(self) -> str:
        return self.info["codebase_version"]

    @property
    def cameras(self) -> dict[str, str]:
        """Current video key -> original camera name."""
        cur = [k for k, f in self.info["features"].items() if f["dtype"] == "video"]
        old = [k for k, f in self.original["features"].items() if f["dtype"] == "video"]
        return {c: o.split(".")[-1] for c, o in zip(cur, old)} if len(cur) == len(old) else {c: c.split(".")[-1] for c in cur}

    def frames(self) -> pd.DataFrame:
        files = sorted((self.root / "data").rglob("*.parquet"))
        cols = ["episode_index", "frame_index", "timestamp", "task_index", "action", "observation.state"]
        return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)

    def video(self, key: str, episode: int) -> tuple[Path, float]:
        """Video file and start time of `episode` within it."""
        if self.version.startswith("v2"):
            chunk = episode // self.info["chunks_size"]
            return self.root / self.info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode), 0.0
        row = self.episodes.loc[self.episodes.episode_index == episode].iloc[0]
        path = self.info["video_path"].format(video_key=key, chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"])
        return self.root / path, float(row[f"videos/{key}/from_timestamp"])


def load(root: str | Path) -> Source:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    bak = root / "meta/info.json.bak"
    original = json.loads(bak.read_text()) if bak.exists() else info
    if info["codebase_version"].startswith("v2"):
        tasks = {t["task_index"]: t["task"] for t in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines()) if t}
        episodes = pd.DataFrame([json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()])
    else:
        t = pd.read_parquet(root / "meta/tasks.parquet")
        tasks = {int(i): str(s) for s, i in zip(t.index, t.task_index)}
        episodes = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))], ignore_index=True)
    return Source(root, info, original, tasks, episodes)


def stack(col: pd.Series) -> np.ndarray:
    return np.stack(col.to_numpy()).astype(np.float64)

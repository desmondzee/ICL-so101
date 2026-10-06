"""Shared fixtures for generation/human-review tests: real store episodes, a fake fal client."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np
from PIL import Image

from sim.train import review
from sim.train.model import EpisodeKey, EpisodeManifest, EpisodeState
from sim.train.store import EpisodeStore, sha256_file

CAMERA = {"pos": [0.48, -0.13, 0.38], "lookat": [0.15, -0.02, 0.0], "fovy": 48.0}


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def video_bytes(seconds: float = 5.0, audio: bool = True, size: str = "64x48") -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "v.mp4"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        return out.read_bytes()


def make_episode(root: Path, task: str, seed: int, *, action_order=("block",),
                 action_text=("Pick up the block and put it inside the bowl.",), approve=True) -> EpisodeKey:
    """A robot_approved episode in the real store, then fal_ready.json rebuilt."""
    store = EpisodeStore(root)
    key = EpisodeKey(task, seed)
    directory = store.episode_dir(key)
    directory.mkdir(parents=True)
    Image.fromarray(np.full((48, 64, 3), seed % 200, np.uint8)).save(directory / "first.png")
    Image.fromarray(np.full((48, 64, 3), seed % 200 + 30, np.uint8)).save(directory / "last.png")
    (directory / "robot_front.mp4").write_bytes(video_bytes(1.0, False))
    (directory / "robot_wrist.mp4").write_bytes(video_bytes(1.0, False))
    (directory / "robot_data.parquet").write_bytes(b"not read by generation")
    (directory / "episode.json").write_text(json.dumps({"visual_config": {"front_camera": CAMERA}, "events": []}))
    (directory / review.REQUEST_NAME).write_text(json.dumps({"key": f"{task}/episode_{seed}"}))
    artifacts = {p.name: sha256_file(p) for p in sorted(directory.iterdir())}
    meta = {"task": task, "seed": seed, "family": "pick_place", "instruction": "Put the block inside the bowl.",
            "action_order": list(action_order), "action_text": list(action_text), "frames": 24, "fps": 24}
    store.create_candidate(EpisodeManifest(key, _h(f"c{task}{seed}"), visual_config_hash=_h(f"v{task}{seed}"),
                                           metadata=meta, artifacts=artifacts))
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"artifact_hashes": artifacts})
    store.transition(key, EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED, {"stage": "automated_qa"})
    if approve:
        store.transition(key, EpisodeState.PHYSICS_APPROVED, EpisodeState.ROBOT_APPROVED, {
            "stage": "robot_review", "decision": "accept", "request_sha256": artifacts[review.REQUEST_NAME],
            "judge": {"reviewer": {"label": "robot-judge:0", "model": "opus", "role": "judge"}},
            "verify": {"reviewer": {"label": "robot-verify:0.0", "model": "opus", "role": "verifier"}},
            "problems": [], "artifact_hashes": dict(artifacts)})
    review.build_fal_ready(root)
    return key


class FakeFal:
    """In-memory fal queue. Never touches the network."""

    def __init__(self, *, submit_error: BaseException | None = None, fail_status: bool = False,
                 statuses: tuple[str, ...] = ("IN_QUEUE", "COMPLETED"), video: bytes | None = None):
        self.submit_error, self.fail_status, self.statuses = submit_error, fail_status, statuses
        self.video = video if video is not None else video_bytes()
        self.submits: list[dict] = []
        self.polls: dict[str, int] = {}
        self.lock = threading.Lock()

    def submit(self, endpoint, payload):
        with self.lock:
            self.submits.append({"endpoint": endpoint, **payload})
            n = len(self.submits)
        if self.submit_error is not None:
            raise self.submit_error
        rid = f"req-{n}"
        return {"request_id": rid, "status_url": f"fake://{rid}/status", "response_url": f"fake://{rid}"}

    def status(self, handle):
        from sim.train.generate import RequestFailed
        if self.fail_status:
            raise RequestFailed("model error")
        with self.lock:
            i = self.polls.get(handle["request_id"], 0)
            self.polls[handle["request_id"]] = i + 1
        return self.statuses[min(i, len(self.statuses) - 1)]

    def result(self, handle):
        return {"video": {"url": f"fake://{handle['request_id']}/video.mp4"}}

    def download(self, url):
        return self.video

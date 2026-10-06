"""Resumable, immutable robot-episode recorder for the simulated training corpus.

Record N automated-QA-approved episodes per qualified task (resumable; reruns skip
published seeds and never touch their bytes):

    python -m sim.train.record --tasks block_in_basket --episodes 10 \\
        [--root data/so101_sim_train_v1] [--qualification ROOT/qualification] \\
        [--seed-start 100000] [--max-attempts 40] [--validation-index .../index.json]
    python -m sim.train.record --verify-all [--root ...]     # re-hash + decode every candidate
    python -m sim.train.record --clean-staging [--root ...]  # drop unpublished crash leftovers

Only tasks admitted by ``sim.train.tasks.qualify.require_qualified_task`` are
recorded. Each episode is rendered into ``staging/<task>/<seed>-<uuid>``; physics
QA, media verification (decode, frame counts, timestamps), review evidence and
hashing happen there; the directory is then atomically renamed to
``candidates/<task>/episode_<seed>`` and registered in the store (manifest with
``config_hash`` and ``visual_config_hash``). Workflow state then advances
deterministically: candidate -> recorded -> physics_approved, or -> rejected.

Episode directory (training format identical to data/so101_sim_val_v2):
    lerobot/                 single-episode LeRobot v3 dataset (front + wrist AV1 640x480 @ 30 fps,
                             action / observation.state in LeRobot degrees with gripper 0-100,
                             action.ee / observation.state.ee from curate.robot.Kinematics)
    robot_front.mp4, robot_wrist.mp4   H.264 yuv420p copies (the v2 per-pair layout), reviewed frames
    robot_data.parquet       the v2 per-pair robot table
    first.png, last.png      robot-free front renders under the frozen visual configuration
    telemetry.npz, physics.json, policy.json   substep telemetry and strict physics audit
    episode.json, media.json  provenance (visual config, poses, oracle events) and media checks
    review/                  dense front/wrist contact sheets + frames.json
    robot_review_request.json  only when automated QA approved the episode
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable
import uuid

import fcntl
import numpy as np

from .model import EpisodeKey, EpisodeManifest, EpisodeRecord, EpisodeState
from .review import REQUEST_NAME, build_sheets, key_name, physics_summary, write_request
from .store import (EpisodeStore, StoreCorruption, _control_path, _lock, atomic_write_json,
                    sha256_file)

ROOT = Path(__file__).resolve().parents[2] / "data" / "so101_sim_train_v1"
DEFAULT_SEED_START = 100_000
MAX_FRAMES = 3600
MAX_VISUAL_RESAMPLES = 8
FPS = 30
PUBLISH_NAME = "publish.json"
VIDEO_NAMES = ("robot_front.mp4", "robot_wrist.mp4")


class RolloutFailed(RuntimeError):
    """A seed produced no publishable recording (visual screening, oracle runtime, frame budget)."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail or {}


@dataclass(frozen=True)
class Admission:
    """Proof that require_qualified_task admitted the task. Build only with ``admit``."""
    task: Any
    report_sha256: str
    source_hash: str


def admit(name: str, qualification_root: str | Path, *, require: Callable | None = None) -> Admission:
    if require is None:
        from .tasks.qualify import require_qualified_task as require
    task = require(name, qualification_root)
    path = Path(qualification_root) / f"{name}.json"
    report = json.loads(path.read_text())
    return Admission(task, sha256_file(path), report["source_hash"])


@dataclass
class Rollout:
    """What a backend left in the staging directory."""
    config_hash: str
    visual_config_hash: str
    frames: int
    width: int
    height: int
    extra: dict = field(default_factory=dict)


def _checkpoint(name: str) -> None:
    """Crash-injection point for tests; a no-op in production."""


def _recorder_hash() -> str:
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("record.py", "review.py", "store.py"):
        digest.update(name.encode())
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


# ----- media ---------------------------------------------------------------------------------------------------
class VideoEncoder:
    """Stream RGB frames into a deterministic H.264 file (no audio, no metadata)."""

    def __init__(self, path: Path, width: int, height: int, fps: int = FPS):
        self.path, self.frames = path, 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
             "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "4", "-map_metadata", "-1",
             "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def verify_media(directory: Path, frames: int, width: int, height: int, fps: int = FPS) -> dict:
    """Every video decodes with the exact frame count/size/rate; tables align with frame indices."""
    import pandas as pd

    report = {"frames": frames, "width": width, "height": height, "fps": fps, "videos": {}, "tables": {}}
    videos = sorted(p for p in directory.rglob("*.mp4"))
    if not {v.name for v in videos} >= set(VIDEO_NAMES):
        raise ValueError("missing robot videos")
    for video in videos:
        streams = probe(video)["streams"]
        if len(streams) != 1 or streams[0]["codec_type"] != "video":
            raise ValueError(f"{video.name}: expected exactly one video stream")
        s = streams[0]
        if (int(s["nb_read_frames"]), s["width"], s["height"], s["r_frame_rate"]) != (frames, width, height, f"{fps}/1"):
            raise ValueError(f"{video.relative_to(directory)}: decoded {s['nb_read_frames']} frames "
                             f"{s['width']}x{s['height']}@{s['r_frame_rate']}, expected {frames} {width}x{height}@{fps}")
        report["videos"][str(video.relative_to(directory))] = {k: s[k] for k in ("codec_name", "pix_fmt")} | {
            "frames": int(s["nb_read_frames"])}
    tables = [directory / "robot_data.parquet", *sorted((directory / "lerobot" / "data").glob("*/*.parquet"))]
    for table in tables:
        if not table.exists():
            raise ValueError(f"missing table {table.name}")
        data = pd.read_parquet(table)
        index = np.arange(frames)
        if (len(data) != frames or not np.array_equal(data["frame_index"].to_numpy(), index)
                or not np.allclose(data["timestamp"].to_numpy(), index / fps, atol=1e-4)):
            raise ValueError(f"{table.relative_to(directory)}: frame indices/timestamps do not match {frames} frames")
        for column in ("action", "observation.state", "action.ee", "observation.state.ee"):
            values = np.stack(data[column].to_numpy())
            if values.shape != (frames, 7 if column.endswith(".ee") else 6) or not np.isfinite(values).all():
                raise ValueError(f"{table.name}: invalid {column}")
        report["tables"][str(table.relative_to(directory))] = {"rows": len(data)}
    return report


# ----- hashing / publication -------------------------------------------------------------------------------------
_STORE_LOCKS = (Path(".manifest.json.lock"), Path(".state.json.lock"))


def write_json(path: Path, value: Any) -> None:
    """Deterministic, exclusive JSON write for staging (no lock side files; the directory publishes atomically)."""
    with Path(path).open("xb") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def hash_tree(directory: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_dir() or relative == Path(PUBLISH_NAME) or relative in _STORE_LOCKS:
            continue
        if path.is_symlink() or _control_path(relative):
            raise ValueError(f"unexpected file in episode: {relative}")
        hashes[relative.as_posix()] = sha256_file(path)
    return hashes


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def automated_qa(directory: Path) -> dict:
    physics = json.loads((directory / "physics.json").read_text())
    episode = json.loads((directory / "episode.json").read_text())
    qa = {"physics_accepted": physics.get("accepted") is True,
          "failed_physics_checks": physics_summary(physics)["failed_checks"],
          "terminal_visibility": episode.get("terminal_screen", {}).get("ok") is True,
          "final_task_success": episode.get("final_task_success") is True}
    qa["accepted"] = qa["physics_accepted"] and qa["terminal_visibility"] and qa["final_task_success"]
    return qa


def _advance(store: EpisodeStore, record: EpisodeRecord) -> EpisodeRecord:
    """Deterministic automated transitions; replays are idempotent in the store."""
    key, manifest = record.manifest.key, record.manifest
    if record.state is EpisodeState.CANDIDATE:
        record = store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED,
                                  {"stage": "recorded", "frames": manifest.metadata["frames"],
                                   "artifact_hashes": dict(manifest.artifacts)})
        _checkpoint("after_recorded")
    if record.state is EpisodeState.RECORDED:
        qa = manifest.metadata["automated_qa"]
        evidence = {"stage": "automated_qa", **qa, "artifact_hashes": {
            name: manifest.artifacts[name] for name in ("physics.json", "policy.json", "telemetry.npz", "episode.json")}}
        target = EpisodeState.PHYSICS_APPROVED if qa["accepted"] else EpisodeState.REJECTED
        record = store.transition(key, EpisodeState.RECORDED, target, evidence)
    return record


def _publish_from_disk(store: EpisodeStore, key: EpisodeKey) -> EpisodeRecord:
    """Finish a publication interrupted after the atomic rename."""
    directory = store.episode_dir(key)
    path = directory / PUBLISH_NAME
    if not path.is_file():
        raise StoreCorruption(f"{directory} exists without manifest or publish record; inspect it manually")
    document = json.loads(path.read_text())
    manifest = EpisodeManifest(EpisodeKey(**document["key"]), document["config_hash"], document["metadata"],
                               document["artifacts"], visual_config_hash=document["visual_config_hash"])
    if manifest.key != key or hash_tree(directory) != manifest.artifacts:
        raise StoreCorruption(f"{directory}: published files do not match their publish record")
    return store.create_candidate(manifest)


def _attempt_path(store: EpisodeStore, key: EpisodeKey) -> Path:
    return store.root / "attempts" / key.task / f"episode_{key.seed}.json"


def record_episode(store: EpisodeStore, admission: Admission, seed: int, backend) -> EpisodeRecord | None:
    """Publish one episode, or resume/return the existing one. Returns None for a failed seed."""
    if not isinstance(admission, Admission):
        raise TypeError("recording requires an Admission from admit()")
    task = admission.task
    key = EpisodeKey(task.name, seed)
    with _lock(store.root / "locks" / key.task / f"episode_{seed}.lock"):
        if store.manifest_path(key).exists():
            return _advance(store, store.load(key))
        if store.episode_dir(key).exists():
            return _advance(store, _publish_from_disk(store, key))
        attempt = _attempt_path(store, key)
        if attempt.exists():
            return None
        staging = store.root / "staging" / key.task / f"{seed}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        try:
            try:
                rollout = backend.record(task, seed, staging, store)
            except RolloutFailed as exc:
                atomic_write_json(attempt, {"schema_version": 1, "key": asdict(key), "reason": exc.reason,
                                            "detail": exc.detail}, overwrite=False)
                return None
            media = verify_media(staging, rollout.frames, rollout.width, rollout.height)
            write_json(staging / "media.json", media)
            episode = json.loads((staging / "episode.json").read_text())
            build_sheets(staging, rollout.frames, episode["events"])
            qa = automated_qa(staging)
            metadata = {
                "schema_version": 1, "task": task.name, "seed": seed, "family": task.family,
                "instruction": task.instruction, "action_order": list(task.action_order),
                "action_text": list(task.action_descriptions()), "frames": rollout.frames, "fps": FPS,
                "width": rollout.width, "height": rollout.height, "automated_qa": qa,
                "qualification": {"report_sha256": admission.report_sha256, "source_hash": admission.source_hash},
                "recorder_hash": _recorder_hash(), **rollout.extra,
            }
            if qa["accepted"]:
                write_request(staging, metadata, hash_tree(staging))
            manifest = EpisodeManifest(key, rollout.config_hash, metadata, hash_tree(staging),
                                       visual_config_hash=rollout.visual_config_hash)
            write_json(staging / PUBLISH_NAME, {"schema_version": 1, **asdict(manifest)})
            _fsync_tree(staging)
            _checkpoint("before_rename")
            target = store.episode_dir(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, target)  # atomic; fails rather than replacing an existing directory
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        _checkpoint("after_rename")
        record = store.create_candidate(manifest)
        _checkpoint("after_candidate")
        return _advance(store, record)


def record_task(store: EpisodeStore, admission: Admission, episodes: int, backend, *,
                seed_start: int = DEFAULT_SEED_START, max_attempts: int | None = None,
                exclude_seeds: set[int] = frozenset(), log=print) -> dict:
    """Walk deterministic seeds until ``episodes`` non-rejected episodes exist (resumable)."""
    task = admission.task
    excluded = set(exclude_seeds) | set(task.qualification_seeds)
    max_attempts = max_attempts or 3 * episodes + 5
    good, tried, seed, summary = 0, 0, seed_start, {"approved": [], "rejected": [], "failed": []}
    while good < episodes and tried < max_attempts:
        if seed in excluded:
            seed += 1
            continue
        record = record_episode(store, admission, seed, backend)
        tried += 1
        if record is None:
            summary["failed"].append(seed)
        elif record.state is EpisodeState.REJECTED:
            summary["rejected"].append(seed)
        else:
            summary["approved"].append(seed)
            good += 1
        log(json.dumps({"task": task.name, "seed": seed, "state": record.state.value if record else "failed",
                        "good": good, "target": episodes}), flush=True)
        seed += 1
    return summary


def clean_staging(root: Path) -> list[str]:
    """Remove unpublished staging directories whose episode lock is free (crash leftovers)."""
    removed = []
    for directory in sorted((Path(root) / "staging").glob("*/*-*")):
        seed = directory.name.split("-")[0]
        lock = Path(root) / "locks" / directory.parent.name / f"episode_{seed}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                shutil.rmtree(directory)
                removed.append(str(directory))
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return removed


def verify_all(root: Path) -> dict:
    """Re-hash (via store.load) and decode every candidate."""
    store = EpisodeStore(root)
    counts: dict[str, int] = {}
    for path in sorted((Path(root) / "candidates").glob("*/episode_*/manifest.json")):
        record = store.load(EpisodeKey(**json.loads(path.read_text())["key"]))
        meta = record.manifest.metadata
        verify_media(path.parent, meta["frames"], meta["width"], meta["height"])
        counts[record.state.value] = counts.get(record.state.value, 0) + 1
    return counts


# ----- MuJoCo backend --------------------------------------------------------------------------------------------
def _versions() -> dict:
    out = {}
    for name in ("mujoco", "so101-nexus", "lerobot", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _log_events(entries: list, frame: int) -> list[dict]:
    events = []
    for entry in entries:
        entry = list(entry)
        if len(entry) == 4 and entry[1] == "grasp":
            events.append({"frame": frame, "kind": "grasp_attempt", "label": f"{entry[0]} ok={bool(entry[3])}"})
        elif len(entry) == 3 and entry[1] == "dropped":
            events.append({"frame": frame, "kind": "dropped", "label": str(entry[0])})
        else:
            events.append({"frame": frame, "kind": "skill", "label": str(entry[0]),
                           "detail": [x if isinstance(x, (int, float, str, bool)) else str(x) for x in entry[1:]]})
    return events


def _gripper_events(gripper: np.ndarray) -> list[dict]:
    """Starts of commanded gripper ramps (LeRobot units: larger = more open)."""
    moving = np.abs(np.diff(gripper)) > 1e-4
    events = []
    for i in range(len(moving)):
        if moving[i] and (i == 0 or not moving[i - 1]):
            events.append({"frame": i + 1, "kind": "gripper_open" if gripper[i + 1] > gripper[i] else "gripper_close",
                           "label": ""})
    return events


def _telemetry_events(telemetry, report, task_objects, n_frames) -> list[dict]:
    a, m = telemetry.arrays, telemetry.manifest
    events = []
    ends = np.flatnonzero(a["frame_end"])
    idx = [m["body_names"].index(name) for name in task_objects]
    grasp = a["grasp"][ends][:, idx]
    previous = np.zeros(len(idx), dtype=bool)
    for row, state in zip(ends, grasp):
        frame = min(int(a["frame"][row]) + 1, n_frames - 1)
        for k, name in enumerate(task_objects):
            if state[k] != previous[k]:
                events.append({"frame": frame, "kind": "grasp" if state[k] else "release", "label": name})
        previous = state
    worst: dict[str, dict] = {}
    for violation in report.violations:
        if violation.get("timestamp") is None:
            continue
        best = worst.get(violation["check"])
        measure = violation.get("measured")
        if best is None or (isinstance(measure, (int, float)) and measure > (best.get("measured") or -np.inf)):
            worst[violation["check"]] = violation
    for check, violation in sorted(worst.items()):
        row = int(np.argmin(np.abs(a["time"] - violation["timestamp"])))
        frame = min(max(int(a["frame"][row]) + 1, 0), n_frames - 1)
        events.append({"frame": frame, "kind": "violation", "label": check,
                       "detail": {k: violation[k] for k in ("measured", "limit", "bodies") if k in violation}})
    return events


class SimBackend:
    """Render one oracle rollout with telemetry and the training-format streams."""

    def __init__(self, *, max_resamples: int = MAX_VISUAL_RESAMPLES):
        self.max_resamples = max_resamples

    @staticmethod
    def _known_visual_hashes(store: EpisodeStore) -> set[str]:
        # Manifest bytes are validated by store.load elsewhere; reading the field avoids re-hashing every video.
        return {json.loads(p.read_text())["visual_config_hash"]
                for p in (store.root / "candidates").glob("*/episode_*/manifest.json")}

    @staticmethod
    def _goal_regions(task, env):
        hook = getattr(task, "goal_regions", None)
        return hook(env) if callable(hook) else env.goal_regions()

    @staticmethod
    def _variation_policy(task):
        from .variation import DEFAULT_POLICY
        hook = getattr(task, "variation_policy", None)
        return hook() if callable(hook) else DEFAULT_POLICY

    def _screen(self, task, seed, env_cls, store):
        from .variation import ScreeningReport, sample_visual_config, save_screening_report, screen_visual_config

        known, attempts = self._known_visual_hashes(store), []
        key = EpisodeKey(task.name, seed)
        policy = self._variation_policy(task)
        for index in range(self.max_resamples):
            config = sample_visual_config(task.name, seed, policy, resample_index=index)
            env = env_cls(render_images=True, visual_config=config)
            try:
                obs, _ = env.reset(seed=seed)
                report = screen_visual_config(config, env, task.task_objects, self._goal_regions(task, env),
                                              policy=policy)
                if config.config_hash in known:
                    report = ScreeningReport(False, (*report.reasons, "duplicate_config"), report.object_pixels,
                                             report.goal_visible_fraction, report.next_resample_index)
                save_screening_report(store, key, config, report)
                attempts.append({"resample_index": index, **report.to_dict()})
                if report.accepted:
                    return config, env, obs, attempts
            except BaseException:
                env.close()
                raise
            env.close()
        raise RolloutFailed("visual_screening_exhausted", {"attempts": attempts})

    def record(self, task, seed, staging: Path, store: EpisodeStore) -> Rollout:
        import pandas as pd
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from curate.robot import Kinematics
        from sim.val.env import HEIGHT, WIDTH
        from sim.val.record import FEATURES, with_ee
        from .physics import audit_trajectory
        from .telemetry import TelemetryCollector
        from .variation import episode_config_hash, screen_visual_config

        env_cls, oracle_cls = task.load_classes()
        config, env, obs, screening = self._screen(task, seed, env_cls, store)
        encoders = []
        try:
            poses_start = env.object_poses()
            identity = episode_config_hash(config, poses_start)
            policy = task.physics_policy(env)
            Image.fromarray(env.render_scene_without_robot("front")).save(staging / "first.png")
            collector = TelemetryCollector(env, config_hash=identity, visual_config_hash=config.config_hash)
            encoders = [VideoEncoder(staging / name, WIDTH, HEIGHT) for name in VIDEO_NAMES]
            dataset = LeRobotDataset.create(repo_id=f"local/so101_sim_train_{task.name}", fps=FPS, features=FEATURES,
                                            root=staging / "lerobot", robot_type="so101_follower", use_videos=True)
            kin = Kinematics()
            oracle = oracle_cls(env, np.random.default_rng(seed))
            states, actions, events, logged = [], [], [], 0
            for i, action in enumerate(oracle.actions()):
                if i >= MAX_FRAMES:
                    raise RolloutFailed("frame_budget_exceeded", {"limit": MAX_FRAMES})
                events += _log_events(oracle.log[logged:], i)
                logged = len(oracle.log)
                state, target = env.lerobot_joints(obs["state"]), env.lerobot_joints(action)
                states.append(state)
                actions.append(target)
                encoders[0].write(obs["front"])
                encoders[1].write(obs["wrist"])
                dataset.add_frame({
                    "action": target.astype(np.float32), "observation.state": state.astype(np.float32),
                    "action.ee": with_ee(kin, target[None])[0], "observation.state.ee": with_ee(kin, state[None])[0],
                    "observation.images.front": obs["front"], "observation.images.wrist": obs["wrist"],
                    "task": task.instruction})
                obs, _, _, _, _ = env.step(action, substep_observer=collector)
            n = len(actions)
            if n < 2:
                raise RolloutFailed("empty_rollout")
            events += _log_events(oracle.log[logged:], n - 1)
            for encoder in encoders:
                encoder.close()
            dataset.save_episode(parallel_encoding=False)  # no process pool: callable from any entry point
            dataset.finalize()
            shutil.rmtree(staging / "lerobot" / "images", ignore_errors=True)

            Image.fromarray(env.render_scene_without_robot("front")).save(staging / "last.png")
            final_success = bool(env.success())
            poses_end = env.object_poses()
            terminal = screen_visual_config(config, env, task.task_objects, self._goal_regions(task, env),
                                            policy=self._variation_policy(task))
            # Goal surfaces are covered by the placed object at the end, so only visibility counts here.
            blocking = [r for r in terminal.reasons if r in ("object_outside_frame", "object_too_small", "config_mismatch")]

            telemetry = collector.finish()
            telemetry.save(staging / "telemetry.npz")
            report = audit_trajectory(telemetry, policy)
            write_json(staging / "physics.json", json.loads(report.to_json()))
            write_json(staging / "policy.json", policy.to_dict())

            states, actions = np.asarray(states), np.asarray(actions)
            table = pd.DataFrame({
                "action": list(actions.astype(np.float32)), "observation.state": list(states.astype(np.float32)),
                "action.ee": list(with_ee(kin, actions)), "observation.state.ee": list(with_ee(kin, states)),
                "timestamp": (np.arange(n) / FPS).astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.zeros(n, dtype=np.int64), "index": np.arange(n, dtype=np.int64),
                "task_index": np.zeros(n, dtype=np.int64)})
            table.to_parquet(staging / "robot_data.parquet", index=False)

            events += _gripper_events(actions[:, 5])
            events += _telemetry_events(telemetry, report, task.task_objects, n)
            events.sort(key=lambda e: (e["frame"], e["kind"], e.get("label", "")))
            write_json(staging / "episode.json", {
                "schema_version": 1, "task": asdict(task), "seed": seed, "instruction": task.instruction,
                "visual_config": config.to_dict(), "config_hash": identity, "visual_config_hash": config.config_hash,
                "object_poses_start": poses_start, "object_poses_end": poses_end,
                "distractors": list(env.active_distractors), "events": events,
                "oracle_log": [[x if isinstance(x, (int, float, str, bool)) else str(x) for x in e] for e in oracle.log],
                "oracle_limited_frames": int(getattr(oracle, "limited", 0)), "frames": n,
                "final_task_success": final_success, "screening": screening,
                "terminal_screen": {"ok": not blocking, **terminal.to_dict()},
                "cameras": {"front": "frozen visual_config.front_camera", "wrist": "fixed attachment transform"},
                "versions": _versions(),
            })
        except BaseException:
            for encoder in encoders:
                encoder.kill()
            raise
        finally:
            env.close()
        return Rollout(identity, config.config_hash, n, WIDTH, HEIGHT,
                       {"arena": config.arena, "versions": _versions()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--qualification", type=Path, help="default: ROOT/qualification")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episodes", type=int, default=3, help="target non-rejected episodes per task")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--validation-index", type=Path, help="exclude its episode seeds")
    parser.add_argument("--verify-all", action="store_true")
    parser.add_argument("--clean-staging", action="store_true")
    args = parser.parse_args(argv)
    if any(part.startswith("so101_sim_val") for part in args.root.resolve().parts):
        parser.error("refusing to write under a validation dataset root")
    if args.verify_all:
        print(json.dumps(verify_all(args.root)))
        return 0
    if args.clean_staging:
        print(json.dumps(clean_staging(args.root), indent=1))
        return 0
    if not args.tasks:
        parser.error("--tasks is required")
    exclude = set()
    if args.validation_index:
        exclude = {row["seed"] for row in json.loads(args.validation_index.read_text())["episodes"]}
    store = EpisodeStore(args.root)
    results = {}
    for name in args.tasks:
        admission = admit(name, args.qualification or args.root / "qualification")
        results[name] = record_task(store, admission, args.episodes, SimBackend(), seed_start=args.seed_start,
                                    max_attempts=args.max_attempts, exclude_seeds=exclude)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

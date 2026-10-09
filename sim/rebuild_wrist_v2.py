"""Staged wrist-video rebuilder: replay released action rows, re-encode clean wrist streams.

The v1 wrist videos include the ``camera_mount`` body's visual mesh
(``wrist_roll_follower_so101_camera_mount``): the wrist camera sits inside its own mount, so the
mount's housing clips the frame into a circular aperture. The mesh geom is visual-only
(contype/conaffinity 0), so changing how it renders changes nothing physical.

The transform sets the wrist camera's absolute near plane to ``TARGET_NEAR_METERS`` (0.02425 m)
via ``model.vis.map.znear = TARGET_NEAR_METERS / model.stat.extent``. The near plane clips the
camera-adjacent mount geometry out of direct view while the mount stays in the scene — it still
occupies its slot in the model, still renders where it is beyond the near plane, and still casts
its shadow. Measured directly-rendered mount pixels are zero at this distance. No geom group is
changed anywhere.

Per episode this writes ``robot_wrist.mp4`` (H.264 640x480@30, yuv420p, h264_nvenc when available
with libx264 fallback) plus ``wrist_v2.json`` (provenance + every input/output hash a resume needs).
Outputs are written to a temp sibling and renamed, so a completed valid output is never overwritten
and an interrupted build leaves only its own partial file.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
from so101_nexus import get_so101_mujoco_model_path

TRANSFORM_VERSION = "absolute-near-plane-v3"

CAMERA_MOUNT_BODY = "camera_mount"
CAMERA_MOUNT_MESH = "wrist_roll_follower_so101_camera_mount"
ROBOT_VISUAL_GROUP = 2          # robot visual geoms (sim/val/scene.py contract)
# Absolute camera near distance in meters: the smallest tested 0.25 mm-grid value with zero
# directly rendered mount pixels (29 px at 0.02375 m, 2 at 0.02400 m, 0 at >= 0.02425 m).
TARGET_NEAR_METERS = 0.02425
WRIST_CAMERA = "wrist_cam"

WIDTH, HEIGHT, FPS = 640, 480, 30
VIDEO_NAME = "robot_wrist.mp4"
META_NAME = "wrist_v2.json"
SHEET_NAME = "robot_wrist_sheet.jpg"
PARQUET_NAME = "robot_data.parquet"

TRAIN_SOURCE_DIR = "sim_train_v1"
VAL_SOURCE_DIR = "sim_val_v2"
TRAIN_META_FILES = ("episode.json", "source.json")
VAL_META_FILES = ("source.json",)
OUTPUT_DIRNAME = "train_v2"
VAL_OUTPUT_DIRNAME = "val_v3"
VAL_META_NAME = "wrist_v3.json"

PILOT_VAL_EPISODE = ("sort_blocks", "episode_005")
PILOT_TRAIN_EPISODE = ("push_cube_into_tape_square", "episode_005")

# ----- helpers ---------------------------------------------------------------------------------


def _sha256_file(path) -> str:
    from sim.train.store import sha256_file

    return sha256_file(path)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----- model seam --------------------------------------------------------------------------------


def camera_mount_visual_geom(model) -> tuple[int, str]:
    """The unique group-2, contype-0 mesh geom on body ``camera_mount``.

    Fails closed: zero or multiple candidate geoms, or a candidate whose mesh is not
    ``wrist_roll_follower_so101_camera_mount``, all raise ``ValueError`` rather than guess.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_MOUNT_BODY)
    if body_id < 0:
        raise ValueError(f"model has no body {CAMERA_MOUNT_BODY!r}")
    candidates = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[g] == 0
        and model.geom_conaffinity[g] == 0
        and int(model.geom_group[g]) == ROBOT_VISUAL_GROUP
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one group-{ROBOT_VISUAL_GROUP} visual mesh geom on body "
            f"{CAMERA_MOUNT_BODY!r}, found {len(candidates)}: {candidates}"
        )
    geom_id = candidates[0]
    mesh_id = int(model.geom_dataid[geom_id])
    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    if mesh_name != CAMERA_MOUNT_MESH:
        raise ValueError(
            f"{CAMERA_MOUNT_BODY} visual geom {geom_id} uses mesh {mesh_name!r}, "
            f"expected {CAMERA_MOUNT_MESH!r}"
        )
    return geom_id, mesh_name


def configure_clean_wrist(model) -> dict:
    """Set the wrist camera's absolute near plane to ``TARGET_NEAR_METERS`` and return provenance.

    ``model.vis.map.znear`` is a factor of ``model.stat.extent``; setting
    ``znear = TARGET_NEAR_METERS / extent`` makes the absolute near distance 0.02425 m, at which
    the camera-adjacent mount mesh clips out of direct view while the mount stays in the scene
    and shadow pass. The mount geom keeps its group (fail-closed identity check only), collision
    groups stay invisible by default, and no physics or camera fields change.
    """
    geom_id, mesh_name = camera_mount_visual_geom(model)
    default = mujoco.MjvOption()
    if default.geomgroup.tolist() != [1, 1, 1, 0, 0, 0]:
        raise RuntimeError(f"unexpected MuJoCo default geom groups: {default.geomgroup.tolist()}")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
    if cam_id < 0:
        raise ValueError(f"model has no camera {WRIST_CAMERA!r}")
    original_group = int(model.geom_group[geom_id])
    extent = float(model.stat.extent)
    original_factor = float(model.vis.map.znear)
    original_near = original_factor * extent
    model.vis.map.znear = TARGET_NEAR_METERS / extent
    assert int(model.geom_group[geom_id]) == original_group, "mount geom group must be unchanged"
    model_path = Path(get_so101_mujoco_model_path())
    return {
        "geom_id": geom_id,
        "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "",
        "mesh_name": mesh_name,
        "body": CAMERA_MOUNT_BODY,
        "geom_group": original_group,
        "near_meters": TARGET_NEAR_METERS,
        "znear_factor": TARGET_NEAR_METERS / extent,
        "extent": extent,
        "original_znear_factor": original_factor,
        "original_near_meters": original_near,
        "camera": {
            "name": WRIST_CAMERA,
            "pos": model.cam_pos[cam_id].tolist(),
            "quat": model.cam_quat[cam_id].tolist(),
            "fovy": float(model.cam_fovy[cam_id]),
        },
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
    }


def clean_wrist_option() -> "mujoco.MjvOption":
    """Default ``MjvOption`` — MuJoCo's defaults already hide collision groups 3/4."""
    option = mujoco.MjvOption()
    if option.geomgroup.tolist() != [1, 1, 1, 0, 0, 0]:
        raise RuntimeError(f"unexpected MuJoCo default geom groups: {option.geomgroup.tolist()}")
    return option


def render_clean_wrist(env, option=None) -> np.ndarray:
    """One wrist frame under the near-plane transform; mirrors ``ValEnv.render_camera``."""
    if option is None:
        option = clean_wrist_option()
    env._renderer_rgb.update_scene(env.data, camera=WRIST_CAMERA, scene_option=option)
    return env._renderer_rgb.render().copy()


# ----- media -------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
             f"color=black:s={WIDTH}x{HEIGHT}:r={FPS}:d=0.2", "-c:v", "h264_nvenc",
             "-pix_fmt", "yuv420p", "-f", "null", "-"],
            capture_output=True, timeout=60)
        return proc.returncode == 0
    except Exception:
        return False


def _resolve_encoder(request: str) -> str:
    if request not in ("auto", "nvenc", "libx264"):
        raise ValueError(f"unknown encoder {request!r}")
    if request == "libx264":
        return "libx264"
    if _nvenc_available():
        return "nvenc"
    if request == "nvenc":
        print("warning: h264_nvenc unavailable; falling back to libx264", file=sys.stderr)
    return "libx264"


class _VideoEncoder:
    """Stream RGB frames into a 640x480@30 H.264 yuv420p file via ffmpeg stdin (cf. record.VideoEncoder)."""

    def __init__(self, path: Path, encoder: str):
        self.path, self.frames = Path(path), 0
        args = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an"]
        if encoder == "nvenc":
            args += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-bf", "0",
                     "-pix_fmt", "yuv420p"]
        else:
            args += ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-threads", "4", "-map_metadata", "-1", "-fflags", "+bitexact",
                     "-flags:v", "+bitexact"]
        args += ["-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def _ffprobe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _write_sheet(frames, path: Path, label: str) -> None:
    from PIL import Image, ImageDraw

    indices = np.linspace(0, len(frames) - 1, 12).round().astype(int)
    thumb = (320, 240)
    canvas = Image.new("RGB", (thumb[0] * 4, thumb[1] * 3), "white")
    for slot, index in enumerate(indices):
        im = Image.fromarray(frames[index]).resize(thumb, Image.Resampling.BILINEAR)
        ImageDraw.Draw(im).rectangle((0, 0, 170, 18), fill="black")
        ImageDraw.Draw(im).text((4, 3), f"{label} f{index}", fill="white")
        canvas.paste(im, ((slot % 4) * thumb[0], (slot // 4) * thumb[1]))
    tmp = path.with_name(path.name + ".tmp.jpg")
    canvas.save(tmp)
    os.replace(tmp, path)


# ----- source metadata ----------------------------------------------------------------------------


def _read_actions(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["action"])
    return np.asarray(table.column("action").to_pylist(), dtype=np.float64)


def _episode_meta(source_dir: Path, split: str) -> dict:
    """Task/episode/seed/visual config plus the source-file hashes resume and verify rely on."""
    source_dir = Path(source_dir)
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    meta_files = TRAIN_META_FILES if split == "train" else VAL_META_FILES
    meta = {"split": split, "episode": source_dir.name, "source_dir": str(source_dir),
            "parquet": source_dir / PARQUET_NAME}
    if not meta["parquet"].is_file():
        raise ValueError(f"missing {meta['parquet']}")
    hashes = {}
    for name in meta_files:
        path = source_dir / name
        if not path.is_file():
            raise ValueError(f"missing {path}")
        hashes[name] = _sha256_file(path)
    meta["metadata_sha256"] = hashes
    meta["parquet_sha256"] = _sha256_file(meta["parquet"])
    if split == "train":
        episode = json.loads((source_dir / "episode.json").read_text())
        task = episode["task"]
        meta["task"] = task["name"] if isinstance(task, dict) else task
        meta["seed"] = int(episode["seed"])
        meta["visual_config"] = episode.get("visual_config")
        meta["object_poses_start"] = episode.get("object_poses_start")
        if source_dir.parent.name != meta["task"]:
            raise ValueError(f"{source_dir}: task {meta['task']!r} does not match directory layout")
    else:
        source = json.loads((source_dir / "source.json").read_text())
        meta["task"] = source_dir.parent.name
        meta["seed"] = int(source["seed"])
        meta["visual_config"] = None
        meta["object_poses_start"] = None
    return meta


# VisualConfig fields added in 7f8ce42 (mid-v1 release). A recorded visual_config without any of
# them was generated by the pre-7f8ce42 code, where TrainEnv inherited ValEnv.place_distractors
# (uniform permutation). Replaying those episodes with the extras-weighted selector draws the RNG
# differently and places different distractors, so they must reset under the legacy selector.
_LEGACY_CONFIG_FIELDS = ("distractor_extras", "primitive_palette", "ambient", "table_surface",
                         "table_tint", "wall_tint", "floor_tint")


def _legacy_distractors(meta: dict) -> bool:
    config = meta.get("visual_config")
    return (meta["split"] == "train" and isinstance(config, dict)
            and not any(k in config for k in _LEGACY_CONFIG_FIELDS))


def _make_env(meta: dict):
    if meta["split"] == "train":
        from sim.train.tasks.catalog import load_train_task
        from sim.train.variation import VisualConfig

        task = load_train_task(meta["task"])
        env_cls, _ = task.load_classes()
        config = VisualConfig.from_dict(meta["visual_config"]) if meta["visual_config"] else None
        env = env_cls(render_images=False, visual_config=config)
        if _legacy_distractors(meta):
            from sim.val.env import ValEnv

            env.place_distractors = ValEnv.place_distractors.__get__(env, type(env))
        return env
    from sim.val.tasks import load as load_val

    env_cls, _ = load_val(meta["task"])
    return env_cls(render_images=False)


def _layout_check(env, meta: dict) -> dict | None:
    """Compare post-reset object poses to the packaged ``object_poses_start`` (train only)."""
    recorded = meta.get("object_poses_start")
    if not recorded:
        return None
    actual = env.object_poses()
    common = sorted(set(recorded) & set(actual))
    diffs = []
    for name in common:
        diffs.append(float(np.max(np.abs(np.asarray(recorded[name], float)
                                        - np.asarray(actual[name], float)))))
    return {"poses_compared": len(common), "poses_missing": sorted(set(recorded) - set(actual)),
            "max_abs_diff": max(diffs) if diffs else None}


# ----- output bookkeeping -------------------------------------------------------------------------


def _expected(meta: dict) -> dict:
    """What a complete valid output must declare for ``meta``'s source under the current transform."""
    return {
        "transform_version": TRANSFORM_VERSION,
        "parquet_sha256": meta["parquet_sha256"],
        "metadata_sha256": meta["metadata_sha256"],
        "model_sha256": _sha256_file(get_so101_mujoco_model_path()),
        "legacy_distractors": _legacy_distractors(meta),
    }


def verify_output(source_dir: Path, output_dir: Path, meta_name: str = META_NAME) -> dict:
    """Check one output dir against its source. Returns ``{ok, checks, errors}``; never writes."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    split = "train" if (source_dir / "episode.json").is_file() else "val"
    errors: list[str] = []
    checks: dict = {}
    try:
        meta = _episode_meta(source_dir, split)
    except Exception as exc:
        return {"ok": False, "split": split, "task": source_dir.parent.name,
                "episode": source_dir.name, "errors": [f"source metadata unreadable: {exc}"],
                "checks": checks}
    video, document = output_dir / VIDEO_NAME, output_dir / meta_name
    if not video.is_file():
        errors.append(f"missing {video}")
    if not document.is_file():
        errors.append(f"missing {document}")
        return {"ok": False, "split": split, "task": meta["task"], "episode": meta["episode"],
                "errors": errors, "checks": checks}
    recorded = json.loads(document.read_text())
    checks["recorded"] = recorded
    expected = _expected(meta)
    for key, want in expected.items():
        got = recorded.get(key)
        ok = got == want
        checks[key] = {"ok": ok, "recorded": got, "expected": want}
        if not ok:
            errors.append(f"{key}: recorded {got!r} != expected {want!r}")
    rows = int(_read_actions(meta["parquet"]).shape[0])
    recorded_frames = recorded.get("output", {}).get("frames")
    checks["frame_count"] = {"ok": recorded_frames == rows, "recorded": recorded_frames,
                             "expected": rows}
    if recorded_frames != rows:
        errors.append(f"frame count: recorded {recorded_frames} != {rows} parquet rows")
    for key, want in (("task", meta["task"]), ("episode", meta["episode"]), ("seed", meta["seed"]),
                      ("split", split)):
        if recorded.get(key) != want:
            errors.append(f"{key}: recorded {recorded.get(key)!r} != {want!r}")
    if video.is_file():
        actual_sha = _sha256_file(video)
        want_sha = recorded.get("output", {}).get("sha256")
        checks["output_sha256"] = {"ok": actual_sha == want_sha, "recorded": want_sha}
        if actual_sha != want_sha:
            errors.append(f"output sha256 does not match {meta_name}")
        try:
            streams = _ffprobe(video)["streams"]
            if len(streams) != 1 or streams[0]["codec_type"] != "video":
                raise ValueError("expected exactly one video stream")
            stream = streams[0]
            probe = {"codec_name": stream["codec_name"], "width": stream["width"],
                     "height": stream["height"], "r_frame_rate": stream["r_frame_rate"],
                     "pix_fmt": stream.get("pix_fmt"), "nb_read_frames": int(stream["nb_read_frames"])}
            checks["ffprobe"] = probe
            if (probe["codec_name"] != "h264" or probe["width"] != WIDTH or probe["height"] != HEIGHT
                    or probe["r_frame_rate"] != f"{FPS}/1"):
                errors.append(f"unexpected stream shape: {probe}")
            if probe["nb_read_frames"] != rows:
                errors.append(f"ffprobe frames {probe['nb_read_frames']} != {rows} parquet rows")
        except Exception as exc:
            errors.append(f"ffprobe failed: {exc}")
            checks["ffprobe"] = {"error": str(exc)}
    return {"ok": not errors, "split": split, "task": meta["task"], "episode": meta["episode"],
            "errors": errors, "checks": checks}


def output_dir_for(output_root: Path, split: str, task: str, episode: str) -> Path:
    """``episodes/<task>/<episode>`` for rebuilt episodes; ``validation_probe/`` for val probes."""
    base = "episodes" if split == "train" else "validation_probe"
    return Path(output_root) / base / task / episode


# ----- replay --------------------------------------------------------------------------------------


def replay_episode(source_dir: Path, output_dir: Path, *, split: str, encoder: str = "auto",
                   sheet: bool = False, meta_name: str = META_NAME) -> dict:
    """Regenerate one episode's wrist video with the normalized absolute near plane.

    Reads the packaged ``action`` rows, resets the exact env/seed (+ packaged visual_config for
    train), renders the wrist frame for each row, then steps ``env.from_lerobot_joints(action)`` —
    so the output frame count equals the parquet row count. A complete, hash-valid output is
    skipped; a stale/partial one is rebuilt through a temp file and atomic rename.
    """
    meta = _episode_meta(Path(source_dir), split)
    output_dir = Path(output_dir)
    prior = verify_output(source_dir, output_dir, meta_name)
    if prior["ok"] and (not sheet or (output_dir / SHEET_NAME).is_file()):
        return {"status": "skipped", "reason": "valid output exists", "task": meta["task"],
                "episode": meta["episode"], "output_dir": str(output_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    codec = _resolve_encoder(encoder)
    env = _make_env(meta)
    tmp_video = output_dir / f"{VIDEO_NAME}.tmp.mp4"
    try:
        env.reset(seed=meta["seed"])
        layout = _layout_check(env, meta)
        if layout and (layout["poses_missing"] or layout["max_abs_diff"] is None
                       or layout["max_abs_diff"] > 1e-3):
            raise RuntimeError(f"reset layout diverges from packaged object_poses_start: {layout}")
        provenance = configure_clean_wrist(env.model)
        option = clean_wrist_option()
        actions = _read_actions(meta["parquet"])
        keep = set(np.linspace(0, len(actions) - 1, 12).round().astype(int)) if sheet else set()
        sampled = []
        enc = _VideoEncoder(tmp_video, codec)
        try:
            for i, row in enumerate(actions):
                frame = render_clean_wrist(env, option)
                enc.write(frame)
                if i in keep:
                    sampled.append(frame)
                env.step(env.from_lerobot_joints(row))
            enc.close()
        except Exception:
            enc.abort()
            raise
        if enc.frames != len(actions):
            raise RuntimeError(f"rendered {enc.frames} frames for {len(actions)} action rows")
    finally:
        env.close()
    os.replace(tmp_video, output_dir / VIDEO_NAME)
    document = {
        "schema": "wrist_v2/1",
        "created_utc": _utcnow(),
        "transform_version": TRANSFORM_VERSION,
        "split": split,
        "task": meta["task"],
        "episode": meta["episode"],
        "seed": meta["seed"],
        "parquet_sha256": meta["parquet_sha256"],
        "metadata_sha256": meta["metadata_sha256"],
        "model_sha256": _sha256_file(get_so101_mujoco_model_path()),
        "legacy_distractors": _legacy_distractors(meta),
        "geom": provenance,
        "layout_check": layout,
        "output": {"file": VIDEO_NAME, "sha256": _sha256_file(output_dir / VIDEO_NAME),
                   "frames": len(actions), "fps": FPS, "width": WIDTH, "height": HEIGHT,
                   "pix_fmt": "yuv420p", "encoder": codec},
    }
    _atomic_json(output_dir / meta_name, document)
    if sheet:
        _write_sheet(sampled, output_dir / SHEET_NAME, f"{meta['task']}/{meta['episode']}")
    return {"status": "rebuilt", "task": meta["task"], "episode": meta["episode"],
            "seed": meta["seed"], "frames": len(actions), "encoder": codec,
            "output_dir": str(output_dir), "layout_check": layout,
            "output_sha256": document["output"]["sha256"], "geom": provenance}


# ----- bulk rebuild and verify ---------------------------------------------------------------------


def _replay_job(payload):
    rel, source_dir, output_dir, split, meta_name, encoder = payload
    try:
        result = replay_episode(source_dir, output_dir, split=split, encoder=encoder,
                                meta_name=meta_name)
        return {"id": rel, **result}
    except Exception as exc:
        return {"id": rel, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


# Each episode constructs an env + EGL renderer; measured RSS grows ~100-150 MB per episode even
# after env.close()+gc (native-side resources), which OOM-killed a worker mid-bulk (24 GB cgroup).
# Workers are recycled every few tasks, and a BrokenProcessPool restarts the pool over the
# remaining jobs — resume is idempotent, so lost in-flight work is simply rebuilt.
WORKER_MAX_TASKS = 15
MAX_POOL_RESTARTS = 4


def rebuild_index(source_root: Path, output_root: Path, *, split: str, meta_name: str = META_NAME,
                  workers: int = 1, encoder: str = "auto") -> dict:
    """Replay every episode in ``source_root/index.json`` into ``output_root/episodes/``.

    ``workers > 1`` uses spawned processes in bounded batches; each worker re-checks resume state
    itself, so an interrupted bulk run can simply be re-run. ``workers <= 1`` stays in-process.
    """
    source_root, output_root = Path(source_root), Path(output_root)
    index = json.loads((source_root / "index.json").read_text())
    jobs = {}
    for entry in index["episodes"]:
        rel = entry["id"]
        jobs[rel] = (rel, str(source_root / "episodes" / rel),
                     str(output_root / "episodes" / rel), split, meta_name, encoder)
    results: dict[str, dict] = {}
    if workers <= 1:
        for job in jobs.values():
            res = _replay_job(job)
            results[res["id"]] = res
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        # A fresh pool per batch bounds each spawned worker to WORKER_MAX_TASKS episodes:
        # equivalent recycling to max_tasks_per_child, which deadlocked on spawn under this
        # Python 3.12.3 build (never spawned a worker; pool.map waited indefinitely).
        restarts = 0
        while len(results) < len(jobs):
            pending = [job for rel, job in jobs.items() if rel not in results]
            batch = pending[: workers * WORKER_MAX_TASKS]
            try:
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=mp.get_context("spawn")) as pool:
                    for res in pool.map(_replay_job, batch, chunksize=1):
                        results[res["id"]] = res
            except BrokenProcessPool:
                restarts += 1
                print(f"warning: worker process died; restarting pool "
                      f"({restarts}/{MAX_POOL_RESTARTS}, {len(results)}/{len(jobs)} done)",
                      file=sys.stderr)
                if restarts > MAX_POOL_RESTARTS:
                    for rel, _ in jobs.items():
                        if rel not in results:
                            results[rel] = {"id": rel, "status": "failed",
                                            "error": "worker process died repeatedly "
                                                     "(BrokenProcessPool)"}
    ordered = list(results.values())
    summary = {"total": len(jobs),
               "rebuilt": sum(1 for r in ordered if r["status"] == "rebuilt"),
               "skipped": sum(1 for r in ordered if r["status"] == "skipped"),
               "failed": [r for r in ordered if r["status"] == "failed"]}
    summary["ok"] = not summary["failed"]
    return summary


def rebuild_train(source_root: Path, output_root: Path, *, workers: int, encoder: str = "auto") -> dict:
    """Train overlay rebuild — delegates to :func:`rebuild_index` with train settings."""
    return rebuild_index(source_root, output_root, split="train", meta_name=META_NAME,
                         workers=workers, encoder=encoder)


def _verify_all(bucket_root: Path, source_dir: str = TRAIN_SOURCE_DIR,
                output_dirname: str = OUTPUT_DIRNAME, meta_name: str = META_NAME) -> dict:
    bucket_root = Path(bucket_root)
    index = json.loads((bucket_root / source_dir / "index.json").read_text())
    report = {"checked": 0, "ok": 0, "failures": []}
    for entry in index["episodes"]:
        rel = entry["id"]
        outcome = verify_output(bucket_root / source_dir / "episodes" / rel,
                                bucket_root / output_dirname / "episodes" / rel, meta_name)
        report["checked"] += 1
        if outcome["ok"]:
            report["ok"] += 1
        else:
            report["failures"].append({"id": rel, "errors": outcome["errors"]})
    return report


# ----- release freeze ------------------------------------------------------------------------------


def _readme(source: str, version: str, meta_name: str, episodes: int, tasks: int,
            include_probe: bool) -> str:
    probe = ("""- `validation_probe/` — regenerated wrist video for `sim_val_v2/episodes/
  sort_blocks/episode_005` kept as transform evidence. It is **not** a training episode and is
  absent from `index.json`.
""" if include_probe else "")
    return f"""# {version} — clean wrist-video overlay for {source}

This is a **slim wrist-video overlay**, not a standalone release. Combine it with `{source}`:
every field, episode metadata, front video, human video and parquet comes from `{source}`
unchanged — the overlay replaces only `episodes/<task>/<episode>/robot_wrist.mp4`.

## What changed and why

v1 wrist videos render the `camera_mount` body's visual mesh
(`wrist_roll_follower_so101_camera_mount`) — the housing around the wrist camera — which occludes
~78% of the frame and leaves only a circular aperture (the "circular clipping" defect). The v3
transform (`absolute-near-plane-v3`) sets the wrist camera's absolute near plane to 0.02425 m
(`model.vis.map.znear = 0.02425 / model.stat.extent`): the near plane clips the camera-adjacent
mount geometry out of direct view while the mount stays in the scene and still casts its shadow —
measured directly-rendered mount pixels are zero at this distance. No geom group is changed;
collision groups 3/4 stay hidden by the MuJoCo default `MjvOption`; physics, collision geoms,
camera pose and every other robot visual are untouched. The packaged `action` rows are replayed
through the same task env, seed and `visual_config` (replay is deterministic: joint states
reproduce packaged `observation.state` to <0.2 deg). Episodes whose packaged `visual_config`
predates the mid-v1 distractor change are replayed with the matching legacy distractor selector
(`legacy_distractors: true` in their metadata).

## Contents

- `episodes/<task>/<episode>/robot_wrist.mp4` — regenerated wrist video: H.264, 640x480, 30 fps,
  yuv420p, exactly one frame per packaged `action` row.
- `episodes/<task>/<episode>/{meta_name}` — per-episode provenance: source parquet and metadata
  SHA256, model SHA256, transform version, mount/near-plane provenance, layout check, output
  SHA256/frame count/encoder.
{probe}- `index.json` — release index (schema `wrist_v2/index/1`), {episodes} episodes across
  {tasks} tasks in `{source}` index order.
- `inventory.json` — every release file with size + SHA256 (schema `wrist_v2/inventory/1`).

## Verify locally

```bash
MUJOCO_GL=egl uv run python -m sim.rebuild_wrist_v2 {'verify' if include_probe else 'verify-val'} --bucket-root <bucket_root>
```

`verify` re-checks every output against the source index: hashes, frame counts, ffprobe decode.
The source `{source}` tree is immutable and is never modified by this pipeline.
"""


_OVERLAYS = {
    "train": {"source": TRAIN_SOURCE_DIR, "output": OUTPUT_DIRNAME, "meta": META_NAME,
              "version": "train_v2",
              "name": "SO-101 simulated training pairs — clean wrist-video overlay",
              "include_probe": True},
    "val": {"source": VAL_SOURCE_DIR, "output": VAL_OUTPUT_DIRNAME, "meta": VAL_META_NAME,
            "version": "val_v3",
            "name": "SO-101 simulated validation pairs — clean wrist-video overlay",
            "include_probe": False},
}


def freeze_overlay(bucket_root: Path, overlay: str = "train") -> dict:
    """Freeze the overlay: full verify gate, then index.json, README.md, inventory.json.

    Refuses to write anything unless verification is checked == ok == episode count.
    Write order is index.json, README.md, inventory.json (last, so it covers the first two);
    the inventory excludes itself to avoid a self-hash paradox. Episode artifacts are untouched.
    """
    cfg = _OVERLAYS[overlay]
    source_name, out_name, meta_name = cfg["source"], cfg["output"], cfg["meta"]
    bucket_root = Path(bucket_root)
    source_index_path = bucket_root / source_name / "index.json"
    source_index = json.loads(source_index_path.read_text())
    expected = len(source_index["episodes"])
    report = _verify_all(bucket_root, source_name, out_name, meta_name)
    if not (report["checked"] == report["ok"] == expected):
        raise RuntimeError(
            f"freeze refused: verify checked={report['checked']} ok={report['ok']} "
            f"expected={expected}; fix or rebuild before freezing")
    out_root = bucket_root / out_name
    episodes = []
    for entry in source_index["episodes"]:
        rel = entry["id"]
        doc = json.loads((out_root / "episodes" / rel / meta_name).read_text())
        out = doc["output"]
        episodes.append({
            "id": rel,
            "source": f"{source_name}/episodes/{rel}",
            "video": f"episodes/{rel}/{VIDEO_NAME}",
            "metadata": f"episodes/{rel}/{meta_name}",
            "sha256": out["sha256"],
            "frames": out["frames"],
            "fps": out["fps"],
            "width": out["width"],
            "height": out["height"],
            "encoder": out["encoder"],
            "seed": doc["seed"],
            "legacy_distractors": doc["legacy_distractors"],
        })
    index = {
        "schema": "wrist_v2/index/1",
        "name": cfg["name"],
        "version": cfg["version"],
        "transform_version": TRANSFORM_VERSION,
        "created_utc": _utcnow(),
        "source_prefix": source_name,
        "source_index_sha256": _sha256_file(source_index_path),
        "model_sha256": _sha256_file(get_so101_mujoco_model_path()),
        "episodes_total": source_index.get("episodes_total"),
        "tasks_total": source_index.get("tasks_total"),
        "families_total": source_index.get("families_total"),
        "episodes": episodes,
    }
    _atomic_json(out_root / "index.json", index)

    readme_path = out_root / "README.md"
    readme_tmp = readme_path.with_name("README.md.tmp")
    readme_tmp.write_text(_readme(source_name, cfg["version"], meta_name,
                                  source_index.get("episodes_total") or expected,
                                  source_index.get("tasks_total") or len(
                                      {e["id"].split("/")[0] for e in episodes}),
                                  cfg["include_probe"]))
    os.replace(readme_tmp, readme_path)

    entries = []
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out_root).as_posix()
        if rel == "inventory.json":
            continue
        entries.append({"path": rel, "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path)})
    inventory = {
        "schema": "wrist_v2/inventory/1",
        "created_utc": _utcnow(),
        "files": entries,
        "totals": {"files": len(entries), "bytes": sum(e["bytes"] for e in entries)},
    }
    _atomic_json(out_root / "inventory.json", inventory)
    return {"ok": True, "verify": {"checked": report["checked"], "ok": report["ok"]},
            "episodes": len(episodes), "files": len(entries),
            "bytes": inventory["totals"]["bytes"],
            "index_sha256": _sha256_file(out_root / "index.json"),
            "readme_sha256": _sha256_file(out_root / "README.md"),
            "inventory_sha256": _sha256_file(out_root / "inventory.json")}


def _pilot(bucket_root: Path, encoder: str) -> dict:
    bucket_root = Path(bucket_root)
    results = {}
    task, episode = PILOT_VAL_EPISODE
    results["validation_probe"] = replay_episode(
        bucket_root / VAL_SOURCE_DIR / "episodes" / task / episode,
        output_dir_for(bucket_root / OUTPUT_DIRNAME, "val", task, episode),
        split="val", encoder=encoder, sheet=True)
    task, episode = PILOT_TRAIN_EPISODE
    results["train_pilot"] = replay_episode(
        bucket_root / TRAIN_SOURCE_DIR / "episodes" / task / episode,
        output_dir_for(bucket_root / OUTPUT_DIRNAME, "train", task, episode),
        split="train", encoder=encoder, sheet=True)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.rebuild_wrist_v2",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "rebuild", "verify", "freeze",
                 "rebuild-val", "verify-val", "freeze-val"):
        p = sub.add_parser(name)
        p.add_argument("--bucket-root", required=True, type=Path)
        if name in ("pilot", "rebuild", "rebuild-val"):
            p.add_argument("--encoder", default="auto", choices=("auto", "nvenc", "libx264"))
        if name in ("rebuild", "rebuild-val"):
            p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "pilot":
        results = _pilot(args.bucket_root, args.encoder)
        print(json.dumps(results, indent=2))
        return 0 if all(r["status"] in ("rebuilt", "skipped") for r in results.values()) else 1
    if args.command == "rebuild":
        summary = rebuild_train(args.bucket_root / TRAIN_SOURCE_DIR,
                                args.bucket_root / OUTPUT_DIRNAME,
                                workers=args.workers, encoder=args.encoder)
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    if args.command == "freeze":
        summary = freeze_overlay(args.bucket_root)
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    if args.command == "rebuild-val":
        summary = rebuild_index(args.bucket_root / VAL_SOURCE_DIR,
                                args.bucket_root / VAL_OUTPUT_DIRNAME,
                                split="val", meta_name=VAL_META_NAME,
                                workers=args.workers, encoder=args.encoder)
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    if args.command == "verify-val":
        report = _verify_all(args.bucket_root, VAL_SOURCE_DIR, VAL_OUTPUT_DIRNAME, VAL_META_NAME)
        print(json.dumps(report, indent=2))
        return 0 if report["checked"] and report["checked"] == report["ok"] else 1
    if args.command == "freeze-val":
        summary = freeze_overlay(args.bucket_root, overlay="val")
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    report = _verify_all(args.bucket_root)
    print(json.dumps(report, indent=2))
    return 0 if report["checked"] and report["checked"] == report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

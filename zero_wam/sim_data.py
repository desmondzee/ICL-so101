"""Convert the simulated SO-101 releases into Zero-WAM LeRobot v2.1 collections.

Sources are immutable: `sim_train_v1` / `sim_val_v2` provide the packaged parquet `action`
and `observation.state` rows, the front robot video, the human video and the index metadata;
the corrected wrist videos come from the `train_v2` / `val_v3` overlays. Nothing is re-encoded —
every media file is hard-linked (or copied as a fallback) so output bytes equal source bytes.

Layout (cf. `zero_wam/so101_smoke_data.py` and upstream's lerobot==0.3.3 loader):

    <output>/<split>/<task>/                     one LeRobot v2.1 task root per source task
        meta/info.json tasks.jsonl episodes.jsonl episodes_stats.jsonl
        data/chunk-000/episode_NNNNNN.parquet
        videos/chunk-000/<obs-key>/episode_NNNNNN.mp4
    <output>/<split>/meta/action_transform.yaml  absolute-joint action mapping
    <output>/<split>/meta/action_stats.json      frozen physical actuator bounds (identical on both splits)
    <output>/<split>/icl_manifest.json           robot<->human pairing for the ICL loader
    <output>/<split>/human_data/so101/run_<split>/samples/<task>__<episode>/generated_video.mp4
    <output>/audit.json                          machine-readable evidence

Action contract: packaged `action[0:5]` (absolute joint degrees) -> `action.hand.position`
channels 0-4; packaged `action[5]` (0-100 gripper) -> `action.effector.position` channel 28.
Both are absolute (`absolute_value` + `use_absolute` + `format: joint`), so the remaining 24
of the model's 30 action channels stay masked/zero. The manifest's `human_video_path` carries
a `run_<split>` segment so upstream `_human_latent_candidate` resolves
`<human_latent_root>/run_<split>/samples/.../generated_video.pth` later.

    python -m zero_wam.sim_data build --bucket-root P --output-root P [--workers N]
    python -m zero_wam.sim_data verify --bucket-root P --output-root P
    python -m zero_wam.sim_data build --verify-only --bucket-root P --output-root P
    python -m zero_wam.sim_data bounds --bucket-root P   # re-derive bounds from a live val env
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from so101_nexus import get_so101_mujoco_model_path

CAMERAS = ("observation.images.front", "observation.images.wrist")
FRONT_KEY, WRIST_KEY = CAMERAS
DATA_COLUMNS = ["action", "observation.state", "timestamp", "frame_index",
                "episode_index", "index", "task_index"]
JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
FPS = 30

# Model action layout (upstream lerobot_action.MODEL_ACTION_LAYOUT): hand.position=14 ch (0-13),
# arm.position=14 ch (14-27), effector.position=2 ch (28-29). Our joints land on 0-4, gripper on 28.
ACTION_TRANSFORM = {
    "states": [
        {"observation.state.hand.position": {"origin_keys":
            [{"observation.state": {"start": 0, "end": 5}}]}},
        {"observation.state.effector.position": {"origin_keys":
            [{"observation.state": {"start": 5, "end": 6}}]}},
    ],
    "actions": [
        {"action.hand.position": {"origin_keys":
            [{"action": {"start": 0, "end": 5}}],
            "absolute_value": True, "use_absolute": True, "format": "joint"}},
        {"action.effector.position": {"origin_keys":
            [{"action": {"start": 5, "end": 6}}],
            "absolute_value": True, "use_absolute": True, "format": "joint"}},
    ],
    "images": [{camera: {"origin_keys": camera}} for camera in CAMERAS],
}

SPLITS = {
    "train": {"source": "sim_train_v1", "wrist": "train_v2", "run": "run_train"},
    "val": {"source": "sim_val_v2", "wrist": "val_v3", "run": "run_val"},
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


HUMAN_DESC_START = "integrated_multimodal_description:"
HUMAN_DESC_END = "overall_soundscape:"
HUMAN_SHOT_PREFIX = "[Shot 1] "


def human_local_instruction(prompt: str) -> str:
    """Detailed human-demo action description: the prompt body between the
    `integrated_multimodal_description:` and `overall_soundscape:` markers,
    without the leading `[Shot 1] ` prefix. Rejects missing markers or empty text."""
    i = prompt.find(HUMAN_DESC_START)
    j = prompt.find(HUMAN_DESC_END)
    if i < 0 or j < 0 or j <= i:
        raise ValueError("human prompt missing description markers")
    body = prompt[i + len(HUMAN_DESC_START):j].strip()
    if body.startswith("[Shot 1]"):
        body = body[len("[Shot 1]"):].strip()
    if not body:
        raise ValueError("human prompt description body is empty")
    return body


def human_local_instruction_sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def episode_detailed_human_instruction(bucket_root: Path, ep_dir: Path) -> str:
    """Detailed human-demo description for one immutable source episode. Train source.json
    carries `human.prompt` directly; val source.json only carries the generated
    `human_video` path (`data/so101_sim_val/human/.../seed_0/video.mp4`), whose sibling
    `prompt.txt` lives under the bucket's `work/` tree."""
    source = json.loads((ep_dir / "source.json").read_text())
    prompt = source.get("human", {}).get("prompt")
    if not prompt:
        hv = source.get("human_video")
        if not hv:
            raise ValueError(f"{ep_dir}: no human prompt or human_video path")
        rel = Path(hv)
        if rel.parts[0] == "data":
            rel = Path("work") / Path(*rel.parts[1:])
        prompt = (bucket_root / rel.with_name("prompt.txt")).read_text()
    return human_local_instruction(prompt)


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _ffprobe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _link_or_copy(source: Path, dest: Path) -> str:
    """Hard-link so bytes are identical by construction; copy if linking is impossible."""
    source, dest = Path(source), Path(dest)
    if dest.exists():
        if (dest.stat().st_ino, dest.stat().st_dev) == (source.stat().st_ino, source.stat().st_dev):
            return "linked"
        if _sha256_file(dest) == _sha256_file(source):
            return "linked"
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
        return "linked"
    except OSError:
        tmp = dest.with_name(dest.name + ".partial")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
        return "copied"


def _read_actions_state(parquet_path: Path):
    table = pq.read_table(parquet_path, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    return actions, state


def _feature_stats(values: np.ndarray) -> dict:
    values = values.reshape(len(values), -1).astype(np.float64)
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "count": [len(values)]}


def _video_stats(video: Path) -> dict:
    """Per-channel stats of frames scaled to [0,1] at 2fps 160x120 (as so101_smoke_data)."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
         "-vf", "fps=2,scale=160:120", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 3) / 255.0
    nest = lambda v: [[[float(x)]] for x in v]
    return {"min": nest(px.min(0)), "max": nest(px.max(0)), "mean": nest(px.mean(0)),
            "std": nest(px.std(0)), "count": [len(px) // (160 * 120)]}


# ----- action stats --------------------------------------------------------------------------------

# Fixed physical action bounds in packaged LeRobot units (arm joints in degrees, gripper 0-100),
# derived from the executable actuator target limits that `sim/val/env.py::from_lerobot_joints`
# maps back to radians. Data quantiles were rejected: a train-only q01/q99 leaves validation
# executable commands unrepresentable (8,845 wrist-roll rows hit the loader's ±2 clip and the
# gripper quantile ceiling ~32.9 is far below physical open=100). These are the exact frozen
# bounds; `python -m zero_wam.sim_data bounds` re-derives them from a live val env and fails on
# drift beyond a small float-path tolerance.
ACTION_Q01 = [-109.99987556, -100.00000051, -96.82987066, -94.99983758, -157.21087394, 0.0]
ACTION_Q99 = [109.99987556, 100.00000051, 96.82987066, 94.99983758, 157.21087394, 100.0]
# Packaged oracle commands are recorded pre-clamp and can exceed actuator bounds by up to
# ~0.0098 deg (wrist_flex pegs at ~95.0096 against the 94.9998 bound; wrist_roll touches
# -157.2109). 0.02 deg (~1e-4 normalized) covers this recording noise while still failing
# closed on real out-of-range data.
ACTION_BOUNDS_TOL = 0.02


def _physical_action_stats() -> dict:
    """The frozen physical-bounds stats consumed by upstream `LeRobotActionProcessor`.

    The loader only requires `method: abs` + `window_size` + `norm_stats.q01/q99`; the other
    fields record where the bounds come from. arm channels get the five joint bounds; the
    effector gets the gripper 0-100 bounds.
    """
    return {
        "method": "abs",
        "window_size": 0,
        "source_method": "physical_actuator_bounds",
        "units": "packaged LeRobot units: arm joints degrees, gripper 0-100",
        "bounds_method": "physical_bounds",
        "provenance": {
            "definition": "executable actuator target limits from "
                          "sim/val/env.py::ValEnv.from_lerobot_joints, mapped back to packaged "
                          "LeRobot units (deg2rad inverse for joints, 0-100 for gripper)",
            "model": str(get_so101_mujoco_model_path()),
            "model_sha256": _sha256_file(get_so101_mujoco_model_path()),
        },
        "bounds": {"q01": ACTION_Q01, "q99": ACTION_Q99},
        "norm_stats": {
            "action.hand.position": {"q01": ACTION_Q01[:5], "q99": ACTION_Q99[:5]},
            "action.effector.position": {"q01": [ACTION_Q01[5]], "q99": [ACTION_Q99[5]]},
        },
    }


def _derive_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Re-derive LeRobot-unit action bounds from a current validation env.

    `from_lerobot_joints` sends row[:5] through deg2rad into the actuator target range and maps
    gripper 0-100 onto the jaw travel — so env target limits inverted to packaged units are the
    physical bounds. Small float-path deviations from the frozen constants are expected.
    """
    from sim.val.tasks import load as load_val

    env_cls, _ = load_val("sort_blocks")
    env = env_cls(render_images=False)
    try:
        lo, hi = np.degrees(env._target_low), np.degrees(env._target_high)
    finally:
        env.close()
    return np.r_[lo[:5], 0.0], np.r_[hi[:5], 100.0]


def _check_within_bounds(actions: np.ndarray, where: str) -> None:
    q01, q99 = np.asarray(ACTION_Q01), np.asarray(ACTION_Q99)
    if (actions < q01 - ACTION_BOUNDS_TOL).any() or (actions > q99 + ACTION_BOUNDS_TOL).any():
        lo_v = actions.min(0); hi_v = actions.max(0)
        raise RuntimeError(
            f"{where}: packaged action rows exceed physical bounds "
            f"(min {np.round(lo_v, 4).tolist()}, max {np.round(hi_v, 4).tolist()}, "
            f"tol {ACTION_BOUNDS_TOL})")


def stats_sha256(stats: dict) -> str:
    """Canonical JSON hash (sorted keys, no whitespace) — distinct from the file's raw SHA256."""
    import hashlib
    return hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()


# ----- per-episode conversion -----------------------------------------------------------------------


def _episode_parquet(actions, state, episode_index: int, task_index: int) -> pd.DataFrame:
    n = len(actions)
    return pd.DataFrame({
        "action": [row.tolist() for row in actions],
        "observation.state": [row.tolist() for row in state],
        "timestamp": (np.arange(n) / FPS).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": np.arange(n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
    })


def _video_feature(codec: str) -> dict:
    return {"dtype": "video", "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": 480, "video.width": 640, "video.codec": codec,
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": FPS, "video.channels": 3, "has_audio": False}}


def convert_episode(source_dir: Path, wrist_video: Path, human_video: Path, task_root: Path,
                    episode_index: int, task_index: int, instruction: str,
                    expected_frames: int | None = None) -> dict:
    """One episode: parquet + linked media + per-episode stats. Returns the episode record."""
    source_dir, task_root = Path(source_dir), Path(task_root)
    actions, state = _read_actions_state(source_dir / "robot_data.parquet")
    if expected_frames is not None and len(actions) != int(expected_frames):
        raise RuntimeError(f"{source_dir}: {len(actions)} action rows, "
                           f"index expects {expected_frames}")
    for name, arr in (("action", actions), ("observation.state", state)):
        if arr.ndim != 2 or arr.shape[1] != 6 or not np.isfinite(arr).all():
            raise RuntimeError(f"{source_dir}: {name} shape/finiteness violation {arr.shape}")
    _check_within_bounds(actions, str(source_dir))

    ep = _episode_parquet(actions, state, episode_index, task_index)
    data_dir = task_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = data_dir / f"episode_{episode_index:06d}.parquet"
    tmp_parquet = parquet_out.with_name(parquet_out.name + ".partial")
    ep.to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, parquet_out)

    videos = {}
    for key, src in ((FRONT_KEY, source_dir / "robot_front.mp4"), (WRIST_KEY, wrist_video)):
        dest = task_root / "videos" / "chunk-000" / key / f"episode_{episode_index:06d}.mp4"
        _link_or_copy(src, dest)
        probe = _ffprobe(dest)
        probe["nb_read_frames"] = int(probe["nb_read_frames"])
        if probe["nb_read_frames"] != len(actions):
            raise RuntimeError(f"{dest}: ffprobe {probe['nb_read_frames']} frames "
                               f"!= {len(actions)} parquet rows")
        videos[key] = {"dest": dest, "probe": probe, "sha256": _sha256_file(dest)}

    ep_stats = {c: _feature_stats(
        np.stack(ep[c].to_numpy()) if ep[c].dtype == object else ep[c].to_numpy().reshape(-1, 1))
        for c in DATA_COLUMNS}
    for key in CAMERAS:
        ep_stats[key] = _video_stats(videos[key]["dest"])
    return {
        "episode_index": episode_index,
        "rows": len(actions),
        "parquet": str(parquet_out),
        "videos": {k: {"path": str(v["dest"]), "sha256": v["sha256"],
                       "codec": v["probe"]["codec_name"], "frames": v["probe"]["nb_read_frames"]}
                   for k, v in videos.items()},
        "ep_stats": ep_stats,
        "human": str(human_video),
        "instruction": instruction,
    }


def build_task(bucket_root: Path, out_split_root: Path, split: str, task: str,
               entries: list[dict], human_root: Path, workers: int) -> dict:
    """Convert every episode of one source task into `<out_split_root>/<task>` atomically."""
    cfg = SPLITS[split]
    source_root = bucket_root / cfg["source"] / "episodes" / task
    wrist_root = bucket_root / cfg["wrist"] / "episodes" / task
    instructions = []
    task_index_of = {}
    for entry in entries:
        text = entry["instruction"]
        if text not in task_index_of:
            task_index_of[text] = len(instructions)
            instructions.append(text)

    def sample_of(entry):
        ep_name = f"episode_{int(entry['episode']):03d}"
        sample = f"{task}__{ep_name}"
        detailed = episode_detailed_human_instruction(bucket_root, source_root / ep_name)
        return {
            "run": cfg["run"], "sample": sample, "sample_id": sample,
            "robot_video_path": f"{task}/videos/chunk-000/{FRONT_KEY}/episode_{int(entry['episode']):06d}.mp4",
            "human_video_path": f"so101/{cfg['run']}/samples/{sample}/generated_video.mp4",
            "robot_task_name": task, "human_text": entry["instruction"],
            "human_local_instruction": detailed,
            "human_local_instruction_sha256": human_local_instruction_sha256(detailed),
            "source": {"robot": f"{cfg['source']}/episodes/{task}/{ep_name}",
                       "wrist_overlay": f"{cfg['wrist']}/episodes/{task}/{ep_name}/robot_wrist.mp4"},
        }

    samples = [sample_of(entry) for entry in entries]

    partial = out_split_root / f"{task}.partial"
    final = out_split_root / task
    marker = final / ".build_ok"
    if marker.is_file():
        return {"task": task, "status": "skipped", "episodes": len(entries),
                "samples": samples}
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    def job(entry):
        ep = int(entry["episode"])
        return entry, convert_episode(
            source_root / f"episode_{ep:03d}",
            wrist_root / f"episode_{ep:03d}" / "robot_wrist.mp4",
            source_root / f"episode_{ep:03d}" / "human.mp4",
            partial, ep, task_index_of[entry["instruction"]], entry["instruction"],
            expected_frames=entry.get("robot_frames"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [fut.result() for fut in
                   (pool.submit(job, e) for e in entries)]
    results.sort(key=lambda r: r[1]["episode_index"])

    # per-episode human media (manifest samples are pure functions of the index, built above)
    for entry, record in results:
        ep_name = f"episode_{int(entry['episode']):03d}"
        sample = f"{task}__{ep_name}"
        human_dest = (human_root / "so101" / cfg["run"] / "samples" / sample /
                      "generated_video.mp4")
        _link_or_copy(source_root / ep_name / "human.mp4", human_dest)
        record["human_sha256"] = _sha256_file(human_dest)

    meta = partial / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    features = {
        "action": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    codecs = {}
    for _, record in results:
        for key in CAMERAS:
            codecs.setdefault(key, record["videos"][key]["codec"])
    for key in CAMERAS:
        features[key] = _video_feature(codecs[key])
    info = {"codebase_version": "v2.1", "robot_type": "so101",
            "total_episodes": len(results),
            "total_frames": sum(r["rows"] for _, r in results),
            "total_tasks": len(instructions),
            "total_videos": len(results) * len(CAMERAS), "total_chunks": 1,
            "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(results)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
    _write_json(meta / "info.json", info)
    with open(partial / "meta" / "tasks.jsonl.tmp", "w") as f:
        for i, text in enumerate(instructions):
            f.write(json.dumps({"task_index": i, "task": text}) + "\n")
    os.replace(partial / "meta" / "tasks.jsonl.tmp", partial / "meta" / "tasks.jsonl")
    with open(partial / "meta" / "episodes.jsonl.tmp", "w") as f:
        for entry, record in results:
            ep = int(entry["episode"]); n = record["rows"]
            f.write(json.dumps({"episode_index": ep, "tasks": [entry["instruction"]],
                                "length": n,
                                "action_config": [{"start_frame": 0, "end_frame": n,
                                                   "action_text": entry["instruction"]}]}) + "\n")
    os.replace(partial / "meta" / "episodes.jsonl.tmp", partial / "meta" / "episodes.jsonl")
    with open(partial / "meta" / "episodes_stats.jsonl.tmp", "w") as f:
        for _, record in results:
            f.write(json.dumps({"episode_index": record["episode_index"],
                                "stats": record["ep_stats"]}) + "\n")
    os.replace(partial / "meta" / "episodes_stats.jsonl.tmp",
               partial / "meta" / "episodes_stats.jsonl")

    marker_tmp = partial / ".build_ok.partial"
    marker_tmp.write_text(_utcnow() + "\n")
    os.replace(marker_tmp, partial / ".build_ok")
    if final.exists():
        shutil.rmtree(final)
    os.replace(partial, final)
    return {"task": task, "status": "built", "episodes": len(results),
            "records": [(e["id"], r["videos"], r["human_sha256"], r["rows"])
                        for e, r in results],
            "samples": samples}


# ----- collection ------------------------------------------------------------------------------------


def _index_tasks(index: dict) -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = {}
    for entry in index["episodes"]:
        tasks.setdefault(entry["task"], []).append(entry)
    for eps in tasks.values():
        eps.sort(key=lambda e: int(e["episode"]))
    return tasks


def _all_source_actions(bucket_root: Path, source_name: str, index: dict) -> np.ndarray:
    parts = []
    for entry in index["episodes"]:
        a, _ = _read_actions_state(bucket_root / source_name / "episodes" / entry["id"]
                                   / "robot_data.parquet")
        parts.append(a)
    return np.concatenate(parts)


def build_split(bucket_root: Path, output_root: Path, split: str, stats: dict,
                workers: int) -> dict:
    cfg = SPLITS[split]
    source_index_path = bucket_root / cfg["source"] / "index.json"
    index = json.loads(source_index_path.read_text())
    tasks = _index_tasks(index)
    split_root = Path(output_root) / split
    split_root.mkdir(parents=True, exist_ok=True)

    summary = {"split": split, "tasks": {}, "samples": [], "order": [e["id"] for e in index["episodes"]]}
    for task, entries in tasks.items():
        result = build_task(bucket_root, split_root, split, task, entries,
                            split_root / "human_data", workers)
        summary["tasks"][task] = result
        summary["samples"].extend(result.get("samples", []))

    # collection meta: transform + stats (val reuses the exact train stats file content)
    _write_text(split_root / "meta" / "action_transform.yaml",
                yaml.safe_dump(ACTION_TRANSFORM, sort_keys=False))
    _write_json(split_root / "meta" / "action_stats.json", stats)
    manifest = {"dataset": "icl-so101-sim", "split": split,
                "transform_version": "absolute-near-plane-v3",
                "manifest_schema": "icl_manifest/2",
                "human_text_field": "human_local_instruction",
                "samples": summary["samples"]}
    _write_json(split_root / "icl_manifest.json", manifest)
    summary["stats_sha256"] = stats_sha256(stats)
    summary["manifest_samples"] = len(summary["samples"])
    return summary


def _leakage(train_index: dict, val_index: dict) -> dict:
    train_tasks = {e["task"] for e in train_index["episodes"]}
    val_tasks = {e["task"] for e in val_index["episodes"]}
    overlap = sorted(train_tasks & val_tasks)
    return {"ok": not overlap, "overlap": overlap,
            "train_tasks": len(train_tasks), "val_tasks": len(val_tasks)}


def build(bucket_root: Path, output_root: Path, workers: int, verify: bool = True) -> dict:
    bucket_root, output_root = Path(bucket_root), Path(output_root)
    train_index = json.loads((bucket_root / SPLITS["train"]["source"] / "index.json").read_text())
    val_index = json.loads((bucket_root / SPLITS["val"]["source"] / "index.json").read_text())
    leakage = _leakage(train_index, val_index)
    if not leakage["ok"]:
        raise RuntimeError(f"held-out validation tasks appear in train: {leakage['overlap']}")

    stats = _physical_action_stats()
    report = {"created_utc": _utcnow(), "output_root": str(output_root),
              "stats_sha256": stats_sha256(stats), "leakage": leakage, "splits": {}}
    for split in ("train", "val"):
        report["splits"][split] = build_split(bucket_root, output_root, split, stats, workers)
    _write_json(output_root / "audit.json", _audit_payload(bucket_root, output_root, report))
    if verify:
        report["verify"] = verify_conversion(bucket_root, output_root)
    return report


def _overlay_episode_sha(bucket_root: Path, split: str, rel: str) -> str | None:
    index_path = bucket_root / SPLITS[split]["wrist"] / "index.json"
    if not index_path.is_file():
        return None
    overlay_index = json.loads(index_path.read_text())
    for entry in overlay_index.get("episodes", []):
        if entry["id"] == rel:
            return entry.get("sha256")
    return None


def _audit_payload(bucket_root: Path, output_root: Path, report: dict) -> dict:
    payload = {"created_utc": _utcnow(), "output_root": str(output_root),
               "stats_sha256": report["stats_sha256"], "leakage": report["leakage"],
               "transform_version": "absolute-near-plane-v3", "splits": {}}
    for split, cfg in SPLITS.items():
        source_index = bucket_root / cfg["source"] / "index.json"
        overlay_index = bucket_root / cfg["wrist"] / "index.json"
        overlay_inv = bucket_root / cfg["wrist"] / "inventory.json"
        entry = {"source": cfg["source"], "wrist_overlay": cfg["wrist"],
                 "source_index_sha256": _sha256_file(source_index)}
        if overlay_index.is_file():
            entry["overlay_index_sha256"] = _sha256_file(overlay_index)
        if overlay_inv.is_file():
            entry["overlay_inventory_sha256"] = _sha256_file(overlay_inv)
        split_summary = report["splits"].get(split, {})
        tasks = split_summary.get("tasks", {})
        entry["episodes"] = int(source_index_episode_count(source_index))
        entry["tasks"] = len(tasks)
        entry["coverage"] = {
            task: {"status": res["status"], "episodes": res["episodes"],
                   "records": res.get("records")} for task, res in tasks.items()}
        entry["manifest_samples"] = split_summary.get("manifest_samples", 0)
        stats_file = output_root / split / "meta" / "action_stats.json"
        if stats_file.is_file():
            entry["action_stats_file_sha256"] = _sha256_file(stats_file)
            entry["action_stats_canonical_sha256"] = stats_sha256(
                json.loads(stats_file.read_text()))
        payload["splits"][split] = entry
    return payload


def source_index_episode_count(path: Path) -> int:
    return len(json.loads(Path(path).read_text())["episodes"])


# ----- verification ----------------------------------------------------------------------------------


def verify_conversion(bucket_root: Path, output_root: Path) -> dict:
    """Validate a converted tree against sources and overlays. Returns a report; `ok` on all pass."""
    bucket_root, output_root = Path(bucket_root), Path(output_root)
    failures: list[str] = []
    report = {"checked": 0, "failures": failures, "splits": {}}

    train_index = json.loads((bucket_root / SPLITS["train"]["source"] / "index.json").read_text())
    val_index = json.loads((bucket_root / SPLITS["val"]["source"] / "index.json").read_text())
    report["leakage"] = _leakage(train_index, val_index)
    if not report["leakage"]["ok"]:
        failures.append(f"held-out val tasks in train: {report['leakage']['overlap']}")

    # stats must be the exact frozen physical actuator bounds, identical bytes on both splits
    stats = _physical_action_stats()
    stats_files = {}
    for split in ("train", "val"):
        stats_path = output_root / split / "meta" / "action_stats.json"
        if not stats_path.is_file():
            failures.append(f"{split}: missing {stats_path}")
            continue
        stats_files[split] = _sha256_file(stats_path)
        if json.loads(stats_path.read_text()) != stats:
            failures.append(f"{split}: action_stats.json does not match frozen "
                            f"physical actuator bounds")
    if len(set(stats_files.values())) > 1:
        failures.append("train/val action_stats.json differ in bytes")
    report["stats_sha256"] = stats_sha256(stats)          # canonical hash
    report["stats_file_sha256"] = stats_files             # raw file bytes

    for split in ("train", "val"):
        cfg = SPLITS[split]
        index = train_index if split == "train" else val_index
        split_root = output_root / split
        tasks = _index_tasks(index)
        on_disk = sorted(p.name for p in split_root.iterdir()
                         if p.is_dir() and not p.name.endswith(".partial"))
        expected_tasks = sorted(tasks)
        extra = [d for d in on_disk if d not in ("human_data", "meta") and d not in expected_tasks]
        missing = [t for t in expected_tasks if not (split_root / t).is_dir()]
        for d in extra:
            failures.append(f"{split}: unexpected task dir {d}")
        for t in missing:
            failures.append(f"{split}: missing task dir {t}")
        n_checked = 0
        mins = np.full(6, np.inf); maxs = np.full(6, -np.inf)
        outside = np.zeros(6, np.int64); clipped = np.zeros(6, np.int64)
        strict_outside = np.zeros(6, np.int64)
        for task, entries in tasks.items():
            task_root = split_root / task
            if not task_root.is_dir():
                continue
            meta = task_root / "meta"
            for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
                if not (meta / name).is_file():
                    failures.append(f"{split}/{task}: missing meta/{name}")
            if (meta / "episodes.jsonl").is_file():
                rows = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
                if len(rows) != len(entries):
                    failures.append(f"{split}/{task}: episodes.jsonl has {len(rows)} rows, "
                                    f"expected {len(entries)}")
            for entry in entries:
                ep = int(entry["episode"])
                rel = entry["id"]
                src_ep = bucket_root / cfg["source"] / "episodes" / rel
                try:
                    per = _verify_episode(bucket_root, cfg, split, task_root, src_ep,
                                          entry, split_root)
                    mins = np.minimum(mins, per["min"]); maxs = np.maximum(maxs, per["max"])
                    outside += per["outside_1"]; clipped += per["clipped_2"]
                    strict_outside += per["strict_outside_1"]
                    n_checked += 1
                except Exception as exc:
                    failures.append(f"{split}/{rel}: {exc}")
        # manifest
        manifest_path = split_root / "icl_manifest.json"
        if not manifest_path.is_file():
            failures.append(f"{split}: missing {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("manifest_schema") != "icl_manifest/2":
                failures.append(f"{split}: manifest_schema "
                                f"{manifest.get('manifest_schema')} != icl_manifest/2")
            if len(manifest.get("samples", [])) != len(index["episodes"]):
                failures.append(f"{split}: manifest has "
                                f"{len(manifest.get('samples', []))} samples, "
                                f"expected {len(index['episodes'])}")
            for sample in manifest.get("samples", []):
                text = sample.get("human_local_instruction")
                if not text or sample.get("human_local_instruction_sha256") != \
                        human_local_instruction_sha256(text):
                    failures.append(f"{split}: sample {sample.get('sample')} missing/invalid "
                                    "human_local_instruction")
                    break
        q01, q99 = np.asarray(ACTION_Q01), np.asarray(ACTION_Q99)
        utilization = {}
        for i, name in enumerate(JOINT_NAMES):
            span = q99[i] - q01[i]
            utilization[name] = {
                "min": float(mins[i]) if np.isfinite(mins[i]) else None,
                "max": float(maxs[i]) if np.isfinite(maxs[i]) else None,
                "range_fraction": ([float((mins[i] - q01[i]) / span),
                                    float((maxs[i] - q01[i]) / span)]
                                   if np.isfinite(mins[i]) else None),
                "outside_1": int(outside[i]), "clipped_2": int(clipped[i]),
                "strict_outside_1": int(strict_outside[i]),
            }
            if outside[i] or clipped[i]:
                failures.append(f"{split}: {name} has {outside[i]} rows outside [-1,1] "
                                f"and {clipped[i]} clipped at ±2 under physical bounds "
                                f"(tolerance {ACTION_BOUNDS_TOL})")
        report["splits"][split] = {"expected_episodes": len(index["episodes"]),
                                   "expected_tasks": len(expected_tasks),
                                   "checked": n_checked,
                                   "utilization": utilization}
        report["checked"] += n_checked
    report["ok"] = not failures
    return report


def _verify_episode(bucket_root: Path, cfg: dict, split: str, task_root: Path,
                    source_dir: Path, entry: dict, split_root: Path) -> dict:
    ep = int(entry["episode"])
    rel = entry["id"]
    actions, state = _read_actions_state(source_dir / "robot_data.parquet")
    _check_within_bounds(actions, f"{split}/{rel}")
    parquet_out = task_root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    if not parquet_out.is_file():
        raise RuntimeError("missing parquet")
    out_actions, out_state = _read_actions_state(parquet_out)
    if len(out_actions) != len(actions):
        raise RuntimeError(f"parquet rows {len(out_actions)} != source {len(actions)}")
    for name, arr in (("action", out_actions), ("observation.state", out_state)):
        if arr.ndim != 2 or arr.shape[1] != 6 or not np.isfinite(arr).all():
            raise RuntimeError(f"{name} invalid")
    if not np.allclose(out_actions, actions) or not np.allclose(out_state, state):
        raise RuntimeError("parquet content differs from source rows")

    probes = {FRONT_KEY: source_dir / "robot_front.mp4",
              WRIST_KEY: bucket_root / cfg["wrist"] / "episodes" / rel / "robot_wrist.mp4"}
    for key, src in probes.items():
        dest = task_root / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
        if not dest.is_file():
            raise RuntimeError(f"missing {key}")
        if _sha256_file(dest) != _sha256_file(src):
            raise RuntimeError(f"{key} hash differs from {'source' if key == FRONT_KEY else 'overlay'}")
        probe = _ffprobe(dest)
        if int(probe["nb_read_frames"]) != len(actions):
            raise RuntimeError(f"{key} frames {probe['nb_read_frames']} != {len(actions)} rows")

    sample = f"{task_root.name}__episode_{ep:03d}"
    human_dest = (split_root / "human_data" / "so101" / cfg["run"] / "samples" / sample /
                  "generated_video.mp4")
    if not human_dest.is_file():
        raise RuntimeError("missing human video")
    if _sha256_file(human_dest) != _sha256_file(source_dir / "human.mp4"):
        raise RuntimeError("human video hash differs from source")
    q01, q99 = np.asarray(ACTION_Q01), np.asarray(ACTION_Q99)
    norm = (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    tol_norm = 2.0 * ACTION_BOUNDS_TOL / (q99 - q01)
    return {"min": actions.min(0), "max": actions.max(0),
            "outside_1": (np.abs(norm) > 1 + tol_norm).sum(0),
            "clipped_2": (np.abs(norm) > 2 + tol_norm).sum(0),
            "strict_outside_1": (np.abs(norm) > 1).sum(0)}


# ----- CLI -------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zero_wam.sim_data",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify", "bounds"):
        p = sub.add_parser(name)
        p.add_argument("--bucket-root", required=True, type=Path)
        if name != "bounds":
            p.add_argument("--output-root", required=True, type=Path)
        if name == "build":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "bounds":
        lo, hi = _derive_action_bounds()
        diff = max(float(np.abs(lo - ACTION_Q01).max()),
                   float(np.abs(hi - ACTION_Q99).max()))
        print(json.dumps({"frozen_q01": ACTION_Q01, "frozen_q99": ACTION_Q99,
                          "derived_q01": lo.tolist(), "derived_q99": hi.tolist(),
                          "max_abs_diff": diff}, indent=2))
        return 0 if diff <= 1e-3 else 1
    if args.command == "verify" or getattr(args, "verify_only", False):
        report = verify_conversion(args.bucket_root, args.output_root)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    report = build(args.bucket_root, args.output_root, workers=args.workers)
    verify = report.get("verify", {})
    compact = {"stats_sha256": report["stats_sha256"], "leakage": report["leakage"],
               "splits": {s: {"tasks": len(r["tasks"]),
                              "manifest_samples": r["manifest_samples"],
                              "status": {t: v["status"] for t, v in r["tasks"].items()}}
                          for s, r in report["splits"].items()},
               "verify": {"checked": verify.get("checked"), "ok": verify.get("ok"),
                          "failures": verify.get("failures")},
               "audit": str(args.output_root / "audit.json")}
    print(json.dumps(compact, indent=2))
    return 0 if verify.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

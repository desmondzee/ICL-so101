"""Convert one screened SO-101 dataset into the curated format.

Curated format (LeRobot v3.0):
- action, observation.state: [6] LeRobot degrees (zero at mid-range, rest pose on the lift/elbow stops), gripper 0-100.
- action.ee, observation.state.ee: [7] gripper-site x, y, z (m, robot base frame), rotation vector, gripper 0-100; FK of the above.
- observation.images.front: the third-person camera; observation.images.wrist when the source has one. Other cameras are dropped.
  640x480 AV1, 30 fps, letterboxed when the source is not 4:3; per-episode camera swaps in the source are undone.
- meta/curation.json: source, revision, units, offsets, camera map, dropped episodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from curate.io import load, stack
from curate.robot import NAMES, Kinematics, rest_offsets
from curate.screen import units as detect_units

EE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
MIN_EPISODE_S = 2.0
MIN_MOTION_DEG = 5.0
MAX_JUMP_DEG = 30.0
WIDTH, HEIGHT = 640, 480
ENCODE = ["-c:v", "libsvtav1", "-preset", "10", "-crf", "30", "-g", "2", "-svtav1-params", "lp=2", "-pix_fmt", "yuv420p", "-an"]
ENCODE_TIMEOUT_S = 300


def _encode(src: Path, dst: Path, frames: int | None = None) -> None:
    """Re-encode to 640x480; with `frames`, trim or repeat the last frame so the video has exactly that many."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    count = ["-frames:v", str(frames)] if frames else []
    if frames:
        vf += ",tpad=stop_mode=clone:stop=5"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vf", vf, *count, *ENCODE, str(dst)]
    for attempt in range(2):
        try:
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=ENCODE_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            if attempt:
                raise


def _reindex_v21(work: Path) -> None:
    """Take episode_index from the file name and rebuild the global index; some sources carry stale values."""
    start = 0
    for f in sorted((work / "data").rglob("episode_*.parquet"), key=lambda f: int(f.stem.split("_")[-1])):
        df = pd.read_parquet(f)
        df["episode_index"] = int(f.stem.split("_")[-1])
        df["index"] = np.arange(start, start + len(df))
        start += len(df)
        df.to_parquet(f)


def _copy_v21(src: Path, work: Path, keep: list[str], swapped: set[int]) -> None:
    shutil.copytree(src / "meta", work / "meta")
    shutil.copytree(src / "data", work / "data")
    _reindex_v21(work)
    other = dict(zip(keep, reversed(keep))) if len(keep) == 2 else {}
    lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()}
    jobs = []
    for chunk in (src / "videos").glob("chunk-*"):
        for key in keep:
            for mp4 in sorted((chunk / key).glob("*.mp4")):
                ep = int(mp4.stem.split("_")[-1])
                source = chunk / other[key] / mp4.name if ep in swapped and key in other else mp4
                jobs.append((source, work / "videos" / chunk.name / key / mp4.name, lengths.get(ep)))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: _encode(*j), jobs))
    info = json.loads((work / "meta/info.json").read_text())
    info["features"] = {k: f for k, f in info["features"].items() if f["dtype"] != "video" or k in keep}
    (work / "meta/info.json").write_text(json.dumps(info, indent=4))
    stats = work / "meta/episodes_stats.jsonl"
    rows = [json.loads(l) for l in stats.read_text().splitlines() if l.strip()]
    for r in rows:
        r["stats"] = {k: v for k, v in r["stats"].items() if k in info["features"]}
    stats.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _to_v30(src: Path, work: Path, keep: list[str], swapped: set[int]) -> None:
    info = json.loads((src / "meta/info.json").read_text())
    if info["codebase_version"].startswith("v3"):
        if swapped:
            raise NotImplementedError("camera swaps in a v3.0 source")
        shutil.copytree(src, work, ignore=shutil.ignore_patterns(".cache", "videos"))
        jobs = [(mp4, work / mp4.relative_to(src)) for key in keep for mp4 in sorted((src / "videos" / key).rglob("*.mp4"))]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda j: _encode(*j), jobs))
        return
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    _copy_v21(src, work, keep, swapped)
    convert_dataset(repo_id=f"local/{work.name}", root=work, push_to_hub=False)
    shutil.rmtree(work.parent / f"{work.name}_old")


def _rename_cameras(root: Path, cams: dict[str, str]) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    videos = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    for key in videos:
        if key not in cams:
            del info["features"][key]
            shutil.rmtree(root / "videos" / key, ignore_errors=True)
    for old, new in cams.items():
        feat = info["features"].pop(old)
        feat["shape"] = [HEIGHT, WIDTH, 3]
        feat["info"] = (feat.get("info") or {}) | {"video.height": HEIGHT, "video.width": WIDTH, "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.fps": info["fps"]}
        info["features"][new] = feat
        if old != new:
            (root / "videos" / old).rename(root / "videos" / new)
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    for f in sorted((root / "meta/episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        drop = [c for c in df.columns for k in videos if k not in cams and (c.startswith(f"videos/{k}/") or c.startswith(f"stats/{k}/"))]
        df = df.drop(columns=drop)
        for old, new in cams.items():
            df = df.rename(columns={c: c.replace(f"/{old}/", f"/{new}/") for c in df.columns})
        df.to_parquet(f)
    stats = json.loads((root / "meta/stats.json").read_text())
    stats = {k: v for k, v in stats.items() if k not in videos or k in cams}
    for old, new in cams.items():
        if old in stats:
            stats[new] = stats.pop(old)
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))


def _joints(root: Path, unit: str, offset: np.ndarray, kin: Kinematics) -> None:
    """Rewrite action and observation.state in place as LeRobot degrees."""
    for f in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for col in ("action", "observation.state"):
            deg = (kin.to_degrees(stack(df[col]), unit) + offset).astype(np.float32)
            df[col] = list(deg)
        df.to_parquet(f)
    info = json.loads((root / "meta/info.json").read_text())
    for col in ("action", "observation.state"):
        info["features"][col]["names"] = NAMES
    info["robot_type"] = "so101_follower"
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))


def _canonical_features(root: Path) -> None:
    """Rewrite feature definitions from one template so every curated dataset can be aggregated."""
    info = json.loads((root / "meta/info.json").read_text())
    video = {
        "dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channels"],
        "info": {"video.height": HEIGHT, "video.width": WIDTH, "video.codec": "av1", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "video.fps": info["fps"], "video.channels": 3, "has_audio": False},
    }
    features = {
        "action": {"dtype": "float32", "shape": [6], "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": [6], "names": NAMES},
        "action.ee": {"dtype": "float32", "shape": [7], "names": EE_NAMES},
        "observation.state.ee": {"dtype": "float32", "shape": [7], "names": EE_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in ("observation.images.front", "observation.images.wrist"):
        if key in info["features"]:
            features[key] = video
    extra = set(info["features"]) - set(features)
    if extra:
        raise ValueError(f"{root}: unexpected features {sorted(extra)}")
    info["features"] = features
    info["robot_type"] = "so101_follower"
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))


def _bad_episodes(root: Path, fps: float) -> dict[int, str]:
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "observation.state"]) for f in sorted((root / "data").rglob("*.parquet"))])
    bad = {}
    for ep, g in df.groupby("episode_index"):
        s = stack(g["observation.state"])[:, :5]
        if len(s) < MIN_EPISODE_S * fps:
            bad[int(ep)] = "short"
        elif np.ptp(s, axis=0).max() < MIN_MOTION_DEG:
            bad[int(ep)] = "still"
        elif len(s) > 1 and np.abs(np.diff(s, axis=0)).max() > MAX_JUMP_DEG:
            bad[int(ep)] = "jump"
    return bad


def convert(src: Path, out: Path, front: str, wrist: str | None, provenance: dict, swapped: set[int] = frozenset(), drop: dict[int, str] | None = None) -> Path:
    from lerobot.datasets.dataset_tools import delete_episodes, modify_features, recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = load(src)
    by_name = {v: k for k, v in source.cameras.items()}
    cams = {by_name[front]: "observation.images.front"} | ({by_name[wrist]: "observation.images.wrist"} if wrist else {})
    names = source.info["features"]["action"]["names"]
    frames = source.frames()
    unit = detect_units(names, stack(frames["observation.state"])[:, :5])
    kin = Kinematics()
    offset = np.zeros(6)
    if unit != "norm":
        offset = rest_offsets(kin.to_degrees(stack(frames["observation.state"]), unit))
        if offset is None:
            raise ValueError(f"{src}: no rest pose to anchor lift/elbow")

    work = out.parent / f"_work_{out.name}"
    for p in (work, work.parent / f"{work.name}_v30", work.parent / f"{work.name}_old"):
        shutil.rmtree(p, ignore_errors=True)
    _to_v30(src, work, list(cams), set(swapped))
    _rename_cameras(work, cams)
    _joints(work, unit, offset, kin)

    ds = LeRobotDataset(f"local/{work.name}", root=work, video_backend="pyav")
    data = pd.concat([pd.read_parquet(f, columns=["action", "observation.state"]) for f in sorted((work / "data").rglob("*.parquet"))])
    ee = {}
    for col in ("action", "observation.state"):
        x = stack(data[col])
        rows = np.empty(len(x), dtype=object)
        rows[:] = list(np.hstack([kin.ee(x[:, :5]), x[:, 5:6]]).astype(np.float32))
        ee[f"{col}.ee"] = (rows, {"dtype": "float32", "shape": [7], "names": EE_NAMES})
    staged = work.parent / f"{work.name}_ee"
    shutil.rmtree(staged, ignore_errors=True)
    ds = modify_features(ds, add_features=ee, output_dir=staged, repo_id=f"local/{staged.name}")
    shutil.rmtree(work)
    ds = recompute_stats(ds)

    bad = _bad_episodes(staged, ds.fps) | {int(k): v for k, v in (drop or {}).items()}
    shutil.rmtree(out, ignore_errors=True)
    if bad:
        delete_episodes(ds, list(bad), output_dir=out, repo_id=f"local/{out.name}")
        shutil.rmtree(staged)
    else:
        staged.rename(out)
    _canonical_features(out)
    curation = provenance | {
        "units": unit,
        "rest_offset_deg": np.round(offset[:5], 2).tolist(),
        "cameras": {"observation.images.front": front, "observation.images.wrist": wrist},
        "dropped_episodes": {str(k): v for k, v in sorted(bad.items())},
        "swapped_camera_episodes": sorted(swapped),
        "format": __doc__.split("Curated format (LeRobot v3.0):")[1].strip(),
    }
    (out / "meta/curation.json").write_text(json.dumps(curation, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--front", required=True, help="original name of the third-person camera")
    p.add_argument("--wrist", help="original name of the wrist camera")
    p.add_argument("--provenance", type=json.loads, default={})
    args = p.parse_args(argv)
    print(convert(args.src, args.out, args.front, args.wrist, args.provenance))
    return 0


if __name__ == "__main__":
    sys.exit(main())

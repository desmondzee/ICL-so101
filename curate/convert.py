"""Convert one screened SO-101 dataset into the curated format.

Curated format (LeRobot v3.0):
- action, observation.state: [6] LeRobot degrees (zero at mid-range, rest pose on the lift/elbow stops), gripper 0-100.
- action.ee, observation.state.ee: [7] gripper-site x, y, z (m, robot base frame), rotation vector, gripper 0-100; FK of the above.
- observation.images.front: the third-person camera; observation.images.wrist when the source has one. Other cameras are dropped.
- meta/curation.json: source, revision, units, offsets, camera map, dropped episodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


def _copy_v21(src: Path, work: Path, keep: list[str]) -> None:
    shutil.copytree(src / "meta", work / "meta")
    shutil.copytree(src / "data", work / "data")
    for chunk in (src / "videos").glob("chunk-*"):
        for key in keep:
            shutil.copytree(chunk / key, work / "videos" / chunk.name / key)
    info = json.loads((work / "meta/info.json").read_text())
    info["features"] = {k: f for k, f in info["features"].items() if f["dtype"] != "video" or k in keep}
    (work / "meta/info.json").write_text(json.dumps(info, indent=4))
    stats = work / "meta/episodes_stats.jsonl"
    rows = [json.loads(l) for l in stats.read_text().splitlines() if l.strip()]
    for r in rows:
        r["stats"] = {k: v for k, v in r["stats"].items() if k in info["features"]}
    stats.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _to_v30(src: Path, work: Path, keep: list[str]) -> None:
    info = json.loads((src / "meta/info.json").read_text())
    if info["codebase_version"].startswith("v3"):
        shutil.copytree(src, work, ignore=shutil.ignore_patterns(".cache"))
        return
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    _copy_v21(src, work, keep)
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
        info["features"][new] = info["features"].pop(old)
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


def convert(src: Path, out: Path, front: str, wrist: str | None, provenance: dict) -> Path:
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
    _to_v30(src, work, list(cams))
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

    bad = _bad_episodes(staged, ds.fps)
    shutil.rmtree(out, ignore_errors=True)
    if bad:
        delete_episodes(ds, list(bad), output_dir=out, repo_id=f"local/{out.name}")
        shutil.rmtree(staged)
    else:
        staged.rename(out)
    curation = provenance | {
        "units": unit,
        "rest_offset_deg": np.round(offset[:5], 2).tolist(),
        "cameras": {"observation.images.front": front, "observation.images.wrist": wrist},
        "dropped_episodes": {str(k): v for k, v in bad.items()},
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

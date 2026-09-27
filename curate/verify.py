"""Check every curated dataset: loads, canonical features, decodable episode ends, sane joints and FK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from curate.io import stack
from curate.robot import NAMES, Kinematics

EE_Z_MIN = -0.03
BELOW_TABLE_OK = {"open_close"}


def verify(root: Path, kin: Kinematics) -> dict:
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{root.name}", root=root, video_backend="pyav")
    problems = []
    if ds.meta.robot_type != "so101_follower" or ds.meta.fps != 30:
        problems.append(f"robot_type/fps {ds.meta.robot_type}/{ds.meta.fps}")
    if ds.meta.features["action"]["names"] != NAMES:
        problems.append("action names")
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
    state, ee = stack(data["observation.state"]), stack(data["observation.state.ee"])
    if np.isnan(state).any() or np.isnan(ee).any():
        problems.append("NaN")
    lo, hi = kin.range_deg[:, 0] - 5, kin.range_deg[:, 1] + 5
    out = ((state[:, :5] < lo) | (state[:, :5] > hi)).any(1).mean()
    if out > 0.01:
        problems.append(f"{out:.1%} frames outside joint limits")
    cur = json.loads((root / "meta/curation.json").read_text())
    warnings = []
    if np.percentile(ee[:, 2], 1) < EE_Z_MIN:
        (warnings if cur.get("family") in BELOW_TABLE_OK else problems).append(f"EE z p1 {np.percentile(ee[:, 2], 1):.3f}")
    ends = data.groupby("episode_index")["index"].agg(["min", "max"])
    for ep, (first, last) in ends.iterrows():
        for i in (first, last):
            try:
                item = ds[int(i)]
                for k in ds.meta.video_keys:
                    if tuple(item[k].shape) != (3, 480, 640):
                        problems.append(f"episode {ep}: {k} shape {tuple(item[k].shape)}")
            except Exception as e:
                problems.append(f"episode {ep} frame {i}: {type(e).__name__}: {str(e)[:80]}")
                break
    return {
        "dataset": root.name, "family": cur.get("family"), "episodes": ds.num_episodes, "frames": ds.num_frames,
        "hours": round(ds.num_frames / ds.fps / 3600, 2), "wrist": "observation.images.wrist" in ds.meta.video_keys,
        "units": cur["units"], "dropped": len(cur["dropped_episodes"]), "swapped": len(cur.get("swapped_camera_episodes", [])),
        "tasks": sorted(set(ds.meta.tasks.index)) if hasattr(ds.meta.tasks, "index") else None, "problems": problems, "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/so101_curated"))
    p.add_argument("--out", type=Path, default=Path("data/so101_curated/_verify.json"))
    args = p.parse_args(argv)
    kin = Kinematics()
    roots = sorted(r for r in args.root.iterdir() if r.is_dir() and not r.name.startswith("_") and (r / "meta/curation.json").exists())
    merged = args.root / "_merged_front"
    rows = [verify(r, kin) for r in roots + ([merged] if (merged / "meta/info.json").exists() and (merged / "meta/curation.json").exists() else [])]
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    bad = 0
    for r in rows:
        bad += bool(r["problems"])
        print(f"{'FAIL' if r['problems'] else 'ok  '} {r['dataset'][:50]:50s} {r['family'] or '':20s} eps={r['episodes']:4d} h={r['hours']:5.2f} wrist={int(r['wrist'])} dropped={r['dropped']:3d} swapped={r['swapped']:2d} {r['problems'][:2]}{' warn ' + str(r['warnings']) if r['warnings'] else ''}")
    per = [r for r in rows if r["family"] != "merged"]
    print(f"per-source total: {len(per)} datasets, {sum(r['episodes'] for r in per)} episodes, {sum(r['hours'] for r in per):.2f} h")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

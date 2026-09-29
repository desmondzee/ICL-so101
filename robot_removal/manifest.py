"""Build hashed input manifests for robot removal.

Each manifest line is one job: dataset/episode IDs, source image path,
SHA-256, dimensions, the robot box hint and task-object boxes from
pair.json, plus provenance fields.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def robot_entry(pair: dict) -> dict | None:
    for e in pair.get("scene", {}).get("entities", []):
        if e.get("id") == "robot" or (e.get("kind") == "actor" and "robot" in e.get("name", "")):
            return e
    return None


def build_manifest(full_root: Path, accepted: set[str] | None = None) -> list[dict]:
    """One record per episode dir containing first.jpg + pair.json."""
    rows = []
    for ep in sorted(full_root.glob("*/*/")):
        first, pair_p = ep / "first.jpg", ep / "pair.json"
        if not first.exists() or not pair_p.exists():
            continue
        dataset_id, episode_id = ep.parent.name, ep.name
        key = f"{dataset_id}/{episode_id}"
        if accepted is not None and key not in accepted:
            continue
        pair = json.loads(pair_p.read_text())
        robot = robot_entry(pair)
        objects = [
            {"id": e["id"], "name": e.get("name"), "box_2d": e["box_2d"]}
            for e in pair.get("scene", {}).get("entities", [])
            if e.get("id") != "robot" and e.get("box_2d")
        ]
        with Image.open(first) as im:
            w, h = im.size
        rows.append(
            {
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "key": key,
                "image": str(first.resolve()),
                "image_sha256": sha256(first),
                "width": w,
                "height": h,
                "pair_json": str(pair_p.resolve()),
                "pair_sha256": sha256(pair_p),
                "robot_box_2d": robot.get("box_2d") if robot else None,
                "robot_name": robot.get("name") if robot else None,
                "object_boxes": objects,
                "task": pair.get("task", {}).get("instruction") or pair.get("task", {}).get("name"),
                "shot": pair.get("scene", {}).get("view", {}).get("shot"),
                "bucket_prefix": f"work/so101_pairs/{key}",
            }
        )
    return rows


def load_accepted_keys(path: Path) -> set[str]:
    """Accepted episode keys from the bucket's accepted.json."""
    return {f"{e['dataset']}/{e['episode']}" for e in json.loads(path.read_text())}


def write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("data/robot_removal/full")
    out = Path(argv[2]) if len(argv) > 2 else Path("data/robot_removal/full_inputs.jsonl")
    accepted_p = Path(argv[3]) if len(argv) > 3 else Path("data/robot_removal/accepted.json")
    accepted = load_accepted_keys(accepted_p) if accepted_p.exists() else None
    rows = build_manifest(root, accepted)
    write_jsonl(rows, out)
    if accepted is not None:
        got = {r["key"] for r in rows}
        print(f"accepted={len(accepted)} downloaded+accepted={len(rows)} missing={len(accepted - got)} extra={len(got - accepted)}")
        for k in sorted(accepted - got)[:20]:
            print("  accepted but missing locally:", k)
    missing_box = [r["key"] for r in rows if not r["robot_box_2d"]]
    print(f"{len(rows)} inputs -> {out}; missing robot_box_2d: {len(missing_box)}")
    for k in missing_box[:20]:
        print("  no robot box:", k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

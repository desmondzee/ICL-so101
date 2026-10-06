"""Robot-review evidence, verdict import and the fal-ready index.

Evidence (built by the recorder inside the staging directory, so it is frozen and
hashed with the episode): dense front and wrist contact sheets covering the start,
every oracle skill transition, gripper commands, grasp/release events, the
neighbourhood of each failed physics check's worst sample, the final settled
second and a 1-second filler; plus ``robot_review_request.json`` (task text,
ordered actions, variation metadata, physics summary, artifact hashes).

Workflow (no paid APIs; reviewers are Opus subagents run by
``sim/train/robot_review_workflow.js``):

    python -m sim.train.review queue  [--root R] [--out args.json]   # pending physics_approved episodes
    # run the workflow with args.json; save its return value as verdicts.json
    python -m sim.train.review import verdicts.json [--root R]       # accept -> robot_approved, reject -> rejected
    python -m sim.train.review fal-ready [--root R]                  # rebuild R/fal_ready.json from manifests

Only ``physics_approved`` + a judge accept confirmed by a distinct verifier, both
bound to the current request hash, advances to ``robot_approved``.
``needs_detail``, a missing verifier or inconsistent output stay pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image, ImageDraw

from .model import EpisodeKey, EpisodeState
from .store import EpisodeStore, EvidenceMismatch, StoreError, atomic_write_json, sha256_file

REQUEST_NAME = "robot_review_request.json"
SHEET_DIR = "review"
THUMB = (320, 240)
SELECT_CHUNK = 40
COLUMNS, ROWS = 6, 4
FILLER_S = 1.0
FINAL_S = 1.0
VERDICTS = ("accept", "reject", "needs_detail")
REVIEW_SCHEMA_VERSION = 1


def key_name(key: EpisodeKey) -> str:
    return f"{key.task}/episode_{key.seed}"


def parse_key(name: str) -> EpisodeKey:
    task, _, episode = name.partition("/")
    if not episode.startswith("episode_") or not episode[8:].isdigit():
        raise ValueError(f"invalid episode key: {name}")
    return EpisodeKey(task, int(episode[8:]))


# ----- evidence -------------------------------------------------------------------------------------------
def select_frames(n_frames: int, events: list[dict], fps: int = 30) -> list[dict]:
    """Ordered unique frames with every reason they were chosen.

    Grasp/release/gripper events get +-6 frame neighbours, violations +-3/+-6, the
    final settled second every 5 frames; a 1-second filler bounds any gap.
    """
    if n_frames < 1:
        raise ValueError("episode has no frames")
    reasons: dict[int, list[str]] = {}

    def add(frame, reason):
        frame = min(max(int(frame), 0), n_frames - 1)
        if reason not in reasons.setdefault(frame, []):
            reasons[frame].append(reason)

    add(0, "start")
    for frame in range(0, n_frames, max(1, int(round(FILLER_S * fps)))):
        add(frame, "filler")
    for event in events:
        kind, frame = event["kind"], event["frame"]
        label = f"{kind}:{event['label']}" if event.get("label") else kind
        add(frame, label)
        if kind in ("grasp", "release", "gripper_close", "gripper_open"):
            for d in (-6, 6):
                add(frame + d, f"{label}{d:+d}")
        elif kind == "violation":
            for d in (-6, -3, 3, 6):
                add(frame + d, f"{label}{d:+d}")
    final = max(1, int(round(FINAL_S * fps)))
    for frame in range(max(0, n_frames - final), n_frames, 5):
        add(frame, "final_second")
    add(n_frames - 1, "final")
    # Fillers are only informative where nothing else was chosen.
    return [{"frame": f, "time_s": round(f / fps, 3),
             "reasons": [r for r in reasons[f] if r != "filler"] or ["filler"]} for f in sorted(reasons)]


def extract_frames(video: Path, frames: list[int], size=THUMB) -> list[Image.Image]:
    """Decode exactly the requested frame indices (one ffmpeg pass, frame-accurate select)."""
    if not frames:
        return []
    if len(frames) > SELECT_CHUNK:
        return [im for i in range(0, len(frames), SELECT_CHUNK) for im in extract_frames(video, frames[i:i + SELECT_CHUNK], size)]
    expr = "+".join(f"eq(n\\,{f})" for f in frames)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-vf",
         f"select='{expr}',scale={size[0]}:{size[1]}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    step = size[0] * size[1] * 3
    if len(raw) != step * len(frames):
        raise ValueError(f"{video}: decoded {len(raw) // step} of {len(frames)} review frames")
    return [Image.frombytes("RGB", size, raw[i * step:(i + 1) * step]) for i in range(len(frames))]


def _sheet(images: list[Image.Image], labels: list[str]) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + COLUMNS - 1) // COLUMNS
    grid = Image.new("RGB", (COLUMNS * w, rows * h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (image, label) in enumerate(zip(images, labels)):
        x, y = (i % COLUMNS) * w, (i // COLUMNS) * h
        grid.paste(image, (x, y))
        lines = [label[j:j + 50] for j in range(0, len(label), 50)][:3]
        draw.rectangle([x, y, x + 6 * max(map(len, lines)) + 6, y + 12 * len(lines) + 2], fill="black")
        for k, line in enumerate(lines):
            draw.text((x + 3, y + 1 + 12 * k), line, fill="white")
    return grid


def build_sheets(directory: Path, n_frames: int, events: list[dict], fps: int = 30) -> dict:
    """Write review/{front,wrist}_NN.jpg and review/frames.json from the frozen videos."""
    selection = select_frames(n_frames, events, fps)
    out = directory / SHEET_DIR
    out.mkdir(exist_ok=True)
    per_page = COLUMNS * ROWS
    pages = [selection[i:i + per_page] for i in range(0, len(selection), per_page)]
    sheets = {}
    for camera in ("front", "wrist"):
        images = extract_frames(directory / f"robot_{camera}.mp4", [s["frame"] for s in selection])
        sheets[camera] = []
        for p, page in enumerate(pages):
            chunk = images[p * per_page:(p + 1) * per_page]
            labels = [f"{camera} f{s['frame']} {s['time_s']:.2f}s {','.join(s['reasons'])}" for s in page]
            name = f"{SHEET_DIR}/{camera}_{p + 1:02d}.jpg"
            _sheet(chunk, labels).save(directory / name, quality=88)
            sheets[camera].append({"path": name, "frames": [s["frame"] for s in page]})
    index = {"fps": fps, "frames": n_frames, "selection": selection, "sheets": sheets}
    write_json(out / "frames.json", index)
    return index


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("xb") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def physics_summary(physics: dict) -> dict:
    counts: dict[str, int] = {}
    for violation in physics.get("violations", []):
        counts[violation["check"]] = counts.get(violation["check"], 0) + 1
    return {"accepted": physics.get("accepted") is True, "checks": physics.get("checks", {}),
            "failed_checks": sorted(k for k, ok in physics.get("checks", {}).items() if not ok),
            "maxima": physics.get("maxima", {}), "violation_counts": dict(sorted(counts.items()))}


def write_request(directory: Path, metadata: dict, artifact_hashes: dict[str, str]) -> dict:
    """The frozen review request; it hashes every other artifact of the episode."""
    episode = json.loads((directory / "episode.json").read_text())
    frames = json.loads((directory / SHEET_DIR / "frames.json").read_text())
    physics = json.loads((directory / "physics.json").read_text())
    request = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "key": key_name(EpisodeKey(metadata["task"], metadata["seed"])),
        "task": metadata["task"], "seed": metadata["seed"], "family": metadata["family"],
        "instruction": metadata["instruction"],
        "ordered_actions": [{"step": i + 1, "object": o, "action": t}
                            for i, (o, t) in enumerate(zip(metadata["action_order"], metadata["action_text"]))],
        "frames": frames["frames"], "fps": frames["fps"], "duration_s": round(frames["frames"] / frames["fps"], 3),
        "variation": {"visual_config": episode["visual_config"], "object_poses_start": episode["object_poses_start"],
                      "object_poses_end": episode["object_poses_end"], "distractors": episode.get("distractors", [])},
        "oracle_events": episode["events"],
        "physics": physics_summary(physics),
        "automated_qa": metadata.get("automated_qa", {}),
        "evidence": {
            "sheets": frames["sheets"], "frame_index": f"{SHEET_DIR}/frames.json",
            "videos": {"front": "robot_front.mp4", "wrist": "robot_wrist.mp4"},
            "robot_free_endpoints": {"first": "first.png", "last": "last.png"},
            "data": "robot_data.parquet", "physics": "physics.json",
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "verdict_values": list(VERDICTS),
    }
    write_json(directory / REQUEST_NAME, request)
    return request


# ----- review queue ----------------------------------------------------------------------------------------
def _records(store: EpisodeStore):
    for path in sorted((store.root / "candidates").glob("*/episode_*/manifest.json")):
        document = json.loads(path.read_text())
        yield store.load(EpisodeKey(**document["key"]))


def queue(root: Path) -> dict:
    """Workflow args for every physics_approved episode awaiting robot review."""
    store = EpisodeStore(root)
    items = []
    for record in _records(store):
        if record.state is EpisodeState.PHYSICS_APPROVED and REQUEST_NAME in record.manifest.artifacts:
            items.append({"key": key_name(record.manifest.key),
                          "dir": str(store.episode_dir(record.manifest.key).resolve()),
                          "request_sha256": record.manifest.artifacts[REQUEST_NAME]})
    return {"root": str(Path(root).resolve()), "items": items}


# ----- verdict import --------------------------------------------------------------------------------------
def _reviewer_ok(review: Any, role: str) -> bool:
    reviewer = review.get("reviewer") if isinstance(review, dict) else None
    return (isinstance(reviewer, dict) and isinstance(reviewer.get("label"), str) and reviewer["label"]
            and reviewer.get("role") == role and isinstance(reviewer.get("model"), str) and reviewer["model"])


def _texts(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(s, str) and s.strip() for s in value)


def decide(entry: dict) -> tuple[str, list[str]]:
    """Recompute the outcome from the raw reviews; never trust a reported 'final'."""
    problems = []
    judge, verify = entry.get("judge"), entry.get("verify")
    if not isinstance(judge, dict) or judge.get("verdict") not in VERDICTS:
        return "pending", ["missing_or_invalid_judge"]
    if not _reviewer_ok(judge, "judge"):
        problems.append("judge_identity")
    if not _texts(judge.get("reasons")):
        problems.append("judge_reasons")
    if not _texts(judge.get("inspected_evidence")):
        problems.append("judge_evidence")
    if problems:
        return "pending", problems
    if judge["verdict"] == "needs_detail":
        return "pending", ["needs_detail"]
    if judge["verdict"] == "reject":
        return "reject", []
    if not isinstance(verify, dict) or type(verify.get("confirmed")) is not bool:
        return "pending", ["missing_verifier"]
    if not _reviewer_ok(verify, "verifier"):
        problems.append("verifier_identity")
    elif verify["reviewer"]["label"] == judge["reviewer"]["label"]:
        problems.append("verifier_not_independent")
    if not _texts(verify.get("reasons")):
        problems.append("verifier_reasons")
    if not _texts(verify.get("inspected_evidence")):
        problems.append("verifier_evidence")
    if problems:
        return "pending", problems
    return ("accept", []) if verify["confirmed"] else ("reject", ["verifier_refuted"])


def import_verdicts(root: Path, results: dict) -> dict:
    """Apply workflow results. Hash/state problems never advance an episode.

    Returns {"robot_approved": [...], "rejected": [...], "pending": {key: reasons}, "refused": {key: reason}}.
    """
    store = EpisodeStore(root)
    payload = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    archive = Path(root) / "reviews" / "robot" / f"import_{hashlib.sha256(payload).hexdigest()}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        atomic_write_json(archive, results, overwrite=False)
    summary = {"robot_approved": [], "rejected": [], "pending": {}, "refused": {}}
    for name, entry in sorted(results.items()):
        try:
            key = parse_key(name)
            if not isinstance(entry, dict) or entry.get("key", name) != name:
                raise ValueError("entry key mismatch")
            record = store.load(key)  # verifies every frozen artifact hash
            expected = record.manifest.artifacts.get(REQUEST_NAME)
            if expected is None:
                raise ValueError("episode has no review request (automated QA did not approve it)")
            if entry.get("request_sha256") != expected:
                raise EvidenceMismatch("verdict was issued for a different review request")
            if sha256_file(store.episode_dir(key) / REQUEST_NAME) != expected:
                raise EvidenceMismatch("review request changed on disk")
        except (StoreError, ValueError, KeyError, OSError) as exc:
            summary["refused"][name] = f"{type(exc).__name__}: {exc}"
            continue
        decision, problems = decide(entry)
        reported = entry.get("final")
        if reported is not None and decision != "pending" and reported != decision:
            decision, problems = "pending", problems + [f"inconsistent_final:{reported}"]
        if decision == "pending":
            summary["pending"][name] = problems
            continue
        evidence = {"stage": "robot_review", "decision": decision, "request_sha256": expected,
                    "judge": entry["judge"], "verify": entry.get("verify"), "problems": problems,
                    "artifact_hashes": dict(record.manifest.artifacts)}
        target = EpisodeState.ROBOT_APPROVED if decision == "accept" else EpisodeState.REJECTED
        try:
            store.transition(key, EpisodeState.PHYSICS_APPROVED, target, evidence)
        except StoreError as exc:
            summary["refused"][name] = f"{type(exc).__name__}: {exc}"
            continue
        summary["robot_approved" if decision == "accept" else "rejected"].append(name)
    return summary


# ----- fal-ready index -------------------------------------------------------------------------------------
def _approval(record) -> dict:
    for entry in record.history:
        if (entry["old_state"], entry["new_state"]) == (EpisodeState.PHYSICS_APPROVED.value, EpisodeState.ROBOT_APPROVED.value):
            return entry["evidence"]
    raise StoreError("robot_approved without a robot review transition")


def build_fal_ready(root: Path) -> dict:
    """Regenerate fal_ready.json from manifests: exactly the robot_approved, hash-valid episodes."""
    root = Path(root)
    store = EpisodeStore(root)
    episodes = []
    for record in _records(store):  # store.load re-hashes every artifact
        if record.state is not EpisodeState.ROBOT_APPROVED:
            continue
        manifest, key = record.manifest, record.manifest.key
        approval = _approval(record)
        if approval.get("decision") != "accept" or approval.get("artifact_hashes") != manifest.artifacts:
            raise EvidenceMismatch(f"approval does not cover current artifacts: {key_name(key)}")
        meta = manifest.metadata
        rel = store.episode_dir(key).relative_to(root)
        artifact = lambda name: {"path": str(rel / name), "sha256": manifest.artifacts[name]}
        episodes.append({
            "key": key_name(key), "task": key.task, "seed": key.seed, "family": meta["family"],
            "instruction": meta["instruction"], "action_order": meta["action_order"], "action_text": meta["action_text"],
            "frames": meta["frames"], "fps": meta["fps"], "robot_duration_s": round(meta["frames"] / meta["fps"], 3),
            "config_hash": manifest.config_hash, "visual_config_hash": manifest.visual_config_hash,
            "manifest_sha256": sha256_file(store.manifest_path(key)),
            "first": artifact("first.png"), "last": artifact("last.png"),
            "robot_front": artifact("robot_front.mp4"), "robot_wrist": artifact("robot_wrist.mp4"),
            "robot_data": artifact("robot_data.parquet"), "review_request": artifact(REQUEST_NAME),
            "reviewers": {"judge": approval["judge"]["reviewer"]["label"],
                          "verifier": approval["verify"]["reviewer"]["label"]},
        })
    index = {"schema_version": 1, "state": EpisodeState.ROBOT_APPROVED.value,
             "episodes_total": len(episodes), "episodes": episodes}
    atomic_write_json(root / "fal_ready.json", index)
    return index


def main(argv=None) -> int:
    from .record import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--root", type=Path, default=ROOT)
    q.add_argument("--out", type=Path)
    i = sub.add_parser("import")
    i.add_argument("verdicts", type=Path)
    i.add_argument("--root", type=Path, default=ROOT)
    f = sub.add_parser("fal-ready")
    f.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "queue":
        value = queue(args.root)
        if args.out:
            args.out.write_text(json.dumps(value, indent=1))
        print(json.dumps(value, indent=1))
    elif args.command == "import":
        summary = import_verdicts(args.root, json.loads(args.verdicts.read_text()))
        build_fal_ready(args.root)
        print(json.dumps(summary, indent=1))
        return 1 if summary["refused"] else 0
    else:
        index = build_fal_ready(args.root)
        print(f"{index['episodes_total']} fal-ready episodes -> {Path(args.root) / 'fal_ready.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

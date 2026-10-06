"""Human-pair review for generated attempts: sheets, workflow queue and two-stage verdict import.

Sheets use the ``humangen/curate_workflow.js`` layout, one folder per generated attempt:
``<root>/review/human/<task>/episode_<seed>_a<N>/`` with ``human.mp4`` (copy of the frozen
attempt video), ``human_frames.jpg`` (12 frames), ``robot_frames.jpg`` (8 frames of the robot
front video, first and last included) and ``info.json`` (instruction, ordered steps, goals,
objects, a note that the scene is a rendered simulation, and the hashes the verdict binds to).

    python -m sim.train.human_review queue [--root R] [--out args.json]
        # sheets for every completed, unreviewed attempt; args.json = {dir, keys, items}
    # Workflow: sim/train/human_review_workflow.js with args.json (Opus judges, 5 per agent;
    # an independent Opus verifier tries to refute every accept). Save its return value.
    python -m sim.train.human_review import verdicts.json [--root R] [--max-attempts 3]

Import recomputes each decision from the raw reviews (``sim.train.review.decide``): an accept
needs a judge accept with task_adherence >= 4 and physics >= 4, confirmed by a verifier with a
different label, both with reasons and inspected evidence, and the echoed ``human_sha256`` /
``info_sha256`` must equal the attempt's stored video hash and the current sheet files. An
accept writes the immutable ``a<N>-review`` record and advances human_complete ->
human_review_approved -> verifier_approved -> accepted. A reject writes ``a<N>-review`` with
its reasons and leaves the episode for a retry (``sim.train.generate``) or, once attempts are
exhausted, for replacement (``generation/replacements.json``). Anything else stays pending.

``humangen/curate_workflow.js`` reads this layout unchanged (args ``{dir, keys}``), but its
output has no reviewer identities, inspected-evidence lists or hash echo, so import keeps such
verdicts pending; use the sim-train copy of the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from .model import EpisodeKey, EpisodeState
from .review import decide, key_name
from .store import EpisodeStore, EvidenceMismatch, StoreError, atomic_write_json, sha256_file

REVIEW_DIR = Path("review") / "human"
MIN_SCORE = 4
NOTE = ("The scene is a rendered simulation (MuJoCo, simplified assets); a realistic human hand in the rendered scene is "
        "expected and fine. The robot reference is a scripted oracle in the same rendered scene.")


def review_key(key: EpisodeKey, attempt: str) -> str:
    return f"{key.task}/episode_{key.seed}_{attempt}"


def parse_review_key(name: str) -> tuple[EpisodeKey, str]:
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_-]*)/episode_(\d+)_(a\d+)", name)
    if not match:
        raise ValueError(f"invalid review key: {name}")
    return EpisodeKey(match.group(1), int(match.group(2))), match.group(3)


def _attempts(record) -> dict[str, dict]:
    return {a["attempt_id"]: a["evidence"] for a in record.human_attempts}


def pending_attempts(store: EpisodeStore) -> list[tuple[EpisodeKey, str, dict]]:
    """Completed attempts without a review record, on episodes still awaiting a human pair."""
    out = []
    for path in sorted((store.root / "candidates").glob("*/episode_*/state.json")):
        if json.loads(path.read_text()).get("state") != EpisodeState.HUMAN_COMPLETE.value:
            continue
        key = EpisodeKey(path.parent.parent.name, int(path.parent.name.split("_")[1]))
        record = store.load(key)
        attempts = _attempts(record)
        for attempt_id, evidence in sorted(attempts.items()):
            if attempt_id.endswith("-complete") and f"{attempt_id[:-9]}-review" not in attempts:
                out.append((key, attempt_id[:-9], evidence))
    return out


def build_sheet(store: EpisodeStore, key: EpisodeKey, attempt: str, complete: dict) -> dict:
    """Write (or verify) one sheet folder; returns the queue item."""
    from humangen.curate_dataset import _duration, _frames, _grid

    record = store.load(key)
    episode_dir = store.episode_dir(key)
    video = episode_dir / complete["video"]
    if sha256_file(video) != complete["video_sha256"]:
        raise EvidenceMismatch(f"attempt video changed: {key_name(key)} {attempt}")
    name = review_key(key, attempt)
    directory = store.root / REVIEW_DIR / name
    info_path = directory / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("human_sha256") != complete["video_sha256"] or sha256_file(directory / "human.mp4") != complete["video_sha256"]:
            raise EvidenceMismatch(f"stale review sheet {name}; delete it to rebuild")
        return {"key": name, "human_sha256": info["human_sha256"], "info_sha256": sha256_file(info_path)}
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video, directory / "human.mp4")
    duration = _duration(directory / "human.mp4")
    times = [(duration - 0.2) * i / 11 for i in range(12)]
    _grid(_frames(directory / "human.mp4", times), [f"human {t:.1f}s" for t in times], 4).save(
        directory / "human_frames.jpg", quality=85)
    meta = record.manifest.metadata
    robot = episode_dir / "robot_front.mp4"
    robot_s = meta["frames"] / meta["fps"]
    rtimes = [(robot_s - 0.1) * i / 7 for i in range(8)]
    _grid(_frames(robot, rtimes), [f"robot {t:.1f}s" for t in rtimes], 4).save(directory / "robot_frames.jpg", quality=85)
    steps = list(meta["action_text"])
    info = {
        "key": name, "episode": key_name(key), "attempt": attempt, "task": key.task, "family": meta["family"],
        "seed": key.seed, "generation_seed": complete["seed"], "instruction": meta["instruction"],
        "steps": steps, "ordered_objects": list(meta["action_order"]),
        "goals": (f"The scene ends as in the robot's last frame: {meta['instruction']} The objects are handled one at a time "
                  f"in exactly this order: {' '.join(steps)}"),
        "objects": ("every object visible in the first frame; the ones named in the steps are the task objects, the rest are "
                    "distractors or fixtures that must not move"),
        "note": NOTE,
        "human_duration_s": round(duration, 2), "robot_duration_s": round(robot_s, 2),
        "human_sha256": complete["video_sha256"],
        "endpoint_sha256": {"first.png": record.manifest.artifacts["first.png"], "last.png": record.manifest.artifacts["last.png"]},
        "robot_front_sha256": record.manifest.artifacts["robot_front.mp4"],
    }
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    return {"key": name, "human_sha256": complete["video_sha256"], "info_sha256": sha256_file(info_path)}


def queue(root: Path) -> dict:
    store = EpisodeStore(root)
    items = [build_sheet(store, key, attempt, complete) for key, attempt, complete in pending_attempts(store)]
    return {"dir": str((Path(root) / REVIEW_DIR).resolve()), "keys": [i["key"] for i in items], "items": items}


def _decision(entry: dict) -> tuple[str, list[str]]:
    """review.decide plus the score gate: a judge accept below MIN_SCORE counts as the judge's reject."""
    judge = entry.get("judge")
    extra = []
    if isinstance(judge, dict) and judge.get("verdict") == "accept":
        scores = (judge.get("task_adherence"), judge.get("physics"))
        if not all(type(s) is int for s in scores):
            return "pending", ["missing_scores"]
        if min(scores) < MIN_SCORE:
            entry, extra = {**entry, "judge": {**judge, "verdict": "reject"}}, ["scores_below_threshold"]
    decision, problems = decide(entry)
    problems = problems + extra
    reported = entry.get("final")
    if reported is not None and decision != "pending" and reported != decision:
        return "pending", problems + [f"inconsistent_final:{reported}"]
    return decision, problems


def import_verdicts(root: Path, results: dict, *, max_attempts: int | None = None) -> dict:
    """Apply workflow results. Hash/state problems never advance an episode."""
    root = Path(root)
    store = EpisodeStore(root)
    payload = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    archive = root / "reviews" / "human" / f"import_{hashlib.sha256(payload).hexdigest()}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        atomic_write_json(archive, results, overwrite=False)
    summary = {"accepted": [], "rejected": {}, "pending": {}, "refused": {}, "replacements": []}
    for name, entry in sorted(results.items()):
        try:
            key, attempt = parse_review_key(name)
            if not isinstance(entry, dict) or entry.get("key", name) != name:
                raise ValueError("entry key mismatch")
            record = store.load(key)  # re-hashes every manifest and attempt artifact
            attempts = _attempts(record)
            complete = attempts.get(f"{attempt}-complete")
            if complete is None:
                raise ValueError("no completed generation for this attempt")
            sheet = root / REVIEW_DIR / name
            current = sha256_file(store.episode_dir(key) / complete["video"])
            if not (entry.get("human_sha256") == complete["video_sha256"] == current == sha256_file(sheet / "human.mp4")):
                raise EvidenceMismatch("verdict, attempt record, episode video and sheet video hashes disagree")
            if entry.get("info_sha256") != sha256_file(sheet / "info.json"):
                raise EvidenceMismatch("review sheet info.json changed since the verdict")
            info = json.loads((sheet / "info.json").read_text())
            if info.get("endpoint_sha256") != {n: record.manifest.artifacts[n] for n in ("first.png", "last.png")}:
                raise EvidenceMismatch("sheet endpoints differ from the frozen episode endpoints")
        except (StoreError, ValueError, KeyError, OSError) as exc:
            summary["refused"][name] = f"{type(exc).__name__}: {exc}"
            continue
        decision, problems = _decision(entry)
        if decision == "pending":
            summary["pending"][name] = problems
            continue
        hashes = {complete["video"]: complete["video_sha256"],
                  **{n: record.manifest.artifacts[n] for n in ("first.png", "last.png", "robot_front.mp4")}}
        judge, verify = entry["judge"], entry.get("verify")
        reasons = list(judge.get("reasons", [])) + (list(verify.get("reasons", [])) if isinstance(verify, dict) else [])
        evidence = {"stage": "human_review", "attempt": attempt, "decision": decision, "problems": problems,
                    "reasons": reasons, "judge": judge, "verify": verify, "human_sha256": complete["video_sha256"],
                    "info_sha256": entry["info_sha256"], "artifact_hashes": hashes}
        try:
            if record.state is not EpisodeState.HUMAN_COMPLETE and f"{attempt}-review" not in attempts:
                raise StoreError(f"episode is {record.state.value}, not human_complete")
            store.record_human_attempt(key, f"{attempt}-review", evidence)
            if decision == "accept":
                common = {"attempt": attempt, "human_sha256": complete["video_sha256"], "artifact_hashes": hashes}
                store.transition(key, EpisodeState.HUMAN_COMPLETE, EpisodeState.HUMAN_REVIEW_APPROVED,
                                 {"stage": "human_review", "judge": judge, **common})
                store.transition(key, EpisodeState.HUMAN_REVIEW_APPROVED, EpisodeState.VERIFIER_APPROVED,
                                 {"stage": "human_verifier", "verify": verify, **common})
                final = store.load(key)
                store.transition(key, EpisodeState.VERIFIER_APPROVED, EpisodeState.ACCEPTED, {
                    "stage": "accepted", "attempt": attempt, "human_sha256": complete["video_sha256"],
                    "request_id": complete["request_id"],
                    "artifact_hashes": {**final.manifest.artifacts, complete["video"]: complete["video_sha256"]}})
        except StoreError as exc:
            summary["refused"][name] = f"{type(exc).__name__}: {exc}"
            continue
        if decision == "accept":
            summary["accepted"].append(name)
        else:
            summary["rejected"][name] = reasons + problems
    if max_attempts is not None:
        from .budget import Ledger
        from .generate import GEN_DIR, mark_replacements
        summary["replacements"] = mark_replacements(root, max_attempts, Ledger(root / GEN_DIR / "ledger.json"))
    return summary


def main(argv=None) -> int:
    from .generate import DEFAULT_MAX_ATTEMPTS
    from .record import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--root", type=Path, default=ROOT)
    q.add_argument("--out", type=Path)
    i = sub.add_parser("import")
    i.add_argument("verdicts", type=Path)
    i.add_argument("--root", type=Path, default=ROOT)
    i.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args(argv)
    if args.command == "queue":
        value = queue(args.root)
        if args.out:
            args.out.write_text(json.dumps(value, indent=1))
        print(json.dumps(value, indent=1))
        return 0
    summary = import_verdicts(args.root, json.loads(args.verdicts.read_text()), max_attempts=args.max_attempts)
    print(json.dumps(summary, indent=1))
    return 1 if summary["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

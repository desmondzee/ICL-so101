"""Recorder/review gates with a fake rendering backend (real store, hashing, ffmpeg and admission)."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sim.train import record as rec
from sim.train import review
from sim.train.model import EpisodeKey, EpisodeState
from sim.train.store import EpisodeStore, EvidenceMismatch, atomic_write_json, sha256_file
from sim.train.tasks import TRAIN_TASKS

TASK = "block_in_bowl"
W, H, N = 64, 48, 40


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


class FakeBackend:
    """Deterministic stand-in for SimBackend: same files, tiny synthetic content."""

    def __init__(self, reject=(), fail=()):
        self.calls, self.reject, self.fail = [], set(reject), set(fail)

    def record(self, task, seed, staging, store):
        self.calls.append(seed)
        if seed in self.fail:
            raise rec.RolloutFailed("oracle_runtime_error", {"seed": seed})
        for name in rec.VIDEO_NAMES:
            encoder = rec.VideoEncoder(staging / name, W, H)
            for i in range(N):
                frame = np.zeros((H, W, 3), np.uint8)
                frame[:, : (i + 1) * W // N] = (seed * 37 % 255, 4 * i, 200 if "front" in name else 50)
                encoder.write(frame)
            encoder.close()
        rows = np.arange(N, dtype=np.float32)
        joints, ee = np.tile(rows[:, None], 6), np.tile(rows[:, None], 7)
        pd.DataFrame({"action": list(joints), "observation.state": list(joints), "action.ee": list(ee),
                      "observation.state.ee": list(ee), "timestamp": (np.arange(N) / 30).astype(np.float32),
                      "frame_index": np.arange(N), "episode_index": np.zeros(N, dtype=np.int64),
                      "index": np.arange(N), "task_index": np.zeros(N, dtype=np.int64)}
                     ).to_parquet(staging / "robot_data.parquet", index=False)
        for name, value in (("first.png", 10), ("last.png", 200)):
            Image.fromarray(np.full((H, W, 3), value, np.uint8)).save(staging / name)
        with (staging / "telemetry.npz").open("xb") as handle:
            np.savez_compressed(handle, time=np.arange(N, dtype=np.float64))
        accepted = seed not in self.reject
        rec.write_json(staging / "physics.json", {
            "accepted": accepted, "checks": {"finite": True, "allowed_contacts": accepted},
            "maxima": {"joint_speed": 1.0}, "config_hash": _h(f"c{seed}"),
            "violations": [] if accepted else [{"check": "allowed_contacts", "timestamp": 0.5}]})
        rec.write_json(staging / "policy.json", {"task_objects": ["block"]})
        events = [{"frame": 5, "kind": "skill", "label": "pregrasp"}, {"frame": 12, "kind": "grasp", "label": "block"},
                  {"frame": 25, "kind": "release", "label": "block"}]
        if not accepted:
            events.append({"frame": 15, "kind": "violation", "label": "allowed_contacts"})
        rec.write_json(staging / "episode.json", {
            "schema_version": 1, "seed": seed, "visual_config": {"arena": "kitchen", "seed": seed},
            "object_poses_start": {"block": [0.2, 0, 0]}, "object_poses_end": {"block": [0.24, 0.07, 0.02]},
            "distractors": [], "events": events, "final_task_success": True, "terminal_screen": {"ok": True}},
            )
        return rec.Rollout(_h(f"c{seed}"), _h(f"v{seed}"), N, W, H, {"arena": "kitchen"})


def fabricate_qualification(root: Path, name=TASK, failures=0) -> Path:
    """Evidence shaped exactly as qualify.py writes it, so the real admission check runs."""
    from sim.train.tasks.qualify import implementation_hash
    task, source = TRAIN_TASKS[name], implementation_hash(name)
    rows = []
    for i, seed in enumerate(task.qualification_seeds):
        ok = i >= failures
        directory = root / name / f"seed_{seed}"
        directory.mkdir(parents=True)
        identity = _h(f"q{seed}")
        (directory / "telemetry.npz").write_bytes(b"fake")
        for file, value in (("physics.json", {"accepted": ok, "checks": {"finite": ok}, "config_hash": identity}),
                            ("policy.json", {}), ("episode.json", {"seed": seed})):
            (directory / file).write_text(json.dumps(value))
        row = {"seed": seed, "source_hash": source, "accepted": ok, "reasons": [] if ok else ["finite"], "config_hash": identity,
               "artifacts": {f: sha256_file(directory / f) for f in
                             ("telemetry.npz", "physics.json", "policy.json", "episode.json")}}
        (directory / "result.json").write_text(json.dumps(row))
        rows.append(row)
    (root / f"{name}.json").write_text(json.dumps({"source_hash": source, "overlap_report": {"accepted": True},
                                                   "qualified": True, "episodes": rows}))
    return root


@pytest.fixture(scope="module")
def admission(tmp_path_factory):
    return rec.admit(TASK, fabricate_qualification(tmp_path_factory.mktemp("qualification")))


def files(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "state.json" and not p.name.endswith(".lock")}


def verdicts(store, seed, *, judge="accept", confirmed=True, same_reviewer=False, request_sha=None, final=None):
    key = EpisodeKey(TASK, seed)
    sha = request_sha or store.load(key).manifest.artifacts[review.REQUEST_NAME]
    entry = {"key": review.key_name(key), "request_sha256": sha,
             "judge": {"verdict": judge, "reasons": ["block lands inside bowl and settles"],
                       "inspected_evidence": ["review/front_01.jpg", "review/wrist_01.jpg"],
                       "reviewer": {"label": "robot-judge:0", "model": "opus", "role": "judge"}}}
    if confirmed is not None:
        entry["verify"] = {"confirmed": confirmed, "reasons": ["checked grasp and release frames"],
                           "inspected_evidence": ["robot_front.mp4 frames 0-39"],
                           "reviewer": {"label": "robot-judge:0" if same_reviewer else "robot-verify:0.0",
                                        "model": "opus", "role": "verifier"}}
    if final:
        entry["final"] = final
    return {entry["key"]: entry}


# ----- admission ------------------------------------------------------------------------------------------------
def test_unqualified_tasks_are_never_recorded(tmp_path, admission):
    with pytest.raises(FileNotFoundError):
        rec.admit("block_out_of_bowl", tmp_path)
    with pytest.raises(TypeError):
        rec.record_episode(EpisodeStore(tmp_path), admission.task, 1, FakeBackend())
    stale = fabricate_qualification(tmp_path / "q")
    report = json.loads((stale / f"{TASK}.json").read_text())
    report["source_hash"] = "0" * 64
    (stale / f"{TASK}.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification"):
        rec.admit(TASK, stale)
    assert admission.task is TRAIN_TASKS[TASK] and len(admission.report_sha256) == 64


def test_admission_rate_admits_43_of_50_and_refuses_42(tmp_path):
    assert rec.admit(TASK, fabricate_qualification(tmp_path / "a", failures=7)).task is TRAIN_TASKS[TASK]
    with pytest.raises(ValueError, match="qualification"):
        rec.admit(TASK, fabricate_qualification(tmp_path / "b", failures=8))


# ----- recording ------------------------------------------------------------------------------------------------
def test_publishes_hashed_episode_with_both_identities_and_evidence(tmp_path, admission):
    store, backend = EpisodeStore(tmp_path), FakeBackend()
    record = rec.record_episode(store, admission, 7, backend)
    assert record.state is EpisodeState.PHYSICS_APPROVED
    directory = store.episode_dir(EpisodeKey(TASK, 7))
    manifest = record.manifest
    assert manifest.config_hash == _h("c7") and manifest.visual_config_hash == _h("v7")
    assert manifest.artifacts == {k: v for k, v in files(directory).items() if k not in ("manifest.json", "publish.json")}
    for name in ("robot_front.mp4", "robot_wrist.mp4", "robot_data.parquet", "first.png", "last.png", "telemetry.npz",
                 "physics.json", "media.json", "review/frames.json", "review/front_01.jpg", "review/wrist_01.jpg",
                 review.REQUEST_NAME):
        assert name in manifest.artifacts, name
    request = json.loads((directory / review.REQUEST_NAME).read_text())
    assert request["artifact_hashes"] == {k: v for k, v in manifest.artifacts.items() if k != review.REQUEST_NAME}
    assert request["ordered_actions"][0]["object"] == "block" and request["instruction"] == TRAIN_TASKS[TASK].instruction
    assert request["physics"]["accepted"] and request["variation"]["visual_config"]["seed"] == 7
    frames = json.loads((directory / "review/frames.json").read_text())
    chosen = [s["frame"] for s in frames["selection"]]
    assert {0, 6, 12, 18, 25, N - 1} <= set(chosen) and max(np.diff(chosen)) <= 30
    assert manifest.metadata["qualification"]["report_sha256"] == admission.report_sha256
    assert json.loads((directory / "media.json").read_text())["videos"]["robot_front.mp4"]["frames"] == N
    assert not list((tmp_path / "staging").rglob("*.mp4"))


def test_physics_rejection_happens_before_any_review(tmp_path, admission):
    store = EpisodeStore(tmp_path)
    record = rec.record_episode(store, admission, 3, FakeBackend(reject={3}))
    assert record.state is EpisodeState.REJECTED
    assert record.history[-1]["evidence"]["failed_physics_checks"] == ["allowed_contacts"]
    assert review.REQUEST_NAME not in record.manifest.artifacts
    frames = json.loads((store.episode_dir(record.manifest.key) / "review/frames.json").read_text())
    assert any("violation:allowed_contacts" in s["reasons"] for s in frames["selection"])
    assert review.queue(tmp_path)["items"] == []
    fake = verdicts(store, 3, request_sha="a" * 64)
    assert list(review.import_verdicts(tmp_path, fake)["refused"]) == [f"{TASK}/episode_3"]
    assert store.load(record.manifest.key).state is EpisodeState.REJECTED
    assert review.build_fal_ready(tmp_path)["episodes"] == []


def test_completed_episode_is_skipped_and_never_rewritten(tmp_path, admission):
    store, backend = EpisodeStore(tmp_path), FakeBackend()
    rec.record_episode(store, admission, 1, backend)
    directory = store.episode_dir(EpisodeKey(TASK, 1))
    before = files(directory)
    mtimes = {p: p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()}
    assert rec.record_episode(store, admission, 1, backend).state is EpisodeState.PHYSICS_APPROVED
    assert backend.calls == [1]
    assert files(directory) == before
    assert {p: p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()} == mtimes


def test_tampered_completed_episode_fails_closed_without_rerecording(tmp_path, admission):
    store, backend = EpisodeStore(tmp_path), FakeBackend()
    rec.record_episode(store, admission, 1, backend)
    video = store.episode_dir(EpisodeKey(TASK, 1)) / "robot_front.mp4"
    video.write_bytes(video.read_bytes() + b"x")
    with pytest.raises(EvidenceMismatch):
        rec.record_episode(store, admission, 1, backend)
    assert backend.calls == [1] and video.read_bytes().endswith(b"x")


def test_rerun_in_fresh_root_is_byte_identical(tmp_path, admission):
    for name in ("a", "b"):
        rec.record_episode(EpisodeStore(tmp_path / name), admission, 5, FakeBackend())
    a, b = (EpisodeStore(tmp_path / n).episode_dir(EpisodeKey(TASK, 5)) for n in ("a", "b"))
    assert files(a) == files(b)


class Crash(Exception):
    pass


@pytest.mark.parametrize("point", ["before_rename", "after_rename", "after_candidate", "after_recorded"])
def test_crash_at_any_point_resumes_to_identical_result(tmp_path, admission, monkeypatch, point):
    clean = tmp_path / "clean"
    rec.record_episode(EpisodeStore(clean), admission, 9, FakeBackend())
    root = tmp_path / "crash"
    store = EpisodeStore(root)

    def crash(name):
        if name == point:
            raise Crash(name)
    monkeypatch.setattr(rec, "_checkpoint", crash)
    with pytest.raises(Crash):
        rec.record_episode(store, admission, 9, FakeBackend())
    key = EpisodeKey(TASK, 9)
    assert store.episode_dir(key).exists() == (point != "before_rename")
    assert not any((root / "staging").rglob("*.mp4"))  # unpublished staging never lingers
    monkeypatch.setattr(rec, "_checkpoint", lambda name: None)
    backend = FakeBackend()
    record = rec.record_episode(store, admission, 9, backend)
    assert record.state is EpisodeState.PHYSICS_APPROVED
    assert backend.calls == ([9] if point == "before_rename" else [])
    assert files(store.episode_dir(key)) == files(EpisodeStore(clean).episode_dir(key))
    assert [(h["old_state"], h["new_state"]) for h in record.history] == [
        ("candidate", "recorded"), ("recorded", "physics_approved")]


def test_killed_staging_is_ignored_and_cleanable(tmp_path, admission):
    leftover = tmp_path / "staging" / TASK / "4-deadbeef"
    leftover.mkdir(parents=True)
    (leftover / "robot_front.mp4").write_bytes(b"partial")
    store = EpisodeStore(tmp_path)
    assert rec.record_episode(store, admission, 4, FakeBackend()).state is EpisodeState.PHYSICS_APPROVED
    assert rec.clean_staging(tmp_path) == [str(leftover)]
    assert not leftover.exists() and store.load(EpisodeKey(TASK, 4)).state is EpisodeState.PHYSICS_APPROVED


def test_published_directory_without_record_fails_closed(tmp_path, admission):
    store = EpisodeStore(tmp_path)
    directory = store.episode_dir(EpisodeKey(TASK, 2))
    directory.mkdir(parents=True)
    (directory / "robot_front.mp4").write_bytes(b"foreign")
    with pytest.raises(Exception, match="inspect"):
        rec.record_episode(store, admission, 2, FakeBackend())
    assert (directory / "robot_front.mp4").read_bytes() == b"foreign"


def test_failed_seed_is_logged_and_task_walk_is_resumable(tmp_path, admission):
    store = EpisodeStore(tmp_path)
    backend = FakeBackend(reject={101}, fail={102})
    quiet = lambda *a, **k: None
    summary = rec.record_task(store, admission, 2, backend, seed_start=100, log=quiet)
    assert summary == {"approved": [100, 103], "rejected": [101], "failed": [102]}
    assert json.loads((tmp_path / "attempts" / TASK / "episode_102.json").read_text())["reason"] == "oracle_runtime_error"
    again = FakeBackend()
    assert rec.record_task(store, admission, 3, again, seed_start=100, log=quiet)["approved"] == [100, 103, 104]
    assert again.calls == [104]
    first = admission.task.qualification_seeds[0]
    summary = rec.record_task(EpisodeStore(tmp_path / "q"), admission, 1, FakeBackend(), seed_start=first, log=quiet)
    assert summary["approved"] == [first + 50]  # qualification seeds are never training episodes


def test_media_verification_rejects_frame_mismatch(tmp_path, admission):
    FakeBackend().record(admission.task, 1, tmp_path, None)
    assert rec.verify_media(tmp_path, N, W, H)["videos"]["robot_wrist.mp4"]["frames"] == N
    with pytest.raises(ValueError, match="decoded"):
        rec.verify_media(tmp_path, N + 1, W, H)


# ----- review import ------------------------------------------------------------------------------------------------
@pytest.fixture
def approved_store(tmp_path, admission):
    store = EpisodeStore(tmp_path)
    for seed in (1, 2):
        rec.record_episode(store, admission, seed, FakeBackend())
    return store


def test_queue_lists_physics_approved_with_request_hash(approved_store):
    items = review.queue(approved_store.root)["items"]
    assert [i["key"] for i in items] == [f"{TASK}/episode_1", f"{TASK}/episode_2"]
    assert items[0]["request_sha256"] == sha256_file(Path(items[0]["dir"]) / review.REQUEST_NAME)


def test_accept_with_matching_hashes_advances_and_fal_ready_lists_only_it(approved_store):
    root = approved_store.root
    summary = review.import_verdicts(root, verdicts(approved_store, 1, final="accept"))
    assert summary["robot_approved"] == [f"{TASK}/episode_1"]
    assert approved_store.load(EpisodeKey(TASK, 1)).state is EpisodeState.ROBOT_APPROVED
    index = review.build_fal_ready(root)
    assert [e["key"] for e in index["episodes"]] == [f"{TASK}/episode_1"]
    entry = index["episodes"][0]
    for name in ("first", "last", "robot_front", "robot_wrist", "robot_data", "review_request"):
        assert sha256_file(root / entry[name]["path"]) == entry[name]["sha256"]
    assert entry["reviewers"] == {"judge": "robot-judge:0", "verifier": "robot-verify:0.0"}
    assert json.loads((root / "fal_ready.json").read_text()) == index
    # Idempotent re-import of the same verdict.
    assert review.import_verdicts(root, verdicts(approved_store, 1))["refused"] == {}


@pytest.mark.parametrize("kwargs,reason", [
    ({"confirmed": None}, "missing_verifier"),
    ({"same_reviewer": True}, "verifier_not_independent"),
    ({"judge": "needs_detail"}, "needs_detail"),
    ({"final": "reject"}, "inconsistent_final:reject"),
])
def test_incomplete_or_inconsistent_reviews_stay_pending(approved_store, kwargs, reason):
    summary = review.import_verdicts(approved_store.root, verdicts(approved_store, 1, **kwargs))
    assert reason in summary["pending"][f"{TASK}/episode_1"]
    assert approved_store.load(EpisodeKey(TASK, 1)).state is EpisodeState.PHYSICS_APPROVED


def test_reviews_without_reasons_or_evidence_stay_pending(approved_store):
    entry = verdicts(approved_store, 1)
    next(iter(entry.values()))["judge"]["inspected_evidence"] = []
    assert "judge_evidence" in review.import_verdicts(approved_store.root, entry)["pending"][f"{TASK}/episode_1"]


@pytest.mark.parametrize("kwargs", [{"judge": "reject"}, {"confirmed": False}])
def test_judge_reject_or_verifier_refutation_rejects(approved_store, kwargs):
    summary = review.import_verdicts(approved_store.root, verdicts(approved_store, 1, **kwargs))
    assert summary["rejected"] == [f"{TASK}/episode_1"]
    assert approved_store.load(EpisodeKey(TASK, 1)).state is EpisodeState.REJECTED
    assert review.build_fal_ready(approved_store.root)["episodes"] == []


def test_stale_request_hash_never_advances(approved_store):
    summary = review.import_verdicts(approved_store.root, verdicts(approved_store, 1, request_sha="b" * 64))
    assert "different review request" in summary["refused"][f"{TASK}/episode_1"]
    assert approved_store.load(EpisodeKey(TASK, 1)).state is EpisodeState.PHYSICS_APPROVED


def test_modified_evidence_after_request_never_advances(approved_store):
    sheet = approved_store.episode_dir(EpisodeKey(TASK, 2)) / "review" / "front_01.jpg"
    good = verdicts(approved_store, 2)
    sheet.write_bytes(sheet.read_bytes()[:-10])
    summary = review.import_verdicts(approved_store.root, good)
    assert "EvidenceMismatch" in summary["refused"][f"{TASK}/episode_2"]
    with pytest.raises(EvidenceMismatch):
        approved_store.load(EpisodeKey(TASK, 2))


def test_fal_ready_refuses_tampered_approved_episode(approved_store):
    review.import_verdicts(approved_store.root, verdicts(approved_store, 1))
    last = approved_store.episode_dir(EpisodeKey(TASK, 1)) / "last.png"
    last.write_bytes(last.read_bytes() + b"\0")
    with pytest.raises(EvidenceMismatch):
        review.build_fal_ready(approved_store.root)


def test_frame_selection_is_dense_around_events():
    events = [{"frame": 50, "kind": "grasp", "label": "a"}, {"frame": 90, "kind": "violation", "label": "penetration"},
              {"frame": 70, "kind": "skill", "label": "lift"}]
    selection = review.select_frames(200, events)
    frames = [s["frame"] for s in selection]
    assert frames[0] == 0 and frames[-1] == 199 and max(np.diff(frames)) <= 30
    assert {44, 50, 56, 70, 84, 87, 90, 93, 96} <= set(frames)
    assert set(range(170, 200, 5)) <= set(frames)
    assert "grasp:a" in next(s for s in selection if s["frame"] == 50)["reasons"]

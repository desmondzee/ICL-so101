"""State-store contracts: skipped gates, changed evidence and partial writes fail."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from sim.train.model import EpisodeKey, EpisodeManifest, EpisodeState
from sim.train.store import (
    EpisodeStore,
    EvidenceMismatch,
    InvalidTransition,
    ManifestConflict,
    StoreCorruption,
    atomic_write_json,
    sha256_file,
)


@pytest.fixture
def candidate(tmp_path):
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 7)
    store.create_candidate(EpisodeManifest(key=key, config_hash="a" * 64, visual_config_hash="a" * 64))
    return store, key


@pytest.mark.parametrize("visual_hash", [None, "", "not-a-hash", "A" * 64])
def test_candidate_requires_a_valid_explicit_visual_hash(tmp_path, visual_hash):
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 7)
    with pytest.raises(ValueError, match="visual_config_hash"):
        store.create_candidate(EpisodeManifest(key, "a" * 64, visual_config_hash=visual_hash))
    assert not store.manifest_path(key).exists()


def test_manifest_without_persisted_visual_hash_fails_closed(candidate):
    store, key = candidate
    manifest_path = store.manifest_path(key)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("visual_config_hash", None)
    manifest_path.write_text(json.dumps(manifest))
    state_path = store.state_path(key)
    state = json.loads(state_path.read_text())
    state["manifest_hash"] = sha256_file(manifest_path)
    state_path.write_text(json.dumps(state))
    with pytest.raises(StoreCorruption, match="visual_config_hash"):
        store.load(key)


def test_cannot_skip_robot_review(candidate):
    store, key = candidate
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.CANDIDATE, EpisodeState.ROBOT_APPROVED, {})
    assert store.load(key).state is EpisodeState.CANDIDATE


def test_ordered_progression_preserves_manifest_and_records_evidence(candidate):
    store, key = candidate
    manifest_before = store.manifest_path(key).read_bytes()
    evidence = {"review": {"verdict": "accept"}, "artifact_hashes": {}}
    for old, new in [
        (EpisodeState.CANDIDATE, EpisodeState.RECORDED),
        (EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED),
        (EpisodeState.PHYSICS_APPROVED, EpisodeState.ROBOT_APPROVED),
    ]:
        store.transition(key, old, new, evidence)
    record = EpisodeStore(store.root).load(key)
    assert record.state is EpisodeState.ROBOT_APPROVED
    assert len(record.history) == 3
    assert record.history[-1]["old_state"] == "physics_approved"
    assert record.history[-1]["new_state"] == "robot_approved"
    assert record.history[-1]["timestamp"].endswith("+00:00")
    assert record.history[-1]["evidence_hashes"]["review"] == hashlib.sha256(
        b'{"verdict":"accept"}'
    ).hexdigest()
    assert store.manifest_path(key).read_bytes() == manifest_before


def test_identical_transition_is_idempotent_after_resume(candidate):
    store, key = candidate
    evidence = {"frames": 10}
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, evidence)
    before = store.state_path(key).read_bytes()
    resumed = EpisodeStore(store.root)
    resumed.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, evidence)
    assert store.state_path(key).read_bytes() == before
    assert len(resumed.load(key).history) == 1


def test_changed_transition_evidence_fails_without_mutation(candidate):
    store, key = candidate
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"frames": 10})
    before = store.state_path(key).read_bytes()
    with pytest.raises(EvidenceMismatch):
        store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"frames": 11})
    assert store.state_path(key).read_bytes() == before


def test_human_completion_needs_two_reviews_before_acceptance(candidate):
    store, key = candidate
    path = [
        EpisodeState.CANDIDATE, EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED,
        EpisodeState.ROBOT_APPROVED, EpisodeState.HUMAN_SUBMITTED, EpisodeState.HUMAN_COMPLETE,
    ]
    for old, new in zip(path, path[1:]):
        store.transition(key, old, new, {})
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.HUMAN_COMPLETE, EpisodeState.ACCEPTED, {})
    store.transition(key, EpisodeState.HUMAN_COMPLETE, EpisodeState.HUMAN_REVIEW_APPROVED, {})
    store.transition(key, EpisodeState.HUMAN_REVIEW_APPROVED, EpisodeState.VERIFIER_APPROVED, {})
    store.transition(key, EpisodeState.VERIFIER_APPROVED, EpisodeState.ACCEPTED, {})
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.ACCEPTED, EpisodeState.REJECTED, {})


def test_rejected_is_terminal(candidate):
    store, key = candidate
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.REJECTED, {"reason": "bad reset"})
    before = store.state_path(key).read_bytes()
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.REJECTED, EpisodeState.RECORDED, {})
    assert store.state_path(key).read_bytes() == before


def test_failed_human_attempt_preserves_robot_approved_source(candidate):
    store, key = candidate
    for old, new in [
        (EpisodeState.CANDIDATE, EpisodeState.RECORDED),
        (EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED),
        (EpisodeState.PHYSICS_APPROVED, EpisodeState.ROBOT_APPROVED),
    ]:
        store.transition(key, old, new, {})
    state_before = store.state_path(key).read_bytes()
    evidence = {"status": "failed", "request_id": "request-1", "reason": "download failed"}
    store.record_human_attempt(key, "attempt-1", evidence)
    store.record_human_attempt(key, "attempt-1", evidence)
    record = store.load(key)
    assert record.state is EpisodeState.ROBOT_APPROVED
    assert len(record.human_attempts) == 1
    assert record.human_attempts[0]["evidence"] == evidence
    assert store.state_path(key).read_bytes() == state_before
    with pytest.raises(EvidenceMismatch):
        store.record_human_attempt(key, "attempt-1", {"status": "failed", "reason": "changed"})


def test_duplicate_candidate_must_match_original_manifest(candidate):
    store, key = candidate
    before = store.manifest_path(key).read_bytes()
    store.create_candidate(EpisodeManifest(key=key, config_hash="a" * 64, visual_config_hash="a" * 64))
    with pytest.raises(ManifestConflict):
        store.create_candidate(EpisodeManifest(key=key, config_hash="b" * 64, visual_config_hash="a" * 64))
    with pytest.raises(ManifestConflict):
        store.create_candidate(EpisodeManifest(key=key, config_hash="a" * 64, metadata={"x": 1},
                                               visual_config_hash="a" * 64))
    assert store.manifest_path(key).read_bytes() == before


@pytest.mark.parametrize("filename", ["manifest.json", "state.json"])
def test_truncated_json_fails_without_rewriting(candidate, filename):
    store, key = candidate
    path = store.episode_dir(key) / filename
    path.write_text('{"truncated":')
    with pytest.raises(StoreCorruption):
        store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {})
    assert path.read_text() == '{"truncated":'


def test_artifact_hash_mismatch_prevents_transition(candidate):
    store, key = candidate
    artifact = store.episode_dir(key) / "robot_front.mp4"
    artifact.write_bytes(b"original")
    evidence = {"artifact_hashes": {"robot_front.mp4": sha256_file(artifact)}}
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, evidence)
    artifact.write_bytes(b"changed")
    before = store.state_path(key).read_bytes()
    with pytest.raises(EvidenceMismatch):
        store.transition(key, EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED, evidence)
    assert store.state_path(key).read_bytes() == before


def test_atomic_write_failure_preserves_previous_document(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"state": "candidate"})
    before = path.read_bytes()

    def interrupt_replace(source, destination):
        raise OSError("simulated interruption before rename")

    monkeypatch.setattr("sim.train.store.os.replace", interrupt_replace)
    with pytest.raises(OSError):
        atomic_write_json(path, {"state": "recorded"})
    assert path.read_bytes() == before
    assert json.loads(path.read_text()) == {"state": "candidate"}
    assert not list(tmp_path.glob("*.tmp"))


def test_final_json_artifacts_are_never_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"version": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"version": 2}, overwrite=False)
    assert json.loads(path.read_text()) == {"version": 1}


def test_sha256_file_uses_content(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("task", ["../outside", "/absolute", "a/b", ""])
def test_episode_task_cannot_escape_store(tmp_path, task):
    with pytest.raises(ValueError):
        EpisodeStore(tmp_path).create_candidate(EpisodeManifest(EpisodeKey(task, 7), "a" * 64,
                                                               visual_config_hash="a" * 64))


def test_concurrent_identical_transitions_append_only_once(candidate):
    store, key = candidate
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(store.transition, key, EpisodeState.CANDIDATE,
                               EpisodeState.RECORDED, {"frames": 10}) for _ in range(4)]
        assert all(future.result().state is EpisodeState.RECORDED for future in futures)
    assert len(store.load(key).history) == 1


@pytest.mark.parametrize("mutation", ["skipped_gate", "changed_evidence", "wrong_state", "changed_manifest"])
def test_corrupt_history_or_manifest_cannot_resume(candidate, mutation):
    store, key = candidate
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"frames": 10})
    path = store.state_path(key)
    data = json.loads(path.read_text())
    if mutation == "skipped_gate":
        data["history"][0]["new_state"] = "robot_approved"
        data["state"] = "robot_approved"
    elif mutation == "changed_evidence":
        data["history"][0]["evidence"]["frames"] = 20
    elif mutation == "wrong_state":
        data["state"] = "accepted"
    else:
        path = store.manifest_path(key)
        data = json.loads(path.read_text())
        data["config_hash"] = "b" * 64
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(StoreCorruption):
        store.create_candidate(EpisodeManifest(key, "a" * 64, visual_config_hash="a" * 64))
    assert path.read_bytes() == before


def test_replay_earlier_transition_does_not_regress_resume(candidate):
    store, key = candidate
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"frames": 10})
    store.transition(key, EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED, {})
    before = store.state_path(key).read_bytes()
    resumed = store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {"frames": 10})
    assert resumed.state is EpisodeState.PHYSICS_APPROVED
    assert store.state_path(key).read_bytes() == before


def test_failed_first_atomic_write_leaves_no_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"

    def interrupt_fsync(descriptor):
        raise OSError("simulated interruption before fsync")

    monkeypatch.setattr("sim.train.store.os.fsync", interrupt_fsync)
    with pytest.raises(OSError):
        atomic_write_json(path, {"config_hash": "a" * 64}, overwrite=False)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_state_is_corruption_not_permission_to_reset(candidate):
    store, key = candidate
    store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, {})
    store.state_path(key).unlink()
    before = store.manifest_path(key).read_bytes()
    with pytest.raises(StoreCorruption):
        store.create_candidate(EpisodeManifest(key, "a" * 64, visual_config_hash="a" * 64))
    assert not store.state_path(key).exists()
    assert store.manifest_path(key).read_bytes() == before


def test_final_json_dangling_symlink_is_not_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"version": 2}, overwrite=False)
    assert path.is_symlink()


def test_candidate_resume_requires_identical_json_types(tmp_path):
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 7)
    store.create_candidate(EpisodeManifest(key, "a" * 64, metadata={"variant": 1}, visual_config_hash="a" * 64))
    before = store.manifest_path(key).read_bytes()
    with pytest.raises(ManifestConflict):
        store.create_candidate(EpisodeManifest(key, "a" * 64, metadata={"variant": True},
                                               visual_config_hash="a" * 64))
    assert store.manifest_path(key).read_bytes() == before


@pytest.mark.parametrize("operation", ["transition", "human_attempt"])
def test_tuple_evidence_has_idempotent_json_replay(candidate, operation):
    store, key = candidate
    evidence = {"inspected_frames": (0, 10, 20), "review": {"issues": ()}}
    normalized = {"inspected_frames": [0, 10, 20], "review": {"issues": []}}
    if operation == "transition":
        def persist(value):
            return store.transition(key, EpisodeState.CANDIDATE, EpisodeState.RECORDED, value)

        path = store.state_path(key)
    else:
        for old, new in [
            (EpisodeState.CANDIDATE, EpisodeState.RECORDED),
            (EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED),
            (EpisodeState.PHYSICS_APPROVED, EpisodeState.ROBOT_APPROVED),
        ]:
            store.transition(key, old, new, {})

        def persist(value):
            return store.record_human_attempt(key, "attempt-tuple", value)

        path = store.episode_dir(key) / "human_attempts" / "attempt-tuple.json"
    persist(evidence)
    before = path.read_bytes()
    persist(evidence)
    persist(normalized)
    assert path.read_bytes() == before
    with pytest.raises(EvidenceMismatch):
        persist({"inspected_frames": (0, 10, 21), "review": {"issues": ()}})
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["transition", "human_attempt"])
@pytest.mark.parametrize("alias", [False, True], ids=["direct", "symlink"])
@pytest.mark.parametrize("control", [
    "state.json", "manifest.json", ".state.json.lock", "scratch.tmp", "human_attempts/existing.json",
])
def test_workflow_files_cannot_be_artifact_evidence(candidate, operation, alias, control):
    store, key = candidate
    for old, new in [
        (EpisodeState.CANDIDATE, EpisodeState.RECORDED),
        (EpisodeState.RECORDED, EpisodeState.PHYSICS_APPROVED),
        (EpisodeState.PHYSICS_APPROVED, EpisodeState.ROBOT_APPROVED),
    ]:
        store.transition(key, old, new, {})
    store.record_human_attempt(key, "existing", {"status": "failed"})
    directory = store.episode_dir(key)
    target = directory / control
    if not target.exists():
        target.write_bytes(b"temporary workflow data")
    name = control
    if alias:
        link = directory / "evidence-alias.json"
        link.symlink_to(target)
        name = link.name
    evidence = {"artifact_hashes": {name: sha256_file(target)}}
    before_state = store.state_path(key).read_bytes()
    before_manifest = store.manifest_path(key).read_bytes()
    before_attempt = (directory / "human_attempts" / "existing.json").read_bytes()
    with pytest.raises(EvidenceMismatch):
        if operation == "transition":
            store.transition(key, EpisodeState.ROBOT_APPROVED, EpisodeState.HUMAN_SUBMITTED, evidence)
        else:
            store.record_human_attempt(key, "new", evidence)
    assert store.state_path(key).read_bytes() == before_state
    assert store.manifest_path(key).read_bytes() == before_manifest
    assert (directory / "human_attempts" / "existing.json").read_bytes() == before_attempt
    assert not (directory / "human_attempts" / "new.json").exists()
    assert store.load(key).state is EpisodeState.ROBOT_APPROVED

"""Human-pair review: sheets, two-stage verdict import, hash binding and the review workflows."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sim.train import generate as gen
from sim.train import human_review as hr
from sim.train.budget import Ledger
from sim.train.model import EpisodeState
from sim.train.store import EpisodeStore, EvidenceMismatch, InvalidTransition
from generation_fixtures import FakeFal, make_episode

REPO = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name("workflow_harness.mjs")
NAME = "block_in_bowl/episode_100000_a1"


@pytest.fixture
def generated(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "price_per_second", lambda at=None, override=None: 0.015)
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    ctx = gen.Context(tmp_path, FakeFal(), Ledger(tmp_path / "generation" / "ledger.json"), EpisodeStore(tmp_path),
                      poll_interval_s=0, sleep=lambda s: None, log=lambda m: None)
    gen.run(ctx, phase="pilot", target=None, cap_usd=120, run_budget_usd=5, workers=1)
    assert EpisodeStore(tmp_path).load(key).state is EpisodeState.HUMAN_COMPLETE
    return tmp_path, key, ctx


def judge(verdict="accept", label="human-judge:0", scores=(5, 5)):
    return {"verdict": verdict, "task_adherence": scores[0], "physics": scores[1], "reasons": ["done in order at 1.0-4.2s"],
            "inspected_evidence": ["human_frames.jpg", "robot_frames.jpg"], "summary": "s",
            "reviewer": {"label": label, "model": "opus", "role": "judge"}}


def verify(confirmed=True, label="human-verify:0.0"):
    return {"confirmed": confirmed, "reasons": ["fps=6 frames 0-5s checked"], "inspected_evidence": ["human.mp4 @6fps"],
            "reviewer": {"label": label, "model": "opus", "role": "verifier"}}


def verdicts(item, j=None, v=None, final=None):
    entry = {"key": item["key"], "human_sha256": item["human_sha256"], "info_sha256": item["info_sha256"],
             "judge": j, "verify": v}
    if final:
        entry["final"] = final
    return {item["key"]: entry}


def test_queue_builds_curate_layout_sheets(generated):
    root, key, _ = generated
    args = hr.queue(root)
    assert args["keys"] == [NAME]
    sheet = Path(args["dir"]) / NAME
    assert {p.name for p in sheet.iterdir()} == {"human.mp4", "human_frames.jpg", "robot_frames.jpg", "info.json"}
    info = json.loads((sheet / "info.json").read_text())
    assert info["steps"] == ["Pick up the block and put it inside the bowl."]
    assert "rendered simulation" in info["note"] and info["instruction"] == "Put the block inside the bowl."
    assert hr.queue(root)["items"] == args["items"]  # idempotent


def test_generation_without_both_review_stages_is_never_accepted(generated):
    root, key, _ = generated
    item = hr.queue(root)["items"][0]
    store = EpisodeStore(root)
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.HUMAN_COMPLETE, EpisodeState.ACCEPTED, {})
    cases = [
        verdicts(item, judge()),                                       # no verifier
        verdicts(item, judge(), verify(label="human-judge:0")),        # verifier not independent
        verdicts(item, {**judge(), "reviewer": None}, verify()),       # unidentified judge
        verdicts(item, judge(), {**verify(), "reasons": []}),          # verifier without reasons
        verdicts(item, judge(), verify(), final="reject"),             # inconsistent final
    ]
    for case in cases:
        summary = hr.import_verdicts(root, case)
        assert NAME in summary["pending"] and not summary["accepted"]
        assert store.load(key).state is EpisodeState.HUMAN_COMPLETE
    summary = hr.import_verdicts(root, verdicts(item, judge(), verify(), final="accept"))
    assert summary["accepted"] == [NAME]
    record = store.load(key)
    assert record.state is EpisodeState.ACCEPTED
    assert [h["new_state"] for h in record.history[-3:]] == ["human_review_approved", "verifier_approved", "accepted"]
    # Replaying the same import is idempotent.
    assert hr.import_verdicts(root, verdicts(item, judge(), verify(), final="accept"))["accepted"] == [NAME]


def test_low_scores_and_verifier_refutation_reject_with_reasons(generated):
    root, key, ctx = generated
    item = hr.queue(root)["items"][0]
    summary = hr.import_verdicts(root, verdicts(item, judge(), verify(confirmed=False)), max_attempts=3)
    assert NAME in summary["rejected"] and "verifier_refuted" in summary["rejected"][NAME]
    attempts = {a["attempt_id"]: a["evidence"] for a in EpisodeStore(root).load(key).human_attempts}
    assert attempts["a1-review"]["decision"] == "reject" and attempts["a1-review"]["reasons"]
    assert EpisodeStore(root).load(key).state is EpisodeState.HUMAN_COMPLETE
    # The next generation run retries with a new seed.
    episodes, _ = gen.load_ready(root)
    plan = gen.plan_requests(episodes, ctx.ledger.snapshot(), target=None, max_attempts=3)
    assert [(p.key, p.kind) for p in plan] == [("block_in_bowl/episode_100000/a2", "retry")]
    assert hr._decision(verdicts(item, judge(scores=(3, 5)), verify())[NAME])[0] == "reject"


def test_modified_video_invalidates_the_verdict(generated):
    root, key, _ = generated
    item = hr.queue(root)["items"][0]
    sheet = root / hr.REVIEW_DIR / NAME / "human.mp4"
    sheet.write_bytes(sheet.read_bytes() + b"x")
    summary = hr.import_verdicts(root, verdicts(item, judge(), verify()))
    assert NAME in summary["refused"]
    assert EpisodeStore(root).load(key).state is EpisodeState.HUMAN_COMPLETE


def test_verdict_for_other_hash_or_changed_info_is_refused(generated):
    root, key, _ = generated
    item = hr.queue(root)["items"][0]
    assert NAME in hr.import_verdicts(root, verdicts({**item, "human_sha256": "0" * 64}, judge(), verify()))["refused"]
    info = root / hr.REVIEW_DIR / NAME / "info.json"
    info.write_text(info.read_text().replace("inside the bowl", "beside the bowl"))
    assert NAME in hr.import_verdicts(root, verdicts(item, judge(), verify()))["refused"]


def test_modified_episode_video_or_endpoint_after_acceptance_fails_closed(generated):
    root, key, _ = generated
    item = hr.queue(root)["items"][0]
    hr.import_verdicts(root, verdicts(item, judge(), verify()))
    store = EpisodeStore(root)
    video = store.episode_dir(key) / "human" / "a1" / "human.mp4"
    original = video.read_bytes()
    video.write_bytes(original + b"x")
    with pytest.raises(EvidenceMismatch):
        store.load(key)
    video.write_bytes(original)
    store.load(key)
    first = store.episode_dir(key) / "first.png"
    first.write_bytes(b"tampered")
    with pytest.raises(EvidenceMismatch):
        store.load(key)


def test_modified_endpoint_before_import_refuses(generated):
    root, key, _ = generated
    item = hr.queue(root)["items"][0]
    (EpisodeStore(root).episode_dir(key) / "last.png").write_bytes(b"tampered")
    assert NAME in hr.import_verdicts(root, verdicts(item, judge(), verify()))["refused"]


def test_robot_verify_all_ignores_human_videos(generated):
    from sim.train import record as rec
    root, key, _ = generated
    directory = EpisodeStore(root).episode_dir(key)
    names = []
    real_probe = rec.probe
    rec.probe = lambda v: names.append(v.relative_to(directory).as_posix()) or real_probe(v)
    try:
        with pytest.raises(Exception):
            rec.verify_media(directory, 24, 64, 48, fps=24)  # fixture tables are dummies
    finally:
        rec.probe = real_probe
    assert names and not any(n.startswith("human/") for n in names)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("workflow", ["sim/train/human_review_workflow.js", "humangen/curate_workflow.js"])
def test_review_workflows_run_on_the_queue(generated, workflow, tmp_path_factory):
    root, key, _ = generated
    args = hr.queue(root)
    args_path = tmp_path_factory.mktemp("wf") / "args.json"
    args_path.write_text(json.dumps(args))
    out = subprocess.run(["node", str(HARNESS), str(REPO / workflow), str(args_path)], capture_output=True, text=True,
                         check=True).stdout
    result = json.loads(out)["result"]
    assert set(result) == {NAME} and result[NAME]["final"] == "accept"
    summary = hr.import_verdicts(root, result)
    if workflow.startswith("sim/"):
        assert summary["accepted"] == [NAME]
        assert EpisodeStore(root).load(key).state is EpisodeState.ACCEPTED
    else:
        # The unchanged curate workflow reads the layout but carries no reviewer identity or hash echo.
        assert NAME in summary["refused"] or NAME in summary["pending"]
        assert EpisodeStore(root).load(key).state is EpisodeState.HUMAN_COMPLETE


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_workflow_rejection_flows_to_a_retry(generated, tmp_path_factory):
    root, key, _ = generated
    args_path = tmp_path_factory.mktemp("wf") / "args.json"
    args_path.write_text(json.dumps(hr.queue(root)))
    out = subprocess.run(["node", str(HARNESS), str(REPO / "sim/train/human_review_workflow.js"), str(args_path), NAME],
                         capture_output=True, text=True, check=True).stdout
    payload = json.loads(out)
    assert payload["calls"] == ["human-judge:0"]  # rejected pairs are not sent to a verifier
    summary = hr.import_verdicts(root, payload["result"])
    assert summary["rejected"][NAME] == ["block never reaches the bowl (4.8s)", "scores_below_threshold"] or \
        "block never reaches the bowl (4.8s)" in summary["rejected"][NAME]

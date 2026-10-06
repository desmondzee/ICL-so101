"""Human-video generation with a fake fal client: prompts, budget gates, resume, retries."""

from datetime import datetime, timezone
import json
import subprocess

import pytest

from humangen import sim_val_demos
from sim.train import generate as gen
from sim.train.budget import Ledger
from sim.train.model import EpisodeState
from sim.train.store import EpisodeStore, sha256_file
from generation_fixtures import FakeFal, make_episode, video_bytes

TWO = ("red_block", "blue_block")
TWO_TEXT = ("Pick up the red block and put it on the green mat.", "Then pick up the blue block and put it on the yellow mat.")


@pytest.fixture(autouse=True)
def promo_price(monkeypatch):
    monkeypatch.setattr(gen, "price_per_second", lambda at=None, override=None: override or 0.015)
    monkeypatch.setattr(gen, "reservation_usd", lambda d, at=None, override=None, margin=1.0: 0.09 if d == 5.0 else 0.15)


def ctx_for(root, client, *, pilot_cap=5.0, timeout=5.0):
    return gen.Context(root, client, Ledger(root / "generation" / "ledger.json", pilot_cap_usd=pilot_cap),
                       EpisodeStore(root), poll_interval_s=0, poll_timeout_s=timeout, sleep=lambda s: None,
                       log=lambda m: None)


def run(ctx, **kw):
    kw.setdefault("phase", "pilot")
    kw.setdefault("target", None)
    kw.setdefault("cap_usd", 120.0)
    kw.setdefault("run_budget_usd", 5.0)
    kw.setdefault("workers", 2)
    return gen.run(ctx, **kw)


def add_review(root, key, attempt, decision):
    EpisodeStore(root).record_human_attempt(key, f"{attempt}-review", {
        "stage": "human_review", "attempt": attempt, "decision": decision, "reasons": ["hand teleports the block at 2.1s"]})


# ----- prompt ---------------------------------------------------------------------------------------------
def test_prompt_lists_the_ordered_actions_one_object_at_a_time():
    action = gen.task_action(list(TWO_TEXT))
    assert action.startswith("first picks up the red block and puts it on the green mat, then picks up the blue block")
    assert "one object at a time" in action
    text = gen.prompt(action, 5.0, "bottom")
    assert text.index("red block") < text.index("blue block")
    assert "single right hand and forearm reaches in empty from the bottom edge" in text
    assert gen.action_clause("Pick up the long block, turn it so it runs left to right, and lay it on the mat.") == \
        "picks up the long block, turns it so it runs left to right, and lays it on the mat"
    assert gen.action_clause("Lift the block out of the bowl and put it on the mat.") == \
        "lifts the block out of the bowl and puts it on the mat"


def test_prompt_reuses_the_v5_validation_format():
    action = gen.task_action(["Pick up the block and put it inside the bowl."])
    assert gen.prompt(action) == sim_val_demos.prompt("Put the block inside the bowl.", action)
    assert "8.00-second mark" in gen.prompt(action, 8.0) and "By 7.00 seconds" in gen.prompt(action, 8.0)


def test_entry_edge_and_duration_policy():
    camera = lambda pos, lookat: {"front_camera": {"pos": pos, "lookat": lookat, "fovy": 48}}
    assert gen.entry_edge(camera([0.48, -0.13, 0.38], [0.15, -0.02, 0])) == "bottom"
    assert gen.entry_edge(camera([-0.3, 0, 0.4], [0.15, 0, 0])) == "top"
    assert gen.entry_edge(camera([0.15, -0.5, 0.4], [0.15, 0, 0])) in ("left", "right")
    assert gen.duration_for("t", 1) == gen.duration_for("t", 2) == 5.0
    assert gen.duration_for("t", 3) == 7.5 and gen.duration_for("t", 9) == 10.0
    assert gen.duration_for("t", 1, {"t": 6}) == 6.0
    with pytest.raises(ValueError):
        gen.duration_for("t", 1, {"t": 30})


# ----- generation -----------------------------------------------------------------------------------------
def test_run_generates_records_and_never_accepts(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    client = FakeFal()
    summary = run(ctx_for(tmp_path, client))
    assert summary["outcomes"] == {"block_in_bowl/episode_100000/a1": "complete"}
    sent = client.submits[0]
    assert sent["endpoint"] == gen.ENDPOINT and sent["resolution"] == "480P" and sent["duration"] == 5
    assert sent["prompt_expansion_mode"] == "disabled" and sent["seed"] == gen.attempt_seed("block_in_bowl/episode_100000", 1)
    record = EpisodeStore(tmp_path).load(key)
    assert record.state is EpisodeState.HUMAN_COMPLETE  # generation alone is never acceptance
    attempts = {a["attempt_id"]: a["evidence"] for a in record.human_attempts}
    assert set(attempts) == {"a1-submitted", "a1-complete"}
    done = attempts["a1-complete"]
    assert done["request_id"] == "req-1" and done["seed"] == sent["seed"]
    assert done["artifact_hashes"]["first.png"] == record.manifest.artifacts["first.png"]
    video = EpisodeStore(tmp_path).episode_dir(key) / done["video"]
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)],
                             capture_output=True, text=True, check=True).stdout.split()
    assert streams == ["video"]  # audio stripped
    assert done["video_sha256"] == sha256_file(video)
    entry = Ledger(tmp_path / "generation" / "ledger.json").get("block_in_bowl/episode_100000/a1")
    assert entry["status"] == "completed" and entry["reconciled_usd"] == pytest.approx(done["probe"]["duration_s"] * 0.015)


def test_uncertain_post_is_never_resubmitted(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    client = FakeFal(submit_error=TimeoutError("read timed out"))
    first = run(ctx_for(tmp_path, client))
    assert first["outcomes"] == {"block_in_bowl/episode_100000/a1": "uncertain"}
    client.submit_error = None
    second = run(ctx_for(tmp_path, client), target=10)
    assert len(client.submits) == 1 and second["reserved"] == 0
    ledger = Ledger(tmp_path / "generation" / "ledger.json")
    assert ledger.get("block_in_bowl/episode_100000/a1")["status"] == "uncertain"
    assert ledger.summary()["committed_usd"] == pytest.approx(0.09)
    assert gen.status_report(tmp_path, ledger, 3)["uncertain_requests"] == ["block_in_bowl/episode_100000/a1"]
    assert EpisodeStore(tmp_path).load(key).state is EpisodeState.ROBOT_APPROVED


def test_crash_during_post_becomes_uncertain_on_resume(tmp_path):
    make_episode(tmp_path, "block_in_bowl", 100000)
    ctx = ctx_for(tmp_path, FakeFal())
    episodes, _ = gen.load_ready(tmp_path)
    plan = gen.plan_requests(episodes, ctx.ledger.snapshot(), target=None, max_attempts=3)
    gen.reserve_plan(ctx, plan, phase="pilot", run_id="crashed", cap_usd=120, run_budget_usd=5,
                     retry_reserve_fraction=0.25, price_override=None)
    ctx.ledger.mark_submitting(plan[0].key)  # process died inside the POST
    client = FakeFal()
    summary = run(ctx_for(tmp_path, client))
    assert summary["resumed"]["recovered"]["uncertain"] == [plan[0].key]
    assert client.submits == []


def test_server_refusal_releases_and_allows_a_new_attempt(tmp_path):
    make_episode(tmp_path, "block_in_bowl", 100000)
    run(ctx_for(tmp_path, FakeFal(submit_error=gen.SubmitRejected(422))))
    ledger = Ledger(tmp_path / "generation" / "ledger.json")
    assert ledger.get("block_in_bowl/episode_100000/a1")["status"] == "released"
    assert ledger.summary()["committed_usd"] == 0
    client = FakeFal()
    summary = run(ctx_for(tmp_path, client))
    assert summary["outcomes"] == {"block_in_bowl/episode_100000/a2": "complete"}


def test_resume_polls_without_resubmitting_and_skips_completed(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    client = FakeFal(statuses=("IN_PROGRESS",))
    summary = run(ctx_for(tmp_path, client, timeout=0))
    assert summary["outcomes"] == {"block_in_bowl/episode_100000/a1": "pending"}
    assert EpisodeStore(tmp_path).load(key).state is EpisodeState.HUMAN_SUBMITTED
    client.statuses = ("COMPLETED",)
    resumed = run(ctx_for(tmp_path, client))
    assert resumed["resumed"]["polled"] == {"block_in_bowl/episode_100000/a1": "complete"}
    assert resumed["reserved"] == 0 and len(client.submits) == 1
    again = run(ctx_for(tmp_path, client))
    assert again["resumed"]["polled"] == {} and again["reserved"] == 0 and len(client.submits) == 1
    assert EpisodeStore(tmp_path).load(key).state is EpisodeState.HUMAN_COMPLETE


def test_crash_after_download_is_completed_from_disk(tmp_path, monkeypatch):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    client = FakeFal()
    real = EpisodeStore.record_human_attempt

    def crash(self, k, attempt_id, evidence):
        if attempt_id.endswith("-complete"):
            raise KeyboardInterrupt
        return real(self, k, attempt_id, evidence)

    monkeypatch.setattr(EpisodeStore, "record_human_attempt", crash)
    with pytest.raises(KeyboardInterrupt):
        run(ctx_for(tmp_path, client))
    monkeypatch.setattr(EpisodeStore, "record_human_attempt", real)
    video = EpisodeStore(tmp_path).episode_dir(key) / "human" / "a1" / "human.mp4"
    before = sha256_file(video)
    summary = run(ctx_for(tmp_path, client))
    assert summary["resumed"]["polled"] == {"block_in_bowl/episode_100000/a1": "complete"}
    assert len(client.submits) == 1 and sha256_file(video) == before


def test_modified_endpoint_blocks_submission(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    (EpisodeStore(tmp_path).episode_dir(key) / "first.png").write_bytes(b"tampered")
    client = FakeFal()
    summary = run(ctx_for(tmp_path, client))
    assert client.submits == [] and summary["reserved"] == 0
    assert "block_in_bowl/episode_100000" in summary["refused_inputs"]


def test_endpoint_changed_between_plan_and_post_is_released(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    ctx = ctx_for(tmp_path, FakeFal())
    episodes, _ = gen.load_ready(tmp_path)
    plan = gen.plan_requests(episodes, ctx.ledger.snapshot(), target=None, max_attempts=3)
    gen.reserve_plan(ctx, plan, phase="pilot", run_id="r", cap_usd=120, run_budget_usd=5, retry_reserve_fraction=0.25,
                     price_override=None)
    (EpisodeStore(tmp_path).episode_dir(key) / "last.png").write_bytes(b"tampered")
    assert gen.submit(ctx, plan[0]) == "endpoint_mismatch"
    assert ctx.client.submits == [] and ctx.ledger.get(plan[0].key)["status"] == "released"


def test_unrobot_approved_episode_is_not_generated(tmp_path):
    make_episode(tmp_path, "block_in_bowl", 100000, approve=False)
    client = FakeFal()
    run(ctx_for(tmp_path, client))
    assert client.submits == []


def test_retries_use_new_seeds_then_mark_replacement(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    client = FakeFal()
    seeds = []
    for n in (1, 2):
        run(ctx_for(tmp_path, client), max_attempts=2)
        seeds.append(client.submits[-1]["seed"])
        add_review(tmp_path, key, f"a{n}", "reject")
    assert len(set(seeds)) == 2
    summary = run(ctx_for(tmp_path, client), max_attempts=2)
    assert summary["reserved"] == 0 and summary["replacements"] == ["block_in_bowl/episode_100000"]
    record = EpisodeStore(tmp_path).load(key)
    assert record.state is EpisodeState.REJECTED
    assert record.history[-1]["evidence"]["decision"] == "replace"
    replacements = json.loads((tmp_path / "generation" / "replacements.json").read_text())
    assert replacements["episodes"][0]["reasons"]["a1-review"] == ["hand teleports the block at 2.1s"]


def test_fal_failure_counts_as_an_attempt_and_keeps_the_reservation(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    summary = run(ctx_for(tmp_path, FakeFal(fail_status=True)), max_attempts=1)
    assert summary["outcomes"] == {"block_in_bowl/episode_100000/a1": "failed"}
    assert Ledger(tmp_path / "generation" / "ledger.json").summary()["committed_usd"] == pytest.approx(0.09)
    assert summary["replacements"] == ["block_in_bowl/episode_100000"]
    assert EpisodeStore(tmp_path).load(key).state is EpisodeState.REJECTED


def test_bad_download_is_recorded_as_failed(tmp_path):
    key = make_episode(tmp_path, "block_in_bowl", 100000)
    summary = run(ctx_for(tmp_path, FakeFal(video=video_bytes(1.0, True))))  # 1 s for a 5 s request
    assert summary["outcomes"] == {"block_in_bowl/episode_100000/a1": "failed"}
    attempts = {a["attempt_id"]: a["evidence"] for a in EpisodeStore(tmp_path).load(key).human_attempts}
    assert attempts["a1-failed"]["reason"].startswith("decode_check")


def test_balanced_coverage_rounds(tmp_path):
    for task, n in (("task_a", 4), ("task_b", 1), ("task_c", 2)):
        for i in range(n):
            make_episode(tmp_path, task, 100000 + i)
    episodes, _ = gen.load_ready(tmp_path)
    plan = gen.plan_requests(episodes, {"requests": {}}, target=5, max_attempts=3)
    tasks = [p.episode.task for p in plan]
    assert sorted(tasks[:3]) == ["task_a", "task_b", "task_c"]
    assert {t: tasks.count(t) for t in set(tasks)} == {"task_a": 2, "task_b": 1, "task_c": 2}


def test_pilot_cap_limits_the_run_and_keeps_balance(tmp_path):
    for task in ("task_a", "task_b", "task_c"):
        for i in range(3):
            make_episode(tmp_path, task, 100000 + i)
    client = FakeFal()
    summary = run(ctx_for(tmp_path, client, pilot_cap=0.27))
    assert summary["reserved"] == 3 and len(client.submits) == 3
    assert sorted(k.split("/")[0] for k in summary["outcomes"]) == ["task_a", "task_b", "task_c"]
    assert Ledger(tmp_path / "generation" / "ledger.json").summary()["pilot_committed_usd"] <= 0.27


def test_dry_run_cli_never_creates_a_client(tmp_path, monkeypatch, capsys):
    make_episode(tmp_path, "block_in_bowl", 100000)
    make_episode(tmp_path, "blocks_onto_mats", 100000, action_order=TWO, action_text=TWO_TEXT)

    def boom(*a, **k):
        raise AssertionError("dry run must not build a fal client")

    monkeypatch.setattr(gen, "HttpFalClient", boom)
    assert gen.main(["plan", "--root", str(tmp_path)]) == 0
    assert gen.main(["run", "--root", str(tmp_path), "--budget", "5"]) == 0  # no --live: dry run
    out = capsys.readouterr().out
    assert "dry run" in out and out.count("block_in_bowl/episode_100000/a1") == 2
    assert not (tmp_path / "generation" / "ledger.json").exists()
    with pytest.raises(SystemExit):
        gen.main(["run", "--root", str(tmp_path), "--live"])  # no positive budget

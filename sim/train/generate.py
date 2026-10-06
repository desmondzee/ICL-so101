"""Credit-limited human-video generation (fal H3 Max Turbo) for robot-approved episodes.

Inputs are only hash-valid robot-approved episodes: those listed in ``<root>/fal_ready.json``
plus episodes that already left ``robot_approved`` for human generation (fal_ready.json only
lists the ``robot_approved`` state) and still carry a robot approval covering their current
artifacts. Every artifact is re-hashed through ``EpisodeStore.load`` and the frozen
``first.png``/``last.png`` bytes are hashed again immediately before each POST.

Request: ``minimax/h3-max-turbo/image-to-video`` at 480P, duration from a per-task policy
(5 s for one or two ordered actions, +2.5 s for each further action, at most 10 s;
``--duration task=seconds`` overrides), prompt expansion disabled, a deterministic seed per
attempt, and the v5 prompt of ``humangen.sim_val_demos`` whose action sentence lists the
task's ordered action descriptions one object at a time, with a right hand entering empty
from the image edge on the camera side of the table (``entry_edge``).

Spending goes through ``sim.train.budget.Ledger`` (``<root>/generation/ledger.json``):
reservations are taken in plan order before any POST, an uncertain POST is never resent,
and unresolved requests keep their reservation. Downloads are decode-checked, audio is
stripped, and the video lands in ``candidates/<task>/episode_<seed>/human/a<N>/``; immutable
store attempt records ``a<N>-submitted``, ``a<N>-complete`` / ``a<N>-failed`` hold request
ids, settings, seeds, endpoint hashes and cost. Generation never accepts anything:
``sim.train.human_review`` does that after a judge and an independent verifier.

Retries: at most ``--max-attempts`` (default 3) counted attempts per episode, each with a new
seed; a new attempt starts only when every earlier one was rejected by review or failed.
Exhausted episodes are moved to ``rejected`` with decision ``replace`` and listed in
``<root>/generation/replacements.json`` for the orchestrator.

    python -m sim.train.generate plan   [--phase pilot|bulk] [--target N] [--cap USD]    # dry run, no fal
    python -m sim.train.generate run --live --budget 5 --phase pilot [--target N]        # paid pilot
    python -m sim.train.generate run --live --budget 60 --phase bulk --target 600 --cap 120
    python -m sim.train.generate resume --live        # poll/download submitted requests only, no new POSTs
    python -m sim.train.generate status               # ledger totals, attempt outcomes, replacements
    python -m sim.train.generate resolve KEY (--request-id ID --status-url U --response-url U | --not-submitted EVIDENCE)
    python -m sim.train.generate reconcile KEY --usd X --source "fal usage export 2026-10-07"
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Protocol
import urllib.error
import uuid

from .budget import (ABSOLUTE_CAP_USD, DEFAULT_PILOT_CAP_USD, DEFAULT_RETRY_RESERVE, Ledger, price_per_second,
                     reservation_usd, totals)
from .model import EpisodeKey, EpisodeState
from .review import _approval, key_name, parse_key
from .store import EpisodeStore, EvidenceMismatch, StoreError, atomic_write_json, sha256_file

ENDPOINT = "minimax/h3-max-turbo/image-to-video"
RESOLUTION = "480P"
DEFAULT_DURATION_S = 5.0
EXTRA_ACTION_S = 2.5
MAX_DURATION_S = 10.0
HOLD_S = 1.0
DEFAULT_MAX_ATTEMPTS = 3
HUMAN_DIR = "human"
GEN_DIR = "generation"
GENERATION_STATES = (EpisodeState.ROBOT_APPROVED, EpisodeState.HUMAN_SUBMITTED, EpisodeState.HUMAN_COMPLETE)
LATER_STATES = tuple(s.value for s in (EpisodeState.HUMAN_SUBMITTED, EpisodeState.HUMAN_COMPLETE,
                                         EpisodeState.HUMAN_REVIEW_APPROVED, EpisodeState.VERIFIER_APPROVED,
                                         EpisodeState.ACCEPTED))
DURATION_TOLERANCE_S = (0.5, 1.5)  # decoded duration within [requested - 0.5, requested + 1.5]


# ----- prompt -------------------------------------------------------------------------------------------------
VERBS = {"do": "does", "go": "goes", "carry": "carries", "have": "has", "be": "is", "fly": "flies", "try": "tries",
         "empty": "empties", "tidy": "tidies", "copy": "copies"}
KNOWN_VERBS = {
    "pick", "put", "place", "lift", "take", "move", "turn", "lay", "set", "stack", "unstack", "push", "slide", "open",
    "close", "insert", "carry", "drop", "rotate", "hang", "pour", "grasp", "grip", "remove", "return", "align", "flip",
    "tilt", "press", "pull", "swap", "transfer", "raise", "lower", "bring", "release", "arrange", "sort", "nest", "fit",
    "rest", "hold", "keep", "leave", "go", "reach", "stand", "drag", "shift", "load", "unload", "fill", "empty",
    "position", "center", "centre", "line", "tuck", "cover", "uncover", "tip", "upend", "spin", "do", "repeat",
}
LEADING = ("first", "then", "next", "finally", "and", "after that", "afterwards")


def third_person(verb: str) -> str:
    v = verb.lower()
    if v in VERBS:
        return VERBS[v]
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def action_clause(text: str) -> str:
    """'Pick up the long block, turn it so it runs left to right, and lay it on the mat.'
    -> 'picks up the long block, turns it so it runs left to right, and lays it on the mat'."""
    clause = " ".join(text.strip().rstrip(".").split())
    lowered = clause.lower()
    for word in LEADING:
        if lowered.startswith(word + " ") or lowered.startswith(word + ", "):
            clause = clause[len(word):].lstrip(" ,")
            lowered = clause.lower()
    words = clause.split(" ")
    out = []
    for i, word in enumerate(words):
        prev = words[i - 1].lower() if i else ""
        bare = word.lower()
        if i == 0:
            out.append(third_person(bare))
        elif (prev in ("and", "then") or words[i - 1].endswith(",")) and bare in KNOWN_VERBS:
            out.append(third_person(bare))
        else:
            out.append(word)
    return " ".join(out)


def task_action(action_texts: list[str]) -> str:
    """The ordered action sentence: one object at a time, in the robot's order."""
    clauses = [action_clause(t) for t in action_texts]
    if not clauses or not all(clauses):
        raise ValueError("an episode needs at least one ordered action description")
    if len(clauses) == 1:
        return clauses[0]
    ordered = ", ".join(["first " + clauses[0]] + ["then " + c for c in clauses[1:]])
    return (f"{ordered}; the hand holds one object at a time and each object moves only while the hand is holding it, "
            "in exactly this order")


def prompt(action: str, duration_s: float = DEFAULT_DURATION_S, edge: str = "bottom") -> str:
    """The v5 prompt of humangen.sim_val_demos.prompt, with the duration and entry edge as parameters."""
    end = float(duration_s)
    action = (f"A person's single right hand and forearm reaches in empty from the {edge} edge, {action}, "
              "then releases its grip and withdraws empty through the same edge.")
    visual = ("The camera remains fixed. The task uses only the objects already visible in Picture 1, with the same count, "
              "appearance and background throughout. One acting hand and forearm performs the task; the rest of that person "
              f"stays outside the image. {action} The hand completes each manipulation directly at the task locations, with "
              f"continuous physical contact while moving the object. By {end - HOLD_S:.2f} seconds, the task is complete and the "
              "hand is in its ending state; the scene remains still for the final second. No other hands or objects enter the "
              "scene at any time.")
    return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
            f"mark of the target video; Picture 2 (from Shot 1) aligns with the {end:.2f}-second mark of the target video.\n\n"
            "integrated_multimodal_description: [Shot 1] Live-action, beginning with the camera framing, lighting, objects and "
            f"spatial arrangement established by Picture 1. {visual} The action continuously brings the scene into the object "
            "arrangement, hand visibility and composition established by Picture 2 at the end of this single static shot.\n\n"
            "overall_soundscape: a quiet room and the soft sounds of the hand handling the objects.")


def entry_edge(visual_config: dict) -> str:
    """Image edge on the person's side of the table.

    The SO-101 base sits on the -x side of the workspace, so a person faces it from +x. The
    front camera looks along (lookat - pos); the person's side projects to the bottom edge when
    the camera looks toward -x, to the top when it looks toward +x, otherwise to a side edge.
    """
    camera = visual_config["front_camera"]
    fx, fy = camera["lookat"][0] - camera["pos"][0], camera["lookat"][1] - camera["pos"][1]
    norm = math.hypot(fx, fy)
    fx, fy = fx / norm, fy / norm
    person = (1.0, 0.0)
    scores = {"bottom": -(person[0] * fx + person[1] * fy), "top": person[0] * fx + person[1] * fy,
              "right": person[0] * fy - person[1] * fx, "left": -(person[0] * fy - person[1] * fx)}
    return max(scores, key=scores.get)


def duration_for(task: str, n_actions: int, overrides: dict[str, float] | None = None) -> float:
    """Per-task duration policy: 5 s for one or two actions, +2.5 s per further action, <= 10 s."""
    if overrides and task in overrides:
        value = float(overrides[task])
    else:
        value = min(MAX_DURATION_S, DEFAULT_DURATION_S + EXTRA_ACTION_S * max(0, n_actions - 2))
    if not 2.0 <= value <= MAX_DURATION_S:
        raise ValueError(f"{task}: duration {value} outside [2, {MAX_DURATION_S}] s")
    return value


def attempt_seed(key: str, attempt: int) -> int:
    return int(hashlib.sha256(f"sim_train_v1|{key}|a{attempt}".encode()).hexdigest()[:8], 16) % (2 ** 31)


# ----- fal client ---------------------------------------------------------------------------------------------
class SubmitRejected(Exception):
    """The server answered the POST with a 4xx: the request was provably not queued."""

    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class RequestFailed(Exception):
    """fal reported that a queued request failed."""


class FalClient(Protocol):
    def submit(self, endpoint: str, payload: dict) -> dict: ...  # {request_id, status_url, response_url}
    def status(self, handle: dict) -> str: ...  # IN_QUEUE | IN_PROGRESS | COMPLETED (raises RequestFailed)
    def result(self, handle: dict) -> dict: ...  # {"video": {"url": ...}, ...}
    def download(self, url: str) -> bytes: ...


class HttpFalClient:
    """fal queue REST client (same protocol as humangen.alternative_demos.run)."""

    def __init__(self, token: str):
        if not token:
            raise ValueError("FAL_API_KEY missing")
        self.token = token

    def submit(self, endpoint: str, payload: dict) -> dict:
        from humangen.alternative_demos import request
        try:
            queue = json.loads(request("https://queue.fal.run/" + endpoint, self.token, payload))
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise SubmitRejected(exc.code) from None
            raise
        return {"request_id": queue["request_id"], "status_url": queue["status_url"],
                "response_url": queue["response_url"]}

    def status(self, handle: dict) -> str:
        from humangen.alternative_demos import request
        value = json.loads(request(handle["status_url"], self.token))
        if value.get("status") in ("FAILED", "ERROR") or value.get("error"):
            raise RequestFailed(str(value.get("error") or value.get("status")))
        return value.get("status", "UNKNOWN")

    def result(self, handle: dict) -> dict:
        from humangen.alternative_demos import request
        try:
            return json.loads(request(handle["response_url"], self.token))
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 422):
                raise RequestFailed(f"result HTTP {exc.code}") from None
            raise

    def download(self, url: str) -> bytes:
        from humangen.alternative_demos import request
        return request(url)


# ----- inputs ---------------------------------------------------------------------------------------------
@dataclass
class ReadyEpisode:
    key: EpisodeKey
    name: str
    task: str
    family: str
    instruction: str
    action_order: list[str]
    action_text: list[str]
    state: EpisodeState
    first_sha256: str
    last_sha256: str
    visual_config: dict
    attempts: list[dict] = field(default_factory=list)


def _verified(store: EpisodeStore, key: EpisodeKey, listed: dict | None) -> ReadyEpisode:
    record = store.load(key)  # re-hashes every manifest and event artifact
    if record.state not in GENERATION_STATES and record.state not in (
            EpisodeState.HUMAN_REVIEW_APPROVED, EpisodeState.VERIFIER_APPROVED, EpisodeState.ACCEPTED):
        raise StoreError(f"{key_name(key)} is {record.state.value}")
    manifest = record.manifest
    approval = _approval(record)
    if approval.get("decision") != "accept" or approval.get("artifact_hashes") != manifest.artifacts:
        raise EvidenceMismatch(f"robot approval does not cover current artifacts: {key_name(key)}")
    if listed is not None:
        for name, field_name in (("first.png", "first"), ("last.png", "last")):
            if listed[field_name]["sha256"] != manifest.artifacts.get(name):
                raise EvidenceMismatch(f"fal_ready {field_name} hash differs from manifest: {key_name(key)}")
        if listed.get("manifest_sha256") != sha256_file(store.manifest_path(key)):
            raise EvidenceMismatch(f"fal_ready manifest hash is stale: {key_name(key)}")
    meta = manifest.metadata
    episode = json.loads((store.episode_dir(key) / "episode.json").read_text())
    return ReadyEpisode(key, key_name(key), key.task, meta["family"], meta["instruction"], list(meta["action_order"]),
                        list(meta["action_text"]), record.state, manifest.artifacts["first.png"],
                        manifest.artifacts["last.png"], episode["visual_config"], list(record.human_attempts))


def load_ready(root: Path) -> tuple[list[ReadyEpisode], dict[str, str]]:
    """Hash-valid robot-approved episodes (fal_ready.json + those already in human generation)."""
    root = Path(root)
    store = EpisodeStore(root)
    listed = {}
    index = root / "fal_ready.json"
    if index.exists():
        for entry in json.loads(index.read_text())["episodes"]:
            listed[entry["key"]] = entry
    out, refused = {}, {}
    for name, entry in sorted(listed.items()):
        try:
            out[name] = _verified(store, parse_key(name), entry)
        except (StoreError, ValueError, KeyError, OSError) as exc:
            refused[name] = f"{type(exc).__name__}: {exc}"
    for path in sorted((root / "candidates").glob("*/episode_*/state.json")):
        state = json.loads(path.read_text())
        name = f"{path.parent.parent.name}/{path.parent.name}"
        if name in out or name in refused or state.get("state") not in LATER_STATES:
            continue
        try:
            out[name] = _verified(store, parse_key(name), None)
        except (StoreError, ValueError, KeyError, OSError) as exc:
            refused[name] = f"{type(exc).__name__}: {exc}"
    return [out[k] for k in sorted(out)], refused


# ----- attempt bookkeeping -------------------------------------------------------------------------------------
def request_key(name: str, attempt: int) -> str:
    return f"{name}/a{attempt}"


def attempt_outcomes(episode: ReadyEpisode, ledger_doc: dict) -> dict[int, str]:
    """Attempt number -> in_flight | uncertain | released | failed | awaiting_review | accepted | rejected."""
    records = {a["attempt_id"]: a["evidence"] for a in episode.attempts}
    numbers = set()
    for key in ledger_doc["requests"]:
        if key.startswith(episode.name + "/a"):
            numbers.add(int(key.rsplit("/a", 1)[1]))
    for attempt_id in records:
        numbers.add(int(attempt_id.split("-")[0][1:]))
    out = {}
    for n in sorted(numbers):
        entry = ledger_doc["requests"].get(request_key(episode.name, n))
        review = records.get(f"a{n}-review")
        if review is not None:
            out[n] = "accepted" if review["decision"] == "accept" else "rejected"
        elif f"a{n}-failed" in records:
            out[n] = "failed"
        elif f"a{n}-complete" in records:
            out[n] = "awaiting_review"
        elif entry is None:
            out[n] = "in_flight"  # store record without ledger entry: inspect, never resubmit
        elif entry["status"] == "released":
            out[n] = "released"
        elif entry["status"] == "uncertain":
            out[n] = "uncertain"
        elif entry["status"] == "failed":
            out[n] = "failed"
        else:
            out[n] = "in_flight"
    return out


def next_step(episode: ReadyEpisode, ledger_doc: dict, max_attempts: int) -> tuple[str, int]:
    """(action, next attempt number); action is first | retry | wait | blocked | done | exhausted."""
    if episode.state not in GENERATION_STATES:
        return "done", 0
    outcomes = attempt_outcomes(episode, ledger_doc)
    values = set(outcomes.values())
    next_n = max(outcomes, default=0) + 1
    if "accepted" in values:
        return "done", 0
    if "uncertain" in values:
        return "blocked", 0
    if values & {"in_flight", "awaiting_review"}:
        return "wait", 0
    counted = sum(1 for v in outcomes.values() if v != "released")
    if counted >= max_attempts:
        return "exhausted", 0
    return ("first" if counted == 0 else "retry"), next_n


@dataclass
class PlannedRequest:
    episode: ReadyEpisode
    attempt: int
    kind: str
    seed: int
    duration_s: float
    edge: str
    prompt: str

    @property
    def key(self) -> str:
        return request_key(self.episode.name, self.attempt)


def plan_requests(episodes: list[ReadyEpisode], ledger_doc: dict, *, target: int | None, max_attempts: int,
                  durations: dict[str, float] | None = None, tasks: set[str] | None = None) -> list[PlannedRequest]:
    """Balanced coverage rounds: each round gives every task one request (retries before new
    episodes), tasks with the fewest pairs in progress or accepted first."""
    queues: dict[str, list[PlannedRequest]] = {}
    progress: dict[str, int] = {}
    for ep in episodes:
        if tasks and ep.task not in tasks:
            continue
        action, n = next_step(ep, ledger_doc, max_attempts)
        progress.setdefault(ep.task, 0)
        if ep.state is not EpisodeState.ROBOT_APPROVED:  # pairs already in progress or accepted
            progress[ep.task] += 1
        if action not in ("first", "retry"):
            continue
        duration = duration_for(ep.task, len(ep.action_text), durations)
        edge = entry_edge(ep.visual_config)
        text = prompt(task_action(ep.action_text), duration, edge)
        queues.setdefault(ep.task, []).append(PlannedRequest(ep, n, action, attempt_seed(ep.name, n), duration, edge, text))
    for task in queues:
        queues[task].sort(key=lambda r: (r.kind != "retry", r.episode.key.seed))
    order = sorted(queues, key=lambda t: (progress.get(t, 0), t))
    plan: list[PlannedRequest] = []
    while any(queues.values()) and (target is None or len(plan) < target):
        for task in order:
            if queues[task] and (target is None or len(plan) < target):
                plan.append(queues[task].pop(0))
    return plan


def simulate_caps(plan: list[PlannedRequest], ledger: Ledger, *, phase: str, cap_usd: float, run_budget_usd: float | None,
                  retry_reserve_fraction: float, price_override: float | None) -> list[dict]:
    """Dry-run of the reservations the run would take, without writing the ledger."""
    document = ledger.snapshot()
    t = totals(document)
    committed, pilot, first, run = t["committed_usd"], t["pilot_committed_usd"], t["first_attempt_committed_usd"], 0.0
    rows = []
    for item in plan:
        reserve = reservation_usd(item.duration_s, override=price_override)
        estimate = round(item.duration_s * price_per_second(override=price_override), 6)
        reason = "ok"
        if committed + reserve > cap_usd + 1e-9:
            reason = "hard_cap"
        elif phase == "pilot" and pilot + reserve > ledger.pilot_cap_usd + 1e-9:
            reason = "pilot_cap"
        elif item.kind == "first" and first + reserve > cap_usd * (1 - retry_reserve_fraction) + 1e-9:
            reason = "retry_reserve"
        elif run_budget_usd is not None and run + reserve > run_budget_usd + 1e-9:
            reason = "run_budget"
        if reason == "ok":
            committed += reserve
            run += reserve
            pilot += reserve if phase == "pilot" else 0
            first += reserve if item.kind == "first" else 0
        rows.append({"key": item.key, "task": item.episode.task, "kind": item.kind, "seed": item.seed,
                     "duration_s": item.duration_s, "edge": item.edge, "reservation_usd": reserve,
                     "estimated_cost_usd": estimate, "fits": reason == "ok", "reason": reason,
                     "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest()})
    return rows


# ----- execution -----------------------------------------------------------------------------------------------
@dataclass
class Context:
    root: Path
    client: FalClient
    ledger: Ledger
    store: EpisodeStore
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 1800.0
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = print


def _endpoint_bytes(store: EpisodeStore, key: EpisodeKey, first_sha: str, last_sha: str) -> tuple[bytes, bytes]:
    """Read the frozen endpoints once and hash exactly the bytes that will be sent."""
    directory = store.episode_dir(key)
    first, last = (directory / "first.png").read_bytes(), (directory / "last.png").read_bytes()
    if hashlib.sha256(first).hexdigest() != first_sha or hashlib.sha256(last).hexdigest() != last_sha:
        raise EvidenceMismatch(f"frozen endpoint changed on disk: {key_name(key)}")
    return first, last


def _data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def settings_for(item: PlannedRequest) -> dict:
    duration = int(item.duration_s) if float(item.duration_s).is_integer() else item.duration_s
    return {"resolution": RESOLUTION, "duration": duration, "prompt_expansion_mode": "disabled", "seed": item.seed}


def _endpoint_hashes(entry: dict) -> dict:
    return {"first.png": entry["first_sha256"], "last.png": entry["last_sha256"]}


def _ensure_submitted(ctx: Context, entry: dict) -> None:
    """Idempotently write the a<N>-submitted record and the robot_approved -> human_submitted edge."""
    key = parse_key(entry["episode"])
    n = entry["attempt"]
    evidence = {"stage": "human_submitted", "attempt": f"a{n}", "ledger_key": entry["key"],
                "request_id": entry["request_id"], "endpoint": entry["endpoint"], "settings": entry["settings"],
                "seed": entry["seed"], "prompt": entry["prompt"], "prompt_sha256": entry["prompt_sha256"],
                "reserved_usd": entry["reserved_usd"], "price_per_s": entry["price_per_s"], "phase": entry["phase"],
                "artifact_hashes": _endpoint_hashes(entry)}
    ctx.store.record_human_attempt(key, f"a{n}-submitted", evidence)
    record = ctx.store.load(key)
    if record.state is EpisodeState.ROBOT_APPROVED:
        ctx.store.transition(key, EpisodeState.ROBOT_APPROVED, EpisodeState.HUMAN_SUBMITTED,
                             {"stage": "human_submitted", "attempt": f"a{n}", "request_id": entry["request_id"],
                              "artifact_hashes": _endpoint_hashes(entry)})


def reserve_plan(ctx: Context, plan: list[PlannedRequest], *, phase: str, run_id: str, cap_usd: float,
                 run_budget_usd: float | None, retry_reserve_fraction: float, price_override: float | None) -> list[PlannedRequest]:
    """Take reservations sequentially in plan order (balanced rounds survive the cap)."""
    reserved = []
    for item in plan:
        entry_meta = {
            "episode": item.episode.name, "task": item.episode.task, "attempt": item.attempt, "kind": item.kind,
            "seed": item.seed, "endpoint": ENDPOINT, "settings": settings_for(item), "duration_s": item.duration_s,
            "edge": item.edge, "prompt": item.prompt, "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
            "first_sha256": item.episode.first_sha256, "last_sha256": item.episode.last_sha256,
            "price_per_s": price_per_second(override=price_override), "price_override": price_override,
        }
        ok, reason = ctx.ledger.reserve(item.key, reservation_usd(item.duration_s, override=price_override), phase=phase,
                                        first_attempt=item.kind == "first", run_id=run_id, cap_usd=cap_usd,
                                        run_budget_usd=run_budget_usd, retry_reserve_fraction=retry_reserve_fraction,
                                        meta=entry_meta)
        if not ok:
            ctx.log(f"stop reserving at {item.key}: {reason}")
            break
        reserved.append(item)
    return reserved


def submit(ctx: Context, item: PlannedRequest) -> str:
    """POST one reserved request. Never retried: an exception after 'submitting' is uncertain."""
    key = item.key
    try:
        first, last = _endpoint_bytes(ctx.store, item.episode.key, item.episode.first_sha256, item.episode.last_sha256)
    except (OSError, EvidenceMismatch) as exc:
        ctx.ledger.release_unsent(key, f"endpoint check failed before POST: {exc}")
        return "endpoint_mismatch"
    payload = {"prompt": item.prompt, "image_url": _data_url(first), "end_image_url": _data_url(last),
               **settings_for(item)}
    ctx.ledger.mark_submitting(key)
    try:
        handle = ctx.client.submit(ENDPOINT, payload)
    except SubmitRejected as exc:
        ctx.ledger.release_unsent(key, f"server refused before queueing: {exc}")
        return "refused"
    except BaseException as exc:  # noqa: BLE001 - delivery unknown: keep the reservation, never resend
        ctx.ledger.mark_uncertain(key, f"{type(exc).__name__} during POST")
        return "uncertain"
    ctx.ledger.mark_submitted(key, handle)
    _ensure_submitted(ctx, ctx.ledger.get(key))
    return "submitted"


def _probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def check_video(video: Path, requested_s: float) -> dict:
    """Exactly one decodable video stream, no audio, a sane duration."""
    probe = _probe(video)
    streams = probe["streams"]
    if [s["codec_type"] for s in streams] != ["video"]:
        raise ValueError(f"expected one video stream and no audio, got {[s['codec_type'] for s in streams]}")
    s = streams[0]
    frames = int(s.get("nb_read_frames") or 0)
    duration = float(probe["format"]["duration"])
    if frames < 1 or not s.get("width") or not s.get("height"):
        raise ValueError("video has no decodable frames")
    low, high = DURATION_TOLERANCE_S
    if not requested_s - low <= duration <= requested_s + high:
        raise ValueError(f"duration {duration:.2f}s outside [{requested_s - low}, {requested_s + high}]")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
                   check=True, capture_output=True)
    return {"codec": s["codec_name"], "width": s["width"], "height": s["height"], "fps": s["r_frame_rate"],
            "frames": frames, "duration_s": round(duration, 4)}


def _fsync_dir(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _cost(entry: dict, decoded_s: float | None) -> float:
    seconds = max(entry["duration_s"], decoded_s or 0.0)
    return round(seconds * entry["price_per_s"], 6)


def finalize(ctx: Context, entry: dict, result: dict) -> str:
    """Download, strip audio, decode-check and record one completed request (idempotent)."""
    key = parse_key(entry["episode"])
    n = entry["attempt"]
    attempts = {a["attempt_id"]: a for a in ctx.store.load(key).human_attempts}
    for done in (f"a{n}-complete", f"a{n}-failed"):
        if done in attempts:
            ev = attempts[done]["evidence"]
            ctx.ledger.mark_completed(entry["key"], ev["cost_usd"], ev["cost_source"])
            return "complete" if done.endswith("complete") else "failed"
    rel = Path(HUMAN_DIR) / f"a{n}"
    final = ctx.store.episode_dir(key) / rel
    failure = None
    if not final.exists():
        work = ctx.root / GEN_DIR / "tmp" / f"{entry['key'].replace('/', '__')}-{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True)
        raw = work / "raw.mp4"
        raw.write_bytes(ctx.client.download(result["video"]["url"]))
        try:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw), "-map", "0:v:0", "-c:v", "copy",
                            "-an", "-sn", "-dn", "-map_metadata", "-1", str(work / "human.mp4")],
                           check=True, capture_output=True)
            probe = check_video(work / "human.mp4", entry["duration_s"])
        except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
            failure = f"decode_check: {type(exc).__name__}: {str(exc)[:300]}"
            bad = ctx.root / GEN_DIR / "bad_downloads" / entry["key"].replace("/", "__")
            bad.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(raw), bad.with_suffix(".mp4"))
            shutil.rmtree(work)
            probe = None
        if failure is None:
            raw.unlink()
            (work / "prompt.txt").write_text(entry["prompt"])
            for name, value in (("request.json", {k: entry[k] for k in ("key", "episode", "attempt", "request_id", "endpoint",
                                                                       "settings", "seed", "prompt_sha256", "first_sha256",
                                                                       "last_sha256", "duration_s", "edge")}),
                                ("response.json", result), ("probe.json", probe)):
                (work / name).write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")
            _fsync_dir(work)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.rename(work, final)
    if failure is not None:
        cost = _cost(entry, None)
        evidence = {"stage": "human_failed", "attempt": f"a{n}", "request_id": entry["request_id"],
                    "reason": failure, "cost_usd": cost, "cost_source": "estimate:requested_duration",
                    "artifact_hashes": _endpoint_hashes(entry)}
        ctx.store.record_human_attempt(key, f"a{n}-failed", evidence)
        ctx.ledger.mark_completed(entry["key"], cost, "estimate:requested_duration")
        return "failed"
    probe = check_video(final / "human.mp4", entry["duration_s"])  # re-check after crash recovery too
    hashes = {str(rel / name): sha256_file(final / name)
              for name in ("human.mp4", "prompt.txt", "request.json", "response.json", "probe.json")}
    cost = _cost(entry, probe["duration_s"])
    evidence = {"stage": "human_complete", "attempt": f"a{n}", "request_id": entry["request_id"],
                "ledger_key": entry["key"], "video": str(rel / "human.mp4"), "video_sha256": hashes[str(rel / "human.mp4")],
                "probe": probe, "settings": entry["settings"], "seed": entry["seed"],
                "prompt_sha256": entry["prompt_sha256"], "cost_usd": cost, "cost_source": "estimate:output_duration",
                "artifact_hashes": {**_endpoint_hashes(entry), **hashes}}
    ctx.store.record_human_attempt(key, f"a{n}-complete", evidence)
    record = ctx.store.load(key)
    if record.state is EpisodeState.HUMAN_SUBMITTED:
        ctx.store.transition(key, EpisodeState.HUMAN_SUBMITTED, EpisodeState.HUMAN_COMPLETE,
                             {"stage": "human_complete", "attempt": f"a{n}", "request_id": entry["request_id"],
                              "artifact_hashes": evidence["artifact_hashes"]})
    ctx.ledger.mark_completed(entry["key"], cost, "estimate:output_duration")
    return "complete"


def poll(ctx: Context, ledger_key: str) -> str:
    """Wait for one submitted request; safe to repeat (GETs only, never a new POST)."""
    entry = ctx.ledger.get(ledger_key)
    if entry is None or entry["status"] != "submitted":
        return entry["status"] if entry else "missing"
    _ensure_submitted(ctx, entry)
    handle = entry["handle"]
    deadline = time.monotonic() + ctx.poll_timeout_s
    try:
        while True:
            status = ctx.client.status(handle)
            if status == "COMPLETED":
                break
            if time.monotonic() >= deadline:
                return "pending"
            ctx.sleep(ctx.poll_interval_s)
        result = ctx.client.result(handle)
    except RequestFailed as exc:
        key = parse_key(entry["episode"])
        ctx.store.record_human_attempt(key, f"a{entry['attempt']}-failed", {
            "stage": "human_failed", "attempt": f"a{entry['attempt']}", "request_id": entry["request_id"],
            "reason": f"fal: {str(exc)[:300]}", "cost_usd": entry["reserved_usd"], "cost_source": "reservation_kept",
            "artifact_hashes": _endpoint_hashes(entry)})
        ctx.ledger.mark_failed(ledger_key, str(exc)[:300])
        return "failed"
    except Exception as exc:  # noqa: BLE001 - transient retrieval error: stays submitted, resumable
        ctx.log(f"{ledger_key}: retrieval error {type(exc).__name__}; resume later")
        return "pending"
    return finalize(ctx, entry, result)


def _safe_poll(ctx: Context, ledger_key: str) -> str:
    try:
        return poll(ctx, ledger_key)
    except Exception as exc:  # noqa: BLE001 - stays 'submitted'; a later resume retries the GETs
        ctx.log(f"{ledger_key}: finalize error {type(exc).__name__}: {exc}")
        return "error"


def submit_and_poll(ctx: Context, item: PlannedRequest) -> tuple[str, str]:
    outcome = submit(ctx, item)
    if outcome != "submitted":
        return item.key, outcome
    return item.key, _safe_poll(ctx, item.key)


def mark_replacements(root: Path, max_attempts: int, ledger: Ledger) -> list[str]:
    """Reject exhausted episodes (decision 'replace') and rewrite generation/replacements.json."""
    store = EpisodeStore(root)
    episodes, _ = load_ready(root)
    doc = ledger.snapshot()
    for ep in episodes:
        action, _ = next_step(ep, doc, max_attempts)
        if action != "exhausted":
            continue
        outcomes = attempt_outcomes(ep, doc)
        reasons = {}
        for a in ep.attempts:
            if a["attempt_id"].endswith(("-review", "-failed")):
                reasons[a["attempt_id"]] = a["evidence"].get("reasons") or a["evidence"].get("reason")
        store.transition(ep.key, ep.state, EpisodeState.REJECTED, {
            "stage": "human_generation", "decision": "replace", "max_attempts": max_attempts,
            "attempts": {f"a{n}": v for n, v in outcomes.items()}, "reasons": reasons, "artifact_hashes": {}})
    return write_replacements(root)


def write_replacements(root: Path) -> list[str]:
    root = Path(root)
    store = EpisodeStore(root)
    items = []
    for path in sorted((root / "candidates").glob("*/episode_*/state.json")):
        state = json.loads(path.read_text())
        if state["state"] != EpisodeState.REJECTED.value or not state["history"]:
            continue
        last = state["history"][-1]["evidence"]
        if last.get("stage") == "human_generation" and last.get("decision") == "replace":
            name = f"{path.parent.parent.name}/{path.parent.name}"
            store.load(parse_key(name))
            items.append({"key": name, "task": path.parent.parent.name, "attempts": last["attempts"],
                          "reasons": last["reasons"]})
    (root / GEN_DIR).mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / GEN_DIR / "replacements.json", {"schema_version": 1, "episodes": items})
    return [i["key"] for i in items]


@contextmanager
def run_lock(root: Path):
    """One generation run per root; crash recovery happens only under this lock."""
    path = Path(root) / GEN_DIR / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another generation run holds the lock") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resume_submitted(ctx: Context, workers: int) -> dict:
    """Recover interrupted reservations and poll every submitted request (no POSTs)."""
    recovered = ctx.ledger.recover_interrupted()
    doc = ctx.ledger.snapshot()
    pending = [k for k, e in doc["requests"].items() if e["status"] == "submitted"]
    for k, e in doc["requests"].items():  # completed in ledger but records lost: rebuild from the episode dir
        if e["status"] == "completed" and not (ctx.store.episode_dir(parse_key(e["episode"])) / HUMAN_DIR / f"a{e['attempt']}").exists():
            attempts = {a["attempt_id"] for a in ctx.store.load(parse_key(e["episode"])).human_attempts}
            if f"a{e['attempt']}-failed" not in attempts:
                ctx.log(f"{k}: completed in ledger without video or failure record; inspect")
    with ThreadPoolExecutor(max(1, workers)) as pool:
        outcomes = dict(zip(pending, pool.map(lambda k: _safe_poll(ctx, k), pending)))
    return {"recovered": recovered, "polled": outcomes}


def run(ctx: Context, *, phase: str, target: int | None, cap_usd: float, run_budget_usd: float,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS, retry_reserve_fraction: float = DEFAULT_RETRY_RESERVE,
        price_override: float | None = None, durations: dict[str, float] | None = None, tasks: set[str] | None = None,
        workers: int = 4, run_id: str | None = None) -> dict:
    if not run_budget_usd > 0:
        raise ValueError("a positive run budget is required")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
    with run_lock(ctx.root):
        resumed = resume_submitted(ctx, workers)
        episodes, refused = load_ready(ctx.root)
        plan = plan_requests(episodes, ctx.ledger.snapshot(), target=target, max_attempts=max_attempts,
                             durations=durations, tasks=tasks)
        reserved = reserve_plan(ctx, plan, phase=phase, run_id=run_id, cap_usd=cap_usd, run_budget_usd=run_budget_usd,
                                retry_reserve_fraction=retry_reserve_fraction, price_override=price_override)
        with ThreadPoolExecutor(max(1, workers)) as pool:
            outcomes = dict(pool.map(lambda item: submit_and_poll(ctx, item), reserved))
        replacements = mark_replacements(ctx.root, max_attempts, ctx.ledger)
    return {"run_id": run_id, "resumed": resumed, "refused_inputs": refused, "planned": len(plan),
            "reserved": len(reserved), "outcomes": outcomes, "replacements": replacements,
            "ledger": ctx.ledger.summary(cap_usd)}


# ----- CLI -----------------------------------------------------------------------------------------------------
def _durations(values: list[str]) -> dict[str, float]:
    out = {}
    for value in values or []:
        task, _, seconds = value.partition("=")
        out[task] = float(seconds)
    return out


def status_report(root: Path, ledger: Ledger, max_attempts: int) -> dict:
    episodes, refused = load_ready(root)
    doc = ledger.snapshot()
    steps: dict[str, int] = {}
    per_task: dict[str, dict[str, int]] = {}
    for ep in episodes:
        action, _ = next_step(ep, doc, max_attempts)
        steps[action] = steps.get(action, 0) + 1
        per_task.setdefault(ep.task, {}).setdefault(action, 0)
        per_task[ep.task][action] += 1
    replacements = json.loads((Path(root) / GEN_DIR / "replacements.json").read_text())["episodes"] \
        if (Path(root) / GEN_DIR / "replacements.json").exists() else []
    uncertain = [k for k, e in doc["requests"].items() if e["status"] == "uncertain"]
    return {"ledger": ledger.summary(), "episodes": len(episodes), "next_steps": steps, "per_task": per_task,
            "uncertain_requests": uncertain, "replacements": [r["key"] for r in replacements], "refused_inputs": refused}


def main(argv=None) -> int:
    from .record import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--root", type=Path, default=ROOT)
        p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
        p.add_argument("--pilot-cap", type=float, default=DEFAULT_PILOT_CAP_USD)

    for name in ("plan", "run"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--phase", choices=("pilot", "bulk"), default="pilot")
        p.add_argument("--target", type=int, help="number of requests this run (coverage rounds across tasks)")
        p.add_argument("--cap", type=float, default=ABSOLUTE_CAP_USD, help=f"hard cap, at most {ABSOLUTE_CAP_USD}")
        p.add_argument("--budget", type=float, help="spending limit for this run (required, > 0, for run)")
        p.add_argument("--retry-reserve", type=float, default=DEFAULT_RETRY_RESERVE)
        p.add_argument("--price-per-second", type=float, help="override the price schedule")
        p.add_argument("--duration", action="append", help="task=seconds override (repeatable)")
        p.add_argument("--tasks", help="comma-separated task subset")
        p.add_argument("--workers", type=int, default=4)
        if name == "run":
            p.add_argument("--live", action="store_true", help="actually call fal (paid)")
    p = sub.add_parser("resume")
    common(p)
    p.add_argument("--live", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("status")
    common(p)
    p = sub.add_parser("resolve")
    common(p)
    p.add_argument("key")
    p.add_argument("--request-id")
    p.add_argument("--status-url")
    p.add_argument("--response-url")
    p.add_argument("--not-submitted", metavar="EVIDENCE")
    p = sub.add_parser("reconcile")
    common(p)
    p.add_argument("key")
    p.add_argument("--usd", type=float, required=True)
    p.add_argument("--source", required=True)
    args = parser.parse_args(argv)

    ledger = Ledger(Path(args.root) / GEN_DIR / "ledger.json", pilot_cap_usd=args.pilot_cap)
    if args.command == "status":
        print(json.dumps(status_report(args.root, ledger, args.max_attempts), indent=1))
        return 0
    if args.command == "resolve":
        if args.not_submitted:
            ledger.resolve_uncertain(args.key, not_submitted_evidence=args.not_submitted)
        else:
            ledger.resolve_uncertain(args.key, handle={"request_id": args.request_id, "status_url": args.status_url,
                                                       "response_url": args.response_url})
        print(json.dumps(ledger.get(args.key), indent=1))
        return 0
    if args.command == "reconcile":
        ledger.reconcile(args.key, args.usd, args.source)
        print(json.dumps(ledger.summary(), indent=1))
        return 0
    if args.command == "plan" or (args.command == "run" and not args.live):
        if args.command == "run":
            print("dry run: pass --live and a positive --budget to submit paid requests")
        cap = min(args.cap, ABSOLUTE_CAP_USD)
        episodes, refused = load_ready(args.root)
        tasks = set(args.tasks.split(",")) if args.tasks else None
        plan = plan_requests(episodes, ledger.snapshot(), target=args.target, max_attempts=args.max_attempts,
                             durations=_durations(args.duration), tasks=tasks)
        rows = simulate_caps(plan, ledger, phase=args.phase, cap_usd=cap, run_budget_usd=args.budget,
                             retry_reserve_fraction=args.retry_reserve, price_override=args.price_per_second)
        for row in rows:
            print(f"{row['key']:<48} {row['kind']:<5} seed={row['seed']:<10} {row['duration_s']:>4}s {row['edge']:<6} "
                  f"est=${row['estimated_cost_usd']:.4f} reserve=${row['reservation_usd']:.2f} "
                  f"{'fits' if row['fits'] else 'SKIP:' + row['reason']}")
        fitting = [r for r in rows if r["fits"]]
        summary = {"fal_ready_episodes": len(episodes), "refused_inputs": refused, "planned_requests": len(rows),
                   "fitting_requests": len(fitting),
                   "per_task": {t: sum(1 for r in fitting if r["task"] == t) for t in sorted({r["task"] for r in rows})},
                   "estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in fitting), 4),
                   "reserved_usd": round(sum(r["reservation_usd"] for r in fitting), 2),
                   "price_per_s_now": price_per_second(override=args.price_per_second), "ledger": ledger.summary(cap)}
        print(json.dumps(summary, indent=1))
        return 0
    if not args.live:
        print("resume contacts fal; pass --live")
        return 2
    if args.command == "run" and not (args.budget and args.budget > 0):
        parser.error("run --live requires a positive --budget")
    from dotenv import dotenv_values
    token = dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("FAL_API_KEY") or os.environ.get("FAL_API_KEY")
    ctx = Context(Path(args.root), HttpFalClient(token), ledger, EpisodeStore(args.root))
    if args.command == "resume":
        with run_lock(ctx.root):
            print(json.dumps(resume_submitted(ctx, args.workers), indent=1))
        return 0
    summary = run(ctx, phase=args.phase, target=args.target, cap_usd=min(args.cap, ABSOLUTE_CAP_USD),
                  run_budget_usd=args.budget, max_attempts=args.max_attempts, retry_reserve_fraction=args.retry_reserve,
                  price_override=args.price_per_second, durations=_durations(args.duration),
                  tasks=set(args.tasks.split(",")) if args.tasks else None, workers=args.workers)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Draft pair records for LeRobot episodes: task, scene and step segments from a VLM (Gemini or OpenRouter), using gripper events."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

from schema.source import Episode, cycles, episodes, fetch, frame, holds, load_episode, local_source, ref, segments, uniform
from schema.validate import bundle, check_pair, check_scene, check_task

TASK_PROMPT = """Write the task schema for this robot dataset task.
Dataset task string: {instruction!r}
Image: {n} frames of episode {episode} in time order, each tile labelled with its frame number; camera {camera}; the robot is a single {robot} arm.
Gripper events (hints, may be incomplete): {events}
- id: {task_id!r}. version: "0.2". instruction: the dataset string, verbatim.
- robot_caption: imperative sentence naming objects by colour and type, e.g. "Put the red cup on the plate."
- roles: every participant, plus role "robot" (kind actor). name: colour + type, at most 4 words.
- steps: one per object manipulation, in the order seen (each object moved is one step; opening a drawer is one step). A tool use is one step covering picking up, using and putting down the tool (e.g. one wipe step); do not add separate lift steps.
- goals: the end state, including released objects not held_by the robot.
- Use role names everywhere. Only facts the task requires.{hint}"""

SCENE_PROMPT = """Annotate the scene at the first frame of a robot episode (image 1, camera {camera}).
The robot is a single {robot} arm: include it as entity "robot" (kind actor).
Images 2-3 (middle and last frame) and image 4 (frames across the episode) show how it unfolds; use them to tell which objects the task moves and to identify objects the robot hides in image 1, never for image 1's state.
- scene.id: {scene_id!r}. scene.version: "0.2". scene.view.camera_key: {camera!r}.
- view.shot: camera placement and framing, e.g. "high front view of a white table".
- entities: task objects, visible distractors, the table or surface they rest on, the robot. Each separately movable object is its own entity (two slippers are two entities).
- name: colour + type, at most 4 words; an entity bound to a role uses the role's name and kind verbatim. grounding: image position, e.g. "lower left".
- attributes: null when not visible; do not guess.
- initial_state: every goal relation below (true/false; null if occluded), plus supported_by facts.
- evidence asset: {asset!r}.
- bindings: each task role to one entity. Task roles: {roles}
Task goals: {goals}"""

SEGMENT_PROMPT = """Find where each task step happens in this robot episode.
Image: {n} frames in time order, each tile labelled with its frame number; the episode has {length} frames at {fps:g} fps.
Gripper hints (may be incomplete or include extra events): closes at frames {closes}; reopens at frames {opens}.
Steps, in order: {steps}
For each step give start (frame where the arm starts moving toward the step's object) and end (frame where the step's result holds and the arm has let go or moved away). A tool step (e.g. wipe) runs from reaching for the tool until it is put down.
Steps are in the given order and do not overlap; frames between steps may belong to neither."""

OUTCOME_PROMPT = """Did this robot episode achieve its task? Image 1: first frame. Image 2: the last seconds of the episode, labelled frames in time order, ending with the last frame. Camera {camera}.
Task: {caption}
For each statement below, in order, say whether it is true at the END of the episode: true, false, or null if it cannot be judged.
An object dropped into a container may be hidden by its rim; use the sequence to judge where it went.
Statements: {goals}"""

ASSIGN_PROMPT = """Each row of the image is one grasp window of a robot episode: the frames where the gripper closes on an object, carries it, and releases it (labelled with frame numbers).
For each window, in order, name which object the gripper is holding. Objects: {objects}"""

FIX_PROMPT = "The JSON failed validation. Return a corrected version.\nErrors:\n{errors}\nJSON:\n{doc}"


class Gemini:
    def __init__(self, model: str, thinking: int | None):
        self.model, self.thinking = model, thinking
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, parts: list[Path | str], schema: dict) -> str:
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=schema, temperature=0.2,
            thinking_config=None if self.thinking is None else types.ThinkingConfig(thinking_budget=self.thinking),
        )
        contents = [types.Part.from_bytes(data=p.read_bytes(), mime_type="image/jpeg") if isinstance(p, Path) else p for p in parts]
        return self.client.models.generate_content(model=self.model, contents=contents, config=config).text


def _nonzero(v) -> bool:
    if isinstance(v, dict):
        return any(_nonzero(x) for x in v.values())
    if isinstance(v, list):
        return any(_nonzero(x) for x in v)
    try:
        return float(v or 0) != 0
    except ValueError:
        return True


class OpenRouter:
    """Free models only: refuses any model with a non-zero price and caps each request at zero cost."""

    URL = "https://openrouter.ai/api/v1/chat/completions"
    FREE = {"max_price": {"prompt": 0, "completion": 0, "image": 0, "request": 0}}

    def __init__(self, model: str, thinking: int | None):
        listing = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
        if model not in listing:
            raise ValueError(f"unknown OpenRouter model {model!r}")
        paid = {k: v for k, v in listing[model]["pricing"].items() if _nonzero(v)}
        if paid:
            raise ValueError(f"{model} is not free on OpenRouter: {paid}")
        self.model = model
        self.reasoning = None if thinking is None else ({"effort": "low"} if thinking == 0 else {"max_tokens": thinking})
        self.headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

    def __call__(self, parts: list[Path | str], schema: dict) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}}
            if isinstance(p, Path) else {"type": "text", "text": p}
            for p in parts
        ] + [{"type": "text", "text": "Reply with only a JSON object that validates against this JSON Schema:\n" + json.dumps(schema)}]
        body = {
            "model": self.model, "temperature": 0.2, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "provider": self.FREE,
        }
        if self.reasoning is not None:
            body["reasoning"] = self.reasoning
        for attempt in range(5):
            r = httpx.post(self.URL, headers=self.headers, json=body, timeout=300)
            if r.status_code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(5 * 2**attempt)
                continue
            r.raise_for_status()
            reply = r.json()
            if "choices" not in reply or not (reply["choices"][0]["message"].get("content") or "").strip():
                if attempt < 4:
                    time.sleep(5 * 2**attempt)
                    continue
                raise RuntimeError(f"OpenRouter error: {reply.get('error', reply)}")
            text = reply["choices"][0]["message"]["content"]
            return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        raise RuntimeError("OpenRouter: retries exhausted")


class Fallback:
    """Primary model per call; the backup answers only when the primary fails or returns invalid JSON."""

    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup
        self.used = {primary.model: 0, backup.model: 0}

    def __call__(self, parts: list[Path | str], schema: dict) -> str:
        try:
            text = self.primary(parts, schema)
            json.loads(text)
            self.used[self.primary.model] += 1
            return text
        except (RuntimeError, httpx.HTTPError, json.JSONDecodeError):
            self.used[self.backup.model] += 1
            return self.backup(parts, schema)


def model(name: str, thinking: int | None, fallback: str | None = None):
    llm = Gemini(name, thinking) if name.startswith("gemini") else OpenRouter(name, thinking)
    return Fallback(llm, model(fallback, 0)) if fallback else llm


def _ask(llm, parts: list[Path | str], schema: dict, check, retries: int = 2) -> dict:
    doc = json.loads(llm(parts, schema))
    for _ in range(retries):
        errors = check(doc)
        if not errors:
            break
        doc = json.loads(llm(parts + [FIX_PROMPT.format(errors="\n".join(errors), doc=json.dumps(doc))], schema))
    return doc


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


GRASP_ACTIONS = {"pick_place", "lift", "stack", "pour", "fold", "wipe"}
TASK_FRAMES, SEGMENT_FRAMES, TILE = 12, 16, (320, 240)


def _scene_frames(ep: Episode, out: Path, camera: str) -> list[Path]:
    return [frame(ep, camera, i, out / f"{n}.jpg") for n, i in (("first", 0), ("middle", ep.length // 2), ("last", ep.length - 1))]


def _strip(ep: Episode, out: Path, camera: str, n: int, name: str) -> tuple[Path, list[int]]:
    """One grid image of n evenly spaced frames, each labelled with its frame number."""
    idx = uniform(ep, n)
    cols = 4
    grid = Image.new("RGB", (cols * TILE[0], -(-n // cols) * TILE[1]), "black")
    draw = ImageDraw.Draw(grid)
    for k, i in enumerate(idx):
        tile = Image.open(frame(ep, camera, i, out / "frames" / f"{i:06d}.jpg")).convert("RGB").resize(TILE)
        x, y = (k % cols) * TILE[0], (k // cols) * TILE[1]
        grid.paste(tile, (x, y))
        draw.rectangle([x, y, x + 92, y + 18], fill="black")
        draw.text((x + 4, y + 3), f"frame {i}", fill="white")
    path = out / f"{name}.jpg"
    grid.save(path, quality=85)
    return path, idx


def _grid(ep: Episode, out: Path, camera: str, rows: list[list[int]], name: str) -> Path:
    """Rows of labelled frames in one image."""
    cols = max(len(r) for r in rows)
    grid = Image.new("RGB", (cols * TILE[0], len(rows) * TILE[1]), "black")
    draw = ImageDraw.Draw(grid)
    for y, row in enumerate(rows):
        for x, i in enumerate(row):
            grid.paste(Image.open(frame(ep, camera, i, out / "frames" / f"{i:06d}.jpg")).convert("RGB").resize(TILE), (x * TILE[0], y * TILE[1]))
            label = f"frame {i}" if len(rows) == 1 or cols > 3 else f"window {y + 1}: frame {i}"
            draw.rectangle([x * TILE[0], y * TILE[1], x * TILE[0] + 7 * len(label) + 8, y * TILE[1] + 18], fill="black")
            draw.text((x * TILE[0] + 4, y * TILE[1] + 3), label, fill="white")
    path = out / f"{name}.jpg"
    grid.save(path, quality=85)
    return path


def _assign_order(llm, ep: Episode, camera: str, out: Path, task: dict, runs: list[tuple[int, int]]) -> None:
    """Reorder task steps so each gripper hold window gets the object actually carried in it."""
    objects = [s["object"] for s in task["steps"]]
    if len(set(objects)) != len(objects):
        return
    names = {r["role"]: r["name"] for r in task["roles"]}
    rows = [[g, (g + r) // 2, max(g, r - 1)] for g, r in runs]
    schema = {"type": "object", "additionalProperties": False, "required": ["windows"], "properties": {
        "windows": {"type": "array", "minItems": len(runs), "maxItems": len(runs), "items": {"enum": objects}}}}
    check = lambda d: [] if sorted(d["windows"]) == sorted(objects) else [f"each object must appear exactly once: {objects}"]
    text = ASSIGN_PROMPT.format(objects="; ".join(f"{o} ({names.get(o, o)})" for o in objects))
    doc = _ask(llm, [_grid(ep, out, camera, rows, "assign_windows"), text], schema, check)
    if not check(doc):
        by_object = {s["object"]: s for s in task["steps"]}
        task["steps"] = [by_object[o] for o in doc["windows"]]
        pos = {s["id"]: i for i, s in enumerate(task["steps"])}
        task["required_order"] = [o for o in task["required_order"] if pos.get(o["earlier"], -1) < pos.get(o["later"], len(pos))]


def _events(ep: Episode) -> tuple[list[int], list[int]]:
    runs = sorted(set(holds(ep)) | set(cycles(ep)))
    return sorted({a for a, _ in runs}), sorted({b for _, b in runs})


def _iou(a, b) -> float:
    if a is None or b is None:
        return float(a is b)
    ih = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iw = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ih * iw
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _agreement(x: tuple[dict, list], y: tuple[dict, list]) -> float:
    """Lowest box IoU over task roles between two scene drafts."""
    def boxes(d):
        ents = {e["id"]: e["box_2d"] for e in d[0]["entities"]}
        return {b["role"]: ents.get(b["entity"]) for b in d[1]}
    bx, by = boxes(x), boxes(y)
    return min((_iou(bx[r], by.get(r)) for r in bx), default=0.0)


def _task_signature(t: dict) -> tuple:
    kinds = tuple(sorted(r["kind"] for r in t["roles"]))
    role_kind = {r["role"]: r["kind"] for r in t["roles"]}
    goals = tuple(sorted((g["relation"], role_kind.get(g["subject"]), role_kind.get(g["object"]), g["value"]) for g in t["goals"]))
    return kinds, tuple(s["action"] for s in t["steps"]), goals


TASK_HINT = """
Another episode of this dataset was annotated with the task below. Reuse its role names, kinds and naming style for the same objects, but describe THIS episode: add or drop roles and steps to match the objects actually manipulated here, in the order seen.
{task}"""
NOTES_HINT = """
Notes on this dataset's task (from a reviewer; follow them): {notes}"""


def draft_task(llm, ep: Episode, camera: str, out: Path, votes: int = 1, hint: dict | None = None, notes: str | None = None) -> dict:
    strip, _ = _strip(ep, out, camera, TASK_FRAMES, "task_strip")
    closes, opens = _events(ep)
    text = TASK_PROMPT.format(
        instruction=ep.task, n=TASK_FRAMES, episode=ep.index, camera=camera, robot=ep.robot, task_id=_slug(ep.task),
        events=f"gripper closes at frames {closes}, reopens at frames {opens}; the gripper clearly held an object {len(holds(ep))} times, usually one step per hold",
        hint=(TASK_HINT.format(task=json.dumps({k: hint[k] for k in ("roles", "steps", "goals")})) if hint else "")
        + (NOTES_HINT.format(notes=notes) if notes else ""),
    )
    drafts = [_ask(llm, [strip, text], bundle("task"), check_task) for _ in range(votes)]
    drafts = [d for d in drafts if not check_task(d)] or drafts
    sigs = [_task_signature(d) for d in drafts]
    return drafts[max(range(len(drafts)), key=lambda i: sigs.count(sigs[i]))]


AGREE_IOU = 0.5


def draft_scene(llm, ep: Episode, camera: str, out: Path, task: dict, votes: int = 1, notes: str | None = None) -> tuple[dict, list]:
    images = _scene_frames(ep, out, camera) + [_strip(ep, out, camera, TASK_FRAMES, "task_strip")[0]]
    schema = bundle("scene")
    defs = schema.pop("$defs")
    wrapper = {
        "type": "object", "additionalProperties": False, "required": ["scene", "bindings"],
        "properties": {"scene": schema, "bindings": {"$ref": "#/$defs/bindings"}}, "$defs": defs,
    }
    roles = json.dumps(task["roles"])
    text = SCENE_PROMPT.format(
        camera=camera, robot=ep.robot, scene_id=f"episode_{ep.index:03d}", asset=ref(ep, camera, 0), roles=roles,
        goals=json.dumps(task["goals"]),
    ) + (NOTES_HINT.format(notes=notes) if notes else "")
    names = {r["role"]: r["name"] for r in task["roles"]}

    def check(d):
        ents = {e["id"]: e["name"] for e in d["scene"]["entities"]}
        bound = {b["role"]: b["entity"] for b in d["bindings"]}
        start = {(c["relation"], c["subject"], c["object"]): c["value"] for c in d["scene"]["initial_state"]}
        goals = [(g["relation"], bound.get(g["subject"]), g["object"] and bound.get(g["object"]), g["value"]) for g in task["goals"]]
        done = all(start.get(g[:3]) == g[3] for g in goals)
        kinds = {e["id"]: e["kind"] for e in d["scene"]["entities"]}
        role_kind = {r["role"]: r["kind"] for r in task["roles"]}
        return check_scene(d["scene"], "$.scene") + (
            ["$.scene.initial_state: every task goal is already true, but image 1 is before the task; describe image 1 only"] if done else []
        ) + [
            f"$.bindings: entity {b['entity']!r} must have kind {role_kind[b['role']]!r} like its role"
            for b in d["bindings"] if b["role"] in role_kind and kinds.get(b["entity"], role_kind[b["role"]]) != role_kind[b["role"]]
        ] + [
            f"$.bindings: entity {b['entity']!r} must be named {names[b['role']]!r}"
            for b in d["bindings"] if b["role"] in names and ents.get(b["entity"], names[b["role"]]) != names[b["role"]]
        ]

    def one():
        doc = _ask(llm, images + [text], wrapper, check)
        return doc["scene"], doc["bindings"]

    drafts = [one() for _ in range(min(votes, 2))]
    if votes > 1 and _agreement(drafts[0], drafts[1]) < AGREE_IOU:
        drafts.append(one())
        return max(drafts, key=lambda d: sum(_agreement(d, o) for o in drafts if o is not d))
    return drafts[0]


PHRASES = {
    "supported_by": "the {s} rests on the {o}", "inside": "the {s} is inside the {o}", "held_by": "the {s} is held by the {o}",
    "touching": "the {s} touches the {o}", "at": "the {s} is at the {o}", "left_of": "the {s} is left of the {o}",
    "right_of": "the {s} is right of the {o}", "in_front_of": "the {s} is in front of the {o}", "behind": "the {s} is behind the {o}",
    "open": "the {s} is open", "switched_on": "the {s} is switched on", "folded": "the {s} is folded", "upright": "the {s} is upright",
    "clean": "the {s} is clean",
}


def _sentence(relation: str, subject: str, obj: str | None) -> str:
    return PHRASES[relation].format(s=subject, o=obj)


BIND_PROMPT = """Image 1 is the first frame of a robot episode with candidate objects boxed and numbered. Image 2 shows, for each task step in order, the frame where the gripper holds that step's object (labelled window 1, 2, ...).
For each step, give the number of the box in image 1 that is the object the gripper picks up in that step. Steps: {steps}"""


def _numbered(out: Path, scene: dict, ids: list[str]) -> Path:
    img = Image.open(out / "first.jpg").convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    ents = {e["id"]: e for e in scene["entities"]}
    for k, i in enumerate(ids):
        y0, x0, y1, x1 = [v * s / 1000 for v, s in zip(ents[i]["box_2d"], (h, w, h, w))]
        d.rectangle([x0, y0, x1, y1], outline="yellow", width=3)
        d.rectangle([x0, y0, x0 + 22, y0 + 18], fill="yellow")
        d.text((x0 + 6, y0 + 3), str(k + 1), fill="black")
    path = out / "bind_boxes.jpg"
    img.save(path, quality=90)
    return path


def bind_by_grasp(llm, ep: Episode, camera: str, out: Path, task: dict, scene: dict, bindings: list, segs: list[dict]) -> list:
    """Rebind step objects to the boxed entities actually picked up, when several same-kind objects could be confused."""
    step_objects = [s["object"] for s in task["steps"]]
    role_kind = {r["role"]: r["kind"] for r in task["roles"]}
    grasps = {s["step"]: s["grasp"] for s in segs}
    steps = [s for s in task["steps"] if grasps.get(s["id"]) is not None]
    cands = [e["id"] for e in scene["entities"] if e["kind"] == "object" and e["box_2d"]]
    if not steps or len(cands) < 2 or len(set(step_objects)) != len(step_objects) or any(role_kind.get(o) != "object" for o in step_objects):
        return bindings
    grid = _grid(ep, out, camera, [[grasps[s["id"]]] for s in steps], "bind_windows")
    schema = {"type": "object", "additionalProperties": False, "required": ["boxes"], "properties": {
        "boxes": {"type": "array", "minItems": len(steps), "maxItems": len(steps), "items": {"type": "integer", "minimum": 1, "maximum": len(cands)}}}}
    check = lambda d: [] if len(set(d["boxes"])) == len(d["boxes"]) else ["each step picks a different box"]
    text = BIND_PROMPT.format(steps="; ".join(f"{k + 1}. {s['action']} {s['object']}" for k, s in enumerate(steps)))
    doc = _ask(llm, [_numbered(out, scene, cands), grid, text], schema, check)
    if check(doc):
        return bindings
    bound = {b["role"]: b["entity"] for b in bindings}
    names = {e["id"]: e for e in scene["entities"]}
    for s, k in zip(steps, doc["boxes"]):
        new, old = cands[k - 1], bound.get(s["object"])
        if new == old:
            continue
        other = next((r for r, e in bound.items() if e == new), None)
        bound[s["object"]] = new
        if other is not None and old is not None:
            bound[other] = old
        if old is not None:
            names[new]["name"], names[old]["name"] = names[old]["name"], names[new]["name"]
    return [{"role": r, "entity": e} for r, e in bound.items()]


def verify_outcome(llm, ep: Episode, camera: str, out: Path, task: dict, scene: dict, bindings: list, notes: str | None = None) -> tuple[str, list]:
    """success if every goal holds in the last frame, failure if any is false, else unknown."""
    names = {e["id"]: e["name"] for e in scene["entities"]}
    bound = {b["role"]: names.get(b["entity"], b["entity"]) for b in bindings}
    goals = "; ".join(
        f"{i + 1}. {_sentence(g['relation'], bound.get(g['subject'], g['subject']), g['object'] and bound.get(g['object'], g['object']))}"
        for i, g in enumerate(task["goals"])
    )
    schema = {"type": "object", "additionalProperties": False, "required": ["goals", "note"], "properties": {
        "goals": {"type": "array", "minItems": len(task["goals"]), "maxItems": len(task["goals"]), "items": {"type": ["boolean", "null"]}},
        "note": {"type": "string"},
    }}
    check = lambda d: [] if len(d["goals"]) == len(task["goals"]) else [f"need {len(task['goals'])} goal verdicts"]
    text = OUTCOME_PROMPT.format(camera=camera, caption=task["robot_caption"], goals=goals) + (NOTES_HINT.format(notes=notes) if notes else "")
    tail = [max(0, ep.length - 1 - round(s * ep.fps)) for s in (3, 2, 1, 0)]
    doc = _ask(llm, [out / "first.jpg", _grid(ep, out, camera, [tail[:2], tail[2:]], "outcome_tail"), text], schema, check)
    holds = [None if v is None else v == g["value"] for v, g in zip(doc["goals"], task["goals"])]
    outcome = "failure" if False in holds else "success" if all(h is True for h in holds) else "unknown"
    evidence = [{"asset": ref(ep, camera, ep.length - 1), "detail": f"VLM goal check on the last frame: {doc['goals']}; {doc['note']}"[:500]}]
    return outcome, evidence


def _in(runs: list[tuple[int, int]], start: int, end: int) -> tuple[int | None, int | None]:
    inside = [r for r in runs if start <= r[0] < end]
    return (inside[0][0], min(inside[-1][1], end - 1)) if inside else (None, None)


def segment_steps(llm, ep: Episode, camera: str, out: Path, task: dict) -> list[dict]:
    """Gripper holds when every step is a grasp and the counts match; otherwise the VLM places steps on a frame strip."""
    steps = [s["id"] for s in task["steps"]]
    runs = holds(ep)
    if all(s["action"] in GRASP_ACTIONS for s in task["steps"]) and len(runs) == len(steps):
        if len(steps) > 1:
            _assign_order(llm, ep, camera, out, task, runs)
        return segments(ep, [s["id"] for s in task["steps"]])
    strip, _ = _strip(ep, out, camera, SEGMENT_FRAMES, "segment_strip")
    closes, opens = _events(ep)
    listed = "; ".join(f"{s['id']}: {s['action']} {s['object']}" + (f" -> {s['destination']}" if s["destination"] else "") for s in task["steps"])
    schema = {
        "type": "object", "additionalProperties": False, "required": ["segments"],
        "properties": {"segments": {"type": "array", "minItems": len(steps), "maxItems": len(steps), "items": {
            "type": "object", "additionalProperties": False, "required": ["step", "start", "end"],
            "properties": {"step": {"enum": steps}, "start": {"type": "integer", "minimum": 0, "maximum": ep.length - 1},
                           "end": {"type": "integer", "minimum": 1, "maximum": ep.length}},
        }}},
    }

    def check(d):
        segs = d["segments"]
        errors = [] if [x["step"] for x in segs] == steps else [f"segments must list steps in order {steps}"]
        errors += [f"{x['step']}: start {x['start']} must be below end {x['end']}" for x in segs if x["start"] >= x["end"]]
        errors += [f"{b['step']} starts before {a['step']} ends" for a, b in zip(segs, segs[1:]) if b["start"] < a["end"]]
        errors += [f"{x['step']}: outside [0, {ep.length}]" for x in segs if x["start"] < 0 or x["end"] > ep.length]
        return errors

    text = SEGMENT_PROMPT.format(n=SEGMENT_FRAMES, length=ep.length, fps=ep.fps, closes=closes, opens=opens, steps=listed)
    doc = _ask(llm, [strip, text], schema, check)
    if check(doc):
        raise ValueError(f"episode {ep.index}: segmentation failed: {check(doc)}")
    runs = sorted(set(holds(ep)) | set(cycles(ep)))
    out_segs = []
    for x in doc["segments"]:
        grasp, release = _in(runs, x["start"], x["end"])
        out_segs.append({"step": x["step"], "start": x["start"], "end": x["end"], "grasp": grasp, "release": release, "source": "vlm"})
    return out_segs


def assemble(ep: Episode, camera: str, task: dict, scene: dict, bindings: list, hand: str, segs: list[dict]) -> dict:
    return {
        "version": "0.2",
        "id": f"{_slug(ep.repo)}_ep{ep.index:03d}_{_slug(camera.split('.')[-1])}",
        "source": {
            "dataset": ep.repo, "revision": ep.revision, "domain": "real", "robot": ep.robot,
            "episode_index": ep.index, "fps": ep.fps, "frames": {"start": 0, "end": ep.length},
            "camera_key": camera, "task": ep.task, "action_field": "action", "outcome": "unknown",
            "evidence": [{"asset": ref(ep, camera, 0), "detail": f"Task string {ep.task!r}; {ep.length} frames at {ep.fps:g} fps."}],
        },
        "scene": scene,
        "task": task,
        "bindings": bindings,
        "segments": segs,
        "generation": {
            "alignment": {"preserve_layout": True, "preserve_camera": True, "variations": []},
            "active_hand": hand, "idle_hand": "rests flat on the table near the far edge", "entry_edge": "far",
            "appearance": None, "duration_s": min(max(ep.length / ep.fps, 5.0), 15.084), "seed": None,
            "edit_model": None, "video_model": None, "edit_prompt": None, "video_prompt": None,
        },
        "human": None,
        "checks": {
            **{k: {"status": "unknown", "evidence": []} for k in ("reference_preservation", "first_frame_visible", "task_consistency")},
            "semantic": {"score": None, "goals": [], "evidence": []},
            "physics": {"score": None, "evidence": []},
        },
        "accepted": None,
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="Hugging Face repo id, or a local curated dataset directory")
    p.add_argument("--root", type=Path, help="local copy of a Hub dataset; default data/<repo name>")
    p.add_argument("--camera", help="default observation.images.front if present, else observation.images.up")
    p.add_argument("--episodes", type=int, nargs="+", help="default: every episode")
    p.add_argument("--hand", choices=["left", "right"], default="right")
    p.add_argument("--model", default="stealth/space-bunny-alpha", help="gemini-* via Gemini API, anything else via OpenRouter (free models only)")
    p.add_argument("--fallback", default="gemini-3.8-flash", help="answers a call only when --model fails; 'none' disables")
    p.add_argument("--votes", type=int, default=2, help="scene drafts to cross-check boxes (3rd on disagreement); the task gets votes+1")
    p.add_argument("--thinking", type=int, default=0, help="thinking/reasoning token budget; 0 = off, -1 = model default (flash-lite needs -1)")
    p.add_argument("--out", type=Path, help="default <root>/pairs")
    args = p.parse_args(argv)

    if (Path(args.dataset) / "meta/curation.json").exists():
        root = Path(args.dataset)
        repo, revision = local_source(root)
    else:
        repo, root = args.dataset, args.root or Path("data") / args.dataset.split("/")[-1]
        revision = fetch(repo, root)
    info = json.loads((root / "meta/info.json").read_text())
    camera = args.camera or ("observation.images.front" if "observation.images.front" in info["features"] else "observation.images.up")
    out = args.out or root / "pairs"
    llm = model(args.model, None if args.thinking < 0 else args.thinking, None if args.fallback == "none" else args.fallback)

    tasks: dict[str, dict] = {}
    failed = 0
    for i in args.episodes or sorted(int(e) for e in episodes(root).episode_index):
        ep_dir = out / f"episode_{i:03d}"
        if (ep_dir / "pair.json").exists():
            print(f"skip {ep_dir / 'pair.json'}")
            continue
        try:
            failed += annotate_episode(llm, root, repo, revision, i, ep_dir, out, camera, tasks, args)
        except Exception as e:
            failed += 1
            print(f"ERROR {ep_dir}: {type(e).__name__}: {e}", flush=True)
    if isinstance(llm, Fallback):
        print(f"calls answered: {llm.used}")
    return 1 if failed else 0


def annotate_episode(llm, root: Path, repo: str, revision: str, i: int, ep_dir: Path, out: Path, camera: str, tasks: dict, args) -> int:
    ep = load_episode(root, repo, revision, i)
    hint_path = out / "tasks" / f"{_slug(ep.task)}.json"
    if ep.task not in tasks and hint_path.exists():
        tasks[ep.task] = json.loads(hint_path.read_text())
    notes_path = hint_path.with_suffix(".notes.txt")
    task = draft_task(llm, ep, camera, ep_dir, args.votes + 1, hint=tasks.get(ep.task), notes=notes_path.read_text().strip() if notes_path.exists() else None)
    if ep.task not in tasks:
        tasks[ep.task] = task
        hint_path.parent.mkdir(parents=True, exist_ok=True)
        hint_path.write_text(json.dumps(task, indent=2) + "\n")
    notes = notes_path.read_text().strip() if notes_path.exists() else None
    scene, bindings = draft_scene(llm, ep, camera, ep_dir, task, args.votes, notes)
    segs = segment_steps(llm, ep, camera, ep_dir, task)
    bindings = bind_by_grasp(llm, ep, camera, ep_dir, task, scene, bindings, segs)
    pair = assemble(ep, camera, task, scene, bindings, args.hand, segs)
    outcome, evidence = verify_outcome(llm, ep, camera, ep_dir, task, scene, bindings, notes)
    pair["source"]["outcome"] = outcome
    pair["source"]["evidence"] += evidence
    errors = check_pair(pair)
    (ep_dir / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")
    print(f"{'FAIL' if errors else 'ok  '} {ep_dir / 'pair.json'} outcome={outcome}", flush=True)
    for e in errors:
        print(f"     {e}")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())

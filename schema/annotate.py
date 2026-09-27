"""Draft pair records for LeRobot episodes: task and scene from a VLM (Gemini or OpenRouter), segments from the gripper."""

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

from schema.source import Episode, fetch, frame, holds, load_episode, ref, segments
from schema.validate import bundle, check_pair, check_scene, check_task

TASK_PROMPT = """Write the task schema for this robot dataset task.
Dataset task string: {instruction!r}
Images: frames {frames} of episode {episode}, camera {camera}; the robot is a single {robot} arm.
- id: {task_id!r}. version: "0.2". instruction: the dataset string, verbatim.
- robot_caption: imperative sentence naming objects by colour and type, e.g. "Put the red cup on the plate."
- roles: every participant, plus role "robot" (kind actor). name: colour + type, at most 4 words.
- steps: one per grasp-release cycle; {n_steps} observed.
- goals: the end state, including released objects not held_by the robot.
- Use role names everywhere. Only facts the task requires."""

SCENE_PROMPT = """Annotate the scene at the first frame of a robot episode (image 1, camera {camera}).
The robot is a single {robot} arm: include it as entity "robot" (kind actor).
Later frames (images 2-3) show how the episode unfolds; use them only to identify objects the robot hides in image 1.
- scene.id: {scene_id!r}. scene.version: "0.2". scene.view.camera_key: {camera!r}.
- view.shot: camera placement and framing, e.g. "high front view of a white table".
- entities: task objects, visible distractors, the table or surface they rest on, the robot.
- name: colour + type, at most 4 words; an entity bound to a role uses the role's name verbatim. grounding: image position, e.g. "lower left".
- attributes: null when not visible; do not guess.
- initial_state: every goal relation below (true/false; null if occluded), plus supported_by facts.
- evidence asset: {asset!r}.
- bindings: each task role to one entity. Task roles: {roles}
Task goals: {goals}"""

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


def model(name: str, thinking: int | None):
    return Gemini(name, thinking) if name.startswith("gemini") else OpenRouter(name, thinking)


def _ask(llm, parts: list[Path | str], schema: dict, check, retries: int = 1) -> dict:
    doc = json.loads(llm(parts, schema))
    for _ in range(retries):
        errors = check(doc)
        if not errors:
            break
        doc = json.loads(llm(parts + [FIX_PROMPT.format(errors="\n".join(errors), doc=json.dumps(doc))], schema))
    return doc


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _key_frames(ep: Episode, out: Path, camera: str) -> dict[str, int]:
    runs = holds(ep)
    idx = {"first": 0, "grasp": runs[0][0], "release": runs[-1][1], "last": ep.length - 1}
    for name, i in idx.items():
        frame(ep, camera, i, out / f"{name}.jpg")
    return idx


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


def draft_task(llm, ep: Episode, camera: str, out: Path, n_steps: int, votes: int = 1) -> dict:
    idx = _key_frames(ep, out, camera)
    text = TASK_PROMPT.format(
        instruction=ep.task, frames=", ".join(f"{k}={v}" for k, v in idx.items()), episode=ep.index,
        camera=camera, robot=ep.robot, task_id=_slug(ep.task), n_steps=n_steps,
    )
    drafts = [_ask(llm, [out / f"{k}.jpg" for k in idx] + [text], bundle("task"), check_task) for _ in range(votes)]
    drafts = [d for d in drafts if not check_task(d)] or drafts
    sigs = [_task_signature(d) for d in drafts]
    return drafts[max(range(len(drafts)), key=lambda i: sigs.count(sigs[i]))]


AGREE_IOU = 0.5


def draft_scene(llm, ep: Episode, camera: str, out: Path, task: dict, votes: int = 1) -> tuple[dict, list]:
    idx = _key_frames(ep, out, camera)
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
    )
    names = {r["role"]: r["name"] for r in task["roles"]}

    def check(d):
        ents = {e["id"]: e["name"] for e in d["scene"]["entities"]}
        return check_scene(d["scene"], "$.scene") + [
            f"$.bindings: entity {b['entity']!r} must be named {names[b['role']]!r}"
            for b in d["bindings"] if b["role"] in names and ents.get(b["entity"], names[b["role"]]) != names[b["role"]]
        ]

    def one():
        doc = _ask(llm, [out / f"{k}.jpg" for k in ("first", "grasp", "last")] + [text], wrapper, check)
        return doc["scene"], doc["bindings"]

    drafts = [one() for _ in range(min(votes, 2))]
    if votes > 1 and _agreement(drafts[0], drafts[1]) < AGREE_IOU:
        drafts.append(one())
        return max(drafts, key=lambda d: sum(_agreement(d, o) for o in drafts if o is not d))
    return drafts[0]


def assemble(ep: Episode, camera: str, task: dict, scene: dict, bindings: list, hand: str) -> dict:
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
        "segments": segments(ep, [s["id"] for s in task["steps"]]),
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
    p.add_argument("repo")
    p.add_argument("--root", type=Path, help="default data/<repo name>")
    p.add_argument("--camera", default="observation.images.up")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=1, help="grasp-release cycles per episode")
    p.add_argument("--hand", choices=["left", "right"], default="right")
    p.add_argument("--model", default="stealth/space-bunny-alpha", help="gemini-* via Gemini API, anything else via OpenRouter (free models only)")
    p.add_argument("--votes", type=int, default=2, help="scene drafts to cross-check boxes (3rd on disagreement); the task gets votes+1")
    p.add_argument("--thinking", type=int, default=0, help="thinking/reasoning token budget; 0 = off, -1 = model default (flash-lite needs -1)")
    p.add_argument("--out", type=Path, help="default <root>/pairs")
    args = p.parse_args(argv)

    root = args.root or Path("data") / args.repo.split("/")[-1]
    out = args.out or root / "pairs"
    revision = fetch(args.repo, root)
    llm = model(args.model, None if args.thinking < 0 else args.thinking)

    tasks: dict[str, dict] = {}
    failed = 0
    for i in args.episodes:
        ep = load_episode(root, args.repo, revision, i)
        ep_dir = out / f"episode_{i:03d}"
        task_path = out / "tasks" / f"{_slug(ep.task)}.json"
        if ep.task not in tasks:
            if task_path.exists():
                tasks[ep.task] = json.loads(task_path.read_text())
            else:
                tasks[ep.task] = draft_task(llm, ep, args.camera, ep_dir, args.steps, args.votes + 1)
                task_path.parent.mkdir(parents=True, exist_ok=True)
                task_path.write_text(json.dumps(tasks[ep.task], indent=2) + "\n")
        scene, bindings = draft_scene(llm, ep, args.camera, ep_dir, tasks[ep.task], args.votes)
        pair = assemble(ep, args.camera, tasks[ep.task], scene, bindings, args.hand)
        errors = check_pair(pair)
        (ep_dir / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")
        failed += bool(errors)
        print(f"{'FAIL' if errors else 'ok  '} {ep_dir / 'pair.json'}")
        for e in errors:
            print(f"     {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

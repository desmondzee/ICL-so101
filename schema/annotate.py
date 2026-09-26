"""Draft pair records for LeRobot episodes: task and scene from Gemini, segments from the gripper."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

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


def _ask(client: genai.Client, model: str, parts: list, schema: dict, check, thinking: int | None, retries: int = 1) -> dict:
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_json_schema=schema, temperature=0.2,
        thinking_config=None if thinking is None else types.ThinkingConfig(thinking_budget=thinking),
    )
    doc = json.loads(client.models.generate_content(model=model, contents=parts, config=config).text)
    for _ in range(retries):
        errors = check(doc)
        if not errors:
            break
        fix = FIX_PROMPT.format(errors="\n".join(errors), doc=json.dumps(doc))
        doc = json.loads(client.models.generate_content(model=model, contents=parts + [fix], config=config).text)
    return doc


def _image(path: Path) -> types.Part:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _key_frames(ep: Episode, out: Path, camera: str) -> dict[str, int]:
    runs = holds(ep)
    idx = {"first": 0, "grasp": runs[0][0], "release": runs[-1][1], "last": ep.length - 1}
    for name, i in idx.items():
        frame(ep, camera, i, out / f"{name}.jpg")
    return idx


def draft_task(client, model: str, ep: Episode, camera: str, out: Path, n_steps: int, thinking: int | None = None) -> dict:
    idx = _key_frames(ep, out, camera)
    text = TASK_PROMPT.format(
        instruction=ep.task, frames=", ".join(f"{k}={v}" for k, v in idx.items()), episode=ep.index,
        camera=camera, robot=ep.robot, task_id=_slug(ep.task), n_steps=n_steps,
    )
    return _ask(client, model, [_image(out / f"{k}.jpg") for k in idx] + [text], bundle("task"), check_task, thinking)


def draft_scene(client, model: str, ep: Episode, camera: str, out: Path, task: dict, thinking: int | None = None) -> tuple[dict, list]:
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

    doc = _ask(client, model, [_image(out / f"{k}.jpg") for k in ("first", "grasp", "last")] + [text], wrapper, check, thinking)
    return doc["scene"], doc["bindings"]


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
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--thinking", type=int, default=0, help="thinking token budget; -1 = model default (flash-lite needs -1)")
    p.add_argument("--out", type=Path, help="default <root>/pairs")
    args = p.parse_args(argv)

    root = args.root or Path("data") / args.repo.split("/")[-1]
    out = args.out or root / "pairs"
    revision = fetch(args.repo, root)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    thinking = None if args.thinking < 0 else args.thinking

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
                tasks[ep.task] = draft_task(client, args.model, ep, args.camera, ep_dir, args.steps, thinking)
                task_path.parent.mkdir(parents=True, exist_ok=True)
                task_path.write_text(json.dumps(tasks[ep.task], indent=2) + "\n")
        scene, bindings = draft_scene(client, args.model, ep, args.camera, ep_dir, tasks[ep.task], thinking)
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

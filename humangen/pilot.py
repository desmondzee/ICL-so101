"""HumanGen pilot: robot first frame -> human first frame (Nano Banana) -> H3 prompt (Space Bunny) -> H3 clip.

Reads the exported accepted pairs (data/so101_export), picks a few episodes per dataset and writes, per episode,
data/humangen_pilot/h3_reference/<dataset>/episode_XXX/: robot_first.jpg, frame.jpg (edited), prompt.json (VLM reply),
meta.json (H3 request), video.mp4 + contact.png (H3), pair.json (generation and human filled in).
Each stage skips work whose output already exists, so a rerun only does what is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from schema.annotate import model as vlm_model, _ask
from schema.source import frame, load_episode, local_source
from schema.validate import check_pair

EDIT_MODEL = "gemini-3.1-flash-image"
PROMPT_MODEL = "stealth/space-bunny-alpha"
VIDEO_MODEL = "reactor/h3-reference-to-video-turbo-realtime"
VARIATIONS = ["table_surface", "lighting", "object_instance", "background"]
VARIATION_SHARE = 0.25
SPEEDUP = 2.0  # a person does the task about twice as fast as the teleoperated robot
FPS_OUT = 24

VARIATION_TEXT = {
    "table_surface": "Change only the table or mat surface under the objects to a different plausible material and colour.",
    "lighting": "Change only the lighting to a different plausible indoor light (warmer or cooler, softer or harder shadows).",
    "object_instance": "Replace {object} with a different instance of the same kind of object (different colour or brand) of the same size, in the same place.",
    "background": "Change only the background beyond the work surface (walls, furniture, floor) to a different plausible room.",
}

EDIT_PROMPT = """Convert this image into a photorealistic view of the same scene from the same camera, with a person instead of the robot arm.
Remove the robot arm, its base, cables and mount completely. Add a person's two bare forearms and hands reaching in from the {edge_text} of the frame, where the robot stood.
The {active} hand rests open on the work surface near {start_place}, ready to act; the {idle} hand {idle_pose}. Neither hand touches any object.
Keep the camera angle, framing and every object exactly where it is and in the same state (open or closed, upright or lying, inside or outside): {objects}.
{states}{variation}No robot parts, no extra limbs, no duplicated or missing objects, no text, no overlays. Keep the 4:3 aspect ratio."""

PROMPT_INSTRUCTION = """You write the prompt for an image-to-video model (Reactor H3). Picture 1 (attached) is the first frame: a person's hands in a scene.
The video must show the person doing this task, the same way a robot did it in the source episode:
Task: {instruction}
Steps, in order (each is one manipulation): {steps}
Objects: {objects}
End state: {goals}
The {active} hand does every step; the {idle} hand {idle_pose} and does not move. The clip lasts {duration} seconds.

Rules: refer to the image as Picture 1. Describe objects by the colour and shape visible in Picture 1. One continuous shot, the camera stays locked, no cuts, no zoom. No extra hands, no duplicate objects. Objects not involved stay put. After the last step the {active} hand withdraws and everything holds still until the clip ends.
Split the clip into timed shots covering 0 to {duration} seconds, one or two shots per step, with realistic human speed.
Return: subject_definitions (one line for the person, one line per task object, one line for objects that stay put), summary (one or two sentences starting "In {duration} seconds,"), detailed_description (scene, camera and constraints, no timings), shots (list of start, end, text), overall_soundscape (the sounds of contact, then "No dialogue. No music."), task_name (short snake_case), dense_caption (one sentence describing the whole video)."""

PROMPT_SCHEMA = {
    "type": "object",
    "required": ["subject_definitions", "summary", "detailed_description", "shots", "overall_soundscape", "task_name", "dense_caption"],
    "properties": {
        "subject_definitions": {"type": "string"},
        "summary": {"type": "string"},
        "detailed_description": {"type": "string"},
        "shots": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["start", "end", "text"],
                  "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}}}},
        "overall_soundscape": {"type": "string"},
        "task_name": {"type": "string"},
        "dense_caption": {"type": "string"},
    },
}


def _seed(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def select(export: Path, per_dataset: int) -> list[dict]:
    rows = json.loads((export / "accepted.json").read_text())
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    picked = []
    for d, rs in sorted(by.items()):
        picked += random.Random(_seed("pilot", d)).sample(rs, min(per_dataset, len(rs)))
    return picked


def entry_edge(scene: dict) -> str:
    """The image edge nearest the robot: the hands come in where the robot stood."""
    robot = next((e for e in scene["entities"] if e["kind"] == "actor" and e["box_2d"]), None)
    if robot is None:
        return "right"
    y0, x0, y1, x1 = robot["box_2d"]
    gaps = {"far": y0, "near": 1000 - y1, "left": x0, "right": 1000 - x1}
    return min(gaps, key=gaps.get)


EDGE_TEXT = {"far": "top edge", "near": "bottom edge", "left": "left edge", "right": "right edge"}


def plan(pair: dict, key: str) -> dict:
    """generation fields decided in code: hand, entry edge, variation, duration, seed."""
    edge = entry_edge(pair["scene"])
    active = "left" if edge == "left" else "right"
    rng = random.Random(_seed(key))
    variations = [rng.choice(VARIATIONS)] if rng.random() < VARIATION_SHARE else []
    if variations == ["object_instance"] and any(s["action"] in ("open", "close") for s in pair["task"]["steps"]):
        variations = ["table_surface"]  # a swapped drawer or lid tends to come back in the wrong state
    robot_s = (pair["source"]["frames"]["end"] - pair["source"]["frames"]["start"]) / pair["source"]["fps"]
    duration = float(min(max(round(robot_s / SPEEDUP), 5), 15))
    return {"edge": edge, "active": active, "idle": "left" if active == "right" else "right",
            "idle_pose": f"rests on the surface near the {EDGE_TEXT[edge]}, out of the way", "variations": variations,
            "duration": duration, "seed": _seed(key) % 2**31}


def _objects(pair: dict) -> list[dict]:
    return [e for e in pair["scene"]["entities"] if e["kind"] != "actor"]


def edit_prompt(pair: dict, p: dict) -> str:
    objs = _objects(pair)
    bound = {b["entity"] for b in pair["bindings"]}
    task_objs = [e for e in objs if e["id"] in bound and e["kind"] in ("object", "container")] or objs
    first = task_objs[0]
    variation = ""
    if p["variations"]:
        v = p["variations"][0]
        variation = VARIATION_TEXT[v].format(object=f"the {first['name']}") + " Everything else stays identical.\n"
    names = {e["id"]: e["name"] for e in pair["scene"]["entities"]}
    facts = [_sentence(c, names) for c in pair["scene"]["initial_state"]
             if c["value"] is not None and c["relation"] in ("open", "switched_on", "folded", "upright", "inside", "clean")]
    states = ("Keep these states exactly: " + "; ".join(facts) + ".\n") if facts else ""
    return EDIT_PROMPT.format(
        states=states, edge_text=EDGE_TEXT[p["edge"]], active=p["active"], idle=p["idle"], idle_pose=p["idle_pose"],
        start_place=f"the {EDGE_TEXT[p['edge']]}", objects="; ".join(f"the {e['name']} ({e['grounding']})" for e in objs),
        variation=variation)


def edit_image(robot_first: Path, prompt: str, out: Path) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    reply = client.models.generate_content(
        model=EDIT_MODEL,
        contents=[types.Part.from_bytes(data=robot_first.read_bytes(), mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="4:3")),
    )
    for part in reply.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            tmp = out.with_suffix(".raw")
            tmp.write_bytes(part.inline_data.data)
            Image.open(tmp).convert("RGB").save(out, quality=95)
            tmp.unlink()
            return
    raise RuntimeError(f"no image returned: {reply.text if hasattr(reply, 'text') else reply}")


def _sentence(g: dict, names: dict) -> str:
    rel = g["relation"].replace("_", " ")
    s, o = names.get(g["subject"], g["subject"]), names.get(g.get("object") or "", g.get("object"))
    neg = "" if g["value"] else "not "
    return f"the {s} is {neg}{rel}" + (f" the {o}" if o else "")


def video_prompt(llm, frame_jpg: Path, pair: dict, p: dict) -> dict:
    names = {e["id"]: (f"person's {p['active']} hand" if e["kind"] == "actor" else e["name"]) for e in pair["scene"]["entities"]}
    roles = {r["role"]: r for r in pair["task"]["roles"]}
    role_name = lambda r: roles[r]["name"] if r in roles else (r or "")
    steps = "; ".join(f"{i + 1}. {s['action'].replace('_', ' ')} the {role_name(s['object'])}"
                      + (f" to/into/onto the {role_name(s['destination'])}" if s.get("destination") else "")
                      for i, s in enumerate(pair["task"]["steps"]))
    text = PROMPT_INSTRUCTION.format(
        instruction=pair["task"]["instruction"], steps=steps,
        objects="; ".join(f"{e['name']} ({e['grounding']})" for e in _objects(pair)),
        goals="; ".join(_sentence(g, names) for g in pair["task"]["goals"]),
        active=p["active"], idle=p["idle"], idle_pose=p["idle_pose"], duration=p["duration"])

    def check(d):
        shots = d.get("shots") or []
        if not shots:
            return ["shots is empty"]
        if abs(shots[0]["start"]) > 0.01 or any(a["end"] > b["start"] + 0.01 for a, b in zip(shots, shots[1:])) or shots[-1]["end"] > p["duration"] + 0.2:
            return ["shots must start at 0, not overlap and end by the clip duration"]
        return []

    return _ask(llm, [frame_jpg, text], PROMPT_SCHEMA, check)


def _ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def render(v: dict) -> str:
    shots = "\n".join(f"[Shot 1, {_ts(s['start'])}-{_ts(s['end'])}] {s['text'].strip()}" for s in v["shots"])
    return (f"subject_definitions:\n{v['subject_definitions'].strip()}\n\nsummary:\n{v['summary'].strip()}\n\n"
            f"detailed_description:\n{v['detailed_description'].strip()}\n\n{shots}\n\noverall_soundscape:\n{v['overall_soundscape'].strip()}")


def fill_pair(pair: dict, p: dict, edit: str, video: str, v: dict, ep_dir: Path) -> dict:
    pair["generation"] = {
        "alignment": {"preserve_layout": True, "preserve_camera": True, "variations": p["variations"]},
        "active_hand": p["active"], "idle_hand": f"the {p['idle']} hand {p['idle_pose']}", "entry_edge": p["edge"],
        "appearance": None, "duration_s": p["duration"], "seed": p["seed"],
        "edit_model": EDIT_MODEL, "video_model": VIDEO_MODEL, "edit_prompt": edit, "video_prompt": video,
    }
    scene = json.loads(json.dumps(pair["scene"]))
    robot_ids = [e["id"] for e in scene["entities"] if e["kind"] == "actor"]
    hand_id = f"{p['active']}_hand"
    for e in scene["entities"]:
        if e["kind"] == "actor":
            e.update(id=hand_id, name=f"{p['active']} hand", grounding=f"entering from the {EDGE_TEXT[p['edge']]}",
                     box_2d=None, visibility="visible", identity_status="known",
                     attributes={"color": None, "material": None, "shape": "human hand"})
    scene["id"] = scene["id"] + "_human"
    ren = lambda x: hand_id if x in robot_ids else x
    for s in scene["initial_state"]:
        s["subject"], s["object"] = ren(s["subject"]), ren(s.get("object"))
    scene["evidence"] = [{"asset": str(ep_dir / "frame.jpg"), "detail": f"Edited from robot_first.jpg with {EDIT_MODEL}."}]
    pair["human"] = {
        "reference_image": str(ep_dir / "frame.jpg"), "scene": scene,
        "entity_map": [{"source": e["id"], "human": ren(e["id"])} for e in pair["scene"]["entities"]],
        "video": str(ep_dir / "video.mp4") if (ep_dir / "video.mp4").exists() else None,
        "task_name": v["task_name"], "dense_caption": v["dense_caption"],
    }
    return pair


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=Path("data/so101_export"))
    ap.add_argument("--out", type=Path, default=Path("data/humangen_pilot/h3_reference"))
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--stages", default="frame,edit,prompt,video", help="comma-separated subset of frame,edit,prompt,video")
    ap.add_argument("--limit", type=int, help="only the first N selected episodes")
    ap.add_argument("--session", type=int, default=10, help="clips per H3 session")
    args = ap.parse_args(argv)
    stages = set(args.stages.split(","))
    picked = select(args.export, args.per_dataset)[: args.limit]
    llm = vlm_model(PROMPT_MODEL, 0, fallback="gemini-3.8-flash") if "prompt" in stages else None
    requests, done = [], []
    for r in picked:
        key = f"{r['dataset']}/{r['episode_index']}"
        ep_dir = args.out / r["dataset"] / f"episode_{r['episode_index']:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        pair = json.loads((args.export / r["pair"]).read_text())
        p = plan(pair, key)
        try:
            if "frame" in stages and not (ep_dir / "robot_first.jpg").exists():
                root = args.export / r["lerobot"]
                ep = load_episode(root, *local_source(root), r["episode_index"])
                frame(ep, pair["source"]["camera_key"], 0, ep_dir / "robot_first.jpg")
            edit = edit_prompt(pair, p)
            if "edit" in stages and not (ep_dir / "frame.jpg").exists():
                edit_image(ep_dir / "robot_first.jpg", edit, ep_dir / "frame.jpg")
            if "prompt" in stages and not (ep_dir / "prompt.json").exists() and (ep_dir / "frame.jpg").exists():
                (ep_dir / "prompt.json").write_text(json.dumps(video_prompt(llm, ep_dir / "frame.jpg", pair, p), indent=1) + "\n")
            if not (ep_dir / "prompt.json").exists():
                continue
            v = json.loads((ep_dir / "prompt.json").read_text())
            video = render(v)
            (ep_dir / "meta.json").write_text(json.dumps({"prompt": video, "duration": p["duration"], "seed": p["seed"], "frame": "frame.jpg"}, indent=1) + "\n")
            filled = fill_pair(pair, p, edit, video, v, ep_dir)
            errs = check_pair(filled)
            if errs:
                logging.warning("%s: pair does not validate: %s", key, errs[:3])
            (ep_dir / "pair.json").write_text(json.dumps(filled, indent=1) + "\n")
            done.append(ep_dir)
            if not (ep_dir / "video.mp4").exists():
                from humangen.generate import VideoRequest
                requests.append(VideoRequest(prompt=video, reference_image=ep_dir / "frame.jpg", duration=p["duration"], seed=p["seed"], output_path=ep_dir / "video.mp4"))
        except Exception as e:  # one bad episode must not stop the pilot
            logging.error("%s: %s", key, e)
    logging.info("%d episodes ready, %d clips to generate", len(done), len(requests))
    if "video" in stages and requests:
        from humangen.generate import generate_videos_sync
        for i in range(0, len(requests), args.session):  # a session per chunk: one Reactor timeout loses at most a chunk
            chunk = requests[i: i + args.session]
            for attempt in range(2):
                chunk = [r for r in chunk if not Path(r.output_path).exists()]
                if not chunk:
                    break
                try:
                    generate_videos_sync(chunk, aspect="4:3")
                except Exception as e:
                    logging.error("H3 session for clips %d-%d failed (attempt %d): %s", i, i + len(chunk), attempt + 1, e)
        for ep_dir in done:
            pair = json.loads((ep_dir / "pair.json").read_text())
            if pair["human"] and (ep_dir / "video.mp4").exists():
                pair["human"]["video"] = str(ep_dir / "video.mp4")
                (ep_dir / "pair.json").write_text(json.dumps(pair, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

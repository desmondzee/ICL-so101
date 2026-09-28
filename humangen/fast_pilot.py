"""FastH3 pilot with start and end frames, on episodes already run through humangen.pilot.

Per episode (data/humangen_pilot/fast_h3_start_end/<dataset>/episode_XXX/): robot_first.jpg and frame.jpg copied from the H3 pilot,
robot_last.jpg (robot episode last frame), end_frame.jpg (Nano Banana: the human end state, matched to frame.jpg),
prompt.json (Space Bunny, FastH3 prompt style), meta.json, video.mp4 (no audio), contact.png, pair.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from humangen import context, frame_check
from humangen.pilot import EDGE_TEXT, EDIT_MODEL, PROMPT_MODEL, _objects, _sentence, edit_image, edit_prompt, fill_pair, plan
from schema.annotate import _ask, model as vlm_model
from schema.source import frame, load_episode, local_source
from schema.validate import check_pair

FAST_MODEL = "reactor/fast-h3"
FAST_MAX = 14.0

END_PROMPT = """Image 1 is the last frame of a robot doing a task. Image 2 is the first frame of the same scene with a person's hands instead of the robot.
Make the last frame of the person's version: the scene of Image 2 (same camera, framing, table, lighting, background, object appearance, and the same two forearms and hands: skin, sleeves, size), with every object placed and in the state it has in Image 1.
End state to show exactly: {goals}.
The task is done: the {active} hand has let go and rests open on the surface near the {edge}; the {idle} hand {idle_pose}. Remove the robot completely.
No robot parts, no extra limbs, no duplicated or missing objects, no text, no overlays. Keep the 4:3 aspect ratio."""

PROMPT_INSTRUCTION = """You write the prompt for an image-to-video model. It animates from the first attached image (start frame) to the second attached image (end frame). The video must show a person's {active} hand doing this task:
Task: {instruction}
Steps, in order: {steps}
Objects: {objects}
End state: {goals}
The {idle} hand {idle_pose} and does not move. The clip lasts {duration} seconds.

Write present tense, as what the camera sees, in this order: 1) the scene and subjects, fully described (the hands, each task object by the colour and shape visible in the images, the surface); 2) the action, beat by beat with rough timing, at natural human speed, the {active} hand touching each object before it moves it, objects moving only when held or pushed, ending exactly as the end frame; 3) the camera: one locked static shot, no cuts, no zoom; 4) sound: "Silent, no music, no dialogue."
Say that there are only two hands and no other objects appear. Keep it under 250 words; put the key action in the first sentence.
Return: prompt, task_name (short snake_case), dense_caption (one sentence describing the whole video)."""

ONE_START = """Convert this robot image into a photorealistic view of the same scene from the same camera, with one person's {active} hand doing the task instead of the robot.
Remove the robot arm, its base, cables and mount completely. Add exactly one bare {active} forearm and hand entering from the {edge} of the frame, where the robot stood; the {active} hand rests open on the surface near that edge, ready to act, not touching any object. The rest of the person is out of frame: no other hand, arm or body part.
Keep the camera angle, framing, every object, its place and its state exactly as in the image. The scene, from its annotation:
{context}
{variation}No robot parts, no text, no overlays. Keep the 4:3 aspect ratio."""

ONE_END = """Image 1 is the last frame of a robot doing a task. Image 2 is the first frame of the same scene with one person's {active} hand instead of the robot.
Make the last frame of the person's version: the scene of Image 2 (same camera, framing, surfaces, lighting, background, object appearance, and the same single {active} forearm and hand: skin, sleeve, size), with every object placed and in the state it has in Image 1. The task is done: the {active} hand has let go and rests open on the surface near the {edge}, not touching any object; no other hand, arm or body part. Remove the robot completely.
The annotation of the task and its end state:
{context}
No robot parts, no text, no overlays. Keep the 4:3 aspect ratio."""

ONE_PROMPT = """You write the prompt for an image-to-video model that animates from the first attached image (start frame) to the second (end frame). The video shows one person's {active} hand doing the task described by this annotation, with the step timings given:
{context}

Write present tense, as what the camera sees, in this order:
1) The scene: the single {active} forearm and hand entering from the {edge}, where it rests at the start and where it rests at the end; each object by the colour and shape visible in the images and where it is.
2) The action, one sentence per step with its time window from the annotation: the hand reaches the object, grasps or pushes it, the object moves only while the hand holds or pushes it, and it ends where the end frame shows it. Describe each moving object's path from its start place to its end place. Objects not in a step stay where they are.
3) The camera: one locked static shot from the same viewpoint.
4) "Silent."
Only the {active} hand and forearm of one person are ever in view. Under 250 words, the key action in the first sentence.
Return: prompt, task_name (short snake_case), dense_caption (one sentence describing the whole video)."""

PROMPT_SCHEMA = {
    "type": "object", "required": ["prompt", "task_name", "dense_caption"],
    "properties": {"prompt": {"type": "string"}, "task_name": {"type": "string"}, "dense_caption": {"type": "string"}},
}


def _goals(pair: dict, active: str) -> str:
    names = {e["id"]: (f"person's {active} hand" if e["kind"] == "actor" else e["name"]) for e in pair["scene"]["entities"]}
    return "; ".join(_sentence(g, names) for g in pair["task"]["goals"])


def edit_end(robot_last: Path, first: Path, prompt: str, out: Path) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    reply = client.models.generate_content(
        model=EDIT_MODEL,
        contents=[types.Part.from_bytes(data=robot_last.read_bytes(), mime_type="image/jpeg"),
                  types.Part.from_bytes(data=first.read_bytes(), mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="4:3")),
    )
    for part in reply.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            tmp = out.with_suffix(".raw")
            tmp.write_bytes(part.inline_data.data)
            Image.open(tmp).convert("RGB").save(out, quality=95)
            tmp.unlink()
            return
    raise RuntimeError("no image returned")


def fast_prompt(llm, first: Path, end: Path, pair: dict, p: dict) -> dict:
    roles = {r["role"]: r["name"] for r in pair["task"]["roles"]}
    steps = "; ".join(f"{i + 1}. {s['action'].replace('_', ' ')} the {roles.get(s['object'], s['object'])}"
                      + (f" to/into/onto the {roles.get(s['destination'], s['destination'])}" if s.get("destination") else "")
                      for i, s in enumerate(pair["task"]["steps"]))
    text = PROMPT_INSTRUCTION.format(
        active=p["active"], idle=p["idle"], idle_pose=p["idle_pose"], duration=p["duration"], instruction=pair["task"]["instruction"],
        steps=steps, objects="; ".join(f"{e['name']} ({e['grounding']})" for e in _objects(pair)), goals=_goals(pair, p["active"]))
    return _ask(llm, [first, end, text], PROMPT_SCHEMA, lambda d: [] if len(d.get("prompt", "").split()) <= 300 else ["prompt is over 300 words"])


def gate(checker, pair: dict, p: dict, d: Path, end_prompt: str, retries: int, start_prompt: str | None = None) -> bool:
    """Check start and end frames; redo a failed edit with the check's issues, up to `retries` times. Writes frame_check.json."""
    history = []
    for attempt in range(retries + 1):
        r = frame_check.check(checker, pair, p, d / "robot_first.jpg", d / "frame.jpg", d / "robot_last.jpg", d / "end_frame.jpg",
                              hands=1 if start_prompt else 2, context=p.get("context"))
        ok_start, ok_end = frame_check.passed(r["start"]), frame_check.passed(r["end"])
        history.append({"attempt": attempt, "start_ok": ok_start, "end_ok": ok_end, "result": r})
        if (ok_start and ok_end) or attempt == retries:
            break
        if not ok_start:  # a new start frame means a new end frame too
            fix = "\nFix these problems of the previous attempt: " + "; ".join(r["start"]["issues"])
            edit_image(d / "robot_first.jpg", (start_prompt or edit_prompt(pair, p)) + fix, d / "frame.jpg")
            edit_end(d / "robot_last.jpg", d / "frame.jpg", end_prompt, d / "end_frame.jpg")
        else:
            fix = "\nFix these problems of the previous attempt: " + "; ".join((r["end"] or {}).get("issues", []))
            edit_end(d / "robot_last.jpg", d / "frame.jpg", end_prompt + fix, d / "end_frame.jpg")
    ok = history[-1]["start_ok"] and history[-1]["end_ok"]
    (d / "frame_check.json").write_text(json.dumps({"passed": ok, "attempts": len(history), "history": history}, indent=1) + "\n")
    return ok


def one_prompt(llm, first: Path, end: Path, p: dict) -> dict:
    text = ONE_PROMPT.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=p["context"])
    return _ask(llm, [first, end, text], PROMPT_SCHEMA, lambda d: [] if len(d.get("prompt", "").split()) <= 300 else ["prompt is over 300 words"])


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="<dataset>/episode_XXX under the H3 pilot folder")
    ap.add_argument("--export", type=Path, default=Path("data/so101_export"))
    ap.add_argument("--pilot", type=Path, default=Path("data/humangen_pilot/h3_reference"))
    ap.add_argument("--out", type=Path, default=Path("data/humangen_pilot/fast_h3_start_end"))
    ap.add_argument("--stages", default="edit,prompt,video")
    ap.add_argument("--no-end", action="store_true", help="start frame only")
    ap.add_argument("--check", action="store_true", help="VLM-check start and end frames, redo failed edits, skip episodes that still fail")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed-frames", type=Path, help="start from the frames of an earlier run (e.g. the unchecked ablation)")
    ap.add_argument("--one-hand", action="store_true", help="only the acting hand in frames and video; timing and all prompts from the schema context")
    args = ap.parse_args(argv)
    stages = set(args.stages.split(","))
    llm = vlm_model(PROMPT_MODEL, 0, fallback="gemini-3.8-flash") if "prompt" in stages else None
    checker = vlm_model(PROMPT_MODEL, 0, fallback="gemini-3.8-flash") if args.check else None
    from humangen.generate import VideoRequest, generate_videos_sync

    requests = []
    for rel in args.episodes:
        src, d = args.pilot / rel, args.out / rel
        d.mkdir(parents=True, exist_ok=True)
        pair = json.loads((src / "pair.json").read_text())
        dataset, idx = rel.split("/")[0], pair["source"]["episode_index"]
        p = plan(pair, f"{dataset}/{idx}")
        p["duration"] = min(p["duration"], FAST_MAX)
        if args.one_hand:
            p["duration"], p["windows"] = context.timing(pair)
            p["idle_pose"] = "is out of frame"
            p["context"] = context.render(pair, f"person's {p['active']} hand", p["duration"], p["windows"])
        try:
            for f in ("robot_first.jpg",) if args.one_hand else ("robot_first.jpg", "frame.jpg"):
                if not (d / f).exists():
                    shutil.copy2(src / f, d / f)
            if not (d / "robot_last.jpg").exists():
                root = args.export / "lerobot" / dataset
                ep = load_episode(root, *local_source(root), idx)
                frame(ep, pair["source"]["camera_key"], ep.length - 1, d / "robot_last.jpg")
            if args.one_hand:
                variation = ""
                if p["variations"]:
                    from humangen.pilot import VARIATION_TEXT
                    first = next((e for e in pair["scene"]["entities"] if e["kind"] in ("object", "container")), None)
                    variation = VARIATION_TEXT[p["variations"][0]].format(object=f"the {first['name'] if first else 'main object'}") + " Everything else stays identical.\n"
                start_prompt = ONE_START.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=p["context"], variation=variation)
                if "edit" in stages and not (d / "frame.jpg").exists():
                    edit_image(d / "robot_first.jpg", start_prompt, d / "frame.jpg")
            end_prompt = ONE_END.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=p["context"]) if args.one_hand else END_PROMPT.format(goals=_goals(pair, p["active"]), active=p["active"], idle=p["idle"],
                                           idle_pose=p["idle_pose"], edge=EDGE_TEXT[p["edge"]])
            if args.seed_frames and not (d / "end_frame.jpg").exists() and (args.seed_frames / rel / "end_frame.jpg").exists():
                shutil.copy2(args.seed_frames / rel / "frame.jpg", d / "frame.jpg")  # reuse the unchecked run's frames; the check decides
                shutil.copy2(args.seed_frames / rel / "end_frame.jpg", d / "end_frame.jpg")
            if "edit" in stages and not args.no_end and not (d / "end_frame.jpg").exists():
                edit_end(d / "robot_last.jpg", d / "frame.jpg", end_prompt, d / "end_frame.jpg")
            if args.check and "edit" in stages and not (d / "frame_check.json").exists():
                if not gate(checker, pair, p, d, end_prompt, args.retries, start_prompt if args.one_hand else None):
                    logging.warning("%s: frames still fail the check after %d retries, skipped", rel, args.retries)
                    continue
            if args.check and not json.loads((d / "frame_check.json").read_text())["passed"]:
                continue
            if "prompt" in stages and not (d / "prompt.json").exists():
                end = d / ("frame.jpg" if args.no_end else "end_frame.jpg")
                v = one_prompt(llm, d / "frame.jpg", end, p) if args.one_hand else fast_prompt(llm, d / "frame.jpg", end, pair, p)
                (d / "prompt.json").write_text(json.dumps(v, indent=1) + "\n")
            if not (d / "prompt.json").exists():
                continue
            v = json.loads((d / "prompt.json").read_text())
            (d / "meta.json").write_text(json.dumps({"prompt": v["prompt"], "duration": p["duration"], "seed": p["seed"],
                                                     "frame": "frame.jpg", "end_frame": None if args.no_end else "end_frame.jpg"}, indent=1) + "\n")
            start_text = start_prompt if args.one_hand else pair["generation"]["edit_prompt"]
            filled = fill_pair(pair, p, start_text + "\n\nEnd frame:\n" + end_prompt, v["prompt"],
                               {"task_name": v["task_name"], "dense_caption": v["dense_caption"]}, d)
            filled["generation"]["video_model"] = FAST_MODEL
            if check_pair(filled):
                logging.warning("%s: pair does not validate: %s", rel, check_pair(filled)[:3])
            (d / "pair.json").write_text(json.dumps(filled, indent=1) + "\n")
            if not (d / "video.mp4").exists():
                requests.append(VideoRequest(prompt=v["prompt"], reference_image=d / "frame.jpg", duration=p["duration"], seed=p["seed"],
                                             output_path=d / "video.mp4", ending_image=None if args.no_end else d / "end_frame.jpg"))
        except Exception as e:
            logging.error("%s: %s", rel, e)
    logging.info("%d clips to generate", len(requests))
    if "video" in stages:
        for i in range(0, len(requests), 10):
            chunk = requests[i: i + 10]
            for attempt in range(2):
                chunk = [r for r in chunk if not Path(r.output_path).exists()]
                if not chunk:
                    break
                try:
                    generate_videos_sync(chunk, aspect="4:3", model=FAST_MODEL)
                except Exception as e:
                    logging.error("FastH3 session %d failed (attempt %d): %s", i, attempt + 1, e)
        for r in requests:
            v = Path(r.output_path)
            if v.exists():  # drop the audio track
                tmp = v.with_name("video_na.mp4")
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(v), "-an", "-c:v", "copy", str(tmp)], check=True)
                tmp.replace(v)
                pair = json.loads((v.parent / "pair.json").read_text())
                pair["human"]["video"] = str(v)
                (v.parent / "pair.json").write_text(json.dumps(pair, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""FastH3 with a keyframe per step: one hand, schema context everywhere, gated frames, two seeds per episode.

Per episode (under --out/<dataset>/episode_XXX/):
  robot_first.jpg, robot_kf1..N.jpg   robot frames at the start and at the end of each step (segment ends)
  frame.jpg, kf1..N.jpg               edited human frames; kf_i copies surfaces and hand from the previous edited frame
  frame_check.json                    Space Bunny gate per frame (start, then each keyframe), with redos
  prompts.json                        Space Bunny FastH3 prompt per step
  cand_s<seed>/video.mp4              the step clips of one seed joined (no audio), judged by humangen/judge_workflow.js
  video.mp4                           the better candidate, copied after judging (--pick)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from humangen import context, edit as scene_edit, frame_check
from humangen.fast_pilot import PROMPT_SCHEMA, edit_end
from humangen.pilot import EDGE_TEXT, PROMPT_MODEL, VARIATION_TEXT, edit_image, fill_pair, plan
from schema.annotate import _ask, model as vlm_model
from schema.source import frame, load_episode, local_source
from schema.validate import check_pair

FAST_MODEL = "reactor/fast-h3"
MIN_STEP_S = 3.0  # human seconds per manipulation at least
CLIP_MIN_S, CLIP_MAX_S = 5.0, 14.0

START = """Convert this robot image into a photorealistic view of the same scene from the same camera, with one person's {active} hand instead of the robot.
Remove the robot arm, its base, cables and mount completely. Add exactly one bare {active} forearm and hand entering from the {edge} of the frame, where the robot stood; the hand rests open on the surface near that edge, not touching any object. The rest of the person is out of frame: no other hand, arm or body part.
Keep the camera angle, framing, every object, its place and its state exactly as in the image. The scene, from its annotation:
{context}
{variation}Keep any black bars at the image edges as in the robot image. No robot parts, no text, no overlays. Keep the 4:3 aspect ratio."""

KEY = """Image 1 is a robot's frame right after step {i} of a task. Image 2 is the person's version of the frame before that step (one {active} hand instead of the robot).
Make the person's version of Image 1: copy Image 2's camera, framing, table surface, lighting, background, object appearance and the single {active} forearm and hand (skin, sleeve, size) exactly, even where they differ from Image 1 on purpose. Place every object where, and in the state, Image 1 shows it. The hand has just let go and rests open near the {edge}, not touching any object; no other hand, arm or body part. Remove the robot completely.
State after step {i}: {facts}.
The objects in the scene:
{context}
Keep any black bars at the image edges as in the robot image. No robot parts, no text, no overlays. Keep the 4:3 aspect ratio."""

START_SINGLE = """Replace the robot arm in this image with one person's bare {active} forearm and hand, entering from the {edge} of the frame where the robot stood. The hand rests open on the surface near that edge, not touching any object. The rest of the person is out of frame: no other hand, arm or body part.
Change nothing else: every object, its place and state, the surfaces, background, lighting, camera and framing stay exactly as in the image. Remove every robot part (arm, gripper, base, cables, mount). The objects in the scene:
{context}
No text, no overlays."""

KEY_SINGLE = """Image 1 is a frame of a scene with a robot arm. Image 2 shows only a person's {active} forearm and hand (a crop, not the scene).
In Image 1, replace the robot arm with that same {active} forearm and hand (same skin, sleeve, size and look as in Image 2), entering from the {edge}. The hand has just let go and rests open near the {edge}, not touching any object; no other hand, arm or body part.
Change nothing else in Image 1: every object stays exactly where and as it is in Image 1, with the same surfaces, background, lighting, camera and framing. Remove every robot part (arm, gripper, base, cables, mount). The objects in the scene:
{context}
No text, no overlays."""

REMOVE = """Remove the robot arm completely: every part of it (arm, gripper, base, cables, mount) and its shadow, and fill in what is behind it (table, background) so it looks as if the robot was never there. Add nothing: no hands, people or objects. Change nothing else: every object, the surfaces, lighting, camera and framing stay exactly as they are{states}."""

START_SHORT = """Remove the robot arm and every part of it (gripper, base, cables, mount). Add one person's bare {active} forearm and hand entering from the {edge}, where the robot was, resting palm down with relaxed fingers on the surface next to the {near}, touching nothing; no other hand or arm. Keep everything else exactly as it is: {keep}{states}. Same camera and framing."""

KEY_SHORT = """Image 1 is the scene. Image 2 shows only a person's {active} forearm and hand. In Image 1, replace the robot arm and every part of it (gripper, base, cables, mount) with that same forearm and hand (same skin, sleeve and size), entering from the {edge}, resting palm down with relaxed fingers on the surface next to the {near}, touching nothing; no other hand or arm. Keep everything else in Image 1 exactly as it is: {keep}{states}. Same camera and framing."""

STEP_PROMPT = """You write the prompt for an image-to-video model that animates from the first attached image to the second. The clip shows one person's {active} hand doing only this step of a task:
Step {i} of {n}: {step}. Clip length {duration}s.
State after the step: {facts}.
The whole task, for context:
{context}

Write present tense, as what the camera sees: 1) the scene: the single {active} forearm and hand entering from the {edge}, where it rests at the start and at the end; each object by the colour and shape visible in the images and where it is; 2) the action with rough timing: the hand reaches the object, grasps or pushes it, the object moves only while the hand holds or pushes it, along its path from its start place to where the second image shows it, then the hand lets go and rests; objects not in this step stay where they are; 3) one locked static shot from the same viewpoint; 4) "Silent."
Only one person's {active} hand and forearm are ever in view. Under 200 words, the key action in the first sentence.
Return: prompt, task_name (short snake_case for the whole task), dense_caption (one sentence describing this step)."""


def _facts(conds: list[dict], names: dict) -> str:
    return "; ".join(context._fact(c, names) for c in conds) or "unchanged"


def end_only_plan(pair: dict) -> list[dict]:
    """One 'step' from the first to the last frame: the end state is the goals, the clip covers the whole task."""
    steps = step_plan(pair)
    last = dict(steps[-1])
    last["after"] = pair["task"]["goals"]
    last["duration"] = float(min(max(6.0, 7.0 * len(steps)), CLIP_MAX_S))
    last["all_steps"] = [s["step"] for s in steps]
    return [last]


def step_plan(pair: dict) -> list[dict]:
    """Per step: robot keyframe index, the state after it (cumulative), and the human clip length."""
    fps, n = pair["source"]["fps"], pair["source"]["frames"]["end"]
    segs = {s["step"]: s for s in pair["segments"]}
    after: dict[tuple, dict] = {}
    out = []
    steps = pair["task"]["steps"]
    for i, st in enumerate(steps):
        seg = segs[st["id"]]
        for c in st["after"]:
            after[(c["relation"], c["subject"], c.get("object"))] = c
        robot_s = ((seg["release"] if seg["release"] is not None else seg["end"]) - (seg["grasp"] if seg["grasp"] is not None else seg["start"])) / fps
        human = max(MIN_STEP_S, robot_s / context.SPEEDUP) + 1.5  # reach and settle around the manipulation
        out.append({"step": st, "kf_index": (n - 1) if i == len(steps) - 1 else min(max(seg["end"] - 1, 0), n - 1),
                    "after": list(after.values()), "duration": round(min(max(human, CLIP_MIN_S), CLIP_MAX_S), 1)})
    return out


CHECK = frame_check.check  # set to frame_check.check3 by --checks 3
HANDS = 1  # hands expected in the edited frames (0 with --no-hand)


def gate(checker, pair, p, pairs_of_frames, edit, retries: int, log: list) -> bool:
    """pairs_of_frames: (robot_prev, human_prev, robot_now, human_now, end_facts or None). edit(issues) redoes human_now."""
    rp, hp, rn, hn, facts = pairs_of_frames
    for attempt in range(retries + 1):
        if not hn.exists():  # the previous edit was rejected as reframed before it was composited
            log.append({"frame": hn.name, "attempt": attempt, "ok": False, "result": {"issues": ["the edit changed the whole scene (reframed)"]}})
            if attempt < retries:
                _safe(edit, "\nChange only the robot arm region; keep the camera and every other pixel of the scene as it is.", attempt + 1)
            continue
        if facts is None:
            r = CHECK(checker, pair, p, rp, hp, hands=HANDS, context=p["scene_context"])
            ok, part = frame_check.passed(r["start"]), r["start"]
        else:
            r = CHECK(checker, pair, p, rp, hp, rn, hn, hands=HANDS, context=p["scene_context"], end_facts=facts)
            ok, part = frame_check.passed(r["end"]), r["end"] or {}
        log.append({"frame": hn.name, "attempt": attempt, "ok": ok, "result": part})
        (hn.parent / "frame_check_partial.json").write_text(json.dumps(log, indent=1, default=str) + "\n")  # survives a later error
        if ok:
            return True
        if attempt < retries:
            _safe(edit, "\nFix these problems of the previous attempt: " + "; ".join(frame_check.blocking_issues(part)), attempt + 1)
    return False


def _safe(edit, fix: str, attempt: int = 0) -> None:
    """Run one edit; a reframed result is removed so the gate counts it as a failed attempt."""
    try:
        edit(fix, attempt)
    except scene_edit.Reframed as e:
        logging.info("edit rejected: %s", e)


def prepare(args, llm, checker, rel: str) -> dict | None:
    src, d = args.pilot / rel, args.out / rel
    d.mkdir(parents=True, exist_ok=True)
    pair = json.loads((src / "pair.json").read_text())
    dataset, idx = rel.split("/")[0], pair["source"]["episode_index"]
    p = plan(pair, f"{dataset}/{idx}")
    p["idle_pose"] = "is out of frame"
    if args.composite:
        p["variations"] = []  # composited frames keep the robot frame's own pixels outside the robot
    hand = f"person's {p['active']} hand"
    names = context.names_for(pair, hand)
    steps = end_only_plan(pair) if args.end_only else step_plan(pair)
    p["duration"] = min(round(sum(s["duration"] for s in steps), 1), 15.084)  # schema caps one clip; the step lengths are in video_prompt
    p["context"] = context.render(pair, hand)
    scene_only = context.render(pair, hand, task=False)
    p["scene_context"] = scene_only
    objects_only = "\n".join(l for l in scene_only.splitlines() if not l.startswith("First-frame state"))
    variation = ""
    if p["variations"]:
        first = next((e for e in pair["scene"]["entities"] if e["kind"] in ("object", "container")), None)
        variation = VARIATION_TEXT[p["variations"][0]].format(object=f"the {first['name'] if first else 'main object'}") + " Everything else stays identical.\n"
    root = args.export / "lerobot" / dataset
    ep = load_episode(root, *local_source(root), idx)
    cam = pair["source"]["camera_key"]
    if not (d / "robot_first.jpg").exists():
        frame(ep, cam, 0, d / "robot_first.jpg")
    for i, s in enumerate(steps, 1):
        if not (d / f"robot_kf{i}.jpg").exists():
            frame(ep, cam, s["kf_index"], d / f"robot_kf{i}.jpg")
    if args.short_prompts:
        keep = context.keep_list(pair)
        st0 = context.risky_states(pair["scene"]["initial_state"], names)
        first_step = (steps[0].get("all_steps") or [steps[0]["step"]])[0]
        start_prompt = START_SHORT.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], keep=keep, states=f"; {st0}" if st0 else "",
                                          near=names.get(first_step["object"], first_step["object"]))
        key_prompts = []
        for s in steps:
            st = context.risky_states(s["after"], names)
            last = s["step"]
            near = names.get(last.get("destination") or last["object"], last.get("destination") or last["object"])
            key_prompts.append(KEY_SHORT.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], keep=keep, states=f"; {st}" if st else "", near=near))
    elif args.composite:
        start_prompt = START_SINGLE.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=scene_only)
        key_prompts = [KEY_SINGLE.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=objects_only) for _ in steps]
    else:
        start_prompt = START.format(active=p["active"], edge=EDGE_TEXT[p["edge"]], context=scene_only, variation=variation)
        key_prompts = [KEY.format(i=i, active=p["active"], edge=EDGE_TEXT[p["edge"]], facts=_facts(s["after"], names), context=objects_only)
                       for i, s in enumerate(steps, 1)]

    def retry_reframed(fn, *a):  # a reframed edit is a failed attempt like any other; the gate decides whether to redo
        return fn(*a)

    def editor_for(attempt: int) -> str:
        return args.fallback_editor if args.fallback_editor and attempt >= args.retries else args.editor

    def remove(robot: Path, out: Path, fix: str, attempt: int, states: str) -> None:
        """No-hand frames: attempt 0 SAM + LaMa (free), then the image editors on the robot frame alone, composited."""
        out.unlink(missing_ok=True)
        if attempt == 0 or args.lama_only:  # free and local; each redo widens the mask to catch robot edges and cables
            return scene_edit.remove_robot(robot, boxes[robot.name], out, dilate=16 + 12 * attempt)
        raw = out.with_name(out.stem + "_raw.jpg")
        scene_edit.edit([robot], REMOVE.format(states=states) + fix, raw, model=args.editor if attempt < args.retries else (args.fallback_editor or args.editor), size=args.size)
        scene_edit.composite(robot, raw, boxes[robot.name], p["edge"], out, objboxes[robot.name])

    def make_start(fix: str = "", attempt: int = 0) -> None:
        if args.no_hand:
            st = context.risky_states(pair["scene"]["initial_state"], names)
            return remove(d / "robot_first.jpg", d / "frame.jpg", fix, attempt, f"; {st}" if st else "")
        if not args.composite:
            return edit_image(d / "robot_first.jpg", start_prompt + fix, d / "frame.jpg")
        def once():
            (d / "frame.jpg").unlink(missing_ok=True)  # a rejected (reframed) edit leaves no frame behind
            scene_edit.edit([d / "robot_first.jpg"], start_prompt + fix, d / "frame_raw.jpg", model=editor_for(attempt), size=args.size)
            scene_edit.composite(d / "robot_first.jpg", d / "frame_raw.jpg", boxes["robot_first.jpg"], p["edge"], d / "frame.jpg", objboxes["robot_first.jpg"])
        retry_reframed(once)

    def make_key(i: int, prev_h: Path, fix: str = "", attempt: int = 0) -> None:
        if args.no_hand:
            st = context.risky_states(steps[i - 1]["after"], names)
            return remove(d / f"robot_kf{i}.jpg", d / f"kf{i}.jpg", fix, attempt, f"; {st}" if st else "")
        rk, hk = d / f"robot_kf{i}.jpg", d / f"kf{i}.jpg"
        if not args.composite:
            return edit_end(rk, prev_h, key_prompts[i - 1] + fix, hk)
        if not (d / "hand_ref.jpg").exists():
            scene_edit.hand_crop(d / "frame.jpg", boxes["robot_first.jpg"], p["edge"], d / "hand_ref.jpg")
        def once():
            hk.unlink(missing_ok=True)
            scene_edit.edit([rk, d / "hand_ref.jpg"], key_prompts[i - 1] + fix, d / f"kf{i}_raw.jpg", model=editor_for(attempt), size=args.size)
            scene_edit.composite(rk, d / f"kf{i}_raw.jpg", boxes[rk.name], p["edge"], hk, objboxes[rk.name])
        retry_reframed(once)

    boxes: dict[str, list[int]] = {}
    objboxes: dict[str, list[list[int]]] = {}
    if args.composite:
        bf = d / "robot_boxes.json"
        saved = json.loads(bf.read_text()) if bf.exists() else {}
        boxes, objboxes = saved.get("robot", {}), saved.get("objects", {})
        obj_names = [e["name"] for e in pair["scene"]["entities"] if e["kind"] in ("object", "container")]
        for f in ["robot_first.jpg"] + [f"robot_kf{i}.jpg" for i in range(1, len(steps) + 1)]:
            if f not in boxes:
                boxes[f] = scene_edit.robot_box(llm, d / f)
            if f not in objboxes:
                objboxes[f] = scene_edit.object_boxes(llm, d / f, obj_names)
        bf.write_text(json.dumps({"robot": boxes, "objects": objboxes}, indent=1) + "\n")
    if not (d / "frame_check.json").exists():
        log: list = []
        ok = True
        if not (d / "frame.jpg").exists():
            _safe(lambda fix, a: make_start(fix, a), "")
        ok = gate(checker, pair, p, (d / "robot_first.jpg", d / "frame.jpg", None, d / "frame.jpg", None), make_start, args.retries, log)
        prev_r, prev_h = d / "robot_first.jpg", d / "frame.jpg"
        for i, s in enumerate(steps, 1):
            if not ok:
                break
            rk, hk = d / f"robot_kf{i}.jpg", d / f"kf{i}.jpg"
            _safe(lambda fix, a, i=i, ph=prev_h: make_key(i, ph, fix, a), "")
            ok = gate(checker, pair, p, (prev_r, prev_h, rk, hk, _facts(s["after"], names)),
                      lambda fix, a, i=i, ph=prev_h: make_key(i, ph, fix, a), args.retries, log)
            prev_r, prev_h = rk, hk
        (d / "frame_check.json").write_text(json.dumps({"passed": ok, "log": log}, indent=1) + "\n")
    if not json.loads((d / "frame_check.json").read_text())["passed"]:
        logging.warning("%s: frames fail the check, skipped", rel)
        return None
    if args.short_prompts:  # template prompts are cheap and deterministic: always rebuild them from the current code
        prompts = [{"prompt": (context.task_prompt(pair, p["active"], EDGE_TEXT[p["edge"]], s["duration"], enters=args.no_hand) if args.end_only
                               else context.step_prompt(pair, s["step"], p["active"], EDGE_TEXT[p["edge"]], s["duration"])),
                    "task_name": context.names_for(pair, hand) and pair["task"]["id"], "dense_caption": f"The person's {p['active']} hand does step {i}: {s['step']['action'].replace('_', ' ')}."}
                   for i, s in enumerate(steps, 1)]
        (d / "prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")
    if not (d / "prompts.json").exists():
        prompts = []
        for i, s in enumerate(steps, 1):
            st = s["step"]
            what = f"{st['action'].replace('_', ' ')} the {names.get(st['object'], st['object'])}" + (f" to the {names.get(st['destination'], st['destination'])}" if st.get("destination") else "")
            text = STEP_PROMPT.format(active=p["active"], i=i, n=len(steps), step=what, duration=s["duration"], facts=_facts(s["after"], names),
                                      context=p["context"], edge=EDGE_TEXT[p["edge"]])
            prev = d / ("frame.jpg" if i == 1 else f"kf{i - 1}.jpg")
            prompts.append(_ask(llm, [prev, d / f"kf{i}.jpg", text], PROMPT_SCHEMA,
                                lambda x: [] if len(x.get("prompt", "").split()) <= 250 else ["prompt is over 250 words"]))
        (d / "prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")
    prompts = json.loads((d / "prompts.json").read_text())
    video_prompt = "\n\n".join(f"[Step {i}, {s['duration']}s]\n{v['prompt']}" for i, (s, v) in enumerate(zip(steps, prompts), 1))
    edit_text = start_prompt + "".join(f"\n\nKeyframe {i}:\n{k}" for i, k in enumerate(key_prompts, 1))
    filled = fill_pair(pair, p, edit_text, video_prompt, {"task_name": prompts[0]["task_name"], "dense_caption": " ".join(v["dense_caption"] for v in prompts)}, d)
    filled["generation"]["video_model"] = FAST_MODEL
    if check_pair(filled):
        logging.warning("%s: pair does not validate: %s", rel, check_pair(filled)[:3])
    (d / "pair.json").write_text(json.dumps(filled, indent=1) + "\n")
    return {"dir": d, "steps": steps, "prompts": prompts, "seed": p["seed"]}


def trim_black(video: Path) -> None:
    """Drop trailing near-black frames the recorder sometimes leaves at the end of a clip."""
    import numpy as np
    from PIL import Image

    tmp = video.with_name(video.stem + "_last.png")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout or 0)
    cut = dur
    for back in (0.04, 0.08, 0.13, 0.17, 0.21, 0.25):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{max(0, dur - back):.3f}", "-i", str(video), "-frames:v", "1", str(tmp)], check=False)
        if tmp.exists() and np.asarray(Image.open(tmp).convert("L")).mean() < 8:
            cut = dur - back
        else:
            break
    tmp.unlink(missing_ok=True)
    if cut < dur:
        out = video.with_name(video.stem + "_trim.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-t", f"{cut:.3f}", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
        out.replace(video)


def generate(jobs: list[dict], seeds: int, no_end: bool = False) -> None:
    from humangen.generate import VideoRequest, generate_videos_sync

    reqs = []
    for j in jobs:
        d = j["dir"]
        for k in range(seeds):
            c = d / f"cand_s{k}"
            c.mkdir(exist_ok=True)
            for i, (s, v) in enumerate(zip(j["steps"], j["prompts"]), 1):
                out = c / f"step{i}.mp4"
                if not out.exists():
                    reqs.append(VideoRequest(prompt=v["prompt"], reference_image=d / ("frame.jpg" if i == 1 else f"kf{i - 1}.jpg"),
                                             ending_image=None if no_end else d / f"kf{i}.jpg", duration=s["duration"], seed=j["seed"] + k, output_path=out))
    logging.info("%d step clips to generate", len(reqs))
    for i in range(0, len(reqs), 10):
        chunk = reqs[i: i + 10]
        for attempt in range(2):
            chunk = [r for r in chunk if not Path(r.output_path).exists()]
            if not chunk:
                break
            try:
                generate_videos_sync(chunk, aspect="4:3", model=FAST_MODEL)
            except Exception as e:
                logging.error("FastH3 session %d failed (attempt %d): %s", i, attempt + 1, e)
    for j in jobs:  # join step clips per seed, drop audio, give each candidate what the judge reads
        d = j["dir"]
        for k in range(seeds):
            c = d / f"cand_s{k}"
            parts = [c / f"step{i}.mp4" for i in range(1, len(j["steps"]) + 1)]
            if not all(x.exists() for x in parts) or (c / "video.mp4").exists():
                continue
            for x in parts:
                trim_black(x)
            (c / "list.txt").write_text("".join(f"file '{x.name}'\n" for x in parts))
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(c / "list.txt"),
                            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(c / "video.mp4")], check=True)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(c / "video.mp4"), "-vf",
                            "fps=2,scale=192:-1,tile=6x4", "-frames:v", "1", str(c / "contact.png")], check=True)
            kfs = [f"kf{i}.jpg" for i in range(1, len(j["steps"]))]  # intermediate keyframes only; the last one is end_frame.jpg
            for f in ["robot_first.jpg", "frame.jpg", "pair.json"] + kfs:
                shutil.copy2(d / f, c / f)
            shutil.copy2(d / f"robot_kf{len(j['steps'])}.jpg", c / "robot_last.jpg")
            shutil.copy2(d / f"kf{len(j['steps'])}.jpg", c / "end_frame.jpg")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="<dataset>/episode_XXX under --pilot")
    ap.add_argument("--export", type=Path, default=Path("data/so101_export"))
    ap.add_argument("--pilot", type=Path, default=Path("data/humangen_pilot/h3_reference"))
    ap.add_argument("--out", type=Path, default=Path("data/humangen_pilot/fast_h3_keyframes"))
    ap.add_argument("--retries", type=int, default=2, help="redos after the first edit: 2 = at most 3 image edits per frame")
    ap.add_argument("--lama-only", action="store_true", help="with --no-hand: every attempt is SAM 3 + LaMa (widening mask), never an image-generation model")
    ap.add_argument("--no-hand", action="store_true", help="frames with the robot removed and no hand: the hand enters empty, does the task and leaves")
    ap.add_argument("--no-end", action="store_true", help="video from the start frame only (no ending frame); frames are still made and checked")
    ap.add_argument("--end-only", action="store_true", help="start and end frames only (no intermediate keyframes); one clip for the whole task")
    ap.add_argument("--short-prompts", action="store_true", help="3-4 sentence edit prompts and template video prompts from the schema (no VLM rewrite)")
    ap.add_argument("--checks", type=int, default=1, help="1 = one Space Bunny frame check, 3 = three focused checks in parallel that must all pass")
    ap.add_argument("--composite", action="store_true", help="edit each robot frame alone and paste its own pixels back outside the robot")
    ap.add_argument("--editor", default=scene_edit.EDITOR, help="image model for --composite")
    ap.add_argument("--fallback-editor", help="image model for the last attempt (e.g. Pro after Lite)")
    ap.add_argument("--size", default=scene_edit.SIZE, help="editor output size for --composite (1K, 2K, 4K)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--stages", default="video", help="'video' to generate after preparing; anything else prepares only")
    ap.add_argument("--pick", type=Path, help="judge results (JSON list from judge_workflow.js): copy each episode's best candidate to video.mp4")
    args = ap.parse_args(argv)
    if args.pick:
        rank = lambda r: (r["semantic"]["score"] == 5 and r["physics"]["score"] >= 3 and all(r[k]["status"] == "pass" for k in ("reference_preservation", "first_frame_visible", "task_consistency")),
                          r["semantic"]["score"] + r["physics"]["score"])
        best: dict[str, dict] = {}
        for r in json.loads(args.pick.read_text()):
            ep = str(Path(r["dir"]).parent)
            if ep not in best or rank(r) > rank(best[ep]):
                best[ep] = r
        for ep, r in best.items():
            shutil.copy2(Path(r["dir"]) / "video.mp4", Path(ep) / "video.mp4")
            (Path(ep) / "judge.json").write_text(json.dumps(r, indent=1) + "\n")
        print(f"picked {len(best)} episodes, {sum(rank(r)[0] for r in best.values())} pass")
        return 0
    stages = set(args.stages.split(","))
    global CHECK, HANDS
    CHECK = frame_check.check3 if args.checks == 3 else frame_check.check
    HANDS = 0 if args.no_hand else 1
    llm = vlm_model(PROMPT_MODEL, 0, fallback="gemini-3.8-flash")
    jobs = []
    for rel in args.episodes:
        try:
            j = prepare(args, llm, llm, rel)  # idempotent: reuses frames, checks and prompts already on disk
            if j:
                jobs.append(j)
        except Exception as e:
            logging.error("%s: %s", rel, e)
    logging.info("%d episodes ready", len(jobs))
    if "video" in stages and jobs:
        generate(jobs, args.seeds, args.no_end)
    return 0


if __name__ == "__main__":
    sys.exit(main())

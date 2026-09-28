"""VLM check (Space Bunny) of the edited human start and end frames against the robot frames and the pair record.

A frame passes when every listed object is there once, in place and state, nothing is added, the camera is unchanged
(apart from the allowed variation), and there are exactly two plausible hands from the expected edge; the end frame must
also show the goals and match the start frame's hands, surfaces and object appearance.
"""

from __future__ import annotations

from pathlib import Path

from humangen.pilot import EDGE_TEXT, _objects, _sentence
from schema.annotate import _ask

FRAME = {
    "type": "object",
    "required": ["objects_ok", "states_ok", "nothing_added", "camera_ok", "hand_count_ok", "hands_plausible", "hands_entry_ok", "issues"],
    "properties": {
        "objects_ok": {"type": "boolean", "description": "every listed object present exactly once, same place (end frame: where the robot's last frame has it)"},
        "states_ok": {"type": "boolean", "description": "object states as required (open/closed, inside, upright...)"},
        "nothing_added": {"type": "boolean", "description": "no new objects, clutter, text, overlays or robot parts"},
        "camera_ok": {"type": "boolean", "description": "same camera angle and framing as the robot frame"},
        "hand_count_ok": {"type": "boolean", "description": "exactly the stated number of human hands and forearms; other body parts (legs, feet, a torso at the frame edge) do not matter"},
        "hands_plausible": {"type": "boolean", "description": "natural anatomy (five fingers, natural wrist), a natural relaxed pose (e.g. palm down on the surface, not awkwardly palm-up or twisted) and a sensible size for the table"},
        "hands_entry_ok": {"type": "boolean", "description": "forearms come in from the expected image edge"},
        "issues": {"type": "array", "items": {"type": "string"}, "description": "each concrete problem, empty if none"},
    },
}
SCHEMA = {
    "type": "object", "required": ["start", "end"],
    "properties": {
        "start": FRAME,
        "end": {"anyOf": [{"type": "null"}, {**FRAME, "required": FRAME["required"] + ["goals_ok", "matches_start"],
                                                 "properties": {**FRAME["properties"],
                                                                "goals_ok": {"type": "boolean", "description": "the end state holds"},
                                                                "matches_start": {"type": "boolean", "description": "same hands, table, lighting, background and object appearance as the start frame"}}}]},
    },
}

PROMPT = """You check edited images for a dataset that turns robot videos into human videos. Attached, in order: {order}.
The edits must show the same scene from the same camera with {hands_text} instead of the robot, coming in from the {edge} of the frame.
Objects (each must appear exactly once): {objects}.
Start frame states: {states}.
{context}{variation}{end}Ignore black letterbox bars at the image edges, present or absent. A forearm entering at the stated edge or an adjacent corner is fine.
Check each edited frame strictly against the rules; report every concrete problem in issues. Do not invent problems."""


def check(llm, pair: dict, p: dict, robot_first: Path, start: Path, robot_last: Path | None = None, end: Path | None = None, hands: int = 2, context: str | None = None, end_facts: str | None = None) -> dict:
    names = {e["id"]: e["name"] for e in pair["scene"]["entities"]}
    hand = {e["id"]: f"person's {p['active']} hand" for e in pair["scene"]["entities"] if e["kind"] == "actor"}
    states = "; ".join(_sentence(c, names) for c in pair["scene"]["initial_state"] if c["value"] is not None) or "none listed"
    variation = ""
    if p["variations"]:
        variation = (f"Allowed difference: the {p['variations'][0].replace('_', ' ')} of the edited frames may differ from the robot frames, "
                     "but it must be identical in every edited frame; any change of surface, lighting or background between edited frames is a failure. Everything else must match the robot frames.\n")
    order = "1) robot first frame, 2) edited start frame"
    parts = [robot_first, start]
    end_text = ""
    if end is not None:
        order += ", 3) robot last frame, 4) edited end frame"
        parts += [robot_last, end]
        goals = end_facts or "; ".join(_sentence(g, {**names, **hand}) for g in pair["task"]["goals"])
        end_text = f"The edited end frame must show the end state of the robot's last frame with the hand(s) at rest, not touching objects: {goals}. It must match the edited start frame's hand(s), table, lighting, background and object appearance.\n"
    hands_text = ("nothing: the robot is simply removed, with no hands, arms or robot parts left" if hands == 0 else
                  "exactly one person's forearm and hand, left or right (no other hand or forearm; other body parts at the frame edge do not matter)" if hands == 1 else "a person's two forearms and hands")
    text = PROMPT.format(context=f"Annotation of the scene and task:\n{context}\n" if context else "", hands_text=hands_text, order=order, edge=EDGE_TEXT[p["edge"]], states=states, variation=variation, end=end_text,
                         objects="; ".join(f"{e['name']} ({e['grounding']})" for e in _objects(pair)))
    return _ask(llm, parts + [text], SCHEMA, lambda d: [] if isinstance(d.get("start"), dict) else ["start is missing"])


INFORMATIONAL = {"issues", "hands_entry_ok"}  # where the forearm comes in from is recorded but does not fail a frame


def passed(frame: dict | None) -> bool:
    return frame is None or all(v for k, v in frame.items() if k not in INFORMATIONAL)


def blocking_issues(frame: dict | None) -> list[str]:
    """Issues worth sending back to the editor: drop the ones about the entry edge or which hand it is."""
    return [i for i in (frame or {}).get("issues", []) if not any(w in i.lower() for w in ("edge", "corner", "left hand", "right hand", "enters from", "entering from"))]


COUNT_PROMPT = """You compare edited images with robot images. Attached, in order: {order}.
Objects that must appear exactly once in every image: {objects}.
The robot arm in each robot image is meant to be replaced by {hands_text} in the edited image: any such hand and forearm is expected and NOT an added object, and the robot arm being gone is not a missing object.
For each edited image, count every listed object and compare with its robot image: an object missing, appearing twice, or in a different place or state is a problem, and so is anything else new (objects, clutter, text, leftover robot parts). Black letterbox bars do not count.
Return per edited image: objects_ok, nothing_added, states_ok and issues (each concrete difference, empty if none). Do not invent problems."""

HAND_PROMPT = """You check edited images where a robot arm was replaced by {hands_text}, coming in from the {edge} or an adjacent corner. Attached, in order: {order}.
For each edited image: camera_ok (same camera angle, zoom and framing as its robot image; ignore black letterbox bars), hand_count_ok (with "nothing": no hands, arms or robot parts at all) (exactly the stated hands and forearms; legs, feet or a person's body at the frame edge do not matter, only hands and forearms count), hands_plausible (five fingers, natural wrist, a natural relaxed pose such as palm down on the surface, not an awkward palm-up or twisted pose, sensible size for the table), hands_entry_ok (enters from the stated edge or adjacent corner), and issues (each concrete problem, empty if none).{match}
Do not invent problems."""

_PART = lambda keys: {"type": "object", "required": keys + ["issues"], "properties": {**{k: {"type": "boolean"} for k in keys}, "issues": {"type": "array", "items": {"type": "string"}}}}


def check3(llm, pair: dict, p: dict, robot_first: Path, start: Path, robot_last: Path | None = None, end: Path | None = None,
           hands: int = 2, context: str | None = None, end_facts: str | None = None) -> dict:
    """Three focused checks in parallel (general, object count, hands and camera); a flag passes only if every check passes it."""
    from concurrent.futures import ThreadPoolExecutor

    order = "1) robot first frame, 2) edited start frame" + (", 3) robot last frame, 4) edited end frame" if end is not None else "")
    parts = [robot_first, start] + ([robot_last, end] if end is not None else [])
    objects = "; ".join(e["name"] for e in _objects(pair))
    hands_text = ("nothing (the robot is simply removed: no hands, arms or robot parts)" if hands == 0 else
                  "exactly one person's forearm and hand (left or right)" if hands == 1 else "a person's two forearms and hands")
    frames = ["start"] + (["end"] if end is not None else [])
    cnt_schema = {"type": "object", "required": frames, "properties": {f: _PART(["objects_ok", "nothing_added", "states_ok"]) for f in frames}}
    hand_keys = ["camera_ok", "hand_count_ok", "hands_plausible", "hands_entry_ok"]
    hand_schema = {"type": "object", "required": frames, "properties": {f: _PART(hand_keys + (["matches_start"] if f == "end" else [])) for f in frames}}
    match = " For the edited end image also matches_start: the same hand (skin, sleeve, size) as the edited start image." if end is not None else ""
    jobs = {
        "general": lambda: check(llm, pair, p, robot_first, start, robot_last, end, hands=hands, context=context, end_facts=end_facts),
        "count": lambda: _ask(llm, parts + [COUNT_PROMPT.format(order=order, objects=objects, hands_text=hands_text)], cnt_schema, lambda d: []),
        "hands": lambda: _ask(llm, parts + [HAND_PROMPT.format(order=order, hands_text=hands_text, edge=EDGE_TEXT[p["edge"]], match=match)], hand_schema, lambda d: []),
    }
    with ThreadPoolExecutor(3) as ex:
        res = {k: f.result() for k, f in {k: ex.submit(v) for k, v in jobs.items()}.items()}
    out = {}
    for f in ("start", "end"):
        if f == "end" and end is None:
            out[f] = None
            continue
        merged: dict = {"issues": []}
        for name, r in res.items():
            part = r.get(f) or {}
            for k, v in part.items():
                if k == "issues":
                    merged["issues"] += [f"[{name}] {i}" for i in v]
                else:
                    merged[k] = merged.get(k, True) and bool(v)
        out[f] = merged
    out["checks"] = res
    return out

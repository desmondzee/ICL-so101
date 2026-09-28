"""Schema context for the generation prompts: the pair record rendered as compact text, and the human clip timing.

Every stage (start edit, end edit, frame check, video prompt) gets the same block, so the scene, task, states and
timing all come from the schema, not from what a model guesses from the images.
"""

from __future__ import annotations

FPS_ROBOT_APPROACH_S = 1.0  # robot seconds kept before the first grasp
HOLD_S = 1.0  # human seconds of stillness after the last release
LEAD_S = 0.5  # human seconds before the hand starts moving
SPEEDUP = 2.0
MIN_S, MAX_S = 5.0, 14.0


def region(box: list[int] | None) -> str:
    if not box:
        return "position not visible"
    y0, x0, y1, x1 = box
    cx, cy = (x0 + x1) / 2000, (y0 + y1) / 2000
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    size = (x1 - x0) * (y1 - y0) / 1e6
    return f"{row} {col} of the image, about {size:.0%} of the frame"


def _fact(c: dict, names: dict) -> str:
    s, o = names.get(c["subject"], c["subject"]), names.get(c.get("object") or "", c.get("object"))
    rel = c["relation"].replace("_", " ")
    return f"{s} is {'' if c['value'] else 'not '}{rel}" + (f" {o}" if o else "")


def names_for(pair: dict, hand: str) -> dict:
    """Entity and role ids to names; the robot becomes the person's hand."""
    names = {e["id"]: e["name"] for e in pair["scene"]["entities"]}
    names |= {r["role"]: r["name"] for r in pair["task"]["roles"]}
    for e in pair["scene"]["entities"]:
        if e["kind"] == "actor":
            names[e["id"]] = hand
    names.update({r["role"]: hand for r in pair["task"]["roles"] if r["kind"] == "actor"})
    return names


def timing(pair: dict) -> tuple[float, list[dict]]:
    """Human clip length and per-step windows (seconds) from the robot segments, sped up and trimmed to the action."""
    fps = pair["source"]["fps"]
    segs = pair["segments"]
    grasps = [s["grasp"] for s in segs if s["grasp"] is not None]
    releases = [s["release"] for s in segs if s["release"] is not None]
    a0 = max(segs[0]["start"], (grasps[0] if grasps else segs[0]["start"]) - FPS_ROBOT_APPROACH_S * fps)
    a1 = releases[-1] if releases else segs[-1]["end"]
    span = (a1 - a0) / fps
    speed = max(SPEEDUP, span / (MAX_S - LEAD_S - HOLD_S))
    duration = min(max(LEAD_S + span / speed + HOLD_S, MIN_S), MAX_S)
    t = lambda f: round(LEAD_S + (min(max(f, a0), a1) - a0) / fps / speed, 1)
    windows = [{"step": s["step"], "start": t(s["start"]), "end": t(s["end"]),
                "grasp": None if s["grasp"] is None else t(s["grasp"]), "release": None if s["release"] is None else t(s["release"])}
               for s in segs]
    windows[0]["start"] = LEAD_S
    return round(duration, 1), windows


def render(pair: dict, hand: str, duration: float | None = None, windows: list[dict] | None = None, task: bool = True) -> str:
    """task=False gives the scene only (camera, objects, first-frame state): image edits given the whole task tend to draw its end state."""
    names = names_for(pair, hand)
    with_task, (scene, task) = task, (pair["scene"], pair["task"])
    lines = [f"Camera: {scene['view']['shot']}, fixed."]
    lines.append("Objects (exactly these, each once):")
    for e in scene["entities"]:
        if e["kind"] == "actor":
            continue
        a = e["attributes"]
        desc = ", ".join(x for x in (a.get("color"), a.get("material"), a.get("shape")) if x)
        lines.append(f"- {e['name']} ({e['kind']}; {desc}): {e['grounding']}; {region(e['box_2d'])}")
    roles = [r for r in task["roles"] if r["kind"] != "actor"]
    if roles:
        lines.append("Roles: " + " ".join(f"{r['name']}: {r['appearance']}" for r in roles))
    known = [c for c in scene["initial_state"] if c["value"] is not None]
    if known:
        lines.append("First-frame state: " + "; ".join(_fact(c, names) for c in known) + ".")
    if not with_task:
        return "\n".join(lines)
    lines.append(f"Task: {task['instruction']}")
    win = {w["step"]: w for w in windows or []}
    for i, s in enumerate(task["steps"], 1):
        what = f"{s['action'].replace('_', ' ')} the {names.get(s['object'], s['object'])}"
        if s.get("destination"):
            what += f" to the {names.get(s['destination'], s['destination'])}"
        line = f"Step {i}: {what}"
        w = win.get(s["id"])
        if w:
            line += f" (from {w['start']}s to {w['end']}s" + (f", grasp at {w['grasp']}s" if w["grasp"] is not None else "") \
                    + (f", release at {w['release']}s" if w["release"] is not None else "") + ")"
        if s["after"]:
            line += "; after it: " + "; ".join(_fact(c, names) for c in s["after"])
        lines.append(line + ".")
    if task["required_order"]:
        lines.append("Order: " + "; ".join(f"{names.get(o['earlier'], o['earlier'])} before {names.get(o['later'], o['later'])}" for o in task["required_order"]) + ".")
    lines.append("End state: " + "; ".join(_fact(g, names) for g in task["goals"]) + ".")
    if task["invariants"]:
        lines.append("Throughout: " + "; ".join(_fact(c, names) for c in task["invariants"]) + ".")
    if duration:
        lines.append(f"Clip length: {duration}s; the hand is still after the last step.")
    return "\n".join(lines)


# Short prompts for the generators: image and video models follow a few concrete sentences better than a schema dump.
# The detailed context above stays with the checks and judges.

VERB = {
    "pick_place": "lifts it and places it {prep} the {dest}", "stack": "lifts it and sets it on top of the {dest}",
    "lift": "lifts it", "slide": "slides it across the surface to the {dest}", "open": "pulls it open",
    "close": "pushes it closed", "press": "presses it down firmly", "pour": "lifts it, tilts it to pour into the {dest} and sets it back down",
    "fold": "folds it over neatly", "wipe": "wipes it across the {dest}",
}


COLOURS = {"black", "white", "grey", "gray", "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "silver", "clear", "transparent", "beige", "tan"}


def _looks(e: dict) -> str:
    """The schema name, with its colour in front when the name has none (e.g. 'can' -> 'white and blue can')."""
    name = e["name"]
    colour = (e["attributes"].get("color") or "").strip()
    if colour and not (set(name.lower().split()) & COLOURS) and len(colour.split()) <= 3:
        return f"{colour} {name}"
    return name


def keep_list(pair: dict) -> str:
    """Object names for 'keep unchanged' in edit prompts (no attributes, regions or roles)."""
    return ", ".join(e["name"] for e in pair["scene"]["entities"] if e["kind"] in ("object", "container", "fixture"))


def risky_states(conds: list[dict], names: dict) -> str:
    """Only the states an image editor tends to get wrong: open/closed, inside, stacked, upright, folded."""
    keep = [c for c in conds if c["value"] is not None and c["relation"] in ("open", "inside", "upright", "folded", "switched_on")
            or (c["relation"] == "supported_by" and c["value"] and names.get(c.get("object") or "", "").lower().find("table") < 0
                and names.get(c.get("object") or "", "").lower().find("surface") < 0)]
    return "; ".join(_fact(c, names) for c in keep)


def step_prompt(pair: dict, step: dict, hand: str, edge: str, duration: float) -> str:
    """A 60-100 word FastH3 prompt for one step, built from the schema (nothing paraphrased)."""
    ents = {e["id"]: e for e in pair["scene"]["entities"]}
    binds = {b["role"]: b["entity"] for b in pair["bindings"]}
    obj = ents.get(binds.get(step["object"], step["object"]))
    dest = ents.get(binds.get(step.get("destination") or "", step.get("destination") or ""))
    what = _looks(obj) if obj else step["object"].replace("_", " ")
    where = f" at the {region(obj['box_2d']).split(' of the image')[0]}" if obj and obj["box_2d"] else ""
    prep = "in" if dest and dest["kind"] == "container" else "on"
    verb = VERB.get(step["action"], "moves it").format(prep=prep, dest=_looks(dest) if dest else "place")
    if step["action"] == "lift" and dest:
        verb += f" and moves it to the {_looks(dest)}"
    return (f"One person's {hand} forearm and hand enters from the {edge} of the frame, reaches the {what}{where}, grasps it, {verb}, "
            f"then lets go and rests near the {edge}. The {what} moves only while the hand holds it. "
            f"Only this one hand moves; every other object stays still. The motion is smooth and natural, over {duration:g} seconds. "
            f"One locked static shot from the same viewpoint. Silent.")


def task_prompt(pair: dict, hand: str, edge: str, duration: float, enters: bool = False) -> str:
    """One FastH3 prompt for the whole task (at most ~120 words): the action first, the object's mechanism and end
    state, then the constraints. Built from the schema, nothing paraphrased."""
    ents = {e["id"]: e for e in pair["scene"]["entities"]}
    binds = {b["role"]: b["entity"] for b in pair["bindings"]}
    parts = []
    for i, st in enumerate(pair["task"]["steps"]):
        obj = ents.get(binds.get(st["object"], st["object"]))
        dest = ents.get(binds.get(st.get("destination") or "", st.get("destination") or ""))
        what = _looks(obj) if obj else st["object"].replace("_", " ")
        prep = "in" if dest and dest["kind"] == "container" else "on"
        verb = VERB.get(st["action"], "moves it").format(prep=prep, dest=_looks(dest) if dest else "place")
        role = next((r for r in pair["task"]["roles"] if r["role"] == st["object"]), None)
        if st["action"] in ("open", "close", "slide", "press") and role:
            # the mechanism, from the role's appearance text: articulated objects need to be told how they move
            how = role["appearance"].rstrip(".").split(";")[0].split(",")[0]  # the mechanism, first clause only
            parts.append(f"{'then ' if i else ''}grasps the {what} ({how}) and {verb.replace('it', 'it', 1)}")
        else:
            where = f" at the {region(obj['box_2d']).split(' of the image')[0]}" if obj and obj["box_2d"] and i == 0 else ""
            oid = obj["id"] if obj else st["object"]
            was_in = [c["object"] for c in pair["scene"]["initial_state"] if c["relation"] == "inside" and c["subject"] == oid and c["value"]]
            out_of = [c for c in was_in if any(g["relation"] == "inside" and g["subject"] == oid and g["object"] == c and g["value"] is False
                                               for g in pair["task"]["goals"] + st["after"])]
            if out_of:  # e.g. a plug that starts inside a socket: say it is pulled out
                verb = f"pulls it out of the {_looks(ents[out_of[0]]) if out_of[0] in ents else out_of[0]} and " + verb.replace("lifts it and ", "")
            parts.append(f"{'then ' if i else ''}grasps the {what}{where}, {verb}")
    action = "; ".join(parts)
    moved_names = [_looks(ents[binds.get(st["object"], st["object"])]) for st in pair["task"]["steps"] if binds.get(st["object"], st["object"]) in ents]
    if enters:  # no hand in the first frame: one hand comes in, does the task on objects already there, and leaves
        # Short, strict and only positive: FastH3 has no negative prompt, so naming unwanted things (text, extra hands,
        # new objects) makes them appear. The judge rejects any clip that adds anything.
        names_ = list(dict.fromkeys(moved_names)) or ["object"]
        moved = names_[0] if len(names_) == 1 else ", ".join(names_[:-1]) + " and " + names_[-1]
        verb, them = ("moves", "it") if len(names_) == 1 else ("move", "them")
        return (f"Fixed camera, one continuous {duration:g}-second shot of this same table. "
                f"A person's {hand} hand reaches in from the {edge}, {action}, then moves back out of the frame. "
                f"Only the {moved} {verb}, and only while the hand is holding or pushing {them}; "
                f"everything else stays exactly where it is. Smooth, natural, real-world motion. "
                f"Sound: a quiet room and the soft sounds of the hand handling the {moved}.")
    return (f"A person's {hand} hand {action}, then lets go and rests still on the table for the final second. "
            f"One continuous {duration:g}-second shot from a fixed camera. "
            f"Only this one hand and forearm, entering from the {edge}, are ever in view: no extra hands, arms or people at any point. "
            f"Nothing else in the scene changes. No object appears, disappears, duplicates or changes shape. "
            f"Objects move only when the hand touches them, with realistic weight, grip and contact, and stay where they are put. Silent.")

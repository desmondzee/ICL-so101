"""JSON Schema validation plus the cross-reference rules JSON Schema cannot express."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).parent
NAMES = ("common", "scene", "task", "pair")
UNARY = {"open", "switched_on", "folded", "upright"}
NEEDS_DESTINATION = {"pick_place", "slide", "stack", "pour"}


@cache
def load(name: str) -> dict:
    return json.loads((ROOT / f"{name}.schema.json").read_text())


@cache
def _registry() -> Registry:
    return Registry().with_resources((load(n)["$id"], Resource.from_contents(load(n))) for n in NAMES)


def bundle(name: str) -> dict:
    """One self-contained schema with local $defs and no const, for LLM structured output."""
    defs = copy.deepcopy(load("common")["$defs"])
    ids = {load(n)["$id"]: n for n in NAMES}

    def rewrite(node):
        if isinstance(node, dict):
            if "const" in node:
                node = {**node, "enum": [node["const"]]}
                del node["const"]
            out = {}
            for key, value in node.items():
                if key in ("$schema", "$id"):
                    continue
                if key == "$ref":
                    base, _, frag = value.partition("#")
                    target = ids.get(base, base)
                    value = f"#{frag}" if target == "common" or not base else f"#/$defs/{target}"
                    if target not in ("common", "") and target not in defs:
                        defs[target] = None
                        defs[target] = rewrite(load(target))
                out[key] = rewrite(value)
            return out
        if isinstance(node, list):
            return [rewrite(v) for v in node]
        return node

    for key in list(defs):
        defs[key] = rewrite(defs[key])
    top = rewrite(load(name))
    top["$defs"] = {**defs, **top.get("$defs", {})}
    return top


def _schema_errors(name: str, doc: dict) -> list[str]:
    validator = Draft202012Validator(load(name), registry=_registry())
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(doc)]


def _dupes(values) -> set:
    seen, dup = set(), set()
    for v in values:
        (dup if v in seen else seen).add(v)
    return dup


def _conditions(conds, names: set[str], where: str) -> list[str]:
    errors = []
    for i, c in enumerate(conds):
        at = f"{where}[{i}]"
        if c["subject"] not in names:
            errors.append(f"{at}: unknown subject {c['subject']!r}")
        if c["relation"] in UNARY:
            if c["object"] is not None:
                errors.append(f"{at}: {c['relation']} is unary, object must be null")
        elif c["object"] is None:
            errors.append(f"{at}: {c['relation']} needs an object")
        elif c["object"] not in names:
            errors.append(f"{at}: unknown object {c['object']!r}")
        elif c["object"] == c["subject"]:
            errors.append(f"{at}: subject and object are the same")
    return errors


def check_scene(scene: dict, where: str = "$") -> list[str]:
    errors = _schema_errors("scene", scene)
    if errors:
        return [f"{where}{e[1:]}" for e in errors]
    ids = [e["id"] for e in scene["entities"]]
    errors += [f"{where}.entities: duplicate id {d!r}" for d in _dupes(ids)]
    errors += _conditions(scene["initial_state"], set(ids), f"{where}.initial_state")
    kinds = {e["id"]: e["kind"] for e in scene["entities"]}
    for i, c in enumerate(scene["initial_state"]):
        if c["relation"] == "held_by" and kinds.get(c["object"]) not in (None, "actor"):
            errors.append(f"{where}.initial_state[{i}]: held_by object must be an actor")
    for i, e in enumerate(scene["entities"]):
        box = e["box_2d"]
        if box is not None and not (box[0] < box[2] and box[1] < box[3]):
            errors.append(f"{where}.entities[{i}].box_2d: expected ymin < ymax and xmin < xmax")
        if e["visibility"] == "occluded" and box is not None:
            errors.append(f"{where}.entities[{i}]: occluded entity has a box")
    return errors


def _cycle(edges: list[tuple[str, str]]) -> bool:
    graph: dict[str, list[str]] = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        state[n] = 1
        for m in graph.get(n, []):
            if state.get(m) == 1 or (m not in state and visit(m)):
                return True
        state[n] = 2
        return False

    return any(n not in state and visit(n) for n in list(graph))


def check_task(task: dict, where: str = "$") -> list[str]:
    errors = _schema_errors("task", task)
    if errors:
        return [f"{where}{e[1:]}" for e in errors]
    roles = [r["role"] for r in task["roles"]]
    names = set(roles)
    steps = [s["id"] for s in task["steps"]]
    errors += [f"{where}.roles: duplicate role {d!r}" for d in _dupes(roles)]
    errors += [f"{where}.steps: duplicate id {d!r}" for d in _dupes(steps)]
    for i, s in enumerate(task["steps"]):
        at = f"{where}.steps[{i}]"
        if s["object"] not in names:
            errors.append(f"{at}: unknown object role {s['object']!r}")
        if s["action"] in NEEDS_DESTINATION:
            if s["destination"] is None:
                errors.append(f"{at}: {s['action']} needs a destination")
            elif s["destination"] not in names:
                errors.append(f"{at}: unknown destination role {s['destination']!r}")
        elif s["destination"] is not None:
            errors.append(f"{at}: {s['action']} takes no destination")
        errors += _conditions(s["after"], names, f"{at}.after")
    errors += _conditions(task["goals"], names, f"{where}.goals")
    errors += _conditions(task["invariants"], names, f"{where}.invariants")
    edges = [(o["earlier"], o["later"]) for o in task["required_order"]]
    for a, b in edges:
        if a not in steps or b not in steps:
            errors.append(f"{where}.required_order: unknown step in {a!r} -> {b!r}")
    if _cycle(edges):
        errors.append(f"{where}.required_order: cycle")
    return errors


def accepts(checks: dict) -> bool | None:
    statuses = [checks[k]["status"] for k in ("reference_preservation", "first_frame_visible", "task_consistency")]
    sem, phys = checks["semantic"]["score"], checks["physics"]["score"]
    if "fail" in statuses or (sem is not None and sem < 5) or (phys is not None and phys < 3):
        return False
    if "unknown" in statuses or sem is None or phys is None:
        return None
    return True


def check_pair(pair: dict) -> list[str]:
    errors = _schema_errors("pair", pair)
    if errors:
        return errors
    scene, task = pair["scene"], pair["task"]
    errors += check_scene(scene, "$.scene") + check_task(task, "$.task")
    if errors:
        return errors

    entities = {e["id"]: e for e in scene["entities"]}
    roles = {r["role"]: r for r in task["roles"]}
    bound = {b["role"]: b["entity"] for b in pair["bindings"]}
    errors += [f"$.bindings: duplicate role {d!r}" for d in _dupes(b["role"] for b in pair["bindings"])]
    errors += [f"$.bindings: entity {d!r} bound twice" for d in _dupes(bound.values())]
    errors += [f"$.bindings: unbound role {r!r}" for r in roles if r not in bound]
    for role, entity in bound.items():
        if role not in roles:
            errors.append(f"$.bindings: unknown role {role!r}")
        elif entity not in entities:
            errors.append(f"$.bindings: role {role!r} bound to unknown entity {entity!r}")
        elif roles[role]["kind"] != entities[entity]["kind"]:
            errors.append(f"$.bindings: role {role!r} is {roles[role]['kind']}, entity {entity!r} is {entities[entity]['kind']}")
    if errors:
        return errors

    start = {(c["relation"], c["subject"], c["object"]): c["value"] for c in scene["initial_state"]}
    goals = [(g["relation"], bound[g["subject"]], g["object"] and bound[g["object"]], g["value"]) for g in task["goals"]]
    if all(start.get(g[:3]) == g[3] for g in goals):
        errors.append("$.task.goals: already satisfied by the scene's initial_state")

    src = pair["source"]["frames"]
    if src["end"] <= src["start"]:
        errors.append("$.source.frames: end must exceed start")
    steps = [s["id"] for s in task["steps"]]
    segs = pair["segments"]
    if sorted(s["step"] for s in segs) != sorted(steps):
        errors.append(f"$.segments: need exactly one segment per step {steps}")
    last = src["start"]
    for i, s in enumerate(segs):
        at = f"$.segments[{i}]"
        if not (src["start"] <= s["start"] < s["end"] <= src["end"]):
            errors.append(f"{at}: [{s['start']}, {s['end']}) outside source frames or empty")
        if s["start"] < last:
            errors.append(f"{at}: overlaps the previous segment")
        last = s["end"]
        for key in ("grasp", "release"):
            if s[key] is not None and not (s["start"] <= s[key] < s["end"]):
                errors.append(f"{at}.{key}: outside the segment")
        if s["grasp"] is not None and s["release"] is not None and s["release"] <= s["grasp"]:
            errors.append(f"{at}: release before grasp")
    position = {s["step"]: i for i, s in enumerate(segs)}
    for o in task["required_order"]:
        if o["earlier"] in position and o["later"] in position and position[o["earlier"]] > position[o["later"]]:
            errors.append(f"$.segments: source violates required order {o['earlier']} -> {o['later']}")

    human, checks = pair["human"], pair["checks"]
    if human is not None:
        errors += check_scene(human["scene"], "$.human.scene")
        human_ids = {e["id"] for e in human["scene"]["entities"]}
        mapped = [m["source"] for m in human["entity_map"]]
        errors += [f"$.human.entity_map: {d!r} mapped twice" for d in _dupes(mapped)]
        errors += [f"$.human.entity_map: human id {d!r} used twice" for d in _dupes(m["human"] for m in human["entity_map"])]
        errors += [f"$.human.entity_map: source entity {e!r} not mapped" for e in entities if e not in mapped]
        for m in human["entity_map"]:
            if m["source"] not in entities:
                errors.append(f"$.human.entity_map: unknown source entity {m['source']!r}")
            if m["human"] not in human_ids:
                errors.append(f"$.human.entity_map: unknown human entity {m['human']!r}")

    for key in ("reference_preservation", "first_frame_visible", "task_consistency"):
        c = checks[key]
        if c["status"] != "unknown" and not c["evidence"]:
            errors.append(f"$.checks.{key}: {c['status']} needs evidence")
        if c["status"] != "unknown" and human is None:
            errors.append(f"$.checks.{key}: reviewed without a human reference")
    for key in ("semantic", "physics"):
        c = checks[key]
        if c["score"] is not None:
            if not c["evidence"]:
                errors.append(f"$.checks.{key}: score needs evidence")
            if human is None or human["video"] is None:
                errors.append(f"$.checks.{key}: scored without a human video")
    goals_verdicts = checks["semantic"]["goals"]
    if goals_verdicts and len(goals_verdicts) != len(task["goals"]):
        errors.append(f"$.checks.semantic.goals: expected {len(task['goals'])} verdicts")
    if checks["semantic"]["score"] == 5 and (not goals_verdicts or set(goals_verdicts) != {"pass"}):
        errors.append("$.checks.semantic: score 5 requires every goal to pass")
    if pair["accepted"] != accepts(checks):
        errors.append(f"$.accepted: expected {accepts(checks)} from the checks")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate pair, scene or task JSON files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--kind", choices=["pair", "scene", "task"], default="pair")
    args = parser.parse_args(argv)
    check = {"pair": check_pair, "scene": check_scene, "task": check_task}[args.kind]
    bad = 0
    for path in args.files:
        errors = check(json.loads(path.read_text()))
        bad += bool(errors)
        print(f"{'FAIL' if errors else 'ok  '} {path}")
        for e in errors:
            print(f"     {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Explicit task semantics and machine-readable held-out overlap checks.

This is structured comparison, not a claim to infer meaning from arbitrary
English. Semantic annotations are part of the reviewed task contract.
"""

from dataclasses import dataclass
import importlib
import re
import unicodedata

from sim.train.model import EpisodeKey


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _semantic_text(text):
    # Colour swaps cannot inflate the number of task definitions.
    colors = {"red", "blue", "green", "yellow", "white", "black", "orange", "purple"}
    return " ".join(t for t in normalize(text.replace("_", " ")).split() if t not in colors)


@dataclass(frozen=True)
class SemanticSignature:
    family: str
    manipulated_objects: tuple[str, ...]
    relation: str
    goal: str
    order: tuple[str, ...]

    def __post_init__(self):
        fields = (self.family, self.relation, self.goal, *self.manipulated_objects, *self.order)
        if not self.manipulated_objects or not self.order or not all(isinstance(s, str) and normalize(s) for s in fields):
            raise ValueError("semantic taxonomy must be nonempty")

    def normalized(self):
        return (_semantic_text(self.family), tuple(sorted(map(_semantic_text, self.manipulated_objects))),
                _semantic_text(self.relation), _semantic_text(self.goal), tuple(map(_semantic_text, self.order)))


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    instruction: str
    family: str
    semantic_signature: SemanticSignature
    task_objects: tuple[str, ...]
    action_order: tuple[str, ...]
    action_text: tuple[str, ...]
    env_class: str
    oracle_class: str
    qualification_seeds: tuple[int, ...] = tuple(range(10000, 10050))

    def __post_init__(self):
        EpisodeKey(self.name, 0)
        if not normalize(self.instruction) or not normalize(self.family):
            raise ValueError("instruction and family are required")
        if (not self.task_objects or len(set(self.task_objects)) != len(self.task_objects)
                or not all(isinstance(n, str) and normalize(n) for n in self.task_objects)):
            raise ValueError("task_objects must be distinct nonempty names")
        if (not self.action_order or set(self.action_order) != set(self.task_objects)
                or len(self.action_order) != len(self.task_objects)
                or len(self.action_text) != len(self.action_order)
                or not all(normalize(t) for t in self.action_text)):
            raise ValueError("declare exactly one action description per ordered task object")
        if not self.qualification_seeds or len(set(self.qualification_seeds)) != len(self.qualification_seeds):
            raise ValueError("qualification seeds must be nonempty and distinct")
        for seed in self.qualification_seeds:
            EpisodeKey(self.name, seed)
        for path in (self.env_class, self.oracle_class):
            if not isinstance(path, str) or len(path.split(":")) != 2:
                raise ValueError("class references must be module:class")

    def load_classes(self):
        def load(path):
            module, name = path.split(":")
            return getattr(importlib.import_module(module), name)
        return load(self.env_class), load(self.oracle_class)

    def action_descriptions(self):
        return self.action_text

    def physics_policy(self, env):
        if tuple(env.task_objects) != self.task_objects:
            raise ValueError("task/environment object contract mismatch")
        return env.physics_policy()


class OrderedCompletion:
    """Track physical grasp→release completions; an invalid order stays invalid."""

    def __init__(self, order):
        self.order, self.completed, self.picked = tuple(order), [], set()
        self.violated = False

    def observe(self, *, held, placed):
        for name in held:
            if name not in self.completed and (len(self.completed) == len(self.order) or name != self.order[len(self.completed)]):
                self.violated = True
        self.picked.update(held)
        for name in self.order:
            if name in self.picked and name in placed and name not in held and name not in self.completed:
                if self.violated or name != self.order[len(self.completed)]:
                    self.violated = True
                else:
                    self.completed.append(name)

    @property
    def complete(self):
        return not self.violated and tuple(self.completed) == self.order


# Reviewed semantic annotations for the existing validation code. Any new
# validation task must supply an annotation rather than silently bypass checks.
VALIDATION_SEMANTICS = {
    "sort_blocks": SemanticSignature("spatial_arrangement", ("block", "block"), "on", "matching plates", ("either",)),
    "stack_bowls": SemanticSignature("container_insertion", ("bowl",), "inside", "bowl", ("bowl",)),
    "mug_on_plate": SemanticSignature("orientation_sensitive_placement", ("mug",), "upright on", "plate", ("mug",)),
    "mugs_in_microwave": SemanticSignature("container_insertion", ("mug", "mug"), "inside", "microwave", ("either",)),
    "pan_on_stove": SemanticSignature("spatial_arrangement", ("frying pan",), "on", "stove", ("frying pan",)),
}


def validate_catalog(definitions, validation_index):
    if not isinstance(validation_index, dict) or not all(isinstance(validation_index.get(k), list) for k in ("tasks", "episodes")):
        raise ValueError("validation index requires tasks and episodes lists")
    names, instructions, semantics, families, seeds = set(), set(), set(), set(), set()
    for row in validation_index["tasks"]:
        name = row["task"]
        signature = SemanticSignature(**row["semantics"]) if "semantics" in row else VALIDATION_SEMANTICS.get(name)
        if signature is None:
            raise ValueError(f"missing validation semantic annotation: {name}")
        names.add(normalize(name))
        semantics.add(signature.normalized())
        families.add(signature.family)
        instructions.update(normalize(s) for s in [row.get("instruction", ""), *row.get("instructions", [])] if s)
    for row in validation_index["episodes"]:
        if type(row.get("seed")) is not int or row["seed"] < 0:
            raise ValueError("invalid validation seed")
        seeds.add(row["seed"])
        if row.get("instruction"):
            instructions.add(normalize(row["instruction"]))
    violations, overlaps = [], []
    seen_names, seen_instructions, seen_semantics = set(), set(), set()
    for task in definitions:
        name, instruction, semantic = normalize(task.name), normalize(task.instruction), task.semantic_signature.normalized()
        for kind, test in (("duplicate_name", name in seen_names), ("duplicate_instruction", instruction in seen_instructions),
                           ("duplicate_semantics", semantic in seen_semantics), ("validation_name", name in names),
                           ("validation_instruction", instruction in instructions), ("validation_semantics", semantic in semantics)):
            if test:
                violations.append({"task": task.name, "kind": kind})
        overlap_seeds = sorted(set(task.qualification_seeds) & seeds)
        if overlap_seeds:
            violations.append({"task": task.name, "kind": "validation_seed", "seeds": overlap_seeds})
        if task.family in families:
            overlaps.append({"task": task.name, "family": task.family})
        seen_names.add(name)
        seen_instructions.add(instruction)
        seen_semantics.add(semantic)
    return {"accepted": not violations, "violations": violations, "family_overlap": overlaps,
            "family_level_isolation": False, "comparison": "reviewed structured semantics; color-insensitive"}

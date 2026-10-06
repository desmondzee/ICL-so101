"""Declarative, fail-closed trajectory gates; policies belong to task definitions.

Defaults are conservative starting points, not pilot-qualified realism claims.
Persist ``policy.to_dict()`` alongside each report when a recorder uses them.
"""

from dataclasses import asdict, dataclass
import json

import numpy as np

from .telemetry import ARRAY_FIELDS, SCHEMA_VERSION, Telemetry, _valid_hash


@dataclass(frozen=True)
class GoalRule:
    body: str
    reference: str | None = None
    offset: tuple[float, float, float] = (0, 0, 0)
    position_tolerance: tuple[float, float, float] = (0.01, 0.01, 0.01)
    upright_cos: float | None = None


@dataclass(frozen=True)
class GraspContactAllowance:
    """Bounded jaw/fingertip contact with a support surface during grasp and place.

    Only contacts between one of ``robot_bodies`` (the two jaw bodies) and one of
    ``surfaces`` qualify, and each sample must satisfy every limit: normal force,
    penetration depth, a contact point within ``proximity`` (xy, m) of some task
    object at that substep (that is, while grasping or placing it), and no
    continuous run of such contact longer than ``max_contact_seconds``. A sample
    outside a limit fails the ``grasp_contact`` check; every other robot contact
    (arm links, wrist, camera mount, self-contact) still fails ``allowed_contacts``.
    Limits are calibrated from measured trajectories; see sim/train/README.md.
    """
    robot_bodies: tuple[str, ...]
    surfaces: tuple[str, ...]
    max_normal_force: float
    max_penetration: float
    max_contact_seconds: float
    proximity: float

    def __post_init__(self):
        for name in ("robot_bodies", "surfaces"):
            names = tuple(getattr(self, name))
            object.__setattr__(self, name, names)
            if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
                raise ValueError(f"{name} must be distinct nonempty body names")
        if set(self.robot_bodies) & set(self.surfaces):
            raise ValueError("a body cannot be both a jaw and a surface")
        for name in ("max_normal_force", "max_penetration", "max_contact_seconds", "proximity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid grasp contact limit: {name}")


@dataclass(frozen=True)
class PhysicsPolicy:
    task_objects: tuple[str, ...]
    support_bodies: tuple[tuple[str, tuple[str, ...]], ...]
    allowed_contacts: tuple[tuple[str, str], ...]
    distractors: tuple[str, ...] = ()
    goals: tuple[GoalRule, ...] = ()
    settled_frames: int = 30
    joint_tolerance: float = 0.01
    max_joint_speed: float = 3.0
    max_joint_acceleration: float = 150.0
    max_linear_speed: float = 1.0
    max_angular_speed: float = 15.0
    max_linear_acceleration: float = 100.0
    max_angular_acceleration: float = 1000.0
    penetration_tolerance: float = 0.003
    distractor_displacement_tolerance: float = 0.005
    distractor_rotation_tolerance: float = 0.1
    support_distance_tolerance: float = 0.001
    min_support_force: float = 0.0
    min_support_upward_cos: float = 0.5
    settled_linear_speed: float = 0.03
    settled_angular_speed: float = 0.5
    grasp_contact: GraspContactAllowance | None = None

    def __post_init__(self):
        if type(self.settled_frames) is not int or self.settled_frames < 30:
            raise ValueError("at least 30 consecutive released success frames are required")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not np.isfinite(value) or value < 0):
                raise ValueError(f"invalid policy threshold: {name}")
        if not self.task_objects or len(set(self.task_objects)) != len(self.task_objects):
            raise ValueError("task_objects must be nonempty and unique")
        supports = dict(self.support_bodies)
        if len(supports) != len(self.support_bodies) or set(supports) != set(self.task_objects):
            raise ValueError("declare support bodies for every task object")
        if any(not names or body in names for body, names in self.support_bodies):
            raise ValueError("objects need a distinct declared support")
        if set(self.task_objects) & set(self.distractors):
            raise ValueError("task objects cannot be distractors")
        if self.min_support_upward_cos > 1:
            raise ValueError("invalid support direction tolerance")
        for goal in self.goals:
            if goal.body not in self.task_objects or len(goal.offset) != 3 or len(goal.position_tolerance) != 3:
                raise ValueError("invalid goal body or vector")
            if not np.isfinite((*goal.offset, *goal.position_tolerance)).all() or min(goal.position_tolerance) < 0:
                raise ValueError("invalid goal tolerance")
            if goal.upright_cos is not None and (not np.isfinite(goal.upright_cos) or not -1 <= goal.upright_cos <= 1):
                raise ValueError("invalid upright tolerance")
        if any(len(pair) != 2 or not all(isinstance(n, str) and n for n in pair) for pair in self.allowed_contacts):
            raise ValueError("allowed_contacts must contain named body pairs")

        if self.grasp_contact is not None and not isinstance(self.grasp_contact, GraspContactAllowance):
            raise ValueError("grasp_contact must be a GraspContactAllowance")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "PhysicsPolicy":
        """Inverse of ``to_dict`` (for re-auditing persisted ``policy.json``)."""
        value = dict(value)
        tuples = lambda items: tuple(tuple(x) if isinstance(x, list) else x for x in items)
        value["task_objects"] = tuple(value["task_objects"])
        value["distractors"] = tuple(value.get("distractors", ()))
        value["support_bodies"] = tuple((body, tuple(names)) for body, names in value["support_bodies"])
        value["allowed_contacts"] = tuples(value["allowed_contacts"])
        value["goals"] = tuple(GoalRule(**{k: tuple(v) if isinstance(v, list) else v for k, v in goal.items()})
                               for goal in value.get("goals", ()))
        if value.get("grasp_contact") is not None:
            value["grasp_contact"] = GraspContactAllowance(**value["grasp_contact"])
        return cls(**value)


@dataclass(frozen=True)
class PhysicsReport:
    accepted: bool
    checks: dict[str, bool]
    maxima: dict[str, float | int | None]
    violations: tuple[dict, ...]
    config_hash: str

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integrity(telemetry, policy):
    """Validate and return safe arithmetic views without altering evidence.

    Continuous fields use IEEE float32/float64; counters and indices use
    signed integers; flags use bool. Unsigned or integer physical values are
    unsupported, rather than relying on NumPy's potentially wrapping math.
    """
    a, m = telemetry.arrays, telemetry.manifest
    if set(a) != ARRAY_FIELDS:
        raise ValueError("missing or unknown schema arrays")
    flags = {"success", "grasp", "frame_end"}
    integers = {"frame", "warnings", "contact_offsets", "contact_geom"}
    normalized = {}
    for key, value in a.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"invalid {key} array")
        if key in flags:
            if value.dtype != np.dtype(bool):
                raise ValueError(f"invalid {key} flag")
            normalized[key] = value
        elif key in integers:
            if value.dtype.kind != "i" or value.dtype.itemsize > 8:
                raise ValueError(f"invalid {key} type: signed integer required")
            normalized[key] = value.astype(np.int64, copy=False)
        else:
            if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
                raise ValueError(f"invalid {key} type: float32/float64 required")
            normalized[key] = value.astype(np.float64, copy=False)
    a = normalized
    if m["schema_version"] != SCHEMA_VERSION or not all(_valid_hash(m[key]) for key in ("config_hash", "visual_config_hash")):
        raise ValueError("unknown schema or missing episode/visual identity")
    bodies, geoms, joints = m["body_names"], m["geom_bodies"], m["joint_names"]
    for names in (bodies, m["geom_names"], joints):
        if not isinstance(names, list) or len(set(names)) != len(names) or not all(isinstance(s, str) and s for s in names):
            raise ValueError("invalid names")
    if len(geoms) != len(m["geom_names"]) or not all(isinstance(s, str) and s for s in geoms):
        raise ValueError("invalid contact body mapping")
    declared = set(policy.task_objects) | set(policy.distractors) | {g.reference for g in policy.goals if g.reference}
    supports = {name for _, names in policy.support_bodies for name in names}
    if not declared <= set(bodies) or not supports <= set(bodies) | set(geoms):
        raise ValueError("policy references missing body")
    allowance = policy.grasp_contact
    if allowance is not None and not set(allowance.robot_bodies) | set(allowance.surfaces) <= set(geoms):
        raise ValueError("grasp contact allowance references missing body")
    n, b, j = len(a["time"]), len(bodies), len(joints)
    if n < 2 or not j or not b:
        raise ValueError("empty trajectory")
    shapes = {"time": (n,), "contact_time": (n,), "frame": (n,), "frame_end": (n,),
              "success": (n,), "grasp": (n, b), "body_pose": (n, b, 7), "body_velocity": (n, b, 6)}
    shapes.update({key: (n, j) for key in ("joint_qpos", "joint_qvel", "joint_target", "actuator_force")})
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f"invalid {key} shape")
    for key in ("qpos", "qvel", "qacc", "warnings"):
        if a[key].ndim != 2 or a[key].shape[0] != n or a[key].shape[1] == 0:
            raise ValueError(f"invalid {key} shape")
    if a["qacc"].shape != a["qvel"].shape:
        raise ValueError("qacc/qvel mismatch")
    if np.any(a["warnings"] < 0):
        raise ValueError("negative warning counters")
    steps, dt = m["substeps"], m["timestep"]
    if type(steps) is not int or steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid sampling declaration")
    if not np.isfinite(a["time"]).all() or not np.allclose(np.diff(a["time"]), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("missing, duplicate or nonfinite timestamps")
    if not np.allclose(a["contact_time"][1:], a["time"][1:] - dt, rtol=1e-9, atol=1e-12):
        raise ValueError("incorrect solver timestamp")
    if not np.array_equal(a["frame"][1:], np.arange(n - 1) // steps):
        raise ValueError("missing control frames")
    if not np.array_equal(a["frame_end"], (np.arange(n) % steps == 0) & (np.arange(n) > 0)):
        raise ValueError("incorrect frame boundaries")
    offsets = a["contact_offsets"]
    if (offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(offsets < 0)
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("invalid contact offsets")
    c = int(offsets[-1])
    for key, shape in (("contact_geom", (c, 2)), ("contact_distance", (c,)), ("contact_force", (c, 6)),
                       ("contact_normal", (c, 3)), ("contact_position", (c, 3))):
        if a[key].shape != shape:
            raise ValueError(f"invalid {key} shape")
    if np.any(a["contact_geom"] < 0) or np.any(a["contact_geom"] >= len(geoms)):
        raise ValueError("contact references absent geom")
    ranges = np.asarray(m["joint_ranges"])
    if ranges.dtype.kind not in "if" or ranges.dtype.itemsize > 8:
        raise ValueError("joint ranges must contain numeric values")
    ranges = ranges.astype(np.float64)
    if ranges.shape != (j, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("invalid joint ranges")
    norms = np.hypot.reduce(a["body_pose"][:, :, 3:], axis=2)
    if np.any(~np.isfinite(norms) | (np.abs(norms - 1) > 1e-3)):
        raise ValueError("invalid object quaternion")
    normals = np.hypot.reduce(a["contact_normal"], axis=1)
    if np.any(~np.isfinite(normals) | (np.abs(normals - 1) > 1e-6)):
        raise ValueError("invalid contact normal")
    return a, ranges


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def audit_trajectory(telemetry: Telemetry, policy: PhysicsPolicy) -> PhysicsReport:
    """Audit every substep, then require a fully supported final success suffix.

    Contact violations carry both integrated timestamp and solver timestamp.
    No contact filtering, numerical replacement, or missing-data acceptance is
    performed. Corrupt telemetry produces a rejection instead of crashing QA.
    """
    checks, maxima, violations = {}, {}, []
    config_hash = telemetry.manifest.get("config_hash", "") if isinstance(telemetry.manifest, dict) else ""
    if not _valid_hash(config_hash):
        config_hash = ""

    def finish():
        return PhysicsReport(bool(checks) and all(checks.values()), checks, maxima, tuple(violations), config_hash)

    try:
        a, ranges = _integrity(telemetry, policy)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        checks["telemetry_integrity"] = False
        violations.append(dict(check="telemetry_integrity", timestamp=None, reason=str(exc)))
        return finish()
    checks["telemetry_integrity"] = True
    m = telemetry.manifest
    n = len(a["time"])
    contact_rows = np.repeat(np.arange(n), np.diff(a["contact_offsets"]))

    def fail(check, row, **detail):
        checks[check] = False
        violations.append(dict(check=check, timestamp=float(a["time"][row]), **detail))

    checks["finite"] = True
    for key, values in a.items():
        bad = np.argwhere(~np.isfinite(values))
        for index in bad:
            row = int(contact_rows[index[0]]) if key in ("contact_geom", "contact_distance", "contact_force", "contact_normal", "contact_position") else int(index[0])
            fail("finite", row, field=key, index=index.tolist())
    if not checks["finite"]:
        return finish()

    def bounded(check, values, limit, rows=None):
        maximum = float(np.max(values)) if values.size else 0.0
        maxima[check] = maximum if np.isfinite(maximum) else None
        checks[check] = True
        for index in np.argwhere((values > limit) | ~np.isfinite(values)):
            row = int(index[0]) if rows is None else int(rows[index[0]])
            measured = float(values[tuple(index)])
            detail = dict(measured=measured if np.isfinite(measured) else None, limit=float(limit), index=index.tolist())
            if check == "penetration":
                detail["solver_timestamp"] = float(a["contact_time"][row])
            fail(check, row, **detail)

    checks["joint_ranges"] = True
    maxima["joint_range_excess"] = 0.0
    for field in ("joint_qpos", "joint_target"):
        excess = np.maximum(ranges[:, 0] - a[field], a[field] - ranges[:, 1])
        nonfinite = ~np.isfinite(excess)
        if nonfinite.any():
            maxima["joint_range_excess"] = None
        elif maxima["joint_range_excess"] is not None:
            maxima["joint_range_excess"] = max(maxima["joint_range_excess"], float(max(0, excess.max())))
        for row, joint in np.argwhere((excess > policy.joint_tolerance) | nonfinite):
            detail = {"reason": "nonfinite derived joint range excess"} if nonfinite[row, joint] else {}
            fail("joint_ranges", row, field=field, joint=m["joint_names"][joint], measured=float(a[field][row, joint]),
                 range=ranges[joint].tolist(), tolerance=policy.joint_tolerance, **detail)

    velocity = a["body_velocity"]
    linear, angular = np.linalg.norm(velocity[:, :, :3], axis=2), np.linalg.norm(velocity[:, :, 3:], axis=2)
    bounded("joint_speed", np.abs(a["joint_qvel"]), policy.max_joint_speed)
    bounded("linear_speed", linear, policy.max_linear_speed)
    bounded("angular_speed", angular, policy.max_angular_speed)
    dt = np.diff(a["time"])
    bounded("joint_acceleration", np.abs(np.diff(a["joint_qvel"], axis=0) / dt[:, None]),
            policy.max_joint_acceleration, np.arange(1, n))
    for label, sl, limit in (("linear", slice(0, 3), policy.max_linear_acceleration),
                             ("angular", slice(3, 6), policy.max_angular_acceleration)):
        acceleration = np.linalg.norm(np.diff(velocity[:, :, sl], axis=0) / dt[:, None, None], axis=2)
        bounded(f"{label}_acceleration", acceleration, limit, np.arange(1, n))
    bounded("simulator_warnings", a["warnings"], 0)
    bounded("penetration", np.maximum(0, -a["contact_distance"]), policy.penetration_tolerance, contact_rows)

    body_index = {body: i for i, body in enumerate(m["body_names"])}
    displacement, rotation = np.zeros((n, len(policy.distractors))), np.zeros((n, len(policy.distractors)))
    for k, body in enumerate(policy.distractors):
        pose = a["body_pose"][:, body_index[body]]
        displacement[:, k] = np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1)
        rotation[:, k] = 2 * np.arccos(np.clip(np.abs(pose[:, 3:] @ pose[0, 3:]), 0, 1))
    bounded("distractor_displacement", displacement, policy.distractor_displacement_tolerance)
    bounded("distractor_rotation", rotation, policy.distractor_rotation_tolerance)

    allowed = {frozenset(pair) for pair in policy.allowed_contacts}
    checks["allowed_contacts"] = True
    supported = np.zeros((n, len(policy.task_objects)), dtype=bool)
    support_rules = dict(policy.support_bodies)
    allowance = policy.grasp_contact
    task_xy = a["body_pose"][:, [body_index[b] for b in policy.task_objects], :2]
    if allowance is not None:
        checks["grasp_contact"] = True
        jaws, surfaces = set(allowance.robot_bodies), set(allowance.surfaces)
        grasp_rows = np.zeros(n, dtype=bool)
        grasp_force, grasp_depth, grasp_samples = 0.0, 0.0, 0
    for contact, (g1, g2) in enumerate(a["contact_geom"]):
        pair = (m["geom_bodies"][g1], m["geom_bodies"][g2])
        row = int(contact_rows[contact])
        detail = dict(bodies=list(pair), geoms=[m["geom_names"][g1], m["geom_names"][g2]],
                      solver_timestamp=float(a["contact_time"][row]), distance=float(a["contact_distance"][contact]),
                      force=a["contact_force"][contact].tolist())
        if (frozenset(pair) not in allowed and allowance is not None
                and len(set(pair) & jaws) == 1 and len(set(pair) & surfaces) == 1):
            force, depth = float(a["contact_force"][contact, 0]), float(max(0.0, -a["contact_distance"][contact]))
            nearest = float(np.min(np.linalg.norm(task_xy[row] - a["contact_position"][contact, :2], axis=1)))
            grasp_rows[row] = True
            grasp_samples += 1
            grasp_force, grasp_depth = max(grasp_force, force), max(grasp_depth, depth)
            reasons = [reason for reason, bad in (
                ("normal_force", not force <= allowance.max_normal_force),
                ("penetration", not depth <= allowance.max_penetration),
                ("task_object_distance", not nearest <= allowance.proximity)) if bad]
            if reasons:
                fail("grasp_contact", row, reasons=reasons, nearest_task_object=nearest, **detail)
        elif frozenset(pair) not in allowed:
            fail("allowed_contacts", row, **detail)
        if a["contact_distance"][contact] <= policy.support_distance_tolerance and a["contact_force"][contact, 0] > policy.min_support_force:
            for k, body in enumerate(policy.task_objects):
                upward_cos = a["contact_normal"][contact, 2] * (-1 if pair[0] == body else 1)
                if upward_cos >= policy.min_support_upward_cos and any(frozenset(pair) == frozenset((body, support)) for support in support_rules[body]):
                    supported[row, k] = True

    if allowance is not None:
        # Continuous runs of allowed jaw/surface contact, measured in substeps.
        longest, start = 0, None
        edges = np.diff(np.r_[0, grasp_rows.astype(np.int8), 0])
        for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            seconds = (end - begin) * float(m["timestep"])
            longest = max(longest, seconds)
            if seconds > allowance.max_contact_seconds + 1e-12:
                fail("grasp_contact", int(begin), reasons=["duration"], measured=seconds,
                     limit=allowance.max_contact_seconds)
        maxima.update(grasp_contact_force=grasp_force, grasp_contact_penetration=grasp_depth,
                      grasp_contact_seconds=longest, grasp_contact_samples=grasp_samples)

    task_indices = [body_index[body] for body in policy.task_objects]
    released = ~a["grasp"][:, task_indices].any(axis=1)
    still = (linear[:, task_indices] <= policy.settled_linear_speed).all(axis=1) & (angular[:, task_indices] <= policy.settled_angular_speed).all(axis=1)
    goal_ok = np.ones(n, dtype=bool)
    for goal in policy.goals:
        pose = a["body_pose"][:, body_index[goal.body]]
        target = np.asarray(goal.offset)
        if goal.reference:
            # Offsets are in world axes, relative to the support's origin.
            target = a["body_pose"][:, body_index[goal.reference], :3] + target
        error = np.abs(pose[:, :3] - target)
        maximum = float(error.max())
        maxima[f"goal_position_error:{goal.body}"] = maximum if np.isfinite(maximum) else None
        goal_ok &= (error <= np.asarray(goal.position_tolerance)).all(axis=1)
        if goal.upright_cos is not None:
            upright = 1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2)
            goal_ok &= upright >= goal.upright_cos

    # All substeps within each frame must pass. An incomplete final frame may
    # never borrow the preceding success suffix to satisfy completion.
    settled = a["success"] & released & supported.all(axis=1) & still & goal_ok
    ends = np.flatnonzero(a["frame_end"])
    frames = [bool(settled[end - m["substeps"] + 1:end + 1].all()) for end in ends]
    suffix = 0
    if ends.size and ends[-1] == n - 1:
        for ok in reversed(frames):
            if not ok: break
            suffix += 1
    maxima["settled_success_frames"] = suffix
    checks["settled_success"] = suffix >= policy.settled_frames
    if not checks["settled_success"]:
        fail("settled_success", n - 1, measured=suffix, required=policy.settled_frames)
    final_start = max(1, n - policy.settled_frames * m["substeps"])
    for check, flags in (("support", supported.all(axis=1)), ("release", released),
                         ("settled_speed", still), ("goal", goal_ok)):
        checks[check] = True
        for row in np.flatnonzero(~flags[final_start:]) + final_start:
            fail(check, int(row))
    return finish()

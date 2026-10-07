"""Task-authoring kit: training env/oracle base classes shared by every family.

A family module (``sim/train/tasks/families/<family>.py``) subclasses
``TrainEnv`` and ``TrainOracle``, declares goals with ``Goal`` during
``layout()``, and exports ``TASKS = [define_task(...), ...]``. The kit derives
the success predicate, the strict physics policy (including the bounded
grasp-contact allowance) and the visibility goal regions from those goals, so
the three can never disagree. See sim/train/README.md.

Validation code is reused, never changed in behaviour: ``TrainEnv`` only
overrides class attributes/hooks of ``ValEnv``; ``TrainOracle`` overrides
``Oracle`` skills in this subclass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from sim.train.physics import GoalRule, GraspContactAllowance, PhysicsPolicy
from sim.train.variation import GoalRegion
from sim.val.env import PARK_X, ValEnv
from sim.val.oracle import (CARRY_Z, CLOSED, FIXED_FACE, MJ_PEAK, OPEN, RELEASE, Oracle, interp_rot, min_jerk,
                            tcp_path, top_down_mat)
from sim.val.scene import CATALOG, Block, Disc, Fixture, Obj, SceneSpec, arena_base  # noqa: F401  (re-exported for families)
from .assets import INVENTORY, Scanned  # noqa: F401  (re-exported for families)
from .schema import OrderedCompletion

# ----- robot constants ---------------------------------------------------------------------------------------

# Validation REST_DEG (0, -99, 95, 64, 0, -9) commands the fixed jaw 8.3 mm into the table, so the
# PD controller presses the folded gripper onto it (17-53 N measured at rest, 2026-10-06) and,
# with +-0.02 rad reset noise, can touch the shoulder. This pose keeps >= 9.2 mm table clearance and
# >= 4.9 mm wrist/shoulder clearance across the reset noise (200 sampled resets).
TRAIN_REST_DEG = (0.0, -99.0, 90.0, 60.0, 0.0, -9.0)
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
TABLE_BODIES = {"kitchen": "table", "living_room": "living_room_table_col"}
COS10 = float(np.cos(np.radians(10)))

# Measured bounded jaw/surface contact allowance; calibration evidence in sim/train/README.md.
GRASP_CONTACT_LIMITS = dict(max_normal_force=6.0, max_penetration=0.0005, max_contact_seconds=1.5, proximity=0.05)

# Reachable workspace (robot base frame, polar about the base). Measured worst-case tilt of a
# top-down TCP (deg) over azimuths -60/0/60:            r = 0.12 0.18 0.22 0.25 0.27 0.29 0.31
#   z 0.015 m:  0  0  0  0  0  4 14     z 0.06 m: 0 0 0 1 5 9 21     z 0.10 m (carry): 5 3 5 9 11 14 18
# Grasp and release poses (z <= 3 cm) are vertical out to r = 0.27; keep objects and targets inside it.
GRASP_REGION = dict(r=(0.14, 0.27), angle=(-65.0, 65.0))
PLACE_REGION = dict(r=(0.13, 0.27), angle=(-70.0, 70.0))
# Keep-outs: robot base, and the hovering folded gripper/wrist-camera mount at rest.
TRAIN_KEEPOUT = ((np.array([0.0, 0.0]), 0.10), (np.array([0.16, 0.0]), 0.06))

# Spatial words in instructions follow the front (viewer) camera, which looks along about -x:
# the viewer's right is world +y, left is -y, "in front"/nearer the viewer is +x (away from the robot)
# and "behind"/farther from the viewer is -x (toward the robot). Never use the robot's own left/right.
VIEW = {"right": np.array([0.0, 1.0]), "left": np.array([0.0, -1.0]),
        "front": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0])}

# Catalog objects no taller than 4 cm at their default 0.5x scale: a carried object clears
# them at the default carry height, so they are safe distractors anywhere on the table.
LOW_DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding",
                   "popcorn", "ramekin", "white_bowl", "red_bowl", "akita_black_bowl", "plate")


# Colour words: an instruction naming any of them keeps every primitive's authored colour.
COLOUR_WORDS = frozenset(("red", "green", "blue", "yellow", "orange", "purple", "violet", "pink", "black", "white",
                          "brown", "grey", "gray", "silver", "gold", "golden", "cyan", "teal", "beige", "tan",
                          "magenta", "maroon", "navy", "turquoise", "colour", "color", "coloured", "colored"))


def mat(name, half=(0.035, 0.035), thickness=0.006, rgba=(0.45, 0.45, 0.5, 1.0), mass=0.05):
    """A thin square mat/coaster/pad (free box). Prefer this over the cylindrical ``Disc``: MuJoCo's
    box-on-thin-cylinder contacts let blocks sink 4-45 mm into a 6 mm disc (measured 2026-10-06)."""
    return Block(name, half=(half[0], half[1], thickness / 2), rgba=rgba, mass=mass, friction=1.0)


# ----- goals -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    """Terminal condition for one task object.

    support: body that must carry the object at the end (``"table"`` means the arena table).
    target:  world xyz of the object's body origin at the goal (``None``: anywhere on the support).
    reference: when set, ``target`` follows this body (offset frozen at ``set_goals`` time).
    tolerance: per-axis |error| bound for ``target``.
    upright_cos: minimum cos of the object's tilt from vertical (``None``: unchecked).
    check: optional extra predicate ``check(env, obj) -> bool`` (orientation, region, ...).
    """
    obj: str
    support: str
    target: tuple[float, float, float] | None = None
    reference: str | None = None
    tolerance: tuple[float, float, float] = (0.012, 0.012, 0.006)
    upright_cos: float | None = COS10
    check: object = None


# ----- environment -------------------------------------------------------------------------------------------

class TrainEnv(ValEnv):
    """Base training environment.

    Subclasses set ``instruction`` and ``task_objects``, implement ``scene_objects()`` (and optionally
    ``scene_fixtures()``) and ``layout()``. ``layout()`` must place every task object/fixture with
    ``self.np_random``, call ``self.place_distractors(placed)`` and finish with ``self.set_goals(...)``.
    """

    rest_deg = TRAIN_REST_DEG
    keepout = TRAIN_KEEPOUT
    n_distractors = (2, 4)
    distractor_region = dict(r=(0.12, 0.32), angle=(-70, 70))
    distractor_pool: tuple[str, ...] = LOW_DISTRACTORS
    task_objects: tuple[str, ...] = ()
    order: tuple[str, ...] | None = None      # required temporal grasp/release order, if any
    surfaces: tuple[str, ...] = ()            # extra bodies the jaws may brush (besides goal supports)
    extra_contacts: tuple[tuple[str, str], ...] = ()  # extra allowed body pairs (e.g. stacked objects)
    arena = "kitchen"                         # default; the episode's VisualConfig decides the arena

    # ----- scene ---------------------------------------------------------------------------------------------
    def scene_objects(self) -> list:
        raise NotImplementedError

    def scene_fixtures(self) -> list:
        return []

    def __init__(self, *args, visual_config=None, **kwargs):
        # make_scene() runs inside ValEnv.__init__ and needs the episode's visual configuration (extra
        # distractors, primitive colours) before ValEnv stores it.
        self._scene_visual_config = visual_config
        super().__init__(*args, visual_config=visual_config, **kwargs)

    def make_scene(self) -> SceneSpec:
        from .assets import EXTRA_DISTRACTORS
        config = getattr(self, "_scene_visual_config", None)
        objects = self._recolour_primitives(list(self.scene_objects()), config)
        fixtures = list(self.scene_fixtures())
        used = {o.name for o in objects} | {f.name for f in fixtures}
        distractors = [Obj(n, n) for n in self.distractor_pool if n not in used]
        if config is not None and config.distractor_extras and max(self.n_distractors) > 0:
            distractors += [EXTRA_DISTRACTORS[n][0]() for n in config.distractor_extras
                            if n not in used and n not in self.distractor_pool]
        self.extra_distractor_names = tuple(d.name for d in distractors if d.name not in self.distractor_pool)
        return SceneSpec(arena=self.arena, objects=objects, fixtures=fixtures, distractors=distractors)

    def _recolour_primitives(self, objects, config):
        """Per-episode colours for primitive blocks/mats when nothing in the task names a colour.

        Only ``Block``/``Disc`` specs whose own name carries no colour word are recoloured (palette colours in
        object order); scanned and LIBERO meshes keep their textures."""
        if config is None or not config.primitive_palette:
            return objects
        words = set(re.findall(r"[a-z]+", self.instruction.lower()))
        if words & COLOUR_WORDS:   # a named colour pins every primitive's colour
            return objects
        out, k = [], 0
        for spec in objects:
            if (isinstance(spec, (Block, Disc))
                    and not set(spec.name.lower().split("_")) & COLOUR_WORDS):
                colour = config.primitive_palette[k % len(config.primitive_palette)]
                spec = replace(spec, rgba=(*colour, spec.rgba[3]))
                k += 1
            out.append(spec)
        return out

    def place_distractors(self, placed):
        """As ValEnv, but per-episode extra distractors come first three times as often as the family pool, so
        the added clutter actually shows up (2-4 active distractors per episode)."""
        k = int(self.np_random.integers(self.n_distractors[0], self.n_distractors[1] + 1))
        pool = list(self.distractor_names)
        extras = set(getattr(self, "extra_distractor_names", ()))
        keys = self.np_random.random(len(pool)) ** np.array([1 / 3 if n in extras else 1.0 for n in pool])
        self.active_distractors = []
        for i in np.argsort(-keys, kind="stable"):
            n = pool[i]
            if len(self.active_distractors) < k:
                try:
                    xy = self.sample_xy(self.footprint(n), placed, **self.distractor_region, clearance=0.03, tries=200)
                except RuntimeError:
                    xy = None
                if xy is not None:
                    placed.append((xy, self.footprint(n)))
                    self.set_object_pose(n, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
                    self.active_distractors.append(n)
                    continue
            self.set_object_pose(n, (PARK_X + i, 0.0), z=self._floor_z)
        return placed

    @property
    def table_body(self) -> str:
        return TABLE_BODIES[arena_base(self.scene.arena)]

    @property
    def floor_body(self) -> str:
        """Body under the off-table park area (parked distractors rest on it)."""
        if not hasattr(self, "_floor_body"):
            # The arena's (finite) floor plane; parking puts distractors on it far off the table.
            self._floor_body = self.model.body(int(self.model.geom_bodyid[self.model.geom("floor").id])).name
        return self._floor_body

    def _body_name(self, name):
        return self.table_body if name == "table" else name

    # ----- layout helpers ------------------------------------------------------------------------------------
    def reset(self, *args, **kwargs):
        self._goals = ()
        self.completion = OrderedCompletion(self.order or self.task_objects)
        return super().reset(*args, **kwargs)

    def set_object_pose(self, name, xy, yaw=0.0, z=None, quat=None):
        """As ValEnv, then refresh kinematics so ``object_pos`` is current inside ``layout()``."""
        super().set_object_pose(name, xy, yaw=yaw, z=z, quat=quat)
        mujoco.mj_kinematics(self.model, self.data)

    def set_fixture_pose(self, name, xy, yaw=0.0, z=0.0):
        super().set_fixture_pose(name, xy, yaw=yaw, z=z)
        mujoco.mj_kinematics(self.model, self.data)

    def place(self, name, placed, region=GRASP_REGION, yaw=None, clearance=0.03, z=None):
        """Sample a collision-free xy in ``region`` for free body ``name``; append it to ``placed``."""
        xy = self.sample_xy(self.footprint(name), placed, clearance=clearance, **region)
        yaw = self.np_random.uniform(-np.pi, np.pi) if yaw is None else yaw
        self.set_object_pose(name, xy, yaw=yaw, z=z)
        placed.append((xy, self.footprint(name)))
        return xy, yaw

    def place_fixture(self, name, placed, region=PLACE_REGION, yaw=None, clearance=0.03, z=0.0):
        xy = self.sample_xy(self.fixture_footprint(name), placed, clearance=clearance, **region)
        yaw = self.np_random.uniform(-np.pi, np.pi) if yaw is None else yaw
        self.set_fixture_pose(name, xy, yaw=yaw, z=z)
        placed.append((xy, self.fixture_footprint(name)))
        return xy, yaw

    def fixture_footprint(self, name) -> float:
        """xy radius of a fixture's collision geometry about its origin (any yaw)."""
        if not hasattr(self, "_fixture_extent"):
            self._fixture_extent = {}
        if name not in self._fixture_extent:
            m = self.model
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
            origin = d.xpos[self._body[name]]
            corners = []
            for g in self._geoms[name]:
                c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
                half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
                corners.extend([c[:2] + s * half[:2] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
            self._fixture_extent[name] = float(np.max(np.linalg.norm(np.asarray(corners) - origin[:2], axis=1)))
        return self._fixture_extent[name]

    def surface_z(self, xy, top=0.5, exclude=()):
        """Height of the first collision surface below (xy, top), ignoring robot and ``exclude`` bodies."""
        mujoco.mj_forward(self.model, self.data)
        geomid = np.zeros(1, dtype=np.int32)
        robot = self.model.body("base").id
        blocked = {g for n in exclude for g in self._geoms.get(n, ())}
        start = np.array([xy[0], xy[1], top])
        for _ in range(16):
            dist = mujoco.mj_ray(self.model, self.data, start, np.array([0, 0, -1.0]), None, 1, -1, geomid)
            if dist < 0:
                return None
            g = int(geomid[0])
            hit = start[2] - dist
            collides = self.model.geom_contype[g] or self.model.geom_conaffinity[g]
            if collides and g not in blocked and not self._in_subtree(self.model.geom_bodyid[g], robot):
                return float(hit)
            start = np.array([xy[0], xy[1], hit - 1e-5])
        return None

    def set_goals(self, *goals: Goal):
        """Freeze this episode's goals (call at the end of ``layout()``)."""
        frozen = []
        for goal in goals:
            if goal.obj not in self.task_objects:
                raise ValueError(f"goal for undeclared task object {goal.obj}")
            offset = None
            if goal.target is not None and goal.reference is not None:
                offset = tuple(float(x) for x in np.asarray(goal.target) - self.object_pos(goal.reference))
            frozen.append((goal, offset))
        if sorted(g.obj for g in goals) != sorted(self.task_objects):
            raise ValueError("declare exactly one goal per task object")
        self._goals = tuple(frozen)

    @property
    def goals(self) -> tuple[Goal, ...]:
        return tuple(g for g, _ in self._goals)

    def goal_target(self, goal) -> np.ndarray | None:
        for g, offset in self._goals:
            if g is goal:
                if g.target is None:
                    return None
                if offset is None:
                    return np.asarray(g.target, float)
                return self.object_pos(g.reference) + offset
        raise KeyError(goal)

    # ----- success helpers -----------------------------------------------------------------------------------
    def _owned_geoms(self, name):
        name = self._body_name(name)
        if name in self._geoms:
            return set(self._geoms[name])
        return {g for g in range(self.model.ngeom) if self.model.body(int(self.model.geom_bodyid[g])).name == name}

    def supported_by(self, name, support, min_upward_cos=0.5) -> bool:
        """Positive-force contact from ``support`` pushing ``name`` upward."""
        geoms, support_geoms = set(self._geoms[name]), self._owned_geoms(support)
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            forward = contact.geom1 in support_geoms and contact.geom2 in geoms
            reverse = contact.geom2 in support_geoms and contact.geom1 in geoms
            if (forward or reverse) and contact.dist <= 0.001:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if force[0] > 0 and contact.frame[2] * (1 if forward else -1) >= min_upward_cos:
                    return True
        return False

    def settled(self, name, linear=0.03, angular=0.5) -> bool:
        v = self.data.qvel[self._dadr[name]:self._dadr[name] + 6]
        return bool(np.linalg.norm(v[:3]) <= linear and np.linalg.norm(v[3:]) <= angular)

    def released(self, name) -> bool:
        return not self.is_grasping(name)

    def upright(self, name, cos=COS10) -> bool:
        return bool(self.data.xmat[self._body[name]][8] >= cos)

    def yaw(self, name) -> float:
        m = self.data.xmat[self._body[name]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def axis_alignment(self, name, axis_world, local_axis=0) -> float:
        """|cos| between the body's local axis and a world direction (1 = parallel, either sense)."""
        m = self.data.xmat[self._body[name]].reshape(3, 3)
        a = np.asarray(axis_world, float)
        return float(abs(m[:, local_axis] @ a) / np.linalg.norm(a))

    def in_region(self, name, center, half_extents) -> bool:
        return bool(np.all(np.abs(self.object_pos(name)[:2] - np.asarray(center)[:2]) <= np.asarray(half_extents)))

    def within_radius(self, name, center, radius) -> bool:
        return bool(np.linalg.norm(self.object_pos(name)[:2] - np.asarray(center)[:2]) <= radius)

    def stacked(self, top, bottom, xy_tolerance=0.012) -> bool:
        return self.supported_by(top, bottom) and self.within_radius(top, self.object_pos(bottom), xy_tolerance)

    def goal_holds(self, goal) -> bool:
        name = goal.obj
        target = self.goal_target(goal)
        if target is not None and np.any(np.abs(self.object_pos(name) - target) > np.asarray(goal.tolerance)):
            return False
        if goal.upright_cos is not None and not self.upright(name, goal.upright_cos):
            return False
        if goal.check is not None and not goal.check(self, name):
            return False
        return self.released(name) and self.settled(name) and self.supported_by(name, goal.support)

    def success(self) -> bool:
        if not self._goals:
            return False
        if not all(self.goal_holds(g) for g in self.goals):
            return False
        return self.order is None or self.completion.complete

    def step(self, action, **kwargs):
        result = super().step(action, **kwargs)
        if self.order is not None and self._goals:
            held = tuple(n for n in self.task_objects if self.is_grasping(n))
            placed = tuple(g.obj for g in self.goals if self.goal_holds(g))
            self.completion.observe(held=held, placed=placed)
        return result

    # ----- physics policy and screening ----------------------------------------------------------------------
    def physics_policy(self) -> PhysicsPolicy:
        """Strict policy from the declared goals; only measured jaw/surface contact is allowed."""
        table = self.table_body
        allowed = {(table, n) for n in self.free_names}
        floor = self.floor_body
        allowed.update((floor, n) for n in self.distractor_names if n not in self.active_distractors)
        supports, rules = [], []
        surfaces = {table, *map(self._body_name, self.surfaces)}
        for goal, offset in self._goals:
            support = self._body_name(goal.support)
            supports.append((goal.obj, (support,)))
            surfaces.add(support)
            allowed.update({(goal.obj, support), *((goal.obj, jaw) for jaw in JAW_BODIES)})
            for extra in self.surfaces:
                allowed.add((goal.obj, self._body_name(extra)))
            if goal.target is not None:
                rules.append(GoalRule(goal.obj, reference=goal.reference,
                                      offset=tuple(float(x) for x in (goal.target if offset is None else offset)),
                                      position_tolerance=tuple(goal.tolerance), upright_cos=goal.upright_cos))
        allowed.update((self._body_name(a), self._body_name(b)) for a, b in self.extra_contacts)
        surfaces -= set(self.task_objects)
        return PhysicsPolicy(
            task_objects=tuple(self.task_objects), support_bodies=tuple(supports),
            allowed_contacts=tuple(sorted(allowed)),
            distractors=tuple(n for n in self.free_names if n not in self.task_objects),
            goals=tuple(rules),
            grasp_contact=GraspContactAllowance(robot_bodies=JAW_BODIES, surfaces=tuple(sorted(surfaces)),
                                                **GRASP_CONTACT_LIMITS))

    def goal_regions(self) -> tuple[GoalRegion, ...]:
        regions = []
        for goal in self.goals:
            target = self.goal_target(goal)
            if target is None:
                center = self.object_pos(self._body_name(goal.support)) if goal.support != "table" else self.object_pos(goal.obj)
                z = self.surface_z(center[:2], exclude=self.task_objects) or 0.0
                center = np.r_[center[:2], z]
            else:
                center = np.r_[target[:2], target[2] - self._extent[goal.obj]["bottom"]]
            support = self._body_name(goal.support)
            body = None if support == self.table_body else support
            regions.append(GoalRegion(body, tuple(tuple(center + [x, y, 0.0]) for x in (-0.008, 0, 0.008)
                                                  for y in (-0.008, 0, 0.008))))
        return tuple(regions)


# ----- oracle ------------------------------------------------------------------------------------------------

class TrainOracle(Oracle):
    """Oracle with motion that stays inside the strict physics gate.

    Differences from the validation ``Oracle`` (validation behaviour is unchanged):
      * Cartesian moves sample the IK path densely, smooth it in joint space and spline it, so a joint
        saturating mid-path (wrist flex at +-95 deg while lifting) no longer stops in one frame; the start
        offset to the commanded joints is blended out with min-jerk instead of jumping.
      * Lower joint speed limits (``vmax``), because at 30 fps the zero-order-hold target staircase alone
        produces ~110 rad/s^2 per rad/s of joint speed in the PD response.
      * Gentle grasps: the fixed finger stops ``grasp_clearance`` from the object (so the closing jaw barely
        slides it), the jaw opens only as wide as needed, and closes over ``close_seconds``.
      * Releases back the fixed finger off the object before lifting, so it does not drag or tip it.
      * Rest is ``env.rest_deg`` (the table-clear training rest pose).
    """

    vmax = np.radians([60.0, 60.0, 60.0, 75.0, 75.0])
    path_samples = 48
    smooth_sigma = 3.0
    grasp_clearance = 0.0015
    open_margin = 0.012         # jaw gap beyond the object width when opening for a grasp
    release_margin = 0.008      # jaw gap beyond the object width when releasing (small: avoids rims/walls)
    close_seconds = 1.2
    release_seconds = 1.0
    release_tolerance = 0.0015
    release_lift_fraction = 0.5  # split a carried tilt between a slight press and a short fall
    max_cartesian_turn = np.radians(60)  # larger end-yaw changes carry in joint space (roll limit)
    release_backoff = 0.004
    carry_z = 0.08              # lower than validation (0.10): less carry tilt, so less pivot in the jaws

    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self._gap_table = None

    # ----- motion --------------------------------------------------------------------------------------------
    def follow(self, pose, seconds, vmax=None, smooth=True):
        vmax = self.vmax if vmax is None else vmax
        k = self.path_samples
        us = np.linspace(0.0, 1.0, k + 1)
        qs = [self.ik(*pose(0.0), seed=self.q)]
        for u in us[1:]:
            qs.append(self.ik(*pose(u), seed=qs[-1]))
        qs = np.asarray(qs)
        if smooth and self.smooth_sigma > 0:
            pad = 6
            ext = np.vstack([2 * qs[0] - qs[pad:0:-1], qs, 2 * qs[-1] - qs[-2:-pad - 2:-1]])
            sm = gaussian_filter1d(ext, self.smooth_sigma, axis=0, mode="nearest")[pad:-pad]
            sm[0], sm[-1] = qs[0], qs[-1]
            qs = sm
        spline = CubicSpline(us, qs, axis=0)
        offset = self.q - qs[0]
        dense = np.linspace(0, 1, 8 * k + 1)
        rate = np.abs(spline(dense, 1) - offset[None] * 30 * dense[:, None] ** 2 * (1 - dense[:, None]) ** 2).max(0)
        seconds = max(seconds, MJ_PEAK * float(np.max(rate / vmax)))
        n = max(int(np.ceil(seconds / self.dt)), 1)
        for i in range(1, n + 1):
            u = min_jerk(i / n)
            self.q = spline(u) + (1 - min_jerk(u)) * offset
            yield self._target()

    def move(self, pos, rot, speed=0.16, tol=0.004, settle=0.6, label="", arc=None, vmax=None, smooth=True):
        start, rot0 = self._cmd_pos.copy(), self._cmd_rot.copy()
        pos = np.asarray(pos, float)
        path, length = tcp_path(start, pos, arc)
        yield from self.follow(lambda u: (path(u), interp_rot(rot0, rot, u)), max(0.35, length / speed),
                               vmax, smooth)
        for _ in range(int(settle / self.dt)):
            if np.linalg.norm(self.tcp() - pos) < tol:
                break
            yield self._target()
        self._cmd_pos, self._cmd_rot = pos.copy(), rot.copy()
        self.log.append((label or "move", round(float(np.linalg.norm(self.tcp() - pos)) * 1000, 1)))

    def joint_move(self, q_target, seconds=1.5, grip=None):
        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        seconds = max(seconds, MJ_PEAK * float(np.max(np.abs(q_target - q0) / self.vmax)))
        n = max(int(np.ceil(seconds / self.dt - 1e-9)), 1)
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            self.q, self.grip = q0 + s * (q_target - q0), g0 + s * (g1 - g0)
            yield self._target()
        d = self.ik_data
        d.qpos[self.env._arm_qpos_addrs] = self.q
        mujoco.mj_kinematics(self.model, d)
        self._cmd_pos = d.site_xpos[self.env._tcp_site_id].copy()
        self._cmd_rot = d.site_xmat[self.env._tcp_site_id].reshape(3, 3).copy()

    def rest(self, seconds=1.8):
        rest = np.radians(np.array(self.env.rest_deg))
        if self.rest_unwind and abs(self.q[4] - rest[4]) > np.radians(20):
            q = self.q.copy()
            q[4] = rest[4]
            yield from self.joint_move(q, 0.6)
        yield from self.joint_move(rest[:5], seconds, grip=rest[5])
        yield from self.wait(0.5)

    def actions(self):
        """Joint targets with the last-resort per-frame speed guard at ``vmax`` (counted in ``limited``)."""
        self.start()
        prev, self.limited = self._target(), 0
        for target in self.plan():
            k = int(np.ceil(float(np.max(np.abs(target[:5] - prev[:5]) / (self.vmax * self.dt))) - 1e-9))
            for i in range(1, k):
                self.limited += 1
                yield prev + (i / k) * (target - prev)
            yield target
            prev = target

    # ----- gripper -------------------------------------------------------------------------------------------
    def jaw_gap(self, angle) -> float:
        """Fingertip gap (m) at gripper joint ``angle`` (rad), from forward kinematics."""
        if self._gap_table is None:
            m, d = self.model, mujoco.MjData(self.model)
            tips = [m.geom(f"{side}_jaw_sph_tip{i}").id for side in ("fixed", "moving") for i in (1, 2, 3)]
            angles = np.linspace(CLOSED, OPEN, 40)
            gaps = []
            for a in angles:
                d.qpos[:] = self.env.data.qpos
                d.qpos[self.env._qpos_addrs[5]] = a
                mujoco.mj_kinematics(m, d)
                p = d.geom_xpos[tips]
                gaps.append(float(np.linalg.norm(p[:3].mean(0) - p[3:].mean(0))))
            self._gap_table = (angles, np.asarray(gaps))
        return float(np.interp(angle, *self._gap_table))

    def open_for(self, width, margin=None) -> float:
        """Smallest gripper angle whose fingertip gap is ``width + margin`` (default ``open_margin``;
        capped at OPEN)."""
        self.jaw_gap(OPEN)
        angles, gaps = self._gap_table
        margin = self.open_margin if margin is None else margin
        return float(min(OPEN, np.interp(width + margin, gaps, angles)))

    def release_for(self, width) -> float:
        return self.open_for(width, self.release_margin)

    # ----- grasp and place -----------------------------------------------------------------------------------
    def grasp_plan(self, center, width, angles):
        best = None
        for a in angles:
            rot = top_down_mat(a)
            pos = center - (FIXED_FACE - width / 2 - self.grasp_clearance) * rot[:, 0]
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Top-down grasp of ``name`` (``width`` along the closing direction, TCP at ``grasp_z``), lifted to
        ``lift_z`` (default ``carry_z``). ``symmetric`` = number of equivalent closing directions (4 for a
        square footprint, 2 for an elongated one, grasped across its narrow side)."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.env.object_pos(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, self.close_seconds, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], lift_z]), rot, speed=0.12, label="lift")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def release(self, open_to=None):
        """Open, slide the fixed finger off the object's face, then the caller retreats."""
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        if self.release_backoff:
            yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.03, tol=0.002,
                                 settle=0.2, label="backoff", smooth=False)

    def place(self, xy, release_z, rot=None, open_to=None, carry_z=None):
        """Carry so the TCP is over ``xy`` (arc about the pan axis when needed), lower to ``release_z``,
        release, retreat upward."""
        carry_z = self.carry_z if carry_z is None else carry_z
        rot = self.carry_rot(xy) if rot is None else rot
        yield from self.move(np.array([xy[0], xy[1], carry_z]), rot, label="carry")
        yield from self.move(np.array([xy[0], xy[1], release_z]), rot, speed=0.06, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def carry_to(self, pos, rot, label="carry"):
        """Carry to ``pos``/``rot``: a Cartesian move (arcs about the pan axis), or a joint-space transit when
        the yaw change exceeds ``max_cartesian_turn`` (a geodesic Cartesian turn can drive the wrist roll
        into its +-157 deg limit mid-path even when the end pose is reachable)."""
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        turn = abs(np.angle(np.exp(1j * (delta - azimuth))))
        if turn <= self.max_cartesian_turn:
            yield from self.move(pos, rot, label=label)
        else:
            yield from self.transit(pos, rot, speed=0.8, label=label)

    def feasible_rotation(self, name, target, rotations, carry_z=None):
        """Among equivalent end rotations of a held object, the one whose carry and release poses the IK
        reaches best (wrist roll is limited to +-157 deg and the TCP is off the roll axis). Ties go to the
        first candidate, so list them by preference (e.g. least turn first)."""
        carry_z = self.carry_z if carry_z is None else carry_z
        held, best = self.held(name), None
        for rot in rotations:
            tcp = np.asarray(target, float) - rot @ held
            err = max(self.solve(np.r_[tcp[:2], carry_z], rot)[1], self.solve(tcp, rot)[1])
            if best is None or err < best[0] - 1e-4:
                best = (err, rot)
        return best[1]

    def _local_points(self, name):
        """Collision-geometry AABB corners of a free body in its own frame."""
        env, m, d = self.env, self.model, self.env.data
        body = env._body[name]
        R, p = d.xmat[body].reshape(3, 3), d.xpos[body]
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        pts = []
        for g in env._geoms[name]:
            world = (corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pts.append((world - p) @ R)
        return np.vstack(pts)

    def release_lift(self, name, rot) -> float:
        """Extra origin height needed so the held object's lowest corner, at the orientation it will have when
        the TCP reaches ``rot``, is no lower than when the object sits flat. Objects pivot a few degrees in
        the jaws while carried (measured up to 7 deg); releasing at the flat height would press a tilted
        object's low edge into its support (3 mm penetration measured)."""
        env = self.env
        R_now = env.data.xmat[env._body[name]].reshape(3, 3)
        R_pred = rot @ self.tcp_rot().T @ R_now
        local = self._local_points(name)
        bottom_tilted = -(local @ R_pred.T)[:, 2].min()
        yaw = np.arctan2(R_pred[1, 0], R_pred[0, 0])
        upright = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bottom_flat = -(local @ upright.T)[:, 2].min()
        return float(max(0.0, bottom_tilted - bottom_flat))

    def held(self, name):
        """Object position relative to the TCP, in the *actual* TCP frame (rigid while held). The arm tilts
        3-11 deg at carry height (IK cannot keep a vertical approach there), so the commanded frame would
        misplace the object by up to ~1 cm once lowered to a vertical release pose."""
        return self.tcp_rot().T @ (self.env.object_pos(name) - self.tcp())

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        """Place a held object so its body origin lands at world ``target`` (xyz), released ``drop`` above.
        The held offset is re-measured over the target after the carry, because objects can slip
        several mm in the jaws while being carried and turned (measured up to 8 mm)."""
        carry_z = self.carry_z if carry_z is None else carry_z
        target = np.asarray(target, float)
        rot = self.carry_rot(target[:2]) if rot is None else rot
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        tcp = target - rot @ self.held(name)
        if np.linalg.norm(tcp[:2] - self._cmd_pos[:2]) > 0.001:
            yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        # Settle tightly before opening: releasing while the arm still lags (the default 4 mm tolerance)
        # drops a tilted object a few mm onto an edge, which slaps flat with a >1000 rad/s^2 spike.
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + lift], rot, speed=0.06, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def pick_and_place(self, name, target, width, grasp_z=None, yaw=None, symmetric=4, rot=None, drop=0.0,
                       carry_z=None):
        """Generic top-down pick of ``name`` and placement of its origin at ``target``. Returns success."""
        if grasp_z is None:
            grasp_z = self.env.object_pos(name)[2] - 0.001
        if yaw is None:
            yaw = self.env.yaw(name)
        ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=symmetric, lift_z=carry_z)
        if not ok:
            return False
        yield from self.place_object(name, target, rot=rot, drop=drop, open_to=self.release_for(width),
                                     carry_z=carry_z)
        return True


# ----- task definition helper --------------------------------------------------------------------------------

def class_path(cls) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def define_task(*, name, instruction, family, env, oracle, objects, relation, goal, steps,
                object_kinds=None, order=None, arenas=("living_room", "kitchen"), **kwargs):
    """Build a ``TaskDefinition`` from classes.

    objects: task-object body names (must equal ``env.task_objects``).
    object_kinds: generic kind per object for semantic comparison (default: the body name).
    relation/goal: the semantic relation and goal (e.g. "inside", "basket").
    steps: ordered per-object action descriptions (one per object, in ``order``).
    order: temporal object order (default ``objects``); use ``("either",)``-style semantics only via
           ``semantic_order``.
    """
    from .schema import SemanticSignature, TaskDefinition
    order = tuple(order or objects)
    kinds = tuple(object_kinds or objects)
    if tuple(env.task_objects) != tuple(objects):
        raise ValueError(f"{name}: objects must equal {env.__name__}.task_objects")
    if env.instruction != instruction:
        raise ValueError(f"{name}: instruction must equal {env.__name__}.instruction")
    semantic_order = kwargs.pop("semantic_order", tuple(kinds[objects.index(o)] for o in order))
    return TaskDefinition(name, instruction, family,
                          SemanticSignature(family, kinds, relation, goal, tuple(semantic_order)),
                          tuple(objects), order, tuple(steps), class_path(env), class_path(oracle),
                          arenas=tuple(arenas), **kwargs)


def step_text(verb, obj, relation=None, target=None) -> str:
    """One action description, e.g. step_text("put", "red block", "inside", "basket")
    -> "Pick up the red block and put it inside the basket."."""
    tail = f" {relation} the {target}" if relation and target else ""
    return f"Pick up the {obj} and {verb} it{tail}."


def then(text: str) -> str:
    """Mark a later step in an ordered sequence: "Then pick up ..."."""
    return "Then " + text[0].lower() + text[1:]


def catalog_names() -> tuple[str, ...]:
    return tuple(CATALOG)



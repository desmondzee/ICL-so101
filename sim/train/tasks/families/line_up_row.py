"""line_up_row: place two or three objects into a straight row, in a stated order.

Left/right/front/behind follow the FRONT camera (the viewer): ``VIEW["right"]`` is world +y, "front" (nearer the
viewer) is +x. Each task fixes the row's anchor (a long mat the row lies on, a plate the row stands behind, or a
cube that heads the row), the row direction and the object order. The row direction is jittered per episode
(``axis_jitter``), and the row's position, the objects' starts, distractors, arena and lighting are randomized.

Strict success (every goal, at every substep of the final 30 frames):

* each object is within ``SLOT_TOL`` of its slot (slot ``i`` is ``i * spacing`` along the row from the first
  one), upright, released, settled and supported by the row's support (the mat, or the table);
* ``row_holds``: the objects' bounding-box centres are in the stated order along the row direction, the middle
  of three lies within ``LINE_TOL`` of the line through the outer two (every object within ``LINE_TOL`` of the
  row's axis), neighbours keep a clear gap of at least ``MIN_GAP`` along the row and do not touch, and the
  centre-to-centre spacings differ by at most ``SPACING_TOL``;
* the objects are placed in the stated temporal order (the kit's ``OrderedCompletion``);
* nothing else moved: the strict physics gate limits distractors, the mat and the anchor cube to 5 mm / 0.1 rad.

Oracle: one pick-and-place per object, with no rest in between. Grasps are centred on the collision bounding box
(LIBERO can origins sit ~2 mm off centre) with a staged close; at the release the jaws close *across* the row,
so the fingers open into free space in front of and behind the object instead of into the gaps between
neighbours (the SO-101 jaws are only +-16 mm wide along the row). Cans use the solid-box collision and the
touch-down release copied from ``unstack`` (LIBERO cans collide as thin plates that hook the fingertips).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.train.tasks.base import (COS10, LOW_DISTRACTORS, TRAIN_KEEPOUT, VIEW, Goal, TrainEnv, TrainOracle,
                                  define_task, mat, then)
from sim.val.oracle import CLOSED, RELEASE, top_down_mat
from sim.val.scene import Block, Fixture, Obj

FAMILY = "line_up_row"
SLOT_TOL = (0.013, 0.013, 0.006)
LINE_TOL = 0.009           # max distance of a centre from the row's line (m)
MIN_GAP = 0.007            # min clear gap between neighbours along the row (m)
SPACING_TOL = 0.012        # max difference between the two centre-to-centre spacings of a three-object row (m)
REACH = dict(r=(0.16, 0.262), angle=(-62.0, 62.0))   # slots: vertical grasp/release poses
START_REGION = dict(r=(0.16, 0.265), angle=(-62.0, 62.0))
REST_CLEAR = 0.006         # extra clearance of slots/objects from the folded-gripper keep-out

RED, GREEN, BLUE = (0.8, 0.12, 0.1, 1.0), (0.2, 0.6, 0.25, 1.0), (0.12, 0.28, 0.82, 1.0)
YELLOW, ORANGE = (0.92, 0.76, 0.12, 1.0), (0.95, 0.5, 0.1, 1.0)
GREY_MAT = (0.5, 0.5, 0.55, 1.0)
HALF = 0.014               # 2.8 cm cube
BAR = (0.024, 0.010, 0.014)  # 4.8 x 2.0 x 2.8 cm long block


# ----- geometry helpers (copied from place_between / unstack) ------------------------------------------------

def extent_along(env, name, u):
    """(min, max) of the collision-geometry AABB corners of ``name`` projected on the horizontal unit vector u."""
    m, d = env.model, env.data
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in env._geoms[name]:
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        pts.append((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ rot.T + d.geom_xpos[g])
    proj = np.vstack(pts)[:, :2] @ np.asarray(u, float)[:2]
    return float(proj.min()), float(proj.max())


def center(env, name):
    """World xy of the centre of ``name``'s collision bounding box."""
    return np.array([np.mean(extent_along(env, name, u)) for u in ((1.0, 0.0), (0.0, 1.0))])


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


def _solidify(env, name):
    """Replace the collision of free body ``name`` by one solid box over its collision AABB (model-only, at init).
    Copied from ``unstack``: LIBERO cans collide as a ring of 1-2 mm plates that the fingertip spheres hook into;
    a single solid box of the same extent grips and releases cleanly. Mass and inertia are unchanged."""
    m = env.model
    geoms = env._geoms[name]
    d = mujoco.MjData(m)
    a = env._qadr[name]
    d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    origin = d.xpos[env._body[name]]
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            v = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    pts = np.vstack(pts) - origin
    lo, hi = pts.min(0), pts.max(0)
    keep = max(geoms, key=lambda g: float(np.prod(m.geom_size[g])))
    parent = int(m.geom_bodyid[keep])
    R = d.xmat[parent].reshape(3, 3)
    centre = origin + 0.5 * (lo + hi)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.T.flatten())
    m.geom_type[keep] = mujoco.mjtGeom.mjGEOM_BOX
    m.geom_size[keep] = 0.5 * (hi - lo)
    m.geom_pos[keep] = R.T @ (centre - d.xpos[parent])
    m.geom_quat[keep] = quat
    m.geom_aabb[keep] = np.r_[np.zeros(3), 0.5 * (hi - lo)]
    m.geom_rbound[keep] = float(np.linalg.norm(0.5 * (hi - lo)))
    for g in geoms:
        if g != keep:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    env._geoms[name] = [keep]


@dataclass(frozen=True)
class Item:
    """How the oracle grasps one task object."""
    width: float                 # extent along the closing direction at the grasp
    symmetric: int = 4           # equivalent closing directions (4 cube, 2 bar/can-as-box)
    yaw_offset: float = 0.0      # closing direction relative to the body yaw
    can: bool = False            # touch-down release with a measured back-off (solid-box LIBERO can)
    long_along_row: bool = False  # bar: long axis along the row (so the jaws close across the row)


CUBE = Item(width=2 * HALF)
CAN = Item(width=0.031, symmetric=2, can=True)
LONG = Item(width=2 * BAR[1], symmetric=2, yaw_offset=np.pi / 2, long_along_row=True)


# ----- environment -------------------------------------------------------------------------------------------

class RowEnv(TrainEnv):
    """``task_objects`` in row order (first slot first); ``order`` is the same (placed first to last)."""

    items: dict = {}
    spacing: tuple = ()             # centre-to-centre distance between consecutive slots (len n-1)
    axis_jitter = 12.0              # deg of random row-direction jitter about the nominal direction
    solid = ()                      # LIBERO cans: solid-box collision
    support = "table"
    head = None                     # a non-task object that heads the row (slot -1), if any
    min_travel = 0.06               # each object starts at least this far from its slot

    def _index_bodies(self):
        super()._index_bodies()
        for name in self.solid:
            _solidify(self, name)
        self._extent = self._measure_extents()

    # nominal row direction (unit xy, first -> last) in world coordinates; subclasses override
    def nominal_axis(self):
        return VIEW["right"]

    def row_axis(self):
        a = np.arctan2(*self.nominal_axis()[::-1]) + np.radians(self.np_random.uniform(-self.axis_jitter,
                                                                                         self.axis_jitter))
        return np.array([np.cos(a), np.sin(a)])

    def slot_offsets(self):
        """Offsets of each task-object slot from the row origin (slot 0 at 0)."""
        return np.r_[0.0, np.cumsum(self.spacing)]

    def slot_ok(self, xy, name):
        r, ang = np.hypot(*xy), np.degrees(np.arctan2(xy[1], xy[0]))
        if not (REACH["r"][0] <= r <= REACH["r"][1] and REACH["angle"][0] <= ang <= REACH["angle"][1]):
            return False
        return all(np.linalg.norm(xy - c) >= q + self.footprint(name) + REST_CLEAR for c, q in TRAIN_KEEPOUT)

    def object_ok(self, name):
        """Extra per-object relation (subclasses): e.g. behind the plate, lying along the row."""
        return True

    def row_holds(self, env=None, obj=None):
        """Goal check for ``obj``: the row prefix up to ``obj`` (in row order, with the head) is in order,
        straight, evenly spaced and gapped, and ``obj`` satisfies ``object_ok``. The last object's check covers
        the whole row; checking prefixes lets the kit's ``OrderedCompletion`` count each object as placed."""
        u = self._row_u
        names = self.task_objects if self.head is None else (self.head, *self.task_objects)
        names = names[:names.index(obj) + 1] if obj is not None else names
        if obj is not None and not self.object_ok(obj):
            return False
        c = [center(self, n) for n in names]
        along = [float(p @ u) for p in c]
        if any(b <= a for a, b in zip(along, along[1:])):
            return False
        normal = np.array([-u[1], u[0]])
        if len(c) >= 3:
            ends = c[-1] - c[0]
            line_n = np.array([-ends[1], ends[0]]) / max(np.linalg.norm(ends), 1e-9)
            if any(abs(float((p - c[0]) @ line_n)) > LINE_TOL for p in c[1:-1]):
                return False
            if np.ptp(np.diff(along)) > SPACING_TOL:
                return False
        across = [float(p @ normal) for p in c]
        if np.ptp(across) > 2 * LINE_TOL:
            return False
        for a, b in zip(names, names[1:]):
            if extent_along(self, b, u)[0] - extent_along(self, a, u)[1] < MIN_GAP or self.touching(a, b):
                return False
        return True

    def place_starts(self, placed, slots):
        for name, slot in zip(self.task_objects, slots):
            for _ in range(60):
                trial = list(placed)
                xy, _ = self.place(name, trial, region=START_REGION)
                if np.linalg.norm(xy - slot) >= self.min_travel:
                    placed[:] = trial
                    break
            else:
                raise RuntimeError("no start position for " + name)

    def freeze_goals(self, slots, z_offsets):
        goals = []
        for name, slot, z in zip(self.task_objects, slots, z_offsets):
            origin = slot + (self.object_pos(name)[:2] - center(self, name))
            reference = None if self.support == "table" else self.support
            goals.append(Goal(name, self.support, target=(*origin, z), reference=reference, tolerance=SLOT_TOL,
                              upright_cos=COS10, check=self.row_holds))
        self.set_goals(*goals)


class RowOracle(TrainOracle):
    touch_depth = 0.001          # staged close (relative_left_right): ease onto the object, then squeeze
    touch_seconds = 0.8
    min_grasp_z = 0.0105         # the jaw hull reaches ~8.2 mm below the TCP
    can_release_margin = 0.020
    can_backoff = 0.006
    can_backoff_speed = 0.012
    can_release_seconds = 1.5

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Top-down pick centred on the collision bounding box, with a staged close."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = center(self.env, name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
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
        item = self._item
        rot = self._cmd_rot
        if not item.can:
            return (yield from super().release(open_to))
        env, name = self.env, self._name
        # Touch down first (<= 3 mm, 0.5 mm steps), so a can held a few degrees tilted is not dropped (unstack).
        support = env._owned_geoms(env.support)
        held = set(env._geoms[name])
        for _ in range(6):
            if any((c.geom1 in held and c.geom2 in support) or (c.geom2 in held and c.geom1 in support)
                   for c in env.data.contact[:env.data.ncon]):
                break
            yield from self.move(self._cmd_pos - [0, 0, 0.0005], rot, speed=0.01, tol=0.0005, settle=0.15,
                                 label="touch", smooth=False)
        yield from self.gripper(RELEASE if open_to is None else open_to, self.can_release_seconds, 0.2)
        tips = [self.model.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
        away = env.data.geom_xpos[tips].mean(0)[:2] - center(env, name)
        away = away / max(np.linalg.norm(away), 1e-6)
        yield from self.move(self._cmd_pos + self.can_backoff * np.r_[away, 0.0], rot, speed=self.can_backoff_speed,
                             tol=0.002, settle=0.2, label="backoff", smooth=False)

    def origin_target(self, name, slot_xy, z, rot):
        """Body-origin target that puts the bounding-box centre on ``slot_xy`` once the TCP turns to ``rot``."""
        env = self.env
        offset = np.r_[env.object_pos(name)[:2] - center(env, name), 0.0]
        return np.r_[slot_xy, z] + (rot @ self.tcp_rot().T) @ offset

    def across_rotations(self, u):
        """The two top-down TCP rotations closing across the row (perpendicular to u), least turn first."""
        across = np.arctan2(u[1], u[0]) + np.pi / 2
        current = np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        return [top_down_mat(t) for t in sorted((across, across + np.pi),
                                                  key=lambda t: abs(_wrap(t - current)))]

    def move_one(self, name, goal):
        env = self.env
        item = env.items[name]
        self._name, self._item = name, item
        grasp_z = max(env.object_pos(name)[2] - 0.001, self.min_grasp_z)
        if item.can:
            grasp_z += 0.003
        ok = yield from self.pick(name, item.width, grasp_z, yaw=env.yaw(name) + item.yaw_offset,
                                  symmetric=item.symmetric)
        if not ok:
            return False
        u = env._row_u
        slot = env._slots[env.task_objects.index(name)]
        z = env.goal_target(goal)[2]
        rots = self.across_rotations(u)
        best = None
        for k, rot in enumerate(rots):
            target = self.origin_target(name, slot, z, rot)
            tcp = target - rot @ self.held(name)
            err = max(self.solve(np.r_[tcp[:2], self.carry_z], rot)[1], self.solve(tcp, rot)[1])
            if best is None or err < best[0] - 1e-3:
                best = (err, rot)
        rot = best[1]
        target = self.origin_target(name, slot, z, rot)
        margin = self.can_release_margin if item.can else self.release_margin
        yield from self.place_object(name, target, rot=rot, open_to=self.open_for(item.width, margin))
        return True

    def plan(self):
        env = self.env
        goals = {g.obj: g for g in env.goals}
        for name in env.order:
            ok = yield from self.move_one(name, goals[name])
            if not ok:
                return
        yield from self.rest()
        yield from self.wait(1.2)


# ----- 1. three cubes left to right along a long mat -----------------------------------------------------------

class BlocksOnMatEnv(RowEnv):
    instruction = "Line up the red, green and blue blocks from left to right along the long mat."
    task_objects = ("red_block", "green_block", "blue_block")
    order = task_objects
    items = {n: CUBE for n in task_objects}
    spacing = (0.047, 0.047)
    support = "mat"
    mat_half = (0.088, 0.03)

    def scene_objects(self):
        return [Block("red_block", half=(HALF,) * 3, rgba=RED), Block("green_block", half=(HALF,) * 3, rgba=GREEN),
                Block("blue_block", half=(HALF,) * 3, rgba=BLUE),
                mat("mat", half=self.mat_half, rgba=GREY_MAT, mass=0.08)]

    def layout(self):
        offsets = self.slot_offsets() - self.slot_offsets()[-1] / 2      # centred on the mat
        for _ in range(400):
            u = self.row_axis()
            rr, aa = self.np_random.uniform(0.19, 0.255), np.radians(self.np_random.uniform(-45, 45))
            c = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            slots = [c + o * u for o in offsets]
            if not all(self.slot_ok(s, n) for s, n in zip(slots, self.task_objects)):
                continue
            normal = np.array([-u[1], u[0]])
            corners = [c + sx * self.mat_half[0] * u + sy * self.mat_half[1] * normal
                       for sx in (-1, 1) for sy in (-1, 1)]
            if all(np.linalg.norm(p - TRAIN_KEEPOUT[1][0]) >= TRAIN_KEEPOUT[1][1] for p in corners):
                break
        else:
            raise RuntimeError("no reachable row")
        self._row_u, self._slots = u, slots
        self.set_object_pose("mat", c, yaw=float(np.arctan2(u[1], u[0])))
        placed = [(c, self.footprint("mat"))]
        self.place_starts(placed, slots)
        self.place_distractors(placed)
        z = float(self.object_pos("mat")[2]) + 0.003 + HALF       # mat origin is its mid-plane
        self.freeze_goals(slots, [z] * 3)


# ----- 2. two cans left to right, in a row behind the plate --------------------------------------------------

class CansBehindPlateEnv(RowEnv):
    instruction = ("Put the soup can and then the tomato sauce can in a row behind the plate, "
                   "with the soup can on the left.")
    task_objects = ("soup_can", "sauce_can")
    order = task_objects
    items = {n: CAN for n in task_objects}
    solid = task_objects
    spacing = (0.052,)
    behind_gap = 0.02          # plate rim to can surface, toward the robot
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce", "plate"))

    def scene_objects(self):
        return [Obj("soup_can", "alphabet_soup"), Obj("sauce_can", "tomato_sauce")]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.6)]

    def layout(self):
        plate_r = self.fixture_footprint("plate") / np.sqrt(2)
        can_r = 0.0155
        for _ in range(600):
            u = self.row_axis()
            rr, aa = self.np_random.uniform(0.23, 0.31), np.radians(self.np_random.uniform(-55, 55))
            p = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if p[0] < 0.17:
                continue
            mid = p + (plate_r + self.behind_gap + can_r) * VIEW["back"]
            slots = [mid + o * u for o in (-self.spacing[0] / 2, self.spacing[0] / 2)]
            if all(self.slot_ok(s, n) for s, n in zip(slots, self.task_objects)):
                break
        else:
            raise RuntimeError("no reachable row")
        self._row_u, self._slots = u, slots
        self.set_fixture_pose("plate", p, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed = [(p, plate_r)] + [(s, 0.022) for s in slots]
        self.place_starts(placed, slots)
        self.place_distractors(placed)
        self.freeze_goals(slots, [self._extent[n]["bottom"] for n in self.task_objects])

    def object_ok(self, name):
        # behind the plate: the can wholly farther from the viewer than the plate, and not touching it
        plate_back = extent_along(self, "plate", VIEW["front"])[0]
        return extent_along(self, name, VIEW["front"])[1] <= plate_back - 0.004 and not self.touching(name, "plate")


class CanRowOracle(RowOracle):
    pass


# ----- 3. a can and a long block lined up behind the cube, going away from the viewer ------------------------

class RowBehindCubeEnv(RowEnv):
    instruction = "Line up the tomato sauce can and then the long block in a straight row behind the yellow cube."
    task_objects = ("sauce_can", "long_block")
    order = task_objects
    items = {"sauce_can": CAN, "long_block": LONG}
    solid = ("sauce_can",)
    head = "cube"
    spacing = (0.056,)            # can -> bar (bar long axis along the row)
    head_spacing = 0.046          # cube -> can
    axis_jitter = 10.0
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce"))

    def nominal_axis(self):
        return VIEW["back"]

    def scene_objects(self):
        return [Block("cube", half=(HALF,) * 3, rgba=YELLOW, mass=0.05), Obj("sauce_can", "tomato_sauce"),
                Block("long_block", half=BAR, rgba=ORANGE)]

    def layout(self):
        for _ in range(600):
            u = self.row_axis()
            rr, aa = self.np_random.uniform(0.24, 0.29), np.radians(self.np_random.uniform(-50, 50))
            head = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            slots = [head + self.head_spacing * u, head + (self.head_spacing + self.spacing[0]) * u]
            bar_end = slots[1] + BAR[0] * u
            if (all(self.slot_ok(s, n) for s, n in zip(slots, self.task_objects))
                    and np.linalg.norm(bar_end - TRAIN_KEEPOUT[1][0]) >= TRAIN_KEEPOUT[1][1] + REST_CLEAR + 0.01
                    and np.hypot(*bar_end) >= 0.15):
                break
        else:
            raise RuntimeError("no reachable row")
        self._row_u, self._slots = u, slots
        self.set_object_pose("cube", head, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed = [(head, self.footprint("cube"))] + [(s, 0.03) for s in slots]
        self.place_starts(placed, slots)
        self.place_distractors(placed)
        self.freeze_goals(slots, [self._extent["sauce_can"]["bottom"], BAR[2]])

    def object_ok(self, name):
        # the long block lies with its long side along the row (within 20 deg)
        return name != "long_block" or self.axis_alignment(name, np.r_[self._row_u, 0.0]) >= np.cos(np.radians(20))


TASKS = [
    define_task(name="line_up_three_blocks_on_mat", instruction=BlocksOnMatEnv.instruction, family=FAMILY,
                env=BlocksOnMatEnv, oracle=RowOracle, objects=BlocksOnMatEnv.task_objects,
                object_kinds=("block", "block", "block"), relation="in a row left to right in stated order on",
                goal="long mat",
                steps=("Pick up the red block and put it at the left end of the long mat.",
                       then("Pick up the green block and put it on the mat just to the right of the red block."),
                       then("Pick up the blue block and put it on the mat just to the right of the green block, "
                            "completing the row."))),
    define_task(name="line_up_cans_behind_plate", instruction=CansBehindPlateEnv.instruction, family=FAMILY,
                env=CansBehindPlateEnv, oracle=CanRowOracle, objects=CansBehindPlateEnv.task_objects,
                object_kinds=("can", "can"), relation="in a row left to right behind", goal="plate",
                steps=("Pick up the soup can and stand it on the table just behind the plate, on the left.",
                       then("Pick up the tomato sauce can and stand it behind the plate to the right of the soup "
                            "can, in a row with it."))),
    define_task(name="line_up_row_behind_cube", instruction=RowBehindCubeEnv.instruction, family=FAMILY,
                env=RowBehindCubeEnv, oracle=RowOracle, objects=RowBehindCubeEnv.task_objects,
                object_kinds=("can", "rectangular block"), relation="in a straight row going back from",
                goal="cube",
                steps=("Pick up the tomato sauce can and stand it on the table just behind the yellow cube.",
                       then("Pick up the long block and lay it lengthwise just behind the tomato sauce can, "
                            "continuing the straight row."))),
]

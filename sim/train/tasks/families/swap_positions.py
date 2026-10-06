"""swap_positions: swap the places of two objects (three moves: one object to a free spot, the other into its
place, then the first into the other's old place).

Each task names two objects A and B (A is the one set aside first). Their start places are randomized per
episode, as are a free spot for A (computed in ``layout()`` so that it exists: reachable, clear of every object,
the rest sweep and both start places), the distractors, arena and lighting.

Strict success (every goal at every substep of the final 30 frames):

* A's bounding-box centre is within ``SWAP_TOL`` of B's start centre and B's within ``SWAP_TOL`` of A's, both
  upright, released, settled and supported by the support of the place they moved into (table, mat or plate);
  the two objects do not touch;
* ``SwapCompletion``: the moves happen as a swap: A is picked and put down (set aside) before B is grasped, B is
  placed at its goal before A is grasped again, A then ends at its goal, and B is never grasped again. Any other
  grasp order is a permanent failure (like the kit's ``OrderedCompletion``, which counts one completion per
  object and so cannot express A's two moves);
* the strict physics gate: mats/plate/distractors move <= 5 mm, no object-object or robot contact.

The kit requires exactly one action text per task object, so the second text covers B's move and A's final move.

Oracle: bounding-box-centred top-down picks with a staged close (copied from ``relative_left_right``); LIBERO
cans use the solid-box collision and the touch-down release with a measured back-off (copied from ``unstack``).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.train.tasks.base import (COS10, LOW_DISTRACTORS, TRAIN_KEEPOUT, Goal, TrainEnv, TrainOracle,
                                  define_task, mat, then)
from sim.val.oracle import CLOSED, RELEASE
from sim.val.scene import Block, Fixture, Obj

FAMILY = "swap_positions"
SWAP_TOL = (0.014, 0.014, 0.006)
REACH = dict(r=(0.165, 0.258), angle=(-60.0, 60.0))
REST_CLEAR = 0.008
HALF = 0.014
RED, BLUE, GREEN, YELLOW = (0.8, 0.12, 0.1, 1.0), (0.12, 0.28, 0.82, 1.0), (0.2, 0.6, 0.25, 1.0), (0.92, 0.76, 0.12, 1.0)
PURPLE = (0.55, 0.25, 0.7, 1.0)


# ----- geometry helpers (copied from place_between / unstack) ------------------------------------------------

def extent_along(env, name, u):
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
    """World xy of the centre of ``name``'s collision bounding box (LIBERO can origins sit ~2 mm off it)."""
    return np.array([np.mean(extent_along(env, name, u)) for u in ((1.0, 0.0), (0.0, 1.0))])


def _solidify(env, name):
    """One solid box over the collision AABB of free body ``name`` (from ``unstack``: LIBERO cans collide as thin
    plates that hook the fingertips)."""
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


class SwapCompletion:
    """Temporal swap order for objects (a, b): a picked and set aside, then b to its goal, then a to its goal."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.state, self.violated = 0, False

    def observe(self, *, held, placed):
        a, b = self.a in held, self.b in held
        s = self.state
        if s == 0:
            if b:
                self.violated = True
            elif a:
                self.state = 1
        elif s == 1:                     # a in the hand, then set aside
            if b:
                self.violated = a or self.violated
                self.state = 2
        elif s == 2:                     # b moving into a's old place
            if a:
                self.violated = True
            elif self.b in placed and not b:
                self.state = 3
        elif s == 3:                     # b done; a must be fetched next
            if b:
                self.violated = True
            elif a:
                self.state = 4
        elif s == 4:
            if b:
                self.violated = True
            elif self.a in placed and not a:
                self.state = 5
        elif b or a:
            self.violated = True

    @property
    def completed(self):
        return {0: [], 1: [], 2: [self.a], 3: [self.a, self.b], 4: [self.a, self.b], 5: [self.a, self.b, self.a]}[self.state]

    @property
    def complete(self):
        return not self.violated and self.state == 5


@dataclass(frozen=True)
class Item:
    width: float
    symmetric: int = 4
    can: bool = False


CUBE = Item(width=2 * HALF)
CAN = Item(width=0.031, symmetric=2, can=True)


# ----- environment -------------------------------------------------------------------------------------------

class SwapEnv(TrainEnv):
    """``task_objects = (a, b)``: a is set aside first."""

    items: dict = {}
    solid = ()
    min_apart = 0.10               # start places at least this far apart (centres)
    aside_clear = 0.045            # free spot: this much clear space beyond the object's footprint

    def _index_bodies(self):
        super()._index_bodies()
        for name in self.solid:
            _solidify(self, name)
        self._extent = self._measure_extents()

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.completion = SwapCompletion(*self.task_objects)
        return result

    def reach_ok(self, xy, radius):
        r, ang = np.hypot(*xy), np.degrees(np.arctan2(xy[1], xy[0]))
        if not (REACH["r"][0] <= r <= REACH["r"][1] and REACH["angle"][0] <= ang <= REACH["angle"][1]):
            return False
        return all(np.linalg.norm(xy - c) >= q + radius + REST_CLEAR for c, q in TRAIN_KEEPOUT)

    def sample_spot(self, radius, placed, clearance):
        for _ in range(400):
            xy = self.sample_xy(radius, placed, clearance=clearance, **REACH)
            if self.reach_ok(xy, radius):
                return xy
        raise RuntimeError("no reachable spot")

    def choose_aside(self, placed):
        a = self.task_objects[0]
        xy = self.sample_spot(self.footprint(a), placed, self.aside_clear)
        self.aside = xy
        placed.append((xy, self.footprint(a) + 0.01))
        return xy

    def swapped(self, env=None, obj=None):
        a, b = self.task_objects
        return not self.touching(a, b)

    def support_top(self, support, xy):
        """Height of ``support``'s upper surface at ``xy`` (now, during layout)."""
        if support == "table":
            return 0.0
        if support in self.free_names:      # a thin mat: its origin is its mid-plane
            return float(self.object_pos(support)[2] + self._extent[support]["height"] / 2)
        return self.surface_z(xy, exclude=self.task_objects)

    def freeze_goals(self, supports, references=(None, None)):
        """Goals: a at b's start centre, b at a's start centre (start centres measured now)."""
        a, b = self.task_objects
        self.start_centre = {n: center(self, n) for n in (a, b)}
        goals = []
        for name, other, support, ref in ((a, b, supports[0], references[0]), (b, a, supports[1], references[1])):
            origin = self.start_centre[other] + (self.object_pos(name)[:2] - center(self, name))
            z = self.support_top(support, self.start_centre[other]) + self._extent[name]["bottom"]
            goals.append(Goal(name, support, target=(*origin, z), reference=ref, tolerance=SWAP_TOL,
                              upright_cos=COS10, check=self.swapped))
        self.set_goals(*goals)


class SwapOracle(TrainOracle):
    touch_depth = 0.001
    touch_seconds = 0.8
    min_grasp_z = 0.0105
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
        item, rot = self._item, self._cmd_rot
        if not item.can:
            return (yield from super().release(open_to))
        env, name = self.env, self._name
        support = env._owned_geoms(self._support)
        held = set(env._geoms[name])
        for _ in range(6):            # touch down first (unstack): a tilted can is not dropped onto an edge
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

    def move_to(self, name, centre_xy, z, support):
        """Pick ``name`` and put its bounding-box centre at ``centre_xy`` with its origin at height ``z``."""
        env = self.env
        item = env.items[name]
        self._name, self._item, self._support = name, item, support
        grasp_z = max(env.object_pos(name)[2] - 0.001, self.min_grasp_z)
        if item.can:
            grasp_z += 0.003
        ok = yield from self.pick(name, item.width, grasp_z, yaw=env.yaw(name), symmetric=item.symmetric)
        if not ok:
            return False
        rot = self.carry_rot(centre_xy)
        offset = np.r_[env.object_pos(name)[:2] - center(env, name), 0.0]
        target = np.r_[centre_xy, z] + (rot @ self.tcp_rot().T) @ offset
        margin = self.can_release_margin if item.can else self.release_margin
        yield from self.place_object(name, target, rot=rot, open_to=self.open_for(item.width, margin))
        return True

    def plan(self):
        env = self.env
        a, b = env.task_objects
        ga, gb = env.goals
        za = env.object_pos(a)[2] + self.aside_z_shift(a)
        if not (yield from self.move_to(a, env.aside, za, "table")):
            return
        if not (yield from self.move_to(b, env.start_centre[a], env.goal_target(gb)[2], gb.support)):
            return
        if not (yield from self.move_to(a, env.start_centre[b], env.goal_target(ga)[2], ga.support)):
            return
        yield from self.rest()
        yield from self.wait(1.2)

    def aside_z_shift(self, name):
        """Origin height change from a's start support to the table at the free spot."""
        env = self.env
        return env._extent[name]["bottom"] - env.object_pos(name)[2]


# ----- 1. cube and soup can on the table ---------------------------------------------------------------------

class CubeCanEnv(SwapEnv):
    instruction = "Swap the places of the red cube and the soup can."
    task_objects = ("cube", "can")
    order = task_objects
    items = {"cube": CUBE, "can": CAN}
    solid = ("can",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce"))

    def scene_objects(self):
        return [Block("cube", half=(HALF,) * 3, rgba=RED), Obj("can", "alphabet_soup")]

    def layout(self):
        placed = []
        for _ in range(100):
            trial = []
            pa = self.sample_spot(0.025, trial, 0.03)
            trial.append((pa, 0.025))
            pb = self.sample_spot(0.025, trial, 0.03)
            if np.linalg.norm(pa - pb) >= self.min_apart:
                break
        else:
            raise RuntimeError("no start places")
        for name, xy in (("cube", pa), ("can", pb)):
            self.set_object_pose(name, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
            self.set_object_pose(name, 2 * xy - center(self, name), yaw=self.yaw(name))
            placed.append((xy, self.footprint(name)))
        self.choose_aside(placed)
        self.place_distractors(placed)
        self.freeze_goals(("table", "table"))


# ----- 2. two blocks on two mats -----------------------------------------------------------------------------

class BlocksOnMatsEnv(SwapEnv):
    instruction = "Swap the blue block and the yellow block so each ends up on the other's mat."
    task_objects = ("blue_block", "yellow_block")
    order = task_objects
    items = {n: CUBE for n in task_objects}
    surfaces = ("green_mat", "purple_mat")
    mat_half = 0.032

    def scene_objects(self):
        return [Block("blue_block", half=(HALF,) * 3, rgba=BLUE), Block("yellow_block", half=(HALF,) * 3, rgba=YELLOW),
                mat("green_mat", half=(self.mat_half,) * 2, rgba=GREEN), mat("purple_mat", half=(self.mat_half,) * 2,
                                                                              rgba=PURPLE)]

    def layout(self):
        placed = []
        r = self.footprint("green_mat")
        for _ in range(100):
            trial = []
            pa = self.sample_spot(r, trial, 0.03)
            trial.append((pa, r))
            pb = self.sample_spot(r, trial, 0.03)
            if np.linalg.norm(pa - pb) >= self.min_apart:
                break
        else:
            raise RuntimeError("no start places")
        for block, m, xy in (("blue_block", "green_mat", pa), ("yellow_block", "purple_mat", pb)):
            self.set_object_pose(m, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
            self.set_object_pose(block, xy + self.np_random.uniform(-0.004, 0.004, 2),
                                 yaw=self.np_random.uniform(-np.pi, np.pi), z=0.006)
            placed.append((xy, r))
        self.choose_aside(placed)
        self.place_distractors(placed)
        self.freeze_goals(("purple_mat", "green_mat"), references=("purple_mat", "green_mat"))


# ----- 3. cube on the plate and soup can on the table --------------------------------------------------------

class PlateCubeCanEnv(SwapEnv):
    instruction = "Swap the green cube on the plate with the tomato sauce can next to it."
    task_objects = ("cube", "can")
    order = task_objects
    items = {"cube": CUBE, "can": CAN}
    solid = ("can",)
    surfaces = ("plate",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce", "plate"))

    def scene_objects(self):
        return [Block("cube", half=(HALF,) * 3, rgba=GREEN), Obj("can", "tomato_sauce")]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.6)]

    def layout(self):
        placed = []
        plate_r = self.fixture_footprint("plate") / np.sqrt(2)
        for _ in range(200):
            trial = []
            pa = self.sample_spot(plate_r, trial, 0.03)
            trial.append((pa, plate_r))
            pb = self.sample_spot(0.025, trial, 0.03)
            if 0.10 <= np.linalg.norm(pa - pb) <= 0.17:
                break
        else:
            raise RuntimeError("no start places")
        self.set_fixture_pose("plate", pa, yaw=self.np_random.uniform(-np.pi, np.pi))
        surface = self.surface_z(pa, exclude=self.task_objects)
        self.set_object_pose("cube", pa + self.np_random.uniform(-0.004, 0.004, 2),
                             yaw=self.np_random.uniform(-np.pi, np.pi), z=surface)
        self.set_object_pose("can", pb, yaw=self.np_random.uniform(-np.pi, np.pi))
        self.set_object_pose("can", 2 * pb - center(self, "can"), yaw=self.yaw("can"))
        placed += [(pa, plate_r), (pb, self.footprint("can"))]
        self.choose_aside(placed)
        self.place_distractors(placed)
        self.freeze_goals(("table", "plate"))


SWAP_STEPS = ("Pick up the {a} and set it down on a free spot of the table.",
              "Then pick up the {b} and put it where the {a} was, then pick up the {a} again and put it where "
              "the {b} was.")


def _steps(a, b):
    return tuple(s.format(a=a, b=b) for s in SWAP_STEPS)


TASKS = [
    define_task(name="swap_cube_and_can", instruction=CubeCanEnv.instruction, family=FAMILY, env=CubeCanEnv,
                oracle=SwapOracle, objects=CubeCanEnv.task_objects, object_kinds=("cube", "can"),
                relation="swap places via a free spot", goal="each other's place on the table",
                steps=_steps("red cube", "soup can")),
    define_task(name="swap_blocks_between_mats", instruction=BlocksOnMatsEnv.instruction, family=FAMILY,
                env=BlocksOnMatsEnv, oracle=SwapOracle, objects=BlocksOnMatsEnv.task_objects,
                object_kinds=("block", "block"), relation="swap places via a free spot", goal="each other's mat",
                steps=(_steps("blue block", "yellow block")[0],
                       then("Pick up the yellow block and put it on the green mat where the blue block was, then "
                            "pick up the blue block again and put it on the purple mat."))),
    define_task(name="swap_plate_cube_and_can", instruction=PlateCubeCanEnv.instruction, family=FAMILY,
                env=PlateCubeCanEnv, oracle=SwapOracle, objects=PlateCubeCanEnv.task_objects,
                object_kinds=("cube", "can"), relation="swap places via a free spot, onto and off",
                goal="plate and its neighbouring spot",
                steps=("Pick up the green cube from the plate and set it down on a free spot of the table.",
                       then("Pick up the tomato sauce can and put it on the plate, then pick up the green cube again "
                            "and put it where the can was."))),
]

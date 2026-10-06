"""place_on_elevated: put an object on top of a raised free support (a book lying flat, a box, a stack of books).

Supports are free bodies (a LIBERO book laid flat, a primitive box), so a placement that knocks them is
caught by the strict physics gate (non-task bodies may move <= 5 mm). Success is the kit's strict goal
(released, settled, upright, supported by the named support, at the target) plus ``on_top``: the
object's geometric centre lies over the support's top face, inset by a margin, so a placement that
overhangs or rests half on the table fails.

Support geometry notes (measured 2026-10-06):
  * LIBERO books stand on edge in ``CATALOG``; laid flat (90 deg about local y) a 0.7x black book is a
    7.7 x 9.4 x 2.0 cm slab and a 0.7x yellow (cream) book 6.9 x 8.5 x 1.7 cm. A book's body origin is
    ~5 cm off its collision box (the spine), so layouts use the collision-box centre, never the origin.
  * The oracle grasps and places by the geometric centre (``center``, from mesh vertices / box corners),
    not ``object_pos``, and picks the closing direction from the minimum caliper width, so objects whose
    origin or hull is offset/rotated inside the body are handled.
  * Supports are kept clear of the rest-pose keep-out (the folded gripper hovers about 1 cm above the
    table at (0.16, 0)), so they sit at |azimuth| >~ 30 deg; objects use r 0.16-0.27, +-62 deg.

Dropped candidates (calibration, 2026-10-06): thin LIBERO boxes (chocolate pudding 1.4 cm, butter 0.9 cm,
cream cheese 1.0 cm) slip/pivot in the jaws (dropped on lift, or rock onto the fixed finger at release,
1100-3000 rad/s^2); the ramekin's thin wall geoms let the jaws sink 2-5 mm; the YCB gelatin box at 0.5x rests
3.6 mm inside the table and creeps; the GSO CoQ10 bottle rocks on its hull indefinitely (never settles).
"""

import mujoco
import numpy as np

from sim.train.tasks.base import Goal, Obj, TrainEnv, TrainOracle, define_task, step_text, then
from sim.val.env import _yaw_quat
from sim.val.oracle import CLOSED
from sim.val.scene import Block

FAMILY = "place_on_elevated"
FLAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])   # lay an on-edge LIBERO book flat
BOOK_SCALE = 0.7
WHITE_BOX = dict(half=(0.035, 0.035, 0.02), rgba=(0.92, 0.92, 0.9, 1.0), mass=0.12, friction=1.2)
# r >= 0.16: a pick at r ~0.15 and large azimuth folded the wrist-camera mount into the shoulder.
OBJECT_REGION = dict(r=(0.16, 0.27), angle=(-62.0, 62.0))
SUPPORT_REGION = dict(r=(0.19, 0.25), angle=(-65.0, 65.0))   # support centre; elevated release reach <= 0.25
ON_TOP_MARGIN = 0.010         # object centre must be this far inside the support's top face


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def on_top(support, margin=ON_TOP_MARGIN):
    """Goal check: the object's geometric centre is over ``support``'s top face (inset by ``margin``) and the
    object's lowest point is no lower than 3 mm below that face."""
    def check(env, name):
        center, axes, half, top = env.top_face(support)
        c = env.center(name)
        local = np.array([(c[:2] - center[:2]) @ axes[:, 0], (c[:2] - center[:2]) @ axes[:, 1]])
        return bool(np.all(np.abs(local) <= half - margin) and env.bottom(name) >= top - 0.003)
    return check


class ElevatedEnv(TrainEnv):
    """Shared helpers for flat supports and geometric centres."""

    # ----- geometry ------------------------------------------------------------------------------------------
    def points(self, name):
        """World points of the object's collision geometry: mesh vertices and box corners (exact), the geom
        AABB corners otherwise (mesh AABBs are conservative for hulls rotated inside their body)."""
        m, d = self.model, self.data
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        pts = []
        for g in self._geoms[name]:
            R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                local = corners * m.geom_size[g]
            else:
                local = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
            pts.append(np.einsum("ij,nj->ni", R, local) + p)
        return np.vstack(pts)

    def _aabb(self, name):
        pts = self.points(name)
        return pts.min(0), pts.max(0)

    def center(self, name):
        """Geometric centre (world AABB centre of the collision geometry)."""
        lo, hi = self._aabb(name)
        return (lo + hi) / 2

    def bottom(self, name):
        return float(self._aabb(name)[0][2])

    def top_z(self, name):
        return float(self._aabb(name)[1][2])

    def _box_geom(self, name):
        geoms = self._geoms[name]
        if len(geoms) != 1 or self.model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"{name}: supports must have one box collision geom")
        return geoms[0]

    def top_face(self, name):
        """(centre xyz of the top face, horizontal xy axes as 2x2 columns, half extents along them, top z)
        of a support's box collision geom (whichever box axis is closest to vertical is the normal)."""
        g = self._box_geom(name)
        R = self.data.geom_xmat[g].reshape(3, 3)
        size = self.model.geom_size[g]
        up = int(np.argmax(np.abs(R[2])))
        others = [i for i in range(3) if i != up]
        top = float(self.data.geom_xpos[g][2] + np.abs(R[2]) @ size)
        axes = np.stack([R[:2, i] for i in others], axis=1)   # horizontal axes as columns (2x2)
        axes /= np.linalg.norm(axes, axis=0, keepdims=True)
        return np.r_[self.data.geom_xpos[g][:2], top], axes, size[others], top

    def support_radius(self, name):
        g = self._box_geom(name)
        size = np.sort(self.model.geom_size[g])
        return float(np.hypot(size[1], size[2]))   # largest two half sizes lie in the horizontal plane

    # ----- placement -----------------------------------------------------------------------------------------
    def lay_flat(self, name, xy, yaw, z=0.0):
        """Rest a LIBERO book flat with its collision-box centre at ``xy`` and its underside at height ``z``."""
        a, v = self._qadr[name], self._dadr[name]
        self.data.qpos[a:a + 3] = [xy[0], xy[1], 0.5]
        self.data.qpos[a + 3:a + 7] = quat_mul(_yaw_quat(yaw), FLAT)
        self.data.qvel[v:v + 6] = 0
        mujoco.mj_kinematics(self.model, self.data)
        lo, hi = self._aabb(name)
        self.data.qpos[a] += xy[0] - (lo[0] + hi[0]) / 2
        self.data.qpos[a + 1] += xy[1] - (lo[1] + hi[1]) / 2
        self.data.qpos[a + 2] += z + 0.0005 - lo[2]
        mujoco.mj_kinematics(self.model, self.data)

    def place_support(self, name, placed, flat_book=True, region=SUPPORT_REGION, z=0.0, xy=None):
        radius = self.support_radius(name)
        if xy is None:
            xy = self.sample_xy(radius, placed, clearance=0.02, **region)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        if flat_book:
            self.lay_flat(name, xy, yaw, z=z)
        else:
            self.set_object_pose(name, xy, yaw=yaw, z=z)
        placed.append((np.asarray(xy, float), radius))
        return np.asarray(xy, float)

    def on_goal(self, obj, support):
        """Goal: ``obj`` resting on ``support``'s top face, centred (target follows the support)."""
        center, _, _, top = self.top_face(support)
        target = (center[0], center[1], top + self.object_pos(obj)[2] - self.bottom(obj))
        return Goal(obj, support, target=target, reference=support, tolerance=(0.025, 0.025, 0.006),
                    check=on_top(support))


class ElevatedOracle(TrainOracle):
    """Grasps and places by the object's geometric centre; lifts clear of tall stacks before resting."""

    grasp_floor = 0.0095          # minimum TCP height above the object's underside (jaw hull reaches 8.2 mm below)
    release_drop = 0.0            # extra release height above the support
    home_z = 0.12                 # raised waypoint before folding to rest, clear of anything placed up high

    def grasp_spec(self, name):
        """(closing width, closing-direction yaw, symmetric) across the object's true narrow horizontal side
        (minimum caliper width of its collision geometry; mesh hulls can be rotated inside the body, so body-
        frame AABBs overstate the width)."""
        xy = self.env.points(name)[:, :2]
        angles = np.radians(np.arange(0.0, 180.0, 1.0))
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        proj = np.einsum("ni,ki->nk", xy, dirs)
        widths = proj.max(0) - proj.min(0)
        k = int(np.argmin(widths))
        narrow, wide = float(widths[k]), float(widths[(k + 90) % 180])
        if wide - narrow < 0.15 * wide:                          # square or round: four equivalent sides
            return narrow, float(angles[k]), 4
        return narrow, float(angles[k]), 2

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """As ``TrainOracle.pick`` but centred on the object's geometric centre rather than its origin."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.env.center(name)
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

    def put_on(self, name, goal, carry_z=None):
        """Pick ``name`` and place its geometric centre over the goal support's top-face centre."""
        env = self.env
        width, yaw, symmetric = self.grasp_spec(name)
        grasp_z = max(env.bottom(name) + self.grasp_floor, env.center(name)[2])   # TCP at the object's middle
        ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=symmetric, lift_z=carry_z)
        if not ok:
            return False
        target = env.goal_target(goal)
        face, _, _, top = env.top_face(goal.support)
        rot = self.carry_rot(face[:2])
        # Origin target that puts the geometric centre over the face centre once turned to ``rot``.
        offset = rot @ self.tcp_rot().T @ (env.center(name) - env.object_pos(name))
        origin = np.r_[face[:2] - offset[:2], target[2]]
        yield from self.place_object(name, origin, rot=rot, drop=self.release_drop, open_to=self.release_for(width),
                                     carry_z=carry_z)
        return True

    def go_rest(self):
        yield from self.move(np.r_[self._cmd_pos[:2], self.home_z], self._cmd_rot, speed=0.10, label="clear")
        yield from self.rest()
        yield from self.wait(1.2)


# ----- 1. milk carton on top of the black book -----------------------------------------------------------------

class MilkOnBookEnv(ElevatedEnv):
    instruction = "Put the milk carton on top of the black book."
    task_objects = ("milk",)
    surfaces = ("black_book",)
    distractor_pool = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn",
                       "ramekin", "red_bowl", "white_bowl", "plate", "akita_black_bowl")

    def scene_objects(self):
        return [Obj("milk", "milk"), Obj("black_book", "black_book", BOOK_SCALE)]

    def layout(self):
        placed = []
        self.place_support("black_book", placed)
        self.place("milk", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        self.set_goals(self.on_goal("milk", "black_book"))


class SingleOracle(ElevatedOracle):
    def plan(self):
        goal = self.env.goals[0]
        ok = yield from self.put_on(goal.obj, goal)
        if ok:
            yield from self.go_rest()


# ----- 2. soup can on top of the white box -------------------------------------------------------------------

class CanOnBoxEnv(ElevatedEnv):
    instruction = "Put the soup can on top of the white box."
    task_objects = ("alphabet_soup",)
    surfaces = ("white_box",)
    distractor_pool = ("cream_cheese", "butter", "chocolate_pudding", "popcorn", "ramekin", "white_bowl",
                       "red_bowl", "akita_black_bowl", "plate")

    def scene_objects(self):
        return [Obj("alphabet_soup", "alphabet_soup"), Block("white_box", **WHITE_BOX)]

    def layout(self):
        placed = []
        self.place_support("white_box", placed, flat_book=False)
        self.place("alphabet_soup", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        self.set_goals(self.on_goal("alphabet_soup", "white_box"))


class TallOracle(SingleOracle):
    """Tall objects (can, carton) on raised supports. Measured on can_on_box: with the kit's default release the
    fixed finger stayed on the can wall and dragged or dropped it on the retreat (1000-2300 rad/s^2, 3-5 mm
    penetration into the box, 11/15 passes). Opening wider lets the fixed finger back off further without the
    moving jaw touching the far side, and releasing with the low edge just touching (no press) avoids
    pressing the can into the box (19/20 on the same layouts).
    Turning the TCP to level the held can before release did not help (17/20)."""

    carry_z = 0.10                # a held can hangs 2 cm below the TCP; clear the 4 cm box
    release_lift_fraction = 1.0
    release_margin = 0.020
    release_backoff = 0.007


# ----- 3. green block on top of a stack of two books ---------------------------------------------------------

class BlockOnBookStackEnv(ElevatedEnv):
    instruction = "Put the green block on top of the stack of books."
    task_objects = ("green_block",)
    surfaces = ("yellow_book",)
    extra_contacts = (("black_book", "yellow_book"),)

    def scene_objects(self):
        return [Block("green_block", half=(0.014,) * 3, rgba=(0.2, 0.6, 0.25, 1.0)),
                Obj("black_book", "black_book", BOOK_SCALE), Obj("yellow_book", "yellow_book", BOOK_SCALE)]

    def layout(self):
        placed = []
        xy = self.place_support("black_book", placed)
        _, _, _, top = self.top_face("black_book")
        jitter = self.np_random.uniform(-0.004, 0.004, 2)
        self.lay_flat("yellow_book", xy + jitter, self.np_random.uniform(-np.pi, np.pi), z=top)
        self.place("green_block", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        self.set_goals(self.on_goal("green_block", "yellow_book"))


# ----- 4. ordered: green block on the box first, then milk carton on the book -------------------------------

class TwoSupportsOrderedEnv(ElevatedEnv):
    instruction = "Put the green block on the white box first, then put the milk carton on the black book."
    task_objects = ("green_block", "milk")
    order = ("green_block", "milk")
    surfaces = ("black_book", "white_box")
    n_distractors = (2, 3)
    distractor_pool = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn",
                       "ramekin", "white_bowl", "red_bowl", "plate")

    def scene_objects(self):
        return [Block("green_block", half=(0.014,) * 3, rgba=(0.2, 0.6, 0.25, 1.0)), Obj("milk", "milk"),
                Obj("black_book", "black_book", BOOK_SCALE), Block("white_box", **WHITE_BOX)]

    def layout(self):
        placed = []
        # One support on each side of the robot (sides swap per seed): both stay clear of the rest keep-out.
        side = 1 if self.np_random.random() < 0.5 else -1
        for name, sign in (("black_book", side), ("white_box", -side)):
            region = dict(r=SUPPORT_REGION["r"], angle=tuple(sorted((25.0 * sign, 65.0 * sign))))
            self.place_support(name, placed, flat_book=name == "black_book", region=region)
        for name in self.task_objects:
            # Wide clearance: the wrist-camera mount brushed the standing carton while picking a block 7 cm away.
            self.place(name, placed, region=OBJECT_REGION, clearance=0.045)
        self.place_distractors(placed)
        self.set_goals(self.on_goal("green_block", "white_box"), self.on_goal("milk", "black_book"))


class OrderedOracle(ElevatedOracle):
    carry_z = 0.10
    # Carry the carton higher so it clears the block already standing on the box (6.8 cm) when the carry arcs
    # past it; at 0.10 the two touched (2 of 50 layouts).
    carry_heights = {"milk": 0.115}

    def plan(self):
        env = self.env
        for name in env.order:
            goal = next(g for g in env.goals if g.obj == name)
            ok = yield from self.put_on(name, goal, carry_z=self.carry_heights.get(name))
            if not ok:
                return
            yield from self.move(np.r_[self._cmd_pos[:2], self.home_z], self._cmd_rot, speed=0.10, label="clear")
        yield from self.rest()
        yield from self.wait(1.2)


TASKS = [
    define_task(name="milk_on_book", instruction=MilkOnBookEnv.instruction, family=FAMILY,
                env=MilkOnBookEnv, oracle=TallOracle, objects=("milk",),
                object_kinds=("milk carton",), relation="on top of", goal="book",
                steps=(step_text("put", "milk carton", "on top of", "black book"),)),
    define_task(name="can_on_box", instruction=CanOnBoxEnv.instruction, family=FAMILY,
                env=CanOnBoxEnv, oracle=TallOracle, objects=("alphabet_soup",),
                object_kinds=("soup can",), relation="on top of", goal="box",
                steps=(step_text("put", "soup can", "on top of", "white box"),)),
    define_task(name="block_on_book_stack", instruction=BlockOnBookStackEnv.instruction, family=FAMILY,
                env=BlockOnBookStackEnv, oracle=SingleOracle, objects=("green_block",),
                object_kinds=("block",), relation="on top of", goal="stack of books",
                steps=(step_text("put", "green block", "on top of", "stack of books"),)),
    define_task(name="block_on_box_then_milk_on_book", instruction=TwoSupportsOrderedEnv.instruction,
                family=FAMILY, env=TwoSupportsOrderedEnv, oracle=OrderedOracle,
                objects=("green_block", "milk"), object_kinds=("block", "milk carton"),
                relation="on top of in temporal order", goal="box then book",
                steps=(step_text("put", "green block", "on top of", "white box"),
                       then(step_text("put", "milk carton", "on top of", "black book")))),
]

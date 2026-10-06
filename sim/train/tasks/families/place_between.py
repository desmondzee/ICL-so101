"""place_between: put an object in the gap between two reference objects.

Each task names one object and one pair of free reference objects. The references are laid out on a line with a
surface-to-surface gap wide enough for the object plus finger clearance; the line's direction, the gap, the pair's
position on the table and the object's start are randomized per episode (the line stays within ``line_from_y``
of the viewer's left-right axis so the front camera sees into the gap, and within ``line_from_tangent`` of the
robot's tangential direction where tall references could meet the forearm).

Strict success, on top of the kit's released/settled/upright/supported-on-the-table goal at the gap midpoint:

* along the reference line, the object centre is within ``ALONG_TOLERANCE`` of the midpoint (roughly midway);
* across the line, it is within ``LATERAL_TOLERANCE`` of the line (actually *between*, not beside the gap);
* it touches neither reference, and keeps at least ``MIN_CLEARANCE`` of clear gap to each one along the line.

The kit's physics policy also fails any episode in which the robot or the object touches a reference. Free
references (cans, plates, bowls) must also not move more than 5 mm / 0.1 rad; the standing books are static
fixtures because free standing books topple by themselves.

Oracle: a top-down pick (centred on the collision bounding box), then a carry that turns the jaws so their
closing direction is perpendicular to the reference line. The SO-101 jaws are only +-16 mm wide across their
closing direction (measured 2026-10-06), so the fingers open and back off parallel to the references. Of the two
such directions it takes the one whose release pose keeps the robot farthest from the references: the
wrist-camera mount sticks out 4-8 cm to one side, 4.5 cm above the TCP.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task,
                                  step_text, top_down_mat)
from sim.val.scene import Block, Fixture, Obj

FAMILY = "place_between"
ALONG_TOLERANCE = 0.015      # |offset from the midpoint| along the reference line (m)
LATERAL_TOLERANCE = 0.012    # |distance from the reference line| (m)
MIN_CLEARANCE = 0.008        # clear gap between the object's and each reference's extents along the line (m)
BOX_TOLERANCE = 0.025        # xy half width of the axis-aligned goal box (physics goal rule); the check is tighter
MID_REGION = dict(r=(0.17, 0.255), angle=(-55.0, 55.0))
REF_REGION = dict(r=(0.11, 0.32), angle=(-80.0, 80.0))
REST_SWEEP = (np.array([0.155, 0.0]), 0.085)  # folded gripper hover / rest sweep (tall-reference keep-out)
HALF = 0.014


@dataclass
class Ball:
    """A free solid ball (sphere geom). ``condim=6`` adds rolling friction, so a released ball comes to rest
    instead of rolling on indefinitely (MuJoCo's default condim 3/4 has no rolling resistance)."""

    name: str
    radius: float = 0.014
    rgba: tuple = (0.95, 0.45, 0.1, 1.0)
    mass: float = 0.02
    friction: float = 1.0
    rolling: float = 0.002

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "geom", name=f"{self.name}_geom", type="sphere", size=f"{self.radius:.6g}",
                      rgba=" ".join(map(str, self.rgba)), mass=repr(self.mass), group="1", condim="6",
                      friction=f"{self.friction} 0.02 {self.rolling}", material="val_plastic")


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
    proj = np.vstack(pts)[:, :2] @ np.asarray(u)[:2]
    return float(proj.min()), float(proj.max())


def center(env, name):
    """World xy of the centre of ``name``'s collision-geometry bounding box (body origins of scanned or LIBERO
    meshes can sit off centre: the black book's by 6 mm, a scanned golf ball's by 1.6 cm)."""
    return np.array([np.mean(extent_along(env, name, u)) for u in ((1.0, 0.0), (0.0, 1.0))])


class BetweenEnv(TrainEnv):
    """One task object and two free references ``refs`` (a, b) on a randomized line."""

    obj: str = ""
    refs: tuple[str, str] = ("", "")
    gap: tuple[float, float] = (0.07, 0.10)          # surface-to-surface gap between the references (m)
    line_from_y: float = 60.0                        # max |angle| of the line from the viewer's left-right (deg)
    line_from_tangent: float = 90.0                  # max |angle| of the line from the robot's tangential (deg)
    ref_yaw_faces_gap: bool = False                  # orient each reference's local x along the line
    tall_refs: bool = False                          # keep the references out of the rest sweep
    upright_cos: float | None = COS10
    z_tolerance = 0.006
    min_travel = 0.08
    mid_region = MID_REGION

    def ref_line(self):
        a, b = (center(self, n) for n in self.refs)
        span = b - a
        return a, b, span / np.linalg.norm(span)

    def between(self, name):
        a, b, u = self.ref_line()
        mid, p = (a + b) / 2, center(self, name)
        along = float((p - mid) @ u)
        lateral = float(abs(np.cross(u, p - mid)))
        if abs(along) > ALONG_TOLERANCE or lateral > LATERAL_TOLERANCE:
            return False
        olo, ohi = extent_along(self, name, u)
        _, ahi = extent_along(self, self.refs[0], u)
        blo, _ = extent_along(self, self.refs[1], u)
        if olo - ahi < MIN_CLEARANCE or blo - ohi < MIN_CLEARANCE:
            return False
        return not (self.touching(name, self.refs[0]) or self.touching(name, self.refs[1]))

    def set_center(self, name, xy, yaw):
        """Put free body ``name`` on the table with its bounding-box centre at ``xy``."""
        put = self.set_fixture_pose if name in self.fixture_names else self.set_object_pose
        put(name, xy, yaw=yaw)
        put(name, np.asarray(xy) + self.object_pos(name)[:2] - center(self, name), yaw=yaw)

    def ref_footprint(self, name):
        return self.fixture_footprint(name) if name in self.fixture_names else self.footprint(name)

    def _ref_yaw(self, u):
        if self.ref_yaw_faces_gap:
            return float(np.arctan2(u[1], u[0]) + self.np_random.uniform(-0.25, 0.25)
                         + np.pi * self.np_random.integers(2))
        return float(self.np_random.uniform(-np.pi, np.pi))

    def _reachable_ref(self, name, p):
        r, ang = np.hypot(*p), np.degrees(np.arctan2(p[1], p[0]))
        if not (REF_REGION["r"][0] <= r <= REF_REGION["r"][1] and REF_REGION["angle"][0] <= ang
                <= REF_REGION["angle"][1]):
            return False
        if any(np.linalg.norm(p - c) < q + self.ref_footprint(name) + 0.02 for c, q in self.keepout):
            return False
        return not (self.tall_refs and np.linalg.norm(p - REST_SWEEP[0]) < REST_SWEEP[1] + self.ref_footprint(name))

    def _layout_refs(self):
        """Sample the gap midpoint, line direction and gap; put each reference's facing edge half a gap from the
        midpoint. Returns (midpoint, line direction a -> b)."""
        rng = self.np_random
        for _ in range(400):
            rr, aa = rng.uniform(*self.mid_region["r"]), np.radians(rng.uniform(*self.mid_region["angle"]))
            mid = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if np.linalg.norm(mid - REST_SWEEP[0]) < REST_SWEEP[1] - 0.02:
                continue
            phi = np.radians(rng.uniform(-self.line_from_y, self.line_from_y))
            u = np.array([-np.sin(phi), np.cos(phi)])                    # phi = 0: world +y (viewer's right)
            tangent = np.array([-np.sin(aa), np.cos(aa)])
            if np.degrees(np.arccos(min(1.0, abs(u @ tangent)))) > self.line_from_tangent:
                continue
            if rng.uniform() < 0.5:
                u = -u                                                   # which reference sits on which side
            half_gap = rng.uniform(*self.gap) / 2
            ok = True
            for name, side in ((self.refs[0], -1), (self.refs[1], 1)):
                yaw = self._ref_yaw(u)
                self.set_center(name, mid, yaw)
                lo, hi = (x - mid @ u for x in extent_along(self, name, u))
                p = mid + side * (half_gap + (hi if side < 0 else -lo)) * u
                if not self._reachable_ref(name, p):
                    ok = False
                    break
                self.set_center(name, p, yaw)
            if ok:
                return mid, u
        raise RuntimeError("no reachable reference line")

    def layout(self):
        mid, u = self._layout_refs()
        placed = [(center(self, n), self.ref_footprint(n)) for n in self.refs]
        placed.append((mid, self.footprint(self.obj) + 0.01))            # keep the gap free
        for _ in range(40):
            trial = list(placed)
            xy, _ = self.place(self.obj, trial)
            if np.linalg.norm(xy - mid) >= self.min_travel:
                placed = trial
                break
        else:
            raise RuntimeError("object start too close to the gap")
        self.place_distractors(placed)
        # The goal box bounds the body origin; it is offset from the centre (the tight test) for scanned meshes.
        offset = float(np.linalg.norm(self.object_pos(self.obj)[:2] - center(self, self.obj)))
        xy_tol = BOX_TOLERANCE + offset
        z = self._extent[self.obj]["bottom"]
        self.set_goals(Goal(self.obj, "table", target=(*mid, z), tolerance=(xy_tol, xy_tol, self.z_tolerance),
                            upright_cos=self.upright_cos, check=lambda env, name: env.between(name)))


class BetweenOracle(TrainOracle):
    """Top-down pick; carry with the closing direction perpendicular to the reference line; release."""

    grasp_symmetric = 4
    min_grasp_z = 0.0095        # the jaw hull reaches ~8.2 mm below the TCP
    close_to = CLOSED           # gripper target when closing on the object
    # Lower by the full tilt depth so a held object that pivoted a few degrees in the jaws during the carry is
    # pressed upright against the table before the jaws open. With the kit's half lift a 3.8 cm can stayed
    # leaning 3 deg on the fixed finger after opening, then slid off and slapped flat (angular acceleration
    # 1090-5480 rad/s^2; 2/12 calibration episodes); pressing passed 12/12 with the same penetration maxima.
    release_lift_fraction = 0.0

    def grasp(self):
        """(width along the closing direction, TCP height, yaw or None) for the pick."""
        env = self.env
        return 2 * HALF, self.mid_height(), env.yaw(env.obj)

    def mid_height(self):
        """TCP height 1 mm below the resting object's mid height, kept clear of the table."""
        return max(self.env.height(self.env.obj) / 2 - 0.001, self.min_grasp_z)

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """The kit's top-down pick, centred on the collision bounding box instead of the body origin."""
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
            yield from self.gripper(self.close_to, self.close_seconds, 0.3)
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

    def plan(self):
        env = self.env
        name = env.obj
        width, grasp_z, yaw = self.grasp()
        ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=self.grasp_symmetric)
        if not ok:
            return
        goal = env.goals[0]
        a, b, u = env.ref_line()
        across = np.arctan2(u[1], u[0]) + np.pi / 2                      # closing direction across the line
        current = np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        candidates = sorted((across, across + np.pi), key=lambda t: abs(np.angle(np.exp(1j * (t - current)))))
        rots = [top_down_mat(t) for t in candidates]
        mid = (a + b) / 2
        open_to = self.release_for(width)
        rot = self.clearest_rotation(name, goal, mid, rots, open_to)
        yield from self.place_object(name, self.origin_target(name, goal, mid, rot), rot=rot, open_to=open_to)
        yield from self.rest()
        yield from self.wait(1.2)

    def clearest_rotation(self, name, goal, mid, rotations, open_to):
        """Of the two closing directions across the line, the reachable one whose release pose (jaws open) keeps
        the robot farthest from both references. The wrist-camera mount sticks out 4-8 cm to one side of the
        jaws, 4.5 cm above the TCP (measured 2026-10-06), so it decides which way the jaws face over tall
        references."""
        held, scored = self.held(name), []
        for k, rot in enumerate(rotations):
            target = self.origin_target(name, goal, mid, rot)
            tcp = target - rot @ held
            carry_err = self.solve(np.r_[tcp[:2], self.carry_z], rot)[1]
            q, err, _ = self.solve(tcp, rot)
            scored.append((max(err, carry_err), self.ref_clearance(q, open_to), k, rot))
        # IK error at the (tilted) carry height is 5-11 mm for every rotation; only a clearly worse one is out.
        least = min(e for e, *_ in scored)
        return min(scored, key=lambda c: (c[0] > least + 0.004, -c[1], c[2]))[3]

    def ref_clearance(self, q, grip):
        """Minimum distance (m) between any robot collision geom and either reference at arm joints ``q``."""
        env, m = self.env, self.model
        if not hasattr(self, "_probe"):
            robot = m.body("base").id
            collide = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
            self._probe = mujoco.MjData(m)
            self._robot_geoms = [g for g in collide if env._in_subtree(m.geom_bodyid[g], robot)]
            self._ref_geoms = [g for n in env.refs for g in env._geoms[n] if g in collide]
        d = self._probe
        d.qpos[:] = env.data.qpos
        d.qpos[env._arm_qpos_addrs] = q
        d.qpos[env._qpos_addrs[5]] = grip
        mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, g, h, 0.3, None) for g in self._robot_geoms for h in self._ref_geoms)

    def origin_target(self, name, goal, mid, rot):
        """Body-origin target that puts the object's centre on ``mid`` once the TCP turns to ``rot``."""
        env = self.env
        offset = np.r_[env.object_pos(name)[:2] - center(env, name), 0.0]
        turn = rot @ self.tcp_rot().T
        return np.r_[mid, env.goal_target(goal)[2]] + turn @ offset


# ----- tasks -------------------------------------------------------------------------------------------------

def _pool(*exclude):
    return tuple(n for n in LOW_DISTRACTORS if n not in exclude)


class BallBetweenCansEnv(BetweenEnv):
    instruction = "Put the orange ball between the soup can and the tomato sauce can."
    task_objects = ("ball",)
    obj, refs = "ball", ("soup_can", "sauce_can")
    gap = (0.065, 0.09)
    line_from_y = 55.0
    tall_refs = True
    upright_cos = None                # a ball has no upright
    distractor_pool = _pool("alphabet_soup", "tomato_sauce")

    def scene_objects(self):
        return [Ball("ball"),
                Obj("soup_can", "alphabet_soup"), Obj("sauce_can", "tomato_sauce")]


class BallOracle(BetweenOracle):
    release_lift_fraction = 0.5       # a ball has no tilt to press out (kit default)

    def grasp(self):
        return 2 * Ball.radius, self.mid_height(), None


class BlockBetweenPlatesEnv(BetweenEnv):
    instruction = "Put the block between the two plates."
    task_objects = ("block",)
    obj, refs = "block", ("plate_a", "plate_b")
    gap = (0.06, 0.085)
    line_from_y = 75.0
    distractor_pool = _pool("plate")

    def scene_objects(self):
        return [Block("block", half=(HALF,) * 3, rgba=(0.15, 0.55, 0.25, 1)),
                Obj("plate_a", "plate"), Obj("plate_b", "plate")]


class SauceBetweenBowlsEnv(BetweenEnv):
    # The akita "black" bowl renders mid grey under the training lights, so the bowls are not named by colour.
    instruction = "Put the tomato sauce can between the two bowls."
    task_objects = ("sauce",)
    obj, refs = "sauce", ("white_bowl", "black_bowl")
    gap = (0.07, 0.095)
    line_from_y = 65.0
    distractor_pool = _pool("alphabet_soup", "tomato_sauce", "white_bowl", "akita_black_bowl", "red_bowl", "ramekin")

    def scene_objects(self):
        return [Obj("sauce", "tomato_sauce"), Obj("white_bowl", "white_bowl"), Obj("black_bowl", "akita_black_bowl")]


class CanOracle(BetweenOracle):
    def grasp(self):
        return 0.031, self.mid_height(), None


class CanBetweenBooksEnv(BetweenEnv):
    instruction = "Put the soup can between the two books."
    task_objects = ("can",)
    obj, refs = "can", ("black_book", "yellow_book")
    gap = (0.075, 0.095)
    line_from_y = 45.0
    line_from_tangent = 40.0
    ref_yaw_faces_gap = True
    tall_refs = True
    mid_region = dict(r=(0.17, 0.235), angle=(-55.0, 55.0))   # the 12.5 cm carry height reaches less far
    distractor_pool = _pool("alphabet_soup", "tomato_sauce")

    def scene_objects(self):
        return [Obj("can", "alphabet_soup")]

    def scene_fixtures(self):
        # Free standing books (1.3 x 5.5 x 6.7 cm) topple on their own a few seconds into an episode (measured
        # 2026-10-06, no robot contact), so the books are static bodies here; any robot or can contact with a
        # book still fails the strict physics gate.
        return [Fixture("black_book", "black_book"), Fixture("yellow_book", "yellow_book")]


class CanBooksOracle(BetweenOracle):
    # The held can rides 7-8 mm lower in the jaws during the carry (measured); at 0.125 its bottom clears the
    # 6.8 cm book by > 3 cm (0.105 brushed it in 4/8 calibration episodes).
    carry_z = 0.125

    def grasp(self):
        # Grasp the 3.8 cm can 7 mm above its middle: the wrist-camera mount's underside (4.5 cm above the TCP)
        # then stays above the 6.1/6.7 cm books.
        return 0.031, self.mid_height() + 0.007, None


TASKS = [
    define_task(name="ball_between_cans", instruction=BallBetweenCansEnv.instruction, family=FAMILY,
                env=BallBetweenCansEnv, oracle=BallOracle, objects=("ball",), object_kinds=("ball",),
                relation="between", goal="two cans",
                steps=(step_text("put", "orange ball", "between", "soup can and the tomato sauce can"),)),
    define_task(name="block_between_plates", instruction=BlockBetweenPlatesEnv.instruction, family=FAMILY,
                env=BlockBetweenPlatesEnv, oracle=BetweenOracle, objects=("block",), object_kinds=("block",),
                relation="between", goal="two plates",
                steps=(step_text("put", "block", "between", "two plates"),)),
    define_task(name="sauce_can_between_bowls", instruction=SauceBetweenBowlsEnv.instruction, family=FAMILY,
                env=SauceBetweenBowlsEnv, oracle=CanOracle, objects=("sauce",), object_kinds=("can",),
                relation="between", goal="two bowls",
                steps=(step_text("put", "tomato sauce can", "between", "two bowls"),)),
    define_task(name="can_between_books", instruction=CanBetweenBooksEnv.instruction, family=FAMILY,
                env=CanBetweenBooksEnv, oracle=CanBooksOracle, objects=("can",), object_kinds=("can",),
                relation="between", goal="two standing books",
                steps=(step_text("put", "soup can", "between", "two books"),)),
]

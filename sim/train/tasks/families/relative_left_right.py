"""relative_left_right: move an object to the viewer's left or right of a reference object.

Left/right always follow the FRONT camera (the human-video viewer), never the robot's own frame:
``VIEW["right"]`` is world +y and ``VIEW["left"]`` is world -y (the front camera looks along about -x, so
image-right is +y for every sampled camera pose). Each task fixes one relation; the reference is a static
fixture, so it cannot be nudged, and the strict success predicate requires, beyond the kit's settled/released/
supported-on-table goal:

* a clear edge-to-edge gap along the viewer's left-right axis (world-y extents of the collision geometry, at
  least ``MIN_GAP``) on the instructed side,
* the object roughly level with the reference in depth (|dx| of the centres at most ``MAX_DEPTH_OFFSET``), so
  the relation reads as left/right and not as in front/behind,
* no contact with the reference (also enforced by the physics policy, which only allows object/table contact).
"""

import mujoco
import numpy as np

from sim.train.tasks.base import GRASP_REGION, VIEW, Goal, TrainEnv, TrainOracle, define_task, step_text
from sim.val.oracle import CLOSED, RELEASE, tcp_path
from sim.val.scene import Fixture, Obj

FAMILY = "relative_left_right"
MIN_GAP = 0.012           # minimum clear edge-to-edge gap along world y (m)
TARGET_GAP = 0.03         # nominal gap the oracle aims for (object footprint radius used as its half width)
MAX_DEPTH_OFFSET = 0.025  # |dx| of object and reference centres (m)
TOLERANCE = (0.015, 0.015, 0.006)
TARGET_REGION = dict(r=(0.175, 0.265), angle=(-62.0, 62.0))
VIEW_HALF_WIDTH = 0.25    # |world y| the front camera shows fully across its sampled poses (checked in previews)
REST_SWEEP = (np.array([0.16, 0.0]), 0.06)    # folded gripper hover/sweep near the rest pose


def _world_y_extent(env, name):
    """(min, max) world y of the collision geometry of a free body or fixture."""
    m, d = env.model, env.data
    lo, hi = np.inf, -np.inf
    for g in env._geoms[name]:
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        center = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
        half = np.abs(rot) @ m.geom_aabb[g, 3:]
        lo, hi = min(lo, center[1] - half[1]), max(hi, center[1] + half[1])
    return lo, hi


def side_gap(env, obj, ref, side):
    """Clear gap (m) between ``obj`` and ``ref`` along the viewer's ``side`` direction (negative: overlap or
    wrong side)."""
    olo, ohi = _world_y_extent(env, obj)
    rlo, rhi = _world_y_extent(env, ref)
    return olo - rhi if side == "right" else rlo - ohi


def horizontal_points(env, name):
    """World xy of the collision-geometry AABB corners of a free body."""
    m, d = env.model, env.data
    pts = []
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for g in env._geoms[name]:
        rot = d.geom_xmat[g].reshape(3, 3)
        pts.append((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ rot.T + d.geom_xpos[g])
    return np.vstack(pts)[:, :2]


def narrow_axis(env, name, steps=36):
    """(angle, width): horizontal direction across which the body is narrowest, and that width."""
    pts = horizontal_points(env, name)
    best = None
    for a in np.linspace(0, np.pi, steps, endpoint=False):
        proj = pts @ np.array([np.cos(a), np.sin(a)])
        width = float(proj.max() - proj.min())
        if best is None or width < best[1]:
            best = (float(a), width)
    return best


class RelativeEnv(TrainEnv):
    """One object beside one static reference fixture, on the viewer's ``side``."""

    obj: str = ""
    ref: str = ""
    side: str = "left"
    ref_region = dict(r=(0.15, 0.32), angle=(-75.0, 75.0))
    min_travel = 0.08           # start at least this far from the goal
    target_gap = TARGET_GAP     # nominal clear gap (larger beside tall references: wrist-camera mount)
    path_clearance = None       # tall references: the carry path must pass this clear of the reference (m)
    start_region = GRASP_REGION
    ref_round = False           # round reference: its true radius is fixture_footprint / sqrt(2)

    def ref_radius(self):
        return self.fixture_footprint(self.ref) / (np.sqrt(2) if self.ref_round else 1.0)

    def relation_holds(self, name):
        dx = abs(self.object_pos(name)[0] - self.object_pos(self.ref)[0])
        return (side_gap(self, name, self.ref, self.side) >= MIN_GAP and dx <= MAX_DEPTH_OFFSET
                and not self.touching(name, self.ref))

    def path_clear(self, start, target, ref_xy):
        """The carry (straight, or arced about the pan axis exactly as ``tcp_path`` will) stays
        ``path_clearance`` clear of the reference, so a held object never passes over a tall reference."""
        if self.path_clearance is None:
            return True
        path, _ = tcp_path(np.r_[start, 0.08], np.r_[target, 0.08])
        reach = self.ref_radius() + self.footprint(self.obj) + self.path_clearance
        return all(np.linalg.norm(path(u)[:2] - ref_xy) >= reach for u in np.linspace(0, 1, 41))

    def layout(self):
        placed = []
        obj_r = self.footprint(self.obj)
        for _ in range(60):
            ref_xy = self.sample_xy(self.ref_radius(), placed, **self.ref_region)
            self.set_fixture_pose(self.ref, ref_xy, yaw=self.np_random.uniform(-np.pi, np.pi))
            lo, hi = _world_y_extent(self, self.ref)
            if lo < -VIEW_HALF_WIDTH or hi > VIEW_HALF_WIDTH:     # keep the whole reference in the front view
                continue
            edge = hi if self.side == "right" else lo
            depth = self.np_random.uniform(-0.4, 0.4) * MAX_DEPTH_OFFSET    # level with the reference, +-1 cm
            target = np.array([ref_xy[0] + depth, edge]) + (obj_r + self.target_gap) * VIEW[self.side]
            radius, azimuth = np.hypot(*target), np.degrees(np.arctan2(target[1], target[0]))
            if not (TARGET_REGION["r"][0] <= radius <= TARGET_REGION["r"][1]
                    and TARGET_REGION["angle"][0] <= azimuth <= TARGET_REGION["angle"][1]):
                continue
            if np.linalg.norm(target - REST_SWEEP[0]) < REST_SWEEP[1] + obj_r:
                continue
            break
        else:
            raise RuntimeError("no reachable side target")
        placed.append((ref_xy, self.ref_radius()))
        placed.append((target, obj_r + 0.005))                     # keep the goal spot free
        for _ in range(30):
            trial = list(placed)
            xy, _ = self.place(self.obj, trial, region=self.start_region)
            if np.linalg.norm(xy - target) >= self.min_travel and self.path_clear(xy, target, ref_xy):
                placed = trial
                break
        else:
            raise RuntimeError("object start too close to the goal")
        self.place_distractors(placed)
        z = self.object_pos(self.obj)[2]
        self.set_goals(Goal(self.obj, "table", target=(*target, z), tolerance=TOLERANCE,
                            check=lambda env, name: env.relation_holds(name)))


class RelativeOracle(TrainOracle):
    min_grasp_z = 0.0105        # TCP height floor: the jaw hull reaches ~8.2 mm below the TCP
    grasp_z = None              # fixed TCP grasp height (default: object centre, floored at min_grasp_z)
    grasp_symmetry = 2          # equivalent closing directions (2: box across its narrow side; 4/8: round)
    max_tilt = np.radians(4)    # grasp/release poses must stay near vertical
    enough_clearance = 0.02     # wrist-camera mount clearance beyond this does not rank grasps
    backoff_speed = 0.03        # as TrainOracle
    touch_depth = 0.001         # staged close: first ease the jaw to this far inside the object width ...
    touch_seconds = 0.9         # ... over this long, then squeeze to CLOSED over close_seconds

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """``TrainOracle.pick`` with a staged close: the moving jaw first eases (min-jerk, ending at rest) to a
        fingertip gap of ``width - touch_depth``, i.e. just onto the object, then squeezes to CLOSED from rest.
        A single 1.2 s close reaches the object near peak jaw speed, which spun light objects (26 g ramekin, 32 g can) at
        1000-2200 rad/s^2 on contact (measured 2026-10-06)."""
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
        """``TrainOracle.release`` with a configurable back-off speed (``backoff_speed``)."""
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        self.log.append(("opened",))
        if self.release_backoff:
            yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=self.backoff_speed,
                                 tol=0.002, settle=0.2, label="backoff", smooth=False)

    def _clearances(self, q):
        """(min distance of the wrist/camera mount/lower arm to the reference, min distance of the camera
        mount to the shoulder/upper arm) at arm joints ``q``, from the collision geometry."""
        env, m, d = self.env, self.model, self.ik_data
        d.qpos[:] = env.data.qpos
        d.qpos[env._arm_qpos_addrs] = q
        mujoco.mj_kinematics(m, d)

        def geoms(*bodies):
            ids = {m.body(b).id for b in bodies}
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]

        def gap(a, b):
            return min(mujoco.mj_geomDistance(m, d, g1, g2, 0.1, None) for g1 in a for g2 in b)

        mount = geoms("camera_mount")
        ref = [g for g in env._geoms[env.ref] if m.geom_contype[g] or m.geom_conaffinity[g]]
        return gap(mount + geoms("wrist", "lower_arm"), ref), gap(mount, geoms("shoulder", "upper_arm"))

    def grasp_score(self, angle, width, grasp_z):
        """Score a grasp closing along ``angle``: reachability of the grasp and release poses first, then
        clearance of the wrist-camera mount from the reference and from the shoulder at the release pose.
        ``place_object`` turns the grasp with the arm's azimuth (``carry_rot``), so the release rotation is
        predictable from the grasp."""
        env = self.env
        obj, target = env.object_pos(env.obj), env.goal_target(env.goals[0])
        rot, grasp, _ = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, [angle])
        turn = np.arctan2(target[1], target[0]) - np.arctan2(obj[1], obj[0])
        c, s = np.cos(turn), np.sin(turn)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        place = np.r_[target[:2] + (rz @ (grasp - obj))[:2], grasp[2] + target[2] - obj[2]]
        q_grasp, err_grasp, tilt_grasp = self.solve(grasp, rot)
        q_place, err_place, tilt_place = self.solve(place, rz @ rot)
        to_ref, to_self = self._clearances(q_place)
        # The camera mount can also fold into the shoulder at the grasp, or on the joint-space transit from
        # rest to the pregrasp (``pick`` transits there, seeded with the grasp's wrist roll).
        q_pre = self.solve(grasp + [0, 0, 0.05], rot, seeds=[q_grasp[4]])[0]
        for q in [q_grasp] + [self.q + u * (q_pre - self.q) for u in np.linspace(0.1, 1.0, 10)]:
            to_self = min(to_self, self._clearances(q)[1])
        err, tilt = max(err_grasp, err_place), max(tilt_grasp, tilt_place)
        feasible = err <= 0.002 and tilt <= self.max_tilt
        clearance = min(to_ref, self.enough_clearance) + min(to_self, self.enough_clearance)
        # As TrainOracle.grasp_plan: position error, tilt and wrist-roll change from the current pose.
        smooth = 200 * err + 2 * tilt + 0.3 * abs(q_grasp[4] - self.q[4])
        return (not feasible, -round(clearance, 3), smooth)

    def choose_grasp(self, grasp_z):
        """(angle, width) of the best grasp among the object's equivalent closing directions."""
        env = self.env
        angle, _ = narrow_axis(env, env.obj)
        pts = horizontal_points(env, env.obj)
        best = None
        for k in range(self.grasp_symmetry):
            a = angle + k * 2 * np.pi / self.grasp_symmetry
            proj = pts @ np.array([np.cos(a), np.sin(a)])
            width = float(proj.max() - proj.min())
            score = self.grasp_score(a, width, grasp_z)
            if best is None or score < best[0]:
                best = (score, a, width)
        return best[1], best[2]

    def plan(self):
        env = self.env
        name = env.obj
        grasp_z = self.grasp_z or max(env.object_pos(name)[2] - 0.001, self.min_grasp_z)
        angle, width = self.choose_grasp(grasp_z)
        ok = yield from self.pick(name, width, grasp_z, yaw=angle, symmetric=1)
        if not ok:
            return
        yield from self.place_object(name, env.goal_target(env.goals[0]), open_to=self.release_for(width))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- tasks -------------------------------------------------------------------------------------------------

class MilkLeftOfBowlEnv(RelativeEnv):
    instruction = "Put the milk carton to the left of the bowl."
    task_objects = ("milk",)
    obj, ref, side = "milk", "bowl", "left"
    ref_round = True
    distractor_pool = ("alphabet_soup", "tomato_sauce", "butter", "chocolate_pudding", "popcorn", "plate",
                       "cream_cheese")

    def scene_objects(self):
        return [Obj("milk", "milk")]

    def scene_fixtures(self):
        return [Fixture("bowl", "white_bowl", 1.1)]


class CanRightOfMugEnv(RelativeEnv):
    instruction = "Put the soup can to the right of the mug."
    task_objects = ("can",)
    obj, ref, side = "can", "mug", "right"
    distractor_pool = ("cream_cheese", "butter", "chocolate_pudding", "popcorn", "plate", "akita_black_bowl")

    def scene_objects(self):
        return [Obj("can", "alphabet_soup")]

    def scene_fixtures(self):
        return [Fixture("mug", "white_yellow_mug")]


class CanRightOfMugOracle(RelativeOracle):
    carry_z = 0.085             # the 5.2 cm mug: the held can's bottom stays >= 1 cm above it
    grasp_symmetry = 8          # round can
    open_margin = 0.018         # the pivoting jaw narrows above the tips: clear the 3.8 cm can top on the way down
    grasp_clearance = 0.003     # the fixed jaw body (above the tips) otherwise grazes the can top while lowering


class JuiceLeftOfBasketEnv(RelativeEnv):
    instruction = "Put the orange juice carton to the left of the basket."
    task_objects = ("juice",)
    obj, ref, side = "juice", "basket", "left"
    target_gap = 0.045          # the wrist-camera mount must stay clear of the 7.1 cm basket wall
    path_clearance = 0.01       # never carry the tall carton over the basket
    start_region = dict(r=(0.17, 0.27), angle=(-58.0, 58.0))   # grasped 3.4 cm up: keep off the base
    distractor_pool = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn",
                       "ramekin")

    def scene_objects(self):
        return [Obj("juice", "orange_juice")]

    def scene_fixtures(self):
        return [Fixture("basket", "basket")]


class MokaRightOfPlateEnv(RelativeEnv):
    instruction = "Put the moka pot to the right of the plate."
    task_objects = ("moka",)
    obj, ref, side = "moka", "plate", "right"
    ref_round = True
    target_gap = 0.025
    distractor_pool = ("alphabet_soup", "tomato_sauce", "popcorn", "butter", "ramekin", "red_bowl")

    def scene_objects(self):
        return [Obj("moka", "moka_pot")]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.65)]


class MokaRightOfPlateOracle(RelativeOracle):
    carry_z = 0.09              # the 7.6 cm pot hangs ~3.7 cm below the TCP: clear the 4 cm distractors
    release_margin = 0.016      # a wider release keeps the rising jaws off the pot's lid and handle
    open_margin = 0.02          # the pivoting jaw's gap narrows above the tips; the pot is 7.6 cm tall


TASKS = [
    define_task(name="milk_left_of_bowl", instruction=MilkLeftOfBowlEnv.instruction, family=FAMILY,
                env=MilkLeftOfBowlEnv, oracle=RelativeOracle, objects=("milk",),
                object_kinds=("milk carton",), relation="left of separated", goal="bowl",
                steps=(step_text("put", "milk carton", "down to the left of", "bowl, leaving a gap"),)),
    define_task(name="can_right_of_mug", instruction=CanRightOfMugEnv.instruction, family=FAMILY,
                env=CanRightOfMugEnv, oracle=CanRightOfMugOracle, objects=("can",),
                object_kinds=("can",), relation="right of separated", goal="mug",
                steps=(step_text("put", "soup can", "down to the right of", "mug, leaving a gap"),)),
    define_task(name="juice_left_of_basket", instruction=JuiceLeftOfBasketEnv.instruction, family=FAMILY,
                env=JuiceLeftOfBasketEnv, oracle=RelativeOracle, objects=("juice",),
                object_kinds=("juice carton",), relation="left of separated", goal="basket",
                steps=(step_text("put", "orange juice carton", "down to the left of", "basket, leaving a gap"),)),
    define_task(name="moka_pot_right_of_plate", instruction=MokaRightOfPlateEnv.instruction, family=FAMILY,
                env=MokaRightOfPlateEnv, oracle=MokaRightOfPlateOracle, objects=("moka",),
                object_kinds=("moka pot",), relation="right of separated", goal="plate",
                steps=(step_text("put", "moka pot", "down to the right of", "plate, leaving a gap"),)),
]

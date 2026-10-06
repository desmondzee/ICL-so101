"""relative_front_behind: put an object in front of / behind a reference object, as seen by the front camera.

"In front of" means nearer the front (viewer) camera, which is world +x (``VIEW["front"]``, away from the robot);
"behind" means farther from the camera, world -x (toward the robot). Every task places one graspable object on
the table on the named side of a static reference fixture, in line with it (laterally centred on it, so the
relation is unambiguous and not "beside"), with a clear gap.

Success (``Goal.check`` on top of the kit's settled / released / table-supported / upright / target-tolerance
predicate, ``relation_holds``):
  * the object's whole footprint lies beyond the reference's footprint along x on the named side (>= 1 cm),
  * its centre is laterally within ``lateral`` of the reference's centre (in line, not diagonal),
  * the minimum collision distance to the reference is >= ``min_gap`` (so it is not touching it),
  * at least 60 % of the object's top surface samples are visible from the episode's front camera, and none of
    them is hidden by the reference (the robot is ignored: the end frame is rendered without it).
The strict physics gate additionally forbids any object-reference or robot-reference contact (the reference is
neither a goal support nor a declared surface).

Layouts sample the goal spot first (reachable, outside the robot keep-outs), derive the reference pose from it,
reject layouts whose goal spot is not clearly visible from the front camera, then place the object and 2-4
distractors. The object never starts on the correct side in line with the reference.
"""

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, LOW_DISTRACTORS, VIEW, Goal, TrainEnv, TrainOracle,
                                  define_task, step_text)
from sim.val.scene import Block, Fixture, Obj

FAMILY = "relative_front_behind"
RED, BLUE = (0.8, 0.1, 0.08, 1), (0.1, 0.25, 0.8, 1)
BAR = (0.024, 0.010, 0.014)       # 4.8 x 2.0 x 2.8 cm long block
BOWLS = ("white_bowl", "red_bowl", "akita_black_bowl")


# ----- geometry helpers ------------------------------------------------------------------------------------

def world_box(env, name):
    """World-frame axis-aligned box (lo, hi) of a free body's or fixture's collision geometry."""
    m, d = env.model, env.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in env._geoms[name]:
        rot = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
        h = np.abs(rot) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def min_distance(env, a, b, cap=0.05):
    """Minimum collision-geometry distance between bodies ``a`` and ``b`` (capped at ``cap``)."""
    best = cap
    for ga in env._geoms[a]:
        for gb in env._geoms[b]:
            best = min(best, mujoco.mj_geomDistance(env.model, env.data, ga, gb, cap, None))
    return float(best)


def visible_fraction(env, points, owner=None, blocker=None):
    """Fraction of world ``points`` the front camera sees (scene geoms only; the robot is ignored).
    A ray that first hits ``owner`` counts as visible. Returns (fraction, hidden_by_blocker)."""
    m, d = env.model, env.data
    cam = d.cam_xpos[env._front_cam_id].copy()
    groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)   # scene visuals; group 2 (robot) and 3 (hulls) off
    robot = m.body("base").id
    owned = set() if owner is None else {g for g in range(m.ngeom)
                                         if env._in_subtree(m.geom_bodyid[g], env._body[owner])}
    blocked = set() if blocker is None else {g for g in range(m.ngeom)
                                             if env._in_subtree(m.geom_bodyid[g], env._body[blocker])}
    geom = np.zeros(1, dtype=np.int32)
    seen, hidden_by_blocker = 0, False
    for p in points:
        start, vec = cam, np.asarray(p, float) - cam
        dist = float(np.linalg.norm(vec))
        direction = vec / dist
        travelled, visible = 0.0, False
        for _ in range(8):                      # step through robot geoms (not excluded by group alone)
            hit = mujoco.mj_ray(m, d, start, direction, groups, 1, -1, geom)
            if hit < 0 or travelled + hit >= dist - 0.003:
                visible = True
                break
            g = int(geom[0])
            if g in owned:
                visible = True
                break
            if env._in_subtree(m.geom_bodyid[g], robot):
                travelled += hit + 1e-4
                start = cam + travelled * direction
                continue
            hidden_by_blocker |= g in blocked
            break
        seen += visible
    return seen / len(points), hidden_by_blocker


def in_frame(env, points, margin=16) -> bool:
    """All world ``points`` project inside the front-camera image with a ``margin`` (pixels)."""
    width, height = env._renderer_rgb.width, env._renderer_rgb.height
    cam = env._front_cam_id
    local = (np.asarray(points, float) - env.data.cam_xpos[cam]) @ env.data.cam_xmat[cam].reshape(3, 3)
    depth = -local[:, 2]
    if np.any(depth <= 0):
        return False
    focal = height / (2 * np.tan(np.radians(env.model.cam_fovy[cam]) / 2))
    u, v = width / 2 + focal * local[:, 0] / depth, height / 2 - focal * local[:, 1] / depth
    return bool(np.all((u >= margin) & (u < width - margin) & (v >= margin) & (v < height - margin)))


def box_corners(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def relation_holds(env, name) -> bool:
    """Correct side with a clear gap, in line with the reference, not touching it, visible and not occluded."""
    ref, side = env.reference, env.side
    olo, ohi = world_box(env, name)
    rlo, rhi = world_box(env, ref)
    beyond = (olo[0] - rhi[0]) if side > 0 else (rlo[0] - ohi[0])
    if beyond < env.min_side_gap:
        return False
    if abs(env.object_pos(name)[1] - 0.5 * (rlo[1] + rhi[1])) > env.lateral:
        return False
    if env.touching(name, ref) or min_distance(env, name, ref) < env.min_gap:
        return False
    c = env.object_pos(name)
    hx, hy = 0.6 * (ohi[:2] - olo[:2]) / 2
    top = ohi[2] - 0.002
    points = [(c[0] + sx * hx, c[1] + sy * hy, top) for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
    fraction, hidden = visible_fraction(env, points, owner=name, blocker=ref)
    return fraction >= 0.6 and not hidden


# ----- environment -----------------------------------------------------------------------------------------

class FrontBehindEnv(TrainEnv):
    """One object to put in front of (side=+1, world +x) or behind (side=-1, world -x) a static reference."""

    obj = "object"
    reference = "reference"
    side = 1
    round_object = False         # footprint() is the bbox corner radius; round objects use footprint/sqrt(2)
    tall_reference = False       # tall references also keep clear of the folded gripper's rest sweep
    gap = 0.032                  # nominal edge gap along x between reference footprint and object footprint
    min_side_gap = 0.01          # success: object footprint beyond the reference footprint along x
    min_gap = 0.012              # success: minimum collision distance object-reference
    lateral = 0.025              # success: |object y - reference centre y|
    target_region = dict(r=(0.165, 0.26), angle=(-58.0, 58.0))
    reference_region = dict(r=(0.10, 0.36), angle=(-72.0, 72.0))
    ref_yaw = np.pi              # reference yaw drawn from [-ref_yaw, ref_yaw]
    start_region = dict(r=(0.14, 0.27), angle=(-65.0, 65.0))

    def object_half(self):
        r = self.footprint(self.obj)
        return r / np.sqrt(2) if self.round_object else r

    def _reference_frame(self, yaw):
        """Reference box centre offset from its origin and half extents (xy) at ``yaw``."""
        self.set_fixture_pose(self.reference, (0.0, 3.0), yaw=yaw)
        mujoco.mj_kinematics(self.model, self.data)
        lo, hi = world_box(self, self.reference)
        origin = self.data.xpos[self._body[self.reference]][:2]
        bottom = lo[2] - self.data.xpos[self._body[self.reference]][2]
        return 0.5 * (lo[:2] + hi[:2]) - origin, 0.5 * (hi[:2] - lo[:2]), bottom

    def _clear(self, xy, radius, extra=()):
        return all(np.linalg.norm(xy - p) >= radius + q + 0.01 for p, q in (*self.keepout, *extra))

    def _in_region(self, xy, region):
        r, a = np.hypot(*xy), np.degrees(np.arctan2(xy[1], xy[0]))
        return region["r"][0] <= r <= region["r"][1] and region["angle"][0] <= a <= region["angle"][1]

    def _on_table(self, centre, half):
        """The reference's footprint corners all lie over the table top (the living-room table ends at x ~0.42)."""
        skip = (*self.free_names, *self.fixture_names)
        return all((z := self.surface_z(centre + (sx * half[0], sy * half[1]), exclude=skip)) is not None and z > -0.005
                   for sx in (-1, 1) for sy in (-1, 1))

    def goal_points(self, target):
        """Goal-spot samples at the object's mid-height (used to screen the goal's visibility)."""
        h, z = self.object_half() * 0.6, self.height(self.obj) / 2
        return [(target[0] + sx * h, target[1] + sy * h, z) for sx in (-1, 0, 1) for sy in (-1, 0, 1)]

    def layout(self):
        obj_half = self.object_half()
        # Tall references also stay out of the folded gripper's rest sweep (x 0.12-0.17, y 0.04-0.08).
        rest_sweep = ((np.array([0.15, 0.06]), 0.04), (np.array([0.15, -0.03]), 0.03)) if self.tall_reference else ()
        for _ in range(60):
            target = self.sample_xy(self.footprint(self.obj), [], clearance=0.01, **self.target_region)
            yaw = self.np_random.uniform(-self.ref_yaw, self.ref_yaw)
            offset, half, z_lo = self._reference_frame(yaw)
            centre = target - self.side * (half[0] + obj_half + self.gap) * VIEW["front"]
            ref_radius = float(np.linalg.norm(half))
            if (self._in_region(centre, self.reference_region) and self._clear(centre, ref_radius, rest_sweep)
                    and self._on_table(centre, half)):
                break
        else:
            raise RuntimeError("no reachable front/behind layout")
        self.set_fixture_pose(self.reference, centre - offset, yaw=yaw, z=max(0.0, -z_lo))
        placed = [(centre, ref_radius), (target, obj_half + 0.012)]
        # The object starts clear of the goal and not already on the named side in line with the reference.
        for _ in range(40):
            xy = self.sample_xy(self.footprint(self.obj), placed, clearance=0.03, **self.start_region)
            ahead = self.side * (xy[0] - centre[0]) > half[0]
            if not (ahead and abs(xy[1] - centre[1]) < half[1] + 0.05):
                break
        else:
            raise RuntimeError("object start already satisfies the relation")
        self.set_object_pose(self.obj, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((xy, self.footprint(self.obj)))
        self.place_distractors(placed)
        mujoco.mj_forward(self.model, self.data)
        # The goal spot (at the object's mid-height and on the table, as the recorder screens it), the reference
        # and the object must be clearly visible and well inside the front image.
        table = [(x, y, 0.001) for x, y, _ in self.goal_points(target)]
        if (visible_fraction(self, self.goal_points(target))[0] < 0.9 or visible_fraction(self, table)[0] < 0.8):
            raise RuntimeError("goal spot not clearly visible from the front camera")
        h = obj_half
        goal_box = ((target[0] - h, target[1] - h, 0.0), (target[0] + h, target[1] + h, self.height(self.obj)))
        if not all(in_frame(self, box_corners(*b)) for b in (goal_box, world_box(self, self.reference),
                                                           world_box(self, self.obj))):
            raise RuntimeError("goal, reference or object near the image border")
        self.set_goals(Goal(self.obj, "table", target=(*target, self._extent[self.obj]["bottom"]),
                            tolerance=(0.012, 0.012, 0.006), upright_cos=COS10, check=relation_holds))


class FrontBehindOracle(TrainOracle):
    """Top-down pick, then placement at the goal spot with the wrist turned so the gripper (its wrist-camera
    mount sticks out ~8 cm sideways, 2.4 cm above the TCP) keeps clear of the reference. Per-task grasp
    parameters live on the env class: ``grasp_width``, ``grasp_yaw_offset`` (added to the object's yaw),
    ``grasp_symmetric`` and ``grasp_z_min``."""

    hand_bodies = ("gripper", "moving_jaw_so101_v1", "camera_mount", "wrist")
    clearance_cap = 0.015        # clear enough: beyond this, prefer the smaller wrist turn

    def _hand_points(self):
        """Collision-box corners of the hand in the TCP frame (measured once at the start pose)."""
        if not hasattr(self, "_hand_local"):
            env, m, d = self.env, self.model, self.env.data
            mujoco.mj_kinematics(m, d)
            tcp, rot = d.site_xpos[env._tcp_site_id], d.site_xmat[env._tcp_site_id].reshape(3, 3)
            corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            boxes = []
            for name in self.hand_bodies:
                body = m.body(name).id
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                        world = (corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T \
                            + d.geom_xpos[g]
                        boxes.append((world - tcp) @ rot)
            self._hand_local = boxes
        return self._hand_local

    def hand_clearance(self, tcp, rot):
        """Smallest AABB separation between the hand (TCP at ``tcp``/``rot``) and the reference."""
        lo_r, hi_r = world_box(self.env, self.env.reference)
        best = self.clearance_cap
        for local in self._hand_points():
            world = local @ rot.T + tcp
            lo, hi = world.min(0), world.max(0)
            best = min(best, float(np.max(np.maximum(lo - hi_r, lo_r - hi))))
        return best

    def placement_rotation(self, name, target):
        """Among wrist turns of the carry rotation (multiples of 45 deg; the goal has no yaw requirement), the
        reachable one with the most hand clearance from the reference; ties go to the smaller turn."""
        base, held = self.carry_rot(target[:2]), self.held(name)
        options = []
        for delta in sorted((k * np.pi / 4 for k in range(-3, 5)), key=abs):
            c, s = np.cos(delta), np.sin(delta)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ base
            tcp = np.asarray(target, float) - rot @ held
            # The release pose must be exact; the carry pose is never exactly vertical at carry height (the
            # IK tilts it, 5-10 mm residual), and the align step corrects that before lowering.
            ok = self.solve(tcp, rot)[1] <= 0.002 and self.solve(np.r_[tcp[:2], self.carry_z], rot)[1] <= 0.015
            options.append((ok, delta, rot, self.hand_clearance(tcp, rot)))
        best = None
        for ok, delta, rot, clear in options:
            score = np.floor(min(clear, self.clearance_cap) / 0.005 + 1e-9)
            if ok and (best is None or score > best[0]):
                best = (score, rot)
        return base if best is None else best[1]

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """The kit's top-down pick, centred on the object's collision-box centre instead of its body origin
        (LIBERO cans sit up to 2 mm off their origin, enough for the fixed finger to land on the rim)."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            lo, hi = world_box(self.env, name)
            centre = 0.5 * (lo[:2] + hi[:2])
            base = np.arctan2(centre[1], centre[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.r_[centre, grasp_z], width, angles)
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

    def plan(self):
        env = self.env
        name = env.obj
        lowest = env.object_pos(name)[2] - env._extent[name]["bottom"]
        grasp_z = max(lowest + env.height(name) / 2 - 0.001, env.grasp_z_min)
        ok = yield from self.pick(name, env.grasp_width, grasp_z, yaw=env.yaw(name) + env.grasp_yaw_offset,
                                  symmetric=env.grasp_symmetric)
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        rot = self.placement_rotation(name, target)
        yield from self.place_object(name, target, rot=rot,
                                     open_to=self.open_for(env.grasp_width,
                                                           getattr(env, "release_margin", self.release_margin)))
        yield from self.rest()
        yield from self.wait(1.2)


class TallReferenceOracle(FrontBehindOracle):
    carry_z = 0.12               # the held 6.6 cm carton (held at mid-height) clears the 7.3 cm basket


# ----- tasks -----------------------------------------------------------------------------------------------

class MilkFrontOfBasketEnv(FrontBehindEnv):
    instruction = "Put the milk carton in front of the basket."
    obj, reference, side = "milk", "basket", 1
    task_objects = ("milk",)
    tall_reference = True
    gap = 0.04
    grasp_width, grasp_yaw_offset, grasp_symmetric, grasp_z_min = 0.027, 0.0, 4, 0.012
    distractor_pool = LOW_DISTRACTORS

    def scene_objects(self):
        return [Obj("milk", "milk")]

    def scene_fixtures(self):
        return [Fixture("basket", "basket", 0.5)]


class JuiceBehindPlateEnv(FrontBehindEnv):
    instruction = "Put the orange juice carton behind the plate."
    obj, reference, side = "juice", "plate", -1
    task_objects = ("juice",)
    grasp_width, grasp_yaw_offset, grasp_symmetric, grasp_z_min = 0.027, 0.0, 4, 0.012
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n != "plate")

    def scene_objects(self):
        return [Obj("juice", "orange_juice")]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.65)]


class BlockOracle(FrontBehindOracle):
    # Release the cube at its flat height: with the kit's half-tilt lift it leans a few tenths of a degree on
    # the fixed finger and drops flat when the finger backs off (1000-1500 rad/s^2 spikes in 2/10 calibration
    # rollouts; 49/50 at 0.0 over seeds 9000-9049).
    release_lift_fraction = 0.0


class CartonOracle(FrontBehindOracle):
    carry_z = 0.095              # the 6.6 cm carton, held at mid-height, clears the <= 4 cm distractors


class BlockBehindBowlEnv(FrontBehindEnv):
    instruction = "Put the red block behind the bowl."
    obj, reference, side = "block", "bowl", -1
    task_objects = ("block",)
    tall_reference = True
    grasp_width, grasp_yaw_offset, grasp_symmetric, grasp_z_min = 0.028, 0.0, 4, 0.012
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in BOWLS)

    def scene_objects(self):
        return [Block("block", half=(0.014,) * 3, rgba=RED)]

    def scene_fixtures(self):
        return [Fixture("bowl", "akita_black_bowl", 0.75)]


class BarFrontOfMugEnv(FrontBehindEnv):
    instruction = "Put the long blue block in front of the mug."
    obj, reference, side = "bar", "mug", 1
    task_objects = ("bar",)
    tall_reference = True
    gap = 0.035
    grasp_width, grasp_yaw_offset, grasp_symmetric, grasp_z_min = 2 * BAR[1], np.pi / 2, 2, 0.012
    distractor_pool = LOW_DISTRACTORS

    def scene_objects(self):
        return [Block("bar", half=BAR, rgba=BLUE)]

    def scene_fixtures(self):
        return [Fixture("mug", "porcelain_mug", 0.5)]


def _task(name, env, oracle, kind, relation, goal, obj_text, ref_text):
    return define_task(name=name, instruction=env.instruction, family=FAMILY, env=env, oracle=oracle,
                       objects=(env.obj,), object_kinds=(kind,), relation=relation, goal=goal,
                       steps=(step_text("put", obj_text, relation, ref_text),))


TASKS = [
    _task("milk_carton_in_front_of_basket", MilkFrontOfBasketEnv, TallReferenceOracle, "milk carton", "in front of",
          "basket", "milk carton", "basket"),
    _task("juice_carton_behind_plate", JuiceBehindPlateEnv, CartonOracle, "juice carton", "behind", "plate",
          "orange juice carton", "plate"),
    _task("block_behind_bowl", BlockBehindBowlEnv, BlockOracle, "block", "behind", "bowl",
          "red block", "bowl"),
    _task("long_block_in_front_of_mug", BarFrontOfMugEnv, FrontBehindOracle, "long block", "in front of", "mug",
          "long blue block", "mug"),
]

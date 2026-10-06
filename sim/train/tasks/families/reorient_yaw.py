"""reorient_yaw: pick an object up, turn it about the vertical in the air and put it down with a required heading.

Distinct from the pilot ``bar_crosswise_on_mat`` (a bar laid crosswise on a mat): each task here has its own object
and heading goal, always judged from the front (viewer) camera or relative to another object:

* ``turn_book_cover_to_camera``: a standing book is turned in place until its cover faces the camera (the book's
  thin axis along the camera's line of sight, either cover).
* ``long_block_lengthwise_on_tray``: a long block is put on a randomly rotated rectangular tray, lined up with the
  tray's long side.
* ``turn_ketchup_label_to_camera``: a standing ketchup bottle is turned in place (up to 180 deg) until its labelled
  front faces the camera (one heading only; the back has no label).

Strict success (beyond the kit's released/settled/supported/upright goal): the heading predicate, and the object
within 2 cm of its goal spot (its start spot for in-place turns, the tray centre for the tray).
"""

import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, GRASP_REGION, LOW_DISTRACTORS, Block, Goal, Obj, TrainEnv,
                                  TrainOracle, define_task, mat)

FAMILY = "reorient_yaw"


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


class YawOracle(TrainOracle):
    """Pick across the object's narrow side, turn it in the air about the vertical, put it down."""

    width = 0.02                 # closing width (narrow side)
    narrow_local = 1             # local axis across which the jaws close (0: x, 1: y)
    grasp_offset = 0.0           # TCP height above the object origin at the grasp
    min_grasp_z = 0.0105
    touch_depth = 0.001
    touch_seconds = 0.9
    lift_z = None                # TCP height while turning (default: carry_z)
    squeeze = None               # close to this far inside the width (None: fully closed)

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Staged-close top-down pick (as relative_left_right): ease onto the object, then squeeze."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.grasp_centre(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
            yield from self.gripper(CLOSED if self.squeeze is None else self.open_for(width, -self.squeeze),
                                    self.close_seconds, 0.3)
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

    def grasp_centre(self, name):
        """World xy of the centre of the object's collision box (scanned origins can sit off-centre)."""
        env, m, d = self.env, self.model, self.env.data
        R, p = d.xmat[env._body[name]].reshape(3, 3), env.object_pos(name)
        corners = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        pts = []
        for g in env._geoms[name]:
            pts.append((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T
                       + d.geom_xpos[g])
        local = (np.vstack(pts) - p) @ R
        mid = (local.min(0) + local.max(0)) / 2
        return p + R @ np.r_[mid[:2], 0.0]

    def desired_yaws(self):
        """World yaws (of the object's local x axis) that satisfy the goal, best first."""
        raise NotImplementedError

    def plan(self):
        env = self.env
        goal = env.goals[0]
        name = goal.obj
        p = env.object_pos(name)
        grasp_z = max(p[2] + self.grasp_offset, self.min_grasp_z)
        yaw0 = env.yaw(name)
        ok = yield from self.pick(name, self.width, grasp_z, yaw=yaw0 + self.narrow_local * np.pi / 2, symmetric=2,
                                  lift_z=self.lift_z)
        if not ok:
            return
        target = env.goal_target(goal)
        current = env.yaw(name)
        rotations = []
        for want in self.desired_yaws():
            d = wrap(want - current)
            rotations.append((abs(d), rotz(d) @ self._cmd_rot))
        rotations = [r for _, r in sorted(rotations, key=lambda t: t[0])]
        rot = self.feasible_rotation(name, target, rotations, carry_z=self.lift_z)
        yield from self.place_object(name, target, rot=rot, open_to=self.release_for(self.width), carry_z=self.lift_z)
        yield from self.rest()
        yield from self.wait(1.2)


# ----- task 1: turn the standing book so its cover faces the camera ----------------------------------------

def cover_faces_viewer(env, name):
    """Book's thin (local x) axis along world x (the front camera's line of sight) within 10 deg, standing."""
    return env.axis_alignment(name, (1.0, 0.0, 0.0), local_axis=0) >= COS10


class BookEnv(TrainEnv):
    instruction = "Turn the book so its cover faces the camera."
    task_objects = ("book",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("plate",))
    start_region = dict(r=(0.17, 0.25), angle=(-50.0, 50.0))

    def scene_objects(self):
        return [Obj("book", "black_book")]

    def layout(self):
        placed = []
        xy, _ = self.place("book", placed, region=self.start_region, yaw=0.0)
        # Start edge-on or oblique: the cover normal 35-90 deg away from the camera axis (either side).
        off = self.np_random.uniform(np.radians(35), np.radians(90)) * self.np_random.choice([-1, 1])
        self.set_object_pose("book", xy, yaw=off + self.np_random.choice([0.0, np.pi]))
        placed[-1] = (xy, self.footprint("book") + 0.01)
        self.place_distractors(placed)
        z = self.object_pos("book")[2]
        self.set_goals(Goal("book", "table", target=(*xy, z), tolerance=(0.02, 0.02, 0.006),
                            check=cover_faces_viewer))


class BookOracle(YawOracle):
    width = 0.0143
    narrow_local = 0             # close across the book's thickness (local x)
    grasp_offset = 0.052         # near the top of the 6.8 cm book (origin at its base)
    open_margin = 0.016
    carry_z = 0.095
    lift_z = 0.095

    def desired_yaws(self):
        return [0.0, np.pi]


# ----- task 2: long block lined up with the rectangular tray -------------------------------------------------

LONG = (0.03, 0.009, 0.009)        # 6.0 x 1.8 x 1.8 cm
TRAY = (0.06, 0.035)               # 12 x 7 cm tray (thin board)
TRAY_T = 0.006


def lengthwise_on(tray):
    def check(env, name):
        return env.axis_alignment(name, env.data.xmat[env._body[tray]].reshape(3, 3)[:, 0]) >= COS10
    return check


class BlockTrayEnv(TrainEnv):
    instruction = "Put the long block on the tray, lined up with the tray's long side."
    task_objects = ("block",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("plate",))

    def scene_objects(self):
        return [Block("block", half=LONG, rgba=(0.75, 0.55, 0.3, 1.0), mass=0.03),
                mat("tray", half=TRAY, thickness=TRAY_T, rgba=(0.55, 0.38, 0.22, 1.0), mass=0.08)]

    def layout(self):
        placed = []
        tray_xy = self.sample_xy(0.075, placed, r=(0.18, 0.25), angle=(-55.0, 55.0))
        tray_yaw = self.np_random.uniform(-np.pi, np.pi)
        self.set_object_pose("tray", tray_xy, yaw=tray_yaw)
        placed.append((tray_xy, 0.07))
        # The block starts within 70 deg of the tray's azimuth: a held block slips in the jaws while the arm turns
        # about the pan axis (11 mm over a 125 deg turn, measured), which left it hanging from the fingertips.
        az = np.degrees(np.arctan2(tray_xy[1], tray_xy[0]))
        lo, hi = GRASP_REGION["angle"]
        region = dict(r=GRASP_REGION["r"], angle=(max(lo, az - 70.0), min(hi, az + 70.0)))
        xy, _ = self.place("block", placed, region=region)
        yaw = tray_yaw + self.np_random.choice([-1, 1]) * self.np_random.uniform(np.radians(35), np.radians(90))
        self.set_object_pose("block", xy, yaw=yaw)
        self.place_distractors(placed)
        self.set_goals(Goal("block", "tray", target=(*tray_xy, TRAY_T + LONG[2]), reference="tray",
                            tolerance=(0.02, 0.02, 0.006), check=lengthwise_on("tray")))


class BlockTrayOracle(YawOracle):
    width = 2 * LONG[1]
    narrow_local = 1
    close_seconds = 1.6
    release_seconds = 1.4
    release_margin = 0.016
    squeeze = 0.008

    def desired_yaws(self):
        t = self.env.yaw("tray")
        return [t, t + np.pi]


# ----- task 3: turn the ketchup bottle so its label faces the camera -------------------------------------------

def label_faces_viewer(env, name):
    """The labelled front (local +x) points at the front camera (world +x) within 12 deg, standing."""
    return env.data.xmat[env._body[name]].reshape(3, 3)[:, 0] @ np.array([1.0, 0.0, 0.0]) >= np.cos(np.radians(12))


class KetchupEnv(TrainEnv):
    instruction = "Turn the ketchup bottle so its label faces the camera."
    task_objects = ("ketchup",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("plate",))
    start_region = dict(r=(0.17, 0.25), angle=(-50.0, 50.0))

    def scene_objects(self):
        return [Obj("ketchup", "ketchup")]

    def layout(self):
        placed = []
        xy, _ = self.place("ketchup", placed, region=self.start_region, yaw=0.0)
        # Label turned 60-180 deg away from the camera, either way.
        off = self.np_random.uniform(np.radians(60), np.pi) * self.np_random.choice([-1, 1])
        self.set_object_pose("ketchup", xy, yaw=off)
        placed[-1] = (xy, self.footprint("ketchup") + 0.01)
        self.place_distractors(placed)
        z = self.object_pos("ketchup")[2]
        self.set_goals(Goal("ketchup", "table", target=(*xy, z), tolerance=(0.02, 0.02, 0.006),
                            check=label_faces_viewer))


class KetchupOracle(YawOracle):
    width = 0.0184
    narrow_local = 0             # close across the bottle's thickness (front/back faces)
    grasp_offset = 0.0           # origin at mid-height
    open_margin = 0.016
    carry_z = 0.095
    lift_z = 0.095

    def desired_yaws(self):
        return [0.0]


TASKS = [
    define_task(name="turn_book_cover_to_camera", instruction=BookEnv.instruction, family=FAMILY, env=BookEnv,
                oracle=BookOracle, objects=("book",), object_kinds=("book",),
                relation="turned in place to face", goal="viewer camera",
                steps=("Pick up the standing book, turn it so its cover faces the camera, and stand it back down.",)),
    define_task(name="long_block_lengthwise_on_tray", instruction=BlockTrayEnv.instruction, family=FAMILY,
                env=BlockTrayEnv, oracle=BlockTrayOracle, objects=("block",), object_kinds=("long block",),
                relation="on aligned lengthwise with", goal="rectangular tray",
                steps=("Pick up the long block, turn it to match the tray's length, and set it on the tray.",)),
    define_task(name="turn_ketchup_label_to_camera", instruction=KetchupEnv.instruction, family=FAMILY, env=KetchupEnv,
                oracle=KetchupOracle, objects=("ketchup",), object_kinds=("ketchup bottle",),
                relation="turned in place so its label faces", goal="viewer camera",
                steps=("Pick up the ketchup bottle, turn it so the label faces the camera, and stand it back down.",)),
]

"""lay_down: pick up a standing tall object and lay it on its side at a goal (on a mat, next to a plate).

The SO-101 is a 5-DoF arm: the gripper approach always lies in the vertical plane through the pan axis, and the
wrist flex saturates at +-95 deg. A ~90 deg pitch of the held object is therefore split between the grasp and the
release (measured reachability, 2026-10-06):

* the grasp approach is tilted 20 deg *toward* the robot (reachable only up to a TCP height of ~5.5 cm, r <= 0.2 m),
  so the hand slides onto the object from the side (the jaw cavity is open along the TCP y axis) instead of
  descending over its top;
* the release approach is pitched ~70 deg away from the robot, which is only reachable for a TCP at r >= ~0.24 m,
  so goals sit at r 0.258-0.276 m and the object starts inside the same azimuth sector (r 0.14-0.19 m);
* the object is grasped at its centre of mass: the jaws pinch it at the fingertip spheres only, so it can pivot
  about the pinch (closing) axis, which is exactly the pitch axis;
* the carry is kept mostly radial: a held object slips in the jaws while the arm turns about the pan axis
  (9.5 mm over a 97 deg turn vs 1.2 mm over a short radial carry, measured), and slipped objects were levered out
  by the palm during the pitch;
* the release pose is re-planned from the measured in-hand pose before the final descent, the grasp closing
  direction is chosen so the wrist-camera mount stays above the table at the release, and the hand is unfolded
  (pitched back to vertical while rising) before the rest move, which otherwise brushes the lower arm against the
  shoulder.

Strict success: the object's tall axis horizontal within 10 deg, at the goal (+-1.5 cm), released, settled and
supported (by the mat / the table without touching the plate).
"""

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, FIXED_FACE, Fixture, Goal, Obj, TrainEnv, TrainOracle,
                                  define_task, mat)
from sim.val.oracle import PAN_AXIS, interp_rot

FAMILY = "lay_down"
SIN10 = float(np.sin(np.radians(10)))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def lying(env, name):
    """The object's tall (local z) axis is horizontal within 10 deg."""
    return abs(env.data.xmat[env._body[name]].reshape(3, 3)[2, 2]) <= SIN10


def item_points(env, name):
    """World points of the collision geometry (mesh vertices, box corners) of a free body."""
    return np.vstack([item_points_geom(env, g) for g in env._geoms[name]])


def local_points(env, name):
    return (item_points(env, name) - env.object_pos(name)) @ env.data.xmat[env._body[name]].reshape(3, 3)


def section(env, name, z):
    """Local-frame points of the collision geoms whose local z-range contains ``z`` (the cross-section there)."""
    m, d = env.model, env.data
    R, p = d.xmat[env._body[name]].reshape(3, 3), env.object_pos(name)
    out = []
    for g in env._geoms[name]:
        pts = (item_points_geom(env, g) - p) @ R
        if pts[:, 2].min() - 1e-4 <= z <= pts[:, 2].max() + 1e-4:
            out.append(pts)
    return np.vstack(out)


def item_points_geom(env, g):
    m, d = env.model, env.data
    Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == 7:
        mid = m.geom_dataid[g]
        local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    else:
        corners = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        local = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
    return local @ Rg.T + pg


class LayEnv(TrainEnv):
    obj = ""
    down_axis = None              # local axis (0/1) that must end facing down (None: either side face)
    start_region = dict(r=(0.14, 0.19), angle=(-55.0, 55.0))
    target_region = dict(r=(0.258, 0.276), angle=(-40.0, 40.0))
    start_sector = 25.0
    face_jitter = 0.22            # rad: start yaw = radial + k*90 deg + U(-j, j)

    def start_yaw(self, xy):
        k = self.np_random.integers(4) if self.down_axis is None else 2 * self.np_random.integers(2) + self.down_axis
        az = np.arctan2(xy[1] - PAN_AXIS[1], xy[0] - PAN_AXIS[0])
        # local axis ``down_axis`` radial: yaw so that local x (k even) or local y (k odd) points radially
        return float(az - k * np.pi / 2 + self.np_random.uniform(-self.face_jitter, self.face_jitter))

    def lying_height(self, axis):
        """Origin height above the support when lying on a face with local normal ``axis`` (0/1) down."""
        pts = local_points(self, self.obj)
        return float(max(-pts[:, axis].min(), pts[:, axis].max()))

    def place_object_standing(self, placed, target_xy):
        """Stand the object inside the target's azimuth sector (+-``start_sector`` deg): turning the arm about the
        pan axis with the object held makes it slip in the jaws (9.5 mm over a 97 deg pan turn, 1.2 mm over a short
        radial carry, measured), so the carry is kept mostly radial."""
        az = np.degrees(np.arctan2(target_xy[1], target_xy[0]))
        lo, hi = self.start_region["angle"]
        region = dict(r=self.start_region["r"], angle=(max(lo, az - self.start_sector), min(hi, az + self.start_sector)))
        xy, _ = self.place(self.obj, placed, region=region, yaw=0.0)
        self.set_object_pose(self.obj, xy, yaw=self.start_yaw(xy))
        # The hand slides on from the side and swings the wrist-camera mount around the object: keep distractors
        # further off than the default clearance.
        placed[-1] = (xy, self.footprint(self.obj) + 0.08)
        return xy, placed


class LayOracle(TrainOracle):
    grasp_tilt = np.radians(20)   # approach tilted toward the robot at the grasp (pitched to ~70 deg at the place)
    grasp_local_z = -0.006
    carry_z = 0.10
    touch_depth = 0.001
    touch_seconds = 0.9
    lift = 0.0015                 # release this far above the seated height
    retreat = 0.05
    side_approach = 0.045
    open_margin = 0.024           # the carton top reaches deep between the jaws, where the pivoting jaw narrows
    pitch_split = 0.6
    carry_speed = 0.16
    carry_phi = 0.0              # approach pitch while carrying (vertical)
    ik_tolerance = 0.0035
    pitch_height = 0.05

    # ----- geometry --------------------------------------------------------------------------------------------
    def _out(self, xy):
        v = np.asarray(xy[:2], float) - PAN_AXIS
        return np.r_[v / np.linalg.norm(v), 0.0]

    def grasp_frame(self):
        """(rot, tcp, width, n_down_local_sign_axis)."""
        env = self.env
        name = env.obj
        R0, p0 = env.data.xmat[env._body[name]].reshape(3, 3), env.object_pos(name)
        out = self._out(p0)
        a = np.cos(self.grasp_tilt) * np.array([0, 0, -1.0]) - np.sin(self.grasp_tilt) * out
        # n2: the local horizontal axis most aligned with the radial direction (it ends facing down)
        axes = [0, 1] if env.down_axis is None else [env.down_axis]
        n2 = max(axes, key=lambda i: abs(R0[:, i] @ out))
        n1 = 1 - n2
        pts = local_points(env, name)
        zc = self.grasp_local_z
        band = section(env, name, zc)
        width = float(np.ptp(band[:, n1]))
        mid1 = float((band[:, n1].min() + band[:, n1].max()) / 2)
        centre_local = np.zeros(3)
        centre_local[2] = zc
        centre_local[n1] = mid1
        centre = p0 + R0 @ centre_local
        self._n2 = n2
        self._s2 = 1.0 if R0[:, n2] @ out < 0 else -1.0     # the face towards the robot ends facing down
        target = env.goal_target(env.goals[0])
        best = None
        for sign in (1, -1):
            x = sign * R0[:, n1]
            x = x - (x @ a) * a
            x /= np.linalg.norm(x)
            rot = np.column_stack([x, np.cross(a, x), a])
            tcp = centre - (FIXED_FACE - width / 2 - self.grasp_clearance) * x
            q, err, tilt = self.solve(tcp, rot)
            # Predict the release pose for this closing direction: the wrist-camera mount must stay off the
            # table and the folded elbow off the shoulder.
            rot_p, tcp_p = self.place_frame(target, rot.T @ R0, rot.T @ (p0 - tcp))
            q_p, err_p = self.solve_branch(tcp_p, rot_p)
            cam, fold = self.clearances(q_p)
            score = (err > 0.003 or err_p > 0.004, -round(min(cam, 0.02) + min(fold, 0.01), 3), err + err_p)
            self.log.append(("grasp_option", sign, round(err * 1000, 1), round(err_p * 1000, 1),
                             round(cam * 1000, 1), round(fold * 1000, 1)))
            if best is None or score < best[0]:
                best = (score, rot, tcp, q, err)
        self.log.append(("grasp_ik_mm", round(best[4] * 1000, 1)))
        return best[1], best[2], best[3], width, n2

    def clearances(self, q):
        """(lowest wrist-camera-mount point above the table, shoulder to lower-arm distance) at joints ``q``."""
        env, m, d = self.env, self.model, self.ik_data
        d.qpos[:] = env.data.qpos
        d.qpos[env._arm_qpos_addrs] = q
        mujoco.mj_kinematics(m, d)

        def geoms(*bodies):
            ids = {m.body(b).id for b in bodies}
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]

        low = min(float((d.geom_xpos[g] - np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
                         + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3])[2]) for g in geoms("camera_mount"))
        fold = min(mujoco.mj_geomDistance(m, d, g1, g2, 0.1, None) for g1 in geoms("shoulder")
                   for g2 in geoms("lower_arm"))
        return low, fold

    def place_frame(self, target, R_rel=None, held=None):
        """TCP (rot, pos) laying the held object with its origin at ``target``: tall axis horizontal and
        pointing at the robot, the face that faced the robot at the grasp facing down. ``R_rel``/``held``: the
        object's pose in the TCP frame (default: measured now)."""
        env = self.env
        name = env.obj
        if R_rel is None:
            R_rel = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
            held = self.held(name)
        n2, s2 = self._n2, self._s2
        out = self._out(target)
        best = None
        heading = np.arctan2(out[1], out[0])
        for _ in range(6):
            o = np.array([np.cos(heading), np.sin(heading), 0.0])
            # world images of local axes: z -> -o (top towards robot), s2*e_n2 -> down
            ez, en2 = -o, np.array([0, 0, -1.0]) * s2
            R_des = np.zeros((3, 3))
            R_des[:, 2] = ez
            R_des[:, n2] = en2
            other = 1 - n2
            R_des[:, other] = np.cross(R_des[:, (other + 1) % 3], R_des[:, (other + 2) % 3])
            rot = R_des @ R_rel.T
            tcp = np.asarray(target, float) - rot @ held
            az_tcp = np.arctan2(tcp[1] - PAN_AXIS[1], tcp[0] - PAN_AXIS[0])
            ah = np.arctan2(rot[1, 2], rot[0, 2])
            heading += float(np.angle(np.exp(1j * (az_tcp - ah))))
        return rot, tcp

    # ----- skills ----------------------------------------------------------------------------------------------
    def tilted_pick(self):
        env = self.env
        name = env.obj
        self._grasp_out = self._out(env.object_pos(name))
        rot, tcp, q, width, n2 = self.grasp_frame()
        self._width = width
        a = rot[:, 2]
        yield from self.gripper(self.open_for(width), 0.4, 0.0)
        # Slide on from the side (the jaw cavity is open along y; the wrist-camera mount is on +y): a tilted TCP
        # cannot rise above the carton's top (wrist flex limit), so a descent along the approach is not possible.
        pre = tcp + self.side_approach * rot[:, 1]
        yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
        yield from self.move(tcp, rot, speed=0.04, tol=0.003, label="grasp")
        yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
        yield from self.gripper(CLOSED, self.close_seconds, 0.3)
        ok = env.is_grasping(name)
        self.log.append((name, "grasp", ok))
        if not ok:
            return False
        yield from self.move(tcp - 0.025 * a, rot, speed=0.05, label="lift")
        # Level the approach to vertical while rising to carry height (a top-down TCP reaches higher).
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], self.pitched(rot, self.carry_phi), speed=0.08, label="raise")
        return env.is_grasping(name)

    def solve_branch(self, pos, rot, ref=None):
        """IK over several arm configurations (elbow/wrist branches): among the accurate solutions the one closest
        to ``ref`` (default: the current joints), else the most accurate."""
        ref = self.q if ref is None else ref
        best = None
        pan0 = self.q[0]
        for lift in (0.0, 0.4, 0.8):
            for elbow in (1.0, 1.4, 1.65):
                for flex in (-1.5, -1.0):
                    for roll in (-1.55, 1.55):
                        q = np.array([pan0, lift, elbow, flex, roll])
                        for _ in range(4):
                            q = self.ik(pos, rot, seed=q)
                        p = self._fk(q)
                        R = self.ik_data.site_xmat[self.env._tcp_site_id].reshape(3, 3)
                        ang = np.arccos(np.clip((np.trace(R.T @ rot) - 1) / 2, -1, 1))
                        err = float(np.linalg.norm(p - pos)) + 0.02 * ang
                        key = (err > self.ik_tolerance, float(np.abs(q - ref).sum()) if err <= self.ik_tolerance else err)
                        if best is None or key < best[0]:
                            best = (key, q, err)
        return best[1], best[2]

    def pitched(self, rot, phi):
        """``rot`` re-pitched so the approach is ``phi`` from straight down (positive: tilted away from the robot),
        keeping the closing axis as close as possible."""
        out = self._out(self._cmd_pos)
        new_a = np.cos(phi) * np.array([0, 0, -1.0]) + np.sin(phi) * out
        x = rot[:, 0] - (rot[:, 0] @ new_a) * new_a
        x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(new_a, x), new_a])

    def lay(self, target, hover=0.012):
        env = self.env
        rot, tcp = self.place_frame(target)
        q, err, tilt = self.solve(tcp, rot)
        self.log.append(("place_ik_mm", round(err * 1000, 1)))
        # Carry upright to above the release point, then pitch the carton over while descending to the hover.
        over = np.r_[tcp[:2], self.carry_z]
        az = np.arctan2(over[1] - PAN_AXIS[1], over[0] - PAN_AXIS[0]) - np.arctan2(
            self._cmd_pos[1] - PAN_AXIS[1], self._cmd_pos[0] - PAN_AXIS[0])
        yield from self.move(over, rotz(az) @ self._cmd_rot, speed=self.carry_speed, label="carry")
        rot, tcp = self.place_frame(target)
        # Pitch most of the way while high (the wrist-flex limit blocks the full pitch there), then finish the
        # pitch while dropping to just above the release pose.
        mid = interp_rot(self._cmd_rot, rot, self.pitch_split)
        yield from self.move(tcp + [0, 0, self.pitch_height], mid, speed=0.05, label="pitch")
        rot, tcp = self.place_frame(target)
        yield from self.move(tcp + [0, 0, hover], rot, speed=0.04, label="pitch2")
        rot, tcp = self.place_frame(target)
        self.log.append(("replan_ik_mm", round(self.solve(tcp, rot)[1] * 1000, 1)))
        yield from self.move(tcp + [0, 0, 0.006], rot, speed=0.04, tol=self.release_tolerance, settle=0.6,
                             label="hover")
        rot, tcp = self.place_frame(target)
        yield from self.move(tcp + [0, 0, self.lift], rot, speed=0.03, tol=self.release_tolerance, settle=0.8,
                             label="lower")
        yield from self.release(self.release_for(self._width))
        # The jaws open sideways: rise straight up off the carton (withdrawing along the approach folds the arm
        # into itself).
        yield from self.move(self._cmd_pos + [0, 0, self.retreat], self._cmd_rot, speed=0.06, label="up")
        # Unfold (pitch the hand back to vertical while rising) before resting: folding straight from the
        # release posture brushes the lower arm against the shoulder.
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z + 0.01], self.pitched(self._cmd_rot, 0.0),
                             speed=0.08, label="unfold")

    def plan(self):
        env = self.env
        ok = yield from self.tilted_pick()
        if not ok:
            return
        yield from self.lay(env.goal_target(env.goals[0]))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- milk carton on the mat --------------------------------------------------------------------------------

MAT_T = 0.006


class MilkOnMatEnv(LayEnv):
    instruction = "Lay the milk carton down on its side on the mat."
    task_objects = ("milk",)
    obj = "milk"
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("plate",))

    def scene_objects(self):
        return [Obj("milk", "milk"), mat("mat", half=(0.05, 0.032), rgba=(0.3, 0.5, 0.7, 1.0))]

    def layout(self):
        placed = []
        mat_xy = self.sample_xy(0.06, placed, **self.target_region)
        az = np.arctan2(mat_xy[1] - PAN_AXIS[1], mat_xy[0] - PAN_AXIS[0])
        self.set_object_pose("mat", mat_xy, yaw=az + self.np_random.uniform(-0.15, 0.15))
        placed.append((mat_xy, 0.06))
        out = np.array([np.cos(az), np.sin(az)])
        placed.append((mat_xy - 0.05 * out, 0.03))      # the gripper at the release
        xy, placed = self.place_object_standing(placed, mat_xy)
        self.place_distractors(placed)
        z = MAT_T + self.lying_height(0)
        self.set_goals(Goal("milk", "mat", target=(*mat_xy, z), reference="mat", tolerance=(0.015, 0.015, 0.006),
                            upright_cos=None, check=lambda env, n: lying(env, n)))


# ----- ketchup bottle on its side next to the plate ---------------------------------------------------------

PLATE_GAP = 0.015                 # clear gap between the laid bottle and the plate rim (nominal)


class KetchupBesidePlateEnv(LayEnv):
    instruction = "Lay the ketchup bottle on its side next to the plate."
    task_objects = ("ketchup",)
    obj = "ketchup"
    down_axis = 1                 # rests on its narrow side (closing across the labelled faces)
    distractor_pool = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn",
                       "ramekin")

    def scene_objects(self):
        return [Obj("ketchup", "ketchup")]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.65)]

    def plate_radius(self):
        return self.fixture_footprint("plate") / np.sqrt(2)

    def beside(self, env=None):
        def check(env, name):
            return lying(env, name) and not env.touching(name, "plate") and \
                env.within_radius(name, env.object_pos("plate"), env.plate_radius() + 0.05)
        return check

    def layout(self):
        placed = []
        target = self.sample_xy(0.045, placed, **self.target_region)
        az = np.arctan2(target[1] - PAN_AXIS[1], target[0] - PAN_AXIS[0])
        tangent = np.array([-np.sin(az), np.cos(az)]) * self.np_random.choice([-1.0, 1.0])
        plate_xy = target + tangent * (self.plate_radius() + 0.014 + PLATE_GAP)
        self.set_fixture_pose("plate", plate_xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed += [(plate_xy, self.plate_radius()), (target, 0.045)]
        xy, placed = self.place_object_standing(placed, target)
        self.place_distractors(placed)
        z = self.lying_height(self.down_axis)
        self.set_goals(Goal("ketchup", "table", target=(*target, z), tolerance=(0.015, 0.015, 0.006),
                            upright_cos=None, check=self.beside()))


class KetchupLayOracle(LayOracle):
    grasp_local_z = -0.0045       # centre of mass
    lift = -0.0025                # seat it: released at the nominal height it hung ~2 mm up and the opening jaw flicked it


TASKS = [
    define_task(name="lay_milk_carton_on_mat", instruction=MilkOnMatEnv.instruction, family=FAMILY,
                env=MilkOnMatEnv, oracle=LayOracle, objects=("milk",), object_kinds=("milk carton",),
                relation="laid on its side on", goal="mat",
                steps=("Pick up the standing milk carton, tip it over and lay it on its side on the mat.",)),
    define_task(name="lay_ketchup_bottle_beside_plate", instruction=KetchupBesidePlateEnv.instruction, family=FAMILY,
                env=KetchupBesidePlateEnv, oracle=KetchupLayOracle, objects=("ketchup",),
                object_kinds=("ketchup bottle",), relation="laid on its side next to", goal="plate",
                steps=("Pick up the standing ketchup bottle, tip it over and lay it on its side next to the plate.",)),
]

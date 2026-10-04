"""Scripted skills for the validation oracles.

The oracle solves IK itself and drives the env in `pd_joint_pos`, so every action it yields is the exact joint
target the robot was commanded (recorded as `action`); `observation.state` is the measured joint state.
Skills are generators of 6-vector joint targets; chain them with `yield from`.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim.val.env import REST_DEG

OPEN, RELEASE, CLOSED = 0.70, 0.55, -0.17
CARRY_Z = 0.10
SPEED = 0.16
FIXED_FACE = 0.0199
ROT_WEIGHT = 0.1
IK_ITERS = 40
# Motion limits, so the recorded motion looks like real SO-101 teleop rather than a fast scripted arm.
PAN_AXIS = np.array([0.0388, 0.0])                   # shoulder-pan axis (xy) in the base frame
VMAX = np.radians([70.0, 70.0, 70.0, 90.0, 90.0])   # peak joint speed (rad/s): pan, lift, elbow, wrist flex, roll
MJ_PEAK = 1.875                                     # peak of d/du min_jerk(u)
ARC_SAG = 0.005                                     # a straight TCP move dipping this much closer to the pan axis than its ends arcs
PATH_SAMPLES = 24                                    # IK samples used to time a Cartesian path


def top_down_mat(angle):
    """TCP rotation for a vertical approach whose closing direction (moving jaw -> fixed finger) is at `angle`."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def interp_rot(r0, r1, s):
    """Geodesic interpolation between rotation matrices."""
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, r0.flatten())
    mujoco.mju_mat2Quat(q1, r1.flatten())
    if np.dot(q0, q1) < 0:
        q1 = -q1
    v = np.zeros(3)
    mujoco.mju_subQuat(v, q1, q0)
    q = q0.copy()
    mujoco.mju_quatIntegrate(q, v, s)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def min_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def tcp_path(start, end, arc=None):
    """Position path u -> xyz (u in [0, 1]) from `start` to `end`, and its length.

    A straight line, unless the line would dip more than ARC_SAG closer to the shoulder-pan axis than its end points
    (or `arc` is True): then radius, azimuth and height about the pan axis are interpolated, so the TCP stays at least
    min(r0, r1) from it. A straight line between the two sides of the robot passes close to the pan axis, a kinematic
    singularity where a tiny sideways step needs a huge pan rotation. Short moves (lift, lower, slide-ins) stay
    straight."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    d0, d1 = start[:2] - PAN_AXIS, end[:2] - PAN_AXIS
    r0, r1 = np.hypot(*d0), np.hypot(*d1)
    a0 = np.arctan2(d0[1], d0[0])
    turn = np.angle(np.exp(1j * (np.arctan2(d1[1], d1[0]) - a0)))
    if arc is None:
        seg = d1 - d0
        t = float(np.clip(-d0 @ seg / max(seg @ seg, 1e-12), 0.0, 1.0))
        closest = float(np.hypot(*(d0 + t * seg)))                  # straight line's closest approach to the axis
        arc = closest < min(r0, r1) - ARC_SAG and min(r0, r1) > 0.05
    if not arc:
        return (lambda u: start + u * (end - start)), float(np.linalg.norm(end - start))

    def path(u):
        r, a = r0 + u * (r1 - r0), a0 + u * turn
        return np.array([PAN_AXIS[0] + r * np.cos(a), PAN_AXIS[1] + r * np.sin(a), start[2] + u * (end[2] - start[2])])

    return path, float(np.hypot(np.hypot(r1 - r0, end[2] - start[2]), 0.5 * (r0 + r1) * abs(turn)))


class Oracle:
    """Base scripted policy. Subclasses implement `plan()` as a generator of joint targets using the skills."""

    rest_unwind = True

    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.ik_data = mujoco.MjData(env.model)
        self.dt = env.control_dt
        self.q = None
        self.grip = None
        self.log = []

    # ----- kinematics -----------------------------------------------------------------------------------------
    def ik(self, pos, rot, seed=None):
        """Arm joint targets putting the TCP at `pos` with orientation `rot` (3x3), seeded from `seed`."""
        env, m, d = self.env, self.model, self.ik_data
        d.qpos[:] = env.data.qpos
        q = (self.q[:5] if seed is None else seed).copy()
        lo, hi = env._target_low[:5], env._target_high[:5]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        dofs = env._arm_qvel_addrs
        for _ in range(IK_ITERS):
            d.qpos[env._arm_qpos_addrs] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, env._tcp_site_id)
            cur = d.site_xmat[env._tcp_site_id].reshape(3, 3)
            err_p = pos - d.site_xpos[env._tcp_site_id]
            err_r = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
            J = np.vstack([jacp[:, dofs], ROT_WEIGHT * jacr[:, dofs]])
            e = np.concatenate([err_p, ROT_WEIGHT * err_r])
            free = np.ones(5, dtype=bool)
            for _ in range(3):
                Jf = J * free
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-4 * np.eye(6), e)
                stuck = ((q <= lo + 1e-6) & (dq < 0)) | ((q >= hi - 1e-6) & (dq > 0))
                if not (stuck & free).any():
                    break
                free &= ~stuck
            q = np.clip(q + dq, lo, hi)
        return q

    def tcp(self):
        return self.env.data.site_xpos[self.env._tcp_site_id].copy()

    def tcp_rot(self):
        return self.env.data.site_xmat[self.env._tcp_site_id].reshape(3, 3).copy()

    def _target(self):
        return np.concatenate([self.q, [self.grip]])

    # ----- skills ---------------------------------------------------------------------------------------------
    def start(self):
        qpos = self.env._get_current_qpos()
        self.q, self.grip = qpos[:5].copy(), float(qpos[5])
        self._cmd_pos, self._cmd_rot = self.tcp(), self.tcp_rot()

    def follow(self, pose, seconds, vmax=VMAX):
        """Track the TCP path `pose(u) -> (pos, rot)`, u in [0, 1], with min-jerk timing over at least `seconds`,
        stretched so that no arm joint exceeds `vmax` (the joint path is sampled through the IK first, starting from
        the settled IK solution at u = 0, so an IK settling step does not count as path speed)."""
        q = self.ik(*pose(0.0), seed=self.q)
        qs = [q]
        for i in range(1, PATH_SAMPLES + 1):
            q = self.ik(*pose(i / PATH_SAMPLES), seed=q)
            qs.append(q)
        rate = np.abs(np.diff(qs, axis=0)).max(0) * PATH_SAMPLES       # max |dq/du| per joint
        seconds = max(seconds, MJ_PEAK * float(np.max(rate / vmax)))
        n = max(int(np.ceil(seconds / self.dt)), 1)
        for i in range(1, n + 1):
            self.q = self.ik(*pose(min_jerk(i / n)))
            yield self._target()

    def move(self, pos, rot, speed=SPEED, tol=0.004, settle=0.6, label="", arc=None, vmax=VMAX):
        """Min-jerk Cartesian move of the TCP to `pos`/`rot`, then wait (up to `settle` s) for the arm to arrive.
        Moves that turn about the pan axis follow an arc around it (`tcp_path`); the timing respects VMAX."""
        start, rot0 = self._cmd_pos.copy(), self._cmd_rot.copy()
        pos = np.asarray(pos, float)
        path, length = tcp_path(start, pos, arc)
        yield from self.follow(lambda u: (path(u), interp_rot(rot0, rot, u)), max(0.35, length / speed), vmax)
        self.q = self.ik(pos, rot)
        for _ in range(int(settle / self.dt)):
            if np.linalg.norm(self.tcp() - pos) < tol:
                break
            yield self._target()
        self._cmd_pos, self._cmd_rot = np.asarray(pos, float).copy(), rot.copy()
        self.log.append((label or "move", round(float(np.linalg.norm(self.tcp() - pos)) * 1000, 1)))

    def gripper(self, value, seconds=0.5, hold=0.2):
        """Ramp the gripper target to `value` (rad) over `seconds`, then hold the pose for `hold` s."""
        g0 = self.grip
        n = max(int(seconds / self.dt), 1)
        for i in range(1, n + 1):
            self.grip = g0 + (value - g0) * min_jerk(i / n)
            yield self._target()
        yield from self.wait(hold)

    def wait(self, seconds):
        for _ in range(int(seconds / self.dt)):
            yield self._target()

    def joint_move(self, q_target, seconds=1.5, grip=None):
        """Min-jerk joint-space move of the arm (and optionally the gripper), slowed down if needed so no arm joint
        exceeds VMAX."""
        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        seconds = max(seconds, MJ_PEAK * float(np.max(np.abs(q_target - q0) / VMAX)))
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
        """Joint-space move to the rest pose. The wrist roll is unwound first, in place: folding the arm with the roll
        far from zero swings the wrist-camera mount into the shoulder (the arm then sticks and snaps free). Oracles
        whose last pose leaves the fingers next to placed objects set `rest_unwind = False`."""
        rest = np.radians(np.array(REST_DEG))
        if self.rest_unwind and abs(self.q[4] - rest[4]) > np.radians(20):
            q = self.q.copy()
            q[4] = rest[4]
            yield from self.joint_move(q, 0.6)
        yield from self.joint_move(rest[:5], seconds, grip=rest[5])
        yield from self.wait(0.5)

    def solve(self, pos, rot, seeds=None):
        """Best IK solution over several wrist-roll seeds: (q, position error m, tilt rad)."""
        best = None
        base = self.q[:5].copy()
        for roll in [base[4]] + list(np.radians([-120, -60, 0, 60, 120])) if seeds is None else seeds:
            q = base.copy()
            q[4] = roll
            for _ in range(3):
                q = self.ik(pos, rot, seed=q)
            err = float(np.linalg.norm(self._fk(q) - pos))
            cand = (err, self._tilt(q), q)
            if best is None or 200 * cand[0] + cand[1] < 200 * best[0] + best[1]:
                best = cand
        return best[2], best[0], best[1]

    def grasp_plan(self, center, width, angles):
        """Choose a top-down closing angle for an object at `center` (TCP height already in z).
        Returns (rot, grasp TCP position, joint solution) for the reachable candidate needing the least roll change."""
        best = None
        for a in angles:
            rot = top_down_mat(a)
            pos = center - (FIXED_FACE - width / 2 - 0.003) * rot[:, 0]
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def transit(self, pos, rot, speed=1.0, q=None, label="transit"):
        """Joint-space min-jerk move to the IK solution for `pos`/`rot` (for large moves; avoids roll wrap-around)."""
        q = self.solve(pos, rot)[0] if q is None else q
        seconds = max(0.6, float(np.max(np.abs(q - self.q))) / speed)
        yield from self.joint_move(q, seconds)
        self._cmd_pos, self._cmd_rot = np.asarray(pos, float).copy(), rot.copy()
        for _ in range(int(0.4 / self.dt)):
            if np.linalg.norm(self.tcp() - pos) < 0.005:
                break
            yield self._target()
        self.log.append((label, round(float(np.linalg.norm(self.tcp() - pos)) * 1000, 1)))

    def _tilt(self, q):
        self._fk(q)
        return float(np.arccos(np.clip(-self.ik_data.site_xmat[self.env._tcp_site_id][8], -1, 1)))

    def _fk(self, q):
        d = self.ik_data
        d.qpos[self.env._arm_qpos_addrs] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.env._tcp_site_id].copy()

    def pick(self, name, width, grasp_z, yaw=None, attempts=3, approach=0.05):
        """Top-down grasp of `name` (size `width` along the closing direction, TCP height `grasp_z`), lifted to
        CARRY_Z. `yaw` is the object's face direction (radians); closing directions yaw + k*pi/2 are considered
        (radial if None). Retries with a fresh object pose on a miss."""
        for attempt in range(attempts):
            obj = self.env.object_pos(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, [base + k * np.pi / 2 for k in range(4)])
            yield from self.gripper(OPEN, 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.08, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, 0.5, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], CARRY_Z]), rot, label="lift")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(OPEN, 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def carry_rot(self, xy):
        """Current TCP orientation turned about z by the change in azimuth to `xy`, so the wrist roll stays put."""
        cur = self._cmd_pos
        d = np.arctan2(xy[1], xy[0]) - np.arctan2(cur[1], cur[0])
        c, s = np.cos(d), np.sin(d)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ self._cmd_rot

    def place(self, xy, release_z, rot=None):
        """Carry the held object so the TCP is over `xy`, lower the TCP to `release_z`, open, retreat upward."""
        rot = self.carry_rot(xy) if rot is None else rot
        yield from self.move(np.array([xy[0], xy[1], CARRY_Z]), rot, label="carry")
        yield from self.move(np.array([xy[0], xy[1], release_z]), rot, speed=0.08, label="lower")
        yield from self.gripper(RELEASE, 0.4, 0.3)
        yield from self.move(np.array([xy[0], xy[1], CARRY_Z]), rot, speed=0.12, label="retreat")

    def held_offset(self, name):
        """TCP minus object position (xy) while holding, so a place target can be given for the object."""
        return (self.tcp() - self.env.object_pos(name))[:2]

    def plan(self):
        raise NotImplementedError

    def actions(self):
        """The plan's joint targets, with a last-resort speed guard: a target further than VMAX allows in one frame
        (e.g. an IK branch switch) is reached through linearly interpolated extra frames. `self.limited` counts them."""
        self.start()
        prev, self.limited = self._target(), 0
        for target in self.plan():
            k = int(np.ceil(float(np.max(np.abs(target[:5] - prev[:5]) / (VMAX * self.dt))) - 1e-9))
            for i in range(1, k):
                self.limited += 1
                yield prev + (i / k) * (target - prev)
            yield target
            prev = target

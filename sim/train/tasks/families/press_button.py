"""press_button: operate the stove's control (LIBERO flat_stove, free on the table).

The flat_stove's only articulated control is a rotary knob (hinge about the vertical, range 0-2.1 rad), not a push
button, so the two tasks turn it (different goals and directions, not colour swaps):

* ``turn_on_stove``  -- the knob starts at off (0 rad); grasp its grip fin and turn it counter-clockwise (seen from
  above) to at least 0.9 rad.
* ``turn_off_stove`` -- the knob starts on (1.1-1.7 rad); turn it back clockwise to off (<= 0.12 rad).

Held-out validation puts a frying pan on this stove; turning the knob is a different skill and goal. The stove is a free
body (see ``Articulated``), so it is the task object of the kit's strict policy: it must stay where it started,
upright, still, touched only by the jaws, and released at the end. The knob angle is the goal's ``check``.

Oracle: top-down pinch across the knob's grip fin (jaws closing across its 1.2 cm thickness), then the TCP pose is
carried rigidly with the knob frame about the knob axis (a Cartesian arc whose yaw turns with the knob, wrist roll
capped at 50 deg/s), release, lift and rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.train.variation import GoalRegion
from sim.val.oracle import interp_rot
from sim.val.scene import ASSETS, _add_assets, _content, _merge_defaults, _soften, load_mjcf

FAMILY = "press_button"


# ----- asset helper (copied per articulated family module; proposed kit addition) --------------------------------

@dataclass
class Articulated:
    """A free articulated LIBERO object (cabinet, fridge, microwave, stove) with its joints kept.

    Mass is put on the free root body (``mass`` at ``com``, box inertia of ``size``); every LIBERO geom gets zero
    density and each jointed part an explicit small inertial (``part_mass``), because LIBERO's 3 kg / 1 kg m^2 part
    inertials are unscaled placeholders. ``joints`` overrides joint attributes by LIBERO joint name (slide ranges
    are not scaled by the loader, so pass them scaled)."""

    name: str
    path: str
    scale: float = 0.5
    mass: float = 1.5
    com: tuple = (0.0, 0.0, 0.03)
    size: tuple = (0.10, 0.10, 0.08)
    part_mass: float = 0.05
    joints: dict = field(default_factory=dict)

    def build_mjcf(self, mj, asset_el, worldbody):
        root = load_mjcf(ASSETS / self.path, self.scale, self.name)
        _soften(root.find("asset"))
        for geom in root.iter("geom"):
            geom.attrib.pop("mass", None)
            geom.set("density", "0")
        for joint in root.iter("joint"):
            local = joint.get("name", "")[len(self.name) + 1:]
            for key, value in self.joints.get(local, {}).items():
                joint.set(key, " ".join(f"{v:.6g}" for v in np.atleast_1d(value)) if not isinstance(value, str)
                          else value)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        sx, sy, sz = self.size
        inertia = [self.mass / 12 * (b * b + c * c) for b, c in ((sy, sz), (sx, sz), (sx, sy))]
        ET.SubElement(body, "inertial", pos=" ".join(map(str, self.com)), mass=str(self.mass),
                      diaginertia=" ".join(f"{i:.6g}" for i in inertia))
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat="1 0 0 0")
        upright.extend(_content(root))
        for part in upright.iter("body"):
            for old in part.findall("inertial"):
                part.remove(old)
            if part.find("joint") is not None:
                part.insert(0, ET.Element("inertial", pos="0 0 0", mass=str(self.part_mass),
                                          diaginertia="2e-5 2e-5 2e-5"))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_box(env, body_id, exclude=()):
    """World AABB (lo, hi) of the collision geoms attached directly to ``body_id``."""
    m, d = env.model, env.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body_id or not (m.geom_contype[g] or m.geom_conaffinity[g]) or g in exclude:
            continue
        c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
        h = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def local_box(env, body_id, frame_body):
    """AABB of ``body_id``'s collision geoms in the frame of ``frame_body`` (corner points)."""
    m, d = env.model, env.data
    R, p = d.xmat[frame_body].reshape(3, 3), d.xpos[frame_body]
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body_id or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        w = (corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append((w - p) @ R)
    pts = np.vstack(pts)
    return pts.min(0), pts.max(0)


# ----- environment -------------------------------------------------------------------------------------------------

STOVE = Articulated("stove", "articulated_objects/flat_stove.xml", 0.5, mass=1.0, com=(0.075, 0.0, 0.0),
                    size=(0.095, 0.095, 0.02), part_mass=0.03,
                    joints={"button": dict(range=(-0.005, 2.1), damping=0.02, frictionloss=0.01)})


class TurnOnEnv(TrainEnv):
    instruction = "Turn the stove knob to switch the stove on."
    task_objects = ("stove",)
    start_q = (0.0, 0.0)
    goal_q = (0.9, 2.1)                                   # success band for the knob angle (rad)
    target_q = 1.35                                       # commanded final knob angle (rad)
    knob_r = (0.16, 0.25)                                 # knob axis radius from the robot base

    def scene_objects(self):
        return [STOVE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._j = m.joint("stove_button").id
        self._knob = m.jnt_bodyid[self._j]
        self._robot_geoms = {g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], m.body("base").id)}
        boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self._knob and m.geom_type[g] == 6
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self._fin = max(boxes, key=lambda g: float(np.prod(m.geom_size[g])) * (1 + 10 * float(m.geom_pos[g][2] > 0.01)))
        self._probe = mujoco.MjData(m)

    def turn(self):
        """Signed knob rotation still to do (rad)."""
        return self.target_q - self._q0

    def knob_q(self, data=None):
        return float((self.data if data is None else data).qpos[self.model.jnt_qposadr[self._j]])

    def set_knob(self, q, data=None):
        data = self.data if data is None else data
        data.qpos[self.model.jnt_qposadr[self._j]] = q
        data.qvel[self.model.jnt_dofadr[self._j]] = 0.0

    def robot_touching(self, name):
        mine = set(self._geoms[name])
        return any((c.geom1 in mine and c.geom2 in self._robot_geoms) or (c.geom2 in mine and c.geom1 in self._robot_geoms)
                   for c in self.data.contact[:self.data.ncon])

    def knob_done(self):
        q = self.knob_q()
        return self.goal_q[0] <= q <= self.goal_q[1] and not self.robot_touching("stove")

    def task_info(self):
        return {"knob_rad": round(self.knob_q(), 3)}

    # ----- knob geometry ------------------------------------------------------------------------------------
    def fin_pose(self, q):
        """(centre, thin axis, top z) of the grip fin at knob angle ``q`` (world, kinematics on a scratch copy)."""
        d = self._probe
        d.qpos[:] = self.data.qpos
        self.set_knob(q, d)
        mujoco.mj_kinematics(self.model, d)
        g = self._fin
        R = d.geom_xmat[g].reshape(3, 3)
        half = self.model.geom_size[g]
        horiz = [i for i in range(3) if abs(R[2, i]) < 0.5]
        vert = [i for i in range(3) if i not in horiz][0]
        thin = min(horiz, key=lambda i: half[i])
        axis = R[:, thin].copy()
        axis[2] = 0.0
        return d.geom_xpos[g].copy(), axis / np.linalg.norm(axis), float(d.geom_xpos[g][2] + half[vert]), 2 * half[thin]

    def grasp_at(self, q, rot0=None, flip=False):
        """TCP pose pinching the fin at knob angle ``q``: closing direction across the fin."""
        centre, axis, top, width = self.fin_pose(q)
        if flip:
            axis = -axis
        rot = top_down_mat(float(np.arctan2(axis[1], axis[0])))
        from sim.val.oracle import FIXED_FACE
        pos = np.r_[centre[:2], top - self.grip_depth] - (FIXED_FACE - width / 2 - 0.0015) * rot[:, 0]
        return pos, rot, width

    grip_depth = 0.008                                   # TCP below the fin's top edge

    def reachable(self, tol=0.002):
        if not hasattr(self, "_ik"):
            self._ik = TrainOracle(self)
        ik = self._ik
        best = None
        for flip in (False, True):
            ik.q = np.radians(np.array(self.rest_deg[:5]))
            q0 = self.knob_q()
            ok = True
            for k, s in enumerate(np.linspace(0, 1, 7)):
                pos, rot, _ = self.grasp_at(q0 + s * self.turn(), flip=flip)
                for p in ((pos + [0, 0, 0.05]) if k == 0 else None, pos):
                    if p is None:
                        continue
                    qq, err, tilt = ik.solve(p, rot, seeds=None if (k == 0 and p is not pos) else [ik.q[4]])
                    if err > tol or tilt > np.radians(8):
                        ok = False
                        break
                    ik.q = qq
                if not ok:
                    break
            if ok:
                return flip
        return None

    # ----- layout -------------------------------------------------------------------------------------------
    def layout(self):
        rng = self.np_random
        q0 = float(rng.uniform(*self.start_q))
        self._q0 = q0
        for _ in range(300):
            yaw = rng.uniform(-np.pi, np.pi)
            rr, aa = rng.uniform(*self.knob_r), np.radians(rng.uniform(-55.0, 55.0))
            knob = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            centre = knob + 0.075 * np.array([np.cos(yaw), np.sin(yaw)])        # burner centre (stove footprint)
            if np.hypot(*centre) > 0.34 or np.hypot(*centre) < 0.14:
                continue
            if any(np.linalg.norm(p - k) < r + 0.06 for k, r in self.keepout for p in (knob, centre)):
                continue
            self.set_object_pose("stove", knob, yaw=yaw)
            self.set_knob(q0)
            mujoco.mj_kinematics(self.model, self.data)
            flip = self.reachable()
            if flip is not None:
                break
        else:
            raise RuntimeError("no reachable stove pose")
        self.flip = flip
        placed = [(knob, 0.045), (centre, 0.07)]
        self.place_distractors(placed)
        self.set_goals(Goal("stove", "table", target=tuple(self.object_pos("stove")),
                            tolerance=(0.005, 0.005, 0.004), upright_cos=float(np.cos(np.radians(5))),
                            check=lambda env, name: env.knob_done()))

    def goal_regions(self):
        """Samples on the knob's top (hub and fin), 3 mm above it."""
        centre, axis, top, _ = self.fin_pose(self.knob_q())
        hub = self.object_pos("stove")
        side = np.array([-axis[1], axis[0], 0.0])
        pts = [tuple(np.r_[(hub[:2] + a * 0.008 * axis[:2] + b * 0.008 * side[:2]), top + 0.003])
               for a in (-1, 0, 1) for b in (-1, 0, 1)]
        return (GoalRegion("stove", tuple(pts)),)


class KnobOracle(TrainOracle):
    vmax = np.radians([60.0, 60.0, 60.0, 75.0, 50.0])
    turn_speed = 0.6                                    # rad/s of the knob


    def unfold_transit(self, pos, rot, label="raise"):
        """Joint-space transit that first unfolds the arm with the wrist roll held, then turns the roll: turning the
        roll while still folded at rest swung the wrist-camera mount into the shoulder (measured on calibration)."""
        q = self.solve(pos, rot)[0]
        held = q.copy()
        held[4] = self.q[4]
        yield from self.joint_move(held, max(0.8, float(np.max(np.abs(held - self.q))) / 1.0))
        yield from self.transit(pos, rot, q=q, label=label)
    def plan(self):
        env = self.env
        q0 = env.knob_q()
        pos, rot, width = env.grasp_at(q0, flip=env.flip)
        yield from self.gripper(self.open_for(width), 0.4, 0.0)
        pre = pos + [0, 0, 0.05]
        yield from self.unfold_transit(pre, rot, label="pregrasp")
        yield from self.move(pos, rot, speed=0.04, tol=0.002, settle=0.8, label="grasp")
        yield from self.gripper(CLOSED, self.close_seconds, 0.3)
        self.log.append(("grasp", env.is_grasping("stove")))
        poses = [env.grasp_at(q0 + s * env.turn(), flip=env.flip) for s in np.linspace(0, 1, 25)]
        pts = np.array([p for p, _, _ in poses])
        rots = [r for _, r, _ in poses]

        def pose(u):
            x = u * (len(poses) - 1)
            i = min(int(x), len(poses) - 2)
            f = x - i
            return pts[i] + f * (pts[i + 1] - pts[i]), interp_rot(rots[i], rots[i + 1], f)

        yield from self.follow(pose, abs(env.turn()) / self.turn_speed)
        self._cmd_pos, self._cmd_rot = pts[-1].copy(), rots[-1].copy()
        yield from self.wait(0.3)
        self.log.append(("knob", round(env.knob_q(), 3)))
        yield from self.gripper(self.open_for(width), 0.8, 0.2)
        yield from self.move(self._cmd_pos + [0, 0, 0.05], self._cmd_rot, speed=0.06, label="lift")
        yield from self.rest()
        yield from self.wait(1.2)


class TurnOffEnv(TurnOnEnv):
    instruction = "Turn the stove knob back to switch the stove off."
    start_q = (1.1, 1.7)
    goal_q = (-0.05, 0.12)                               # the 0 rad stop is a soft joint limit (a few mrad give)
    target_q = 0.0                                       # stop at the stop (the knob lags 0.04 rad): driving into it
                                                         # overshot to -0.1..-0.3 rad while held


TASKS = [
    define_task(name="turn_on_stove", instruction=TurnOnEnv.instruction, family=FAMILY, env=TurnOnEnv,
                oracle=KnobOracle, objects=("stove",), object_kinds=("stove knob",), relation="turned on",
                goal="stove", steps=("Grasp the stove knob and turn it until the stove is on.",)),
    define_task(name="turn_off_stove", instruction=TurnOffEnv.instruction, family=FAMILY, env=TurnOffEnv,
                oracle=KnobOracle, objects=("stove",), object_kinds=("stove knob",), relation="turned off",
                goal="stove", steps=("Grasp the stove knob and turn it back until the stove is off.",)),
]

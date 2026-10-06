"""door_close: swing an open hinged door shut (articulated LIBERO microwave / short fridge, free on the table).

Tasks (different fixtures, hinge sides and door sizes):

* ``close_microwave_door`` -- LIBERO microwave at 0.5x (18.5 x 12 x 9.5 cm, 13 cm door hinged on the viewer's left
  of its front); the door starts 30-60 deg open.
* ``close_fridge_door``    -- LIBERO short_fridge at 0.5x (10 x 9.5 x 13 cm, 10 cm door hinged on the right);
  the door starts 30-60 deg open.

The fixture is a free body (see ``Articulated``), so it is the task object of the kit's strict policy: it must stay
where it started (GoalRule), upright, still, touched only by the jaws. The articulated goal (door joint within
0.05 rad of closed, robot not touching the fixture) is the goal's ``check``, part of the per-substep ``success``.

Oracle: top-down gripper with jaws closed, the flat side of the fingers parallel to the door's outer face (wrist-camera
mount on the far side); it comes down beside the open door, closes the measured gap, then follows the door's arc about
its hinge (TCP yaw turning with the door) to just past closed, backs off along the door normal, lifts and rests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.train.variation import GoalRegion
from sim.val.scene import ASSETS, _add_assets, _content, _merge_defaults, _soften, load_mjcf

FAMILY = "door_close"
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "camera_mount",
                "moving_jaw_so101_v1")


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

@dataclass(frozen=True)
class DoorKind:
    spec: Articulated
    joint: str                    # LIBERO joint name
    door: str                     # LIBERO door body name
    sign: float                   # sign of the joint change that opens the door (closed at q = 0)
    noun: str


MICROWAVE = DoorKind(
    Articulated("fixture", "articulated_objects/microwave.xml", 0.4, mass=1.6, com=(0.0, 0.0, 0.036),
                size=(0.148, 0.096, 0.076), part_mass=0.06,
                joints={"microjoint": dict(range=(-2.094, 0.0), damping=0.05, frictionloss=0.005)}),
    joint="microjoint", door="microdoorroot", sign=-1.0, noun="microwave")

FRIDGE = DoorKind(
    Articulated("fixture", "articulated_objects/short_fridge.xml", 0.4, mass=1.6, com=(0.0, 0.0, 0.05),
                size=(0.08, 0.076, 0.104), part_mass=0.06,
                joints={"fridge_door_joint_0": dict(limited="true", range=(0.0, 2.0), damping=0.05,
                                                    frictionloss=0.005)}),
    joint="fridge_door_joint_0", door="door", sign=1.0, noun="fridge")


class DoorCloseEnv(TrainEnv):
    instruction = "Close the microwave door."
    task_objects = ("fixture",)
    kind = MICROWAVE
    open_deg = (30.0, 60.0)
    closed_tolerance = 0.05                               # rad
    # Fixture origin (polar about the robot base) and front normal direction (deg, world). Mapped over 500 random
    # poses: the hinge-left microwave is reachable (IK along the whole arc, keep-outs clear) only on the robot's
    # right/viewer's left, facing 60-110 deg.
    region = dict(r=(0.17, 0.32), angle=(-72.0, -40.0))
    facing = (60.0, 110.0)
    frame_xy = dict(x_min=0.09, y_min=-0.22)              # fixture origin bounds that keep it in the front view
    approach_gap = 0.02
    approach_back = 0.03
    back_off = 0.015                                      # retreat along the closed door's normal
    lift = 0.04                                           # then straight up before resting
    slab = 0.0161
    slab_lateral = 0.012
    contact_fraction = 0.45                               # contact point along the door from the hinge
    push_z = 0.040                                        # TCP height above the fixture base
    overpush = 0.0                                        # rad past closed (the joint limit stops the door)
    press = 0.0015                                        # slab pressed this far into the door face on the arc

    def scene_objects(self):
        return [self.kind.spec]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._j = m.joint(f"fixture_{self.kind.joint}").id
        self._door = m.body(f"fixture_{self.kind.door}").id
        self._robot_geoms = {g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], m.body("base").id)}
        self._probe = mujoco.MjData(m)
        boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self._door and m.geom_type[g] == 6
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self._slab_geom = max(boxes, key=lambda g: float(np.prod(m.geom_size[g])))

    # ----- articulation -------------------------------------------------------------------------------------
    def door_q(self, data=None):
        return float((self.data if data is None else data).qpos[self.model.jnt_qposadr[self._j]])

    def set_door(self, q, data=None):
        data = self.data if data is None else data
        data.qpos[self.model.jnt_qposadr[self._j]] = q
        data.qvel[self.model.jnt_dofadr[self._j]] = 0.0

    def opening(self):
        return self.kind.sign * self.door_q()

    def robot_touching(self, name):
        mine = set(self._geoms[name])
        return any((c.geom1 in mine and c.geom2 in self._robot_geoms) or (c.geom2 in mine and c.geom1 in self._robot_geoms)
                   for c in self.data.contact[:self.data.ncon])

    def door_closed(self):
        return abs(self.door_q()) <= self.closed_tolerance and not self.robot_touching("fixture")

    def task_info(self):
        return {"door_opening_deg": round(float(np.degrees(self.opening())), 2)}

    # ----- door geometry (kinematics on a scratch copy) -----------------------------------------------------
    def _frame(self, q):
        d = self._probe
        d.qpos[:] = self.data.qpos
        self.set_door(q, d)
        mujoco.mj_kinematics(self.model, d)
        return d

    def face(self):
        """Contact point and outer-face normal of the door slab, in the door body frame (from the closed pose)."""
        m = self.model
        d = self._frame(0.0)
        Rb, pb = d.xmat[self._door].reshape(3, 3), d.xpos[self._door]
        g = self._slab_geom
        Rg = Rb.T @ d.geom_xmat[g].reshape(3, 3)
        cg = Rb.T @ (d.geom_xpos[g] - pb)
        out = Rb.T @ np.r_[self.front_dir(), 0.0]            # outward front normal (door frame)
        k = int(np.argmax(np.abs(Rg.T @ out)))                # slab's thin axis
        half = m.geom_size[g]
        normal = Rg[:, k] * np.sign(Rg[:, k] @ out)
        hinge = Rb.T @ (d.xanchor[self._j] - pb)
        axes = [i for i in range(3) if i != k and abs(Rg[2, i]) < 0.5]
        length_axis = Rg[:, axes[0]]
        ends = [cg + s * half[axes[0]] * length_axis for s in (-1, 1)]
        far = max(ends, key=lambda e: np.linalg.norm((e - hinge)[:2]))
        along = far - hinge
        along[2] = 0.0
        hinge_face = hinge.copy()
        point = hinge_face + self.contact_fraction * along
        point = point - (point - cg) @ normal * normal + half[k] * normal   # onto the outer face
        return point, normal

    def keepout_clear(self, margin=0.01):
        """Every collision box of the fixture (at the current state) clears the keep-out discs (exact box distance;
        the fixture's bounding radius would exclude every pose in front of the robot)."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        for g in self._geoms["fixture"]:
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            half = m.geom_aabb[g, 3:]
            for centre, radius in self.keepout:
                local = R.T @ (np.r_[centre, c[2]] - c)
                closest = c + R @ np.clip(local, -half, half)
                if np.linalg.norm(closest[:2] - centre) < radius + margin:
                    return False
        return True

    def front_dir(self):
        yaw = self._yaw
        return np.array([np.cos(yaw - np.pi / 2), np.sin(yaw - np.pi / 2)])

    def tcp_at(self, q, offset):
        """TCP pose (pos, rot) with the slab ``offset`` off the door face at door angle ``q``."""
        point, normal = self._face
        d = self._frame(q)
        Rb, pb = d.xmat[self._door].reshape(3, 3), d.xpos[self._door]
        n = Rb @ normal
        n = np.r_[n[:2] / np.linalg.norm(n[:2]), 0.0]
        c = pb + Rb @ point
        rot = top_down_mat(float(np.arctan2(n[0], -n[1])))
        pos = c + offset * n - self.slab_lateral * rot[:, 0]
        pos[2] = self.object_pos("fixture")[2] - self._extent["fixture"]["bottom"] + self.push_z
        return pos, rot, n

    def arc(self, offset, n=12):
        q0 = self.door_q()
        q1 = -self.kind.sign * self.overpush
        return [self.tcp_at(q0 + (q1 - q0) * s, offset) for s in np.linspace(0, 1, n)]

    def top(self):
        return float(self.object_pos("fixture")[2] - self._extent["fixture"]["bottom"] + self._extent["fixture"]["height"])

    def approach_poses(self, start, n):
        pre = start + self.approach_back * n
        # 3 cm over the fixture, capped at 12 cm: the IK cannot hold a vertical TCP at 16 cm over the 13 cm fridge
        # (17-45 mm error); nothing tall lies under the pre-push point, which is beyond the open door's outer face.
        return np.r_[pre[:2], min(max(start[2] + 0.05, self.top() + 0.03), 0.12)], pre

    def reachable(self, tol=0.002):
        if not hasattr(self, "_ik"):
            self._ik = TrainOracle(self)
        ik = self._ik
        ik.q = np.radians(np.array(self.rest_deg[:5]))
        start, rot, n = self.tcp_at(self.door_q(), self.slab + self.approach_gap)
        high, low = self.approach_poses(start, n)
        poses = [(high, rot, 0.015), (low, rot, tol), (start, rot, tol)]
        poses += [(p, r, tol) for p, r, _ in self.arc(self.slab)]
        end, rot_end, n_end = self.arc(self.slab)[-1]
        back = end + self.back_off * n_end
        poses.append((back, rot_end, tol))
        poses.append((back + [0, 0, self.lift], rot_end, 0.006))
        for k, (pos, rot, t) in enumerate(poses):
            q, err, tilt = ik.solve(pos, rot, seeds=None if k == 0 else [ik.q[4]])
            if err > t or tilt > np.radians(12 if t > 0.01 else 8):
                return False
            ik.q = q
        return True

    # ----- layout -------------------------------------------------------------------------------------------
    def layout(self):
        rng = self.np_random
        self.set_door(0.0)
        opening = np.radians(rng.uniform(*self.open_deg))
        for _ in range(300):
            rr, aa = rng.uniform(*self.region["r"]), np.radians(rng.uniform(*self.region["angle"]))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if xy[0] < self.frame_xy["x_min"] or xy[1] < self.frame_xy["y_min"]:
                continue
            self._yaw = np.radians(rng.uniform(*self.facing)) + np.pi / 2   # LIBERO fronts face local -y
            self.set_door(0.0)
            self.set_object_pose("fixture", xy, yaw=self._yaw)
            self._face = self.face()
            self.set_door(self.kind.sign * opening)
            mujoco.mj_kinematics(self.model, self.data)
            if not self.keepout_clear():
                continue   # the fixture (door open) must clear the robot base and the folded gripper at rest
            start, rot, n = self.tcp_at(self.door_q(), self.slab + self.approach_gap)
            if np.hypot(*start[:2]) < 0.13 or np.linalg.norm(start[:2] - [0.16, 0]) < 0.07:
                continue
            if self.reachable():
                break
        else:
            raise RuntimeError("no reachable door pose")
        placed = [(xy, self.footprint("fixture"))]
        # keep distractors off the door's swept region and the approach
        for p, _, _ in self.arc(self.slab, n=5):
            placed.append((p[:2], 0.05))
        placed.append((start[:2] * 0.5, 0.04))
        self.place_distractors(placed)
        self.set_goals(Goal("fixture", "table", target=tuple(self.object_pos("fixture")),
                            tolerance=(0.005, 0.005, 0.004), upright_cos=float(np.cos(np.radians(5))),
                            check=lambda env, name: env.door_closed()))

    def goal_regions(self):
        """Samples on the door slab as it currently is, 3 mm proud of whichever face (outer or inner) looks toward
        the front camera: an open door can show either face to the viewer."""
        point, normal = self._face
        d = self._frame(self.door_q())
        Rb, pb = d.xmat[self._door].reshape(3, 3), d.xpos[self._door]
        n = Rb @ normal
        c0 = pb + Rb @ point
        cam = self.data.cam_xpos[self.model.camera("front").id]
        thick = 2 * float(self.model.geom_size[self._slab_geom][int(np.argmax(np.abs(
            self.data.geom_xmat[self._slab_geom].reshape(3, 3).T @ n)))])
        if (cam - c0) @ n < 0:
            c0, n = c0 - thick * n, -n
        side = np.cross([0, 0, 1.0], n)
        z0 = self.object_pos("fixture")[2] - self._extent["fixture"]["bottom"] + self.push_z
        pts = []
        for a_ in (-0.02, 0.0, 0.02):
            for b_ in (-0.015, 0.0, 0.015):
                c = c0 + 0.003 * n + a_ * side
                c[2] = z0 + b_
                pts.append(tuple(c))
        return (GoalRegion("fixture", tuple(pts)),)


class DoorCloseOracle(TrainOracle):
    push_speed = 0.025
    # The arc turns the TCP yaw with the door; near the top-down pose that is mostly wrist roll, which at the kit's
    # 75 deg/s peaked at 150-162 rad/s^2 (zero-order-hold staircase), so the roll is capped lower here.
    vmax = np.radians([60.0, 60.0, 60.0, 75.0, 50.0])

    def gap_to(self, geom):
        """Measured clearance (m) between the jaws' collision geoms and the door slab (not its handle)."""
        m, d = self.model, self.env.data
        jaws = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in ("gripper", "moving_jaw_so101_v1")
                and (m.geom_contype[g] or m.geom_conaffinity[g])]
        return min(mujoco.mj_geomDistance(m, d, g, geom, 0.2, None) for g in jaws)


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
        start, rot, n = env.tcp_at(env.door_q(), env.slab + env.approach_gap)
        high, low = env.approach_poses(start, n)
        # Rise clear of the fixture first: a direct joint-space transit from the folded rest pose sweeps the jaws
        # low through an open door / drawer front (measured: door knocked 2 deg, fixture shoved at 0.16 m/s).
        lift = np.r_[0.20 * high[:2] / np.linalg.norm(high[:2]), high[2]]
        yield from self.unfold_transit(lift, rot)
        yield from self.move(high, rot, speed=0.10, tol=0.004, label="above")
        yield from self.move(low, rot, speed=0.06, tol=0.003, label="down")
        yield from self.move(start, rot, speed=0.04, tol=0.002, settle=0.8, label="in")
        gap = self.gap_to(env._slab_geom)
        self.log.append(("gap_mm", round(gap * 1000, 1)))
        offset = env.slab + env.approach_gap - gap - env.press
        q0, q1 = env.door_q(), -env.kind.sign * env.overpush
        # touch the face, then follow the door's arc about the hinge
        touch = offset + env.press + 0.001       # stop 1 mm short; the press builds up over the arc's first fifth
        p, r, _ = env.tcp_at(q0, touch)
        yield from self.move(p, r, speed=0.02, tol=0.002, label="touch")
        poses = [env.tcp_at(q0 + (q1 - q0) * s, touch + (offset - touch) * min(1.0, s / 0.2))
                 for s in np.linspace(0, 1, 25)]
        pos_s = np.array([p for p, _, _ in poses])
        rots = [r for _, r, _ in poses]
        length = float(np.sum(np.linalg.norm(np.diff(pos_s, axis=0), axis=1)))

        def pose(u):
            x = u * (len(poses) - 1)
            i = min(int(x), len(poses) - 2)
            f = x - i
            from sim.val.oracle import interp_rot
            return pos_s[i] + f * (pos_s[i + 1] - pos_s[i]), interp_rot(rots[i], rots[i + 1], f)

        yield from self.follow(pose, max(1.0, length / self.push_speed))
        self._cmd_pos, self._cmd_rot = pos_s[-1].copy(), rots[-1].copy()
        yield from self.wait(0.3)
        n_end = poses[-1][2]
        back = pos_s[-1] + env.back_off * n_end
        yield from self.move(back, rots[-1], speed=0.04, label="back off")
        yield from self.move(back + [0, 0, env.lift], rots[-1], speed=0.06, label="lift")
        yield from self.rest()
        yield from self.wait(1.2)


class FridgeDoorCloseEnv(DoorCloseEnv):
    instruction = "Close the fridge door."
    kind = FRIDGE
    region = dict(r=(0.17, 0.32), angle=(-50.0, -10.0))   # mapped: the hinge-right fridge reaches best here
    facing = (55.0, 100.0)


TASKS = [
    define_task(name="close_microwave_door", instruction=DoorCloseEnv.instruction, family=FAMILY,
                env=DoorCloseEnv, oracle=DoorCloseOracle, objects=("fixture",), object_kinds=("door",),
                relation="swung closed", goal="microwave",
                steps=("Push the open microwave door around its hinge until it is shut.",)),
    define_task(name="close_fridge_door", instruction=FridgeDoorCloseEnv.instruction, family=FAMILY,
                env=FridgeDoorCloseEnv, oracle=DoorCloseOracle, objects=("fixture",), object_kinds=("door",),
                relation="swung closed", goal="fridge",
                steps=("Push the open fridge door around its hinge until it is shut.",)),
]

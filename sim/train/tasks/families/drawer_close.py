"""drawer_close: push an open cabinet drawer shut (articulated LIBERO cabinets, free on the table).

Tasks (different fixtures and drawers, not colour swaps):

* ``close_short_cabinet_drawer`` -- LIBERO short_cabinet at 0.5x (10 x 9.5 x 7.7 cm, three 2 cm drawers with small
  pull handles); the top or middle drawer starts 2.5-4.5 cm open.
* ``close_white_cabinet_drawer`` -- LIBERO white_cabinet at 0.5x (12.7 x 9.5 x 10.9 cm, three 3.5 cm drawers with
  bar handles); the bottom or middle drawer starts 2.5-4.5 cm open.

The cabinet is a *free* body (its own free joint, 1.5-2 kg) so contacts of its drawers with the table and the robot are
simulated (a static LIBERO fixture's moving parts are parent-filtered against every world-welded geom), and so it is
a task object of the kit's strict physics policy: it must stay on the table where it started (GoalRule tolerance),
upright and still, and only the jaws may touch it. The articulated goal (drawer joint within 4 mm of closed, every
other drawer still closed, robot not touching the cabinet) is the goal's ``check`` and therefore part of the
per-substep ``success`` the audit requires over the final 30 frames.

Oracle: top-down gripper (jaws closed), lowered 2 cm in front of the open drawer at drawer height, with the flat side
of the fingers facing the drawer and the wrist-camera mount on the far side, then a slow straight push along the
drawer axis to 1.5 mm past flush, a short hold, a straight back-off, lift and rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import COS10, Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.train.variation import GoalRegion
from sim.val.scene import ASSETS, _add_assets, _content, _merge_defaults, _soften, load_mjcf

FAMILY = "drawer_close"
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
class CabinetKind:
    spec: Articulated
    drawers: dict                 # drawer body (local LIBERO name) -> joint (local LIBERO name)
    openable: tuple               # drawer bodies that may start open
    sign: float                   # +1: opening increases the joint (short cabinet), -1: decreases (white cabinet)
    noun: str


SHORT_CABINET = CabinetKind(
    Articulated("cabinet", "articulated_objects/short_cabinet.xml", 0.5, mass=1.5, com=(0.0, 0.0, 0.035),
                size=(0.10, 0.095, 0.077),
                joints={j: dict(range=(-0.0025, 0.07), damping=20.0, frictionloss=0.3)
                        for j in ("top_region", "middle_level", "bottom_region")}),
    drawers={"drawer_top": "top_region", "drawer_middle": "middle_level", "drawer_bottom": "bottom_region"},
    openable=("drawer_top", "drawer_middle"), sign=1.0, noun="short cabinet")

WHITE_CABINET = CabinetKind(
    Articulated("cabinet", "articulated_objects/white_cabinet.xml", 0.5, mass=2.0, com=(0.0, 0.008, 0.05),
                size=(0.127, 0.095, 0.109),
                joints={j: dict(range=(-0.07, 0.005), damping=20.0, frictionloss=0.3)
                        for j in ("top_level", "middle_level", "bottom_level")}),
    drawers={"cabinet_top": "top_level", "cabinet_middle": "middle_level", "cabinet_bottom": "bottom_level"},
    openable=("cabinet_middle", "cabinet_bottom"), sign=-1.0, noun="cabinet")


class DrawerCloseEnv(TrainEnv):
    instruction = "Push the open drawer of the short cabinet closed."
    task_objects = ("cabinet",)
    kind = SHORT_CABINET
    open_range = (0.025, 0.045)
    closed_tolerance = 0.004
    region = dict(r=(0.18, 0.30), angle=(-55.0, -20.0))  # cabinet origin (polar about the robot base)
    facing = (25.0, 75.0)                                 # drawer opening direction (deg, world)
    approach_gap = 0.02                                   # slab-to-drawer gap before the push
    overpush = 0.0015                                     # push target past flush
    slab = 0.0161                                         # fingers' flat side from the TCP (along -y_tcp)
    slab_lateral = 0.012                                  # slab centre along +x_tcp
    push_r = (0.13, 0.265)                                # TCP radii allowed along the push
    push_height = 0.011                                   # TCP above the drawer's bottom edge
    approach_back = 0.03                                  # pre-push pose this far beyond the start along the axis

    def scene_objects(self):
        return [self.kind.spec]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._joint = {b: m.joint(f"cabinet_{j}").id for b, j in self.kind.drawers.items()}
        self._drawer_body = {b: m.body(f"cabinet_{b}").id for b in self.kind.drawers}
        self._robot_geoms = {g for g in range(m.ngeom)
                             if self._in_subtree(m.geom_bodyid[g], m.body("base").id)}

    # ----- articulation -------------------------------------------------------------------------------------
    def drawer_q(self, drawer):
        return float(self.data.qpos[self.model.jnt_qposadr[self._joint[drawer]]])

    def set_drawer(self, drawer, opening):
        self.data.qpos[self.model.jnt_qposadr[self._joint[drawer]]] = self.kind.sign * opening
        self.data.qvel[self.model.jnt_dofadr[self._joint[drawer]]] = 0.0

    def opening(self, drawer):
        return self.kind.sign * self.drawer_q(drawer)

    def robot_touching(self, name):
        mine = set(self._geoms[name])
        for c in self.data.contact[:self.data.ncon]:
            if (c.geom1 in mine and c.geom2 in self._robot_geoms) or (c.geom2 in mine and c.geom1 in self._robot_geoms):
                return True
        return False

    def all_closed(self, env=None, name=None):
        return (all(self.opening(b) <= self.closed_tolerance for b in self.kind.drawers)
                and not self.robot_touching("cabinet"))

    def task_info(self):
        return {f"{b}_opening": round(self.opening(b), 4) for b in self.kind.drawers}

    # ----- geometry -----------------------------------------------------------------------------------------
    def out_dir(self):
        """World unit vector the drawers open along (horizontal)."""
        j = self._joint[self.target]
        axis = self.data.xaxis[j] * self.kind.sign
        return axis[:2] / np.linalg.norm(axis[:2])

    def front_box(self, drawer):
        """Drawer collision box in the cabinet frame (lo, hi) at the current opening."""
        return local_box(self, self._drawer_body[drawer], self._body["cabinet"])

    def push_plan(self):
        """(rot, start TCP, end TCP) for pushing the target drawer closed from its current opening."""
        mujoco.mj_kinematics(self.model, self.data)
        u = np.r_[self.out_dir(), 0.0]
        lo, hi = self.front_box(self.target)
        R, p = self.data.xmat[self._body["cabinet"]].reshape(3, 3), self.data.xpos[self._body["cabinet"]]
        u_local = R.T @ u
        # Drawer front (handle tip) along u, lateral centre and bottom height, in the cabinet frame.
        corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        front = float(np.max(corners @ u_local))
        centre = 0.5 * (lo + hi)
        lateral = centre - (centre @ u_local) * u_local
        lateral[2] = 0.0
        rot =top_down_mat(float(np.arctan2(u[0], -u[1])))
        opening = self.opening(self.target)
        contact = p + R @ (lateral + (front - opening - self.overpush) * u_local)
        end = np.r_[contact[:2], p[2] + lo[2] + self.push_height] + self.slab * u - self.slab_lateral * rot[:, 0]
        start = end + (opening + self.overpush + self.approach_gap) * u
        return rot, start, end

    def cabinet_top(self):
        return float(self.data.xpos[self._body["cabinet"]][2] + self._extent["cabinet"]["height"]
                     - self._extent["cabinet"]["bottom"])

    def approach_poses(self, start):
        """Clear pre-push pose beyond the open drawer's front (3 cm above the drawer's top edge; the IK cannot hold
        a vertical TCP 3 cm above the 11 cm white cabinet: 13-16 mm error), and its lowered twin."""
        u = np.r_[self.out_dir(), 0.0]
        pre = start + self.approach_back * u
        _, hi = self.front_box(self.target)
        top = float(self.data.xpos[self._body["cabinet"]][2] + hi[2])
        return np.r_[pre[:2], max(start[2] + 0.05, top + 0.03)], pre

    def reachable(self, rot, start, end, tol=0.002):
        """IK check of every pose the oracle visits (position error, robot-frame tilt)."""
        if not hasattr(self, "_ik"):
            self._ik = TrainOracle(self)
        ik = self._ik
        ik.q = np.radians(np.array(self.rest_deg[:5]))
        high, low = self.approach_poses(start)
        for k, pos in enumerate((high, low, start, end)):
            q, err, tilt = ik.solve(pos, rot, seeds=None if k == 0 else [ik.q[4]])
            # The high waypoint is only a transit target (measured 6-7 mm IK error at 11-14 cm): looser bound.
            if err > (0.012 if k == 0 else tol) or tilt > np.radians(12 if k == 0 else 8):
                return False
            ik.q = q
        return True

    # ----- layout -------------------------------------------------------------------------------------------
    def layout(self):
        rng = self.np_random
        for b in self.kind.drawers:
            self.set_drawer(b, 0.0)
        self.target = self.kind.openable[int(rng.integers(len(self.kind.openable)))]
        opening = float(rng.uniform(*self.open_range))
        for _ in range(400):
            rr, aa = rng.uniform(*self.region["r"]), np.radians(rng.uniform(*self.region["angle"]))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if any(np.linalg.norm(xy - k) < self.footprint("cabinet") + kr + 0.01 for k, kr in self.keepout):
                continue   # the fixture must clear the robot base and the folded gripper at rest
            psi = np.radians(rng.uniform(*self.facing))
            yaw = psi + np.pi / 2           # LIBERO drawers open along local -y
            self.set_object_pose("cabinet", xy, yaw=yaw)
            self.set_drawer(self.target, opening)
            mujoco.mj_kinematics(self.model, self.data)
            rot, start, end = self.push_plan()
            radii = [np.hypot(*start[:2]), np.hypot(*end[:2])]
            if (min(radii) >= self.push_r[0] and max(radii) <= self.push_r[1]
                    and np.linalg.norm(start[:2] - [0.16, 0]) > 0.07 and self.reachable(rot, start, end)):
                break
        else:
            raise RuntimeError("no reachable cabinet pose")
        self._plan = (rot, start, end)
        placed = [(xy, self.footprint("cabinet"))]
        u = np.r_[self.out_dir(), 0.0]
        for s in np.linspace(0.0, 1.0, 4):
            placed.append((start[:2] + s * 0.06 * u[:2], 0.045))
        placed.append((start[:2] * 0.5, 0.04))
        self.place_distractors(placed)
        self.set_goals(Goal("cabinet", "table", target=tuple(self.object_pos("cabinet")),
                            tolerance=(0.005, 0.005, 0.004), upright_cos=float(np.cos(np.radians(5))),
                            check=lambda env, name: env.all_closed()))

    def goal_regions(self):
        """Samples on the target drawer's front (handle) face as it currently is, 3 mm proud of it."""
        mujoco.mj_kinematics(self.model, self.data)
        R, p = self.data.xmat[self._body["cabinet"]].reshape(3, 3), self.data.xpos[self._body["cabinet"]]
        lo, hi = self.front_box(self.target)
        u_local = R.T @ np.r_[self.out_dir(), 0.0]
        corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        front = float(np.max(corners @ u_local))
        centre = 0.5 * (lo + hi)
        side = np.cross([0, 0, 1.0], u_local)
        half_w = 0.3 * abs((hi - lo) @ side)
        half_h = 0.3 * (hi[2] - lo[2])
        points = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                local = centre - (centre @ u_local) * u_local + (front + 0.003) * u_local + a * half_w * side
                local[2] = centre[2] + b * half_h
                points.append(tuple(p + R @ local))
        return (GoalRegion("cabinet", tuple(points)),)


class DrawerCloseOracle(TrainOracle):
    push_speed = 0.03

    def gap_to(self, body_id):
        """Measured clearance (m) between the jaws' collision geoms and ``body_id``'s geoms, at the current state."""
        env, m, d = self.env, self.model, self.env.data
        jaws = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in ("gripper", "moving_jaw_so101_v1")
                and (m.geom_contype[g] or m.geom_conaffinity[g])]
        mine = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body_id and (m.geom_contype[g] or m.geom_conaffinity[g])]
        return min(mujoco.mj_geomDistance(m, d, g, h, 0.2, None) for g in jaws for h in mine)


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
        rot, start, end = env.push_plan()
        u = np.r_[env.out_dir(), 0.0]
        high, low = env.approach_poses(start)
        # Rise clear of the fixture first: a direct joint-space transit from the folded rest pose sweeps the jaws
        # low through an open door / drawer front (measured: door knocked 2 deg, fixture shoved at 0.16 m/s).
        lift = np.r_[0.20 * high[:2] / np.linalg.norm(high[:2]), high[2]]
        yield from self.unfold_transit(lift, rot)
        yield from self.move(high, rot, speed=0.10, tol=0.004, label="above")
        yield from self.move(low, rot, speed=0.06, tol=0.003, label="down")
        yield from self.move(start, rot, speed=0.04, tol=0.002, settle=0.8, label="in")
        # Close the loop on the measured jaw-to-drawer clearance (the jaw hulls are not flat boxes).
        gap = self.gap_to(env._drawer_body[env.target])
        self.log.append(("gap_mm", round(gap * 1000, 1)))
        end = self._cmd_pos - (gap + env.opening(env.target) + env.overpush) * u
        yield from self.move(end, rot, speed=self.push_speed, tol=0.002, settle=0.8, label="push")
        yield from self.wait(0.3)
        back = end + (0.035 + env.approach_back) * u
        yield from self.move(back, rot, speed=0.04, label="back off")
        yield from self.move(np.r_[back[:2], high[2]], rot, speed=0.08, label="lift")
        yield from self.rest()
        yield from self.wait(1.2)


class WhiteDrawerCloseEnv(DrawerCloseEnv):
    instruction = "Push the open drawer of the cabinet closed."
    kind = WHITE_CABINET
    region = dict(r=(0.20, 0.30), angle=(-50.0, -20.0))
    facing = (55.0, 100.0)   # taller cabinet: keep it inside the front camera's frame


TASKS = [
    define_task(name="close_short_cabinet_drawer", instruction=DrawerCloseEnv.instruction, family=FAMILY,
                env=DrawerCloseEnv, oracle=DrawerCloseOracle, objects=("cabinet",), object_kinds=("drawer",),
                relation="pushed closed", goal="short cabinet",
                steps=("Push the open drawer of the short cabinet until it is closed.",)),
    define_task(name="close_white_cabinet_drawer", instruction=WhiteDrawerCloseEnv.instruction, family=FAMILY,
                env=WhiteDrawerCloseEnv, oracle=DrawerCloseOracle, objects=("cabinet",), object_kinds=("drawer",),
                relation="pushed closed", goal="cabinet",
                steps=("Push the open drawer of the cabinet until it is closed.",)),
]

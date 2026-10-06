"""drawer_open: pull a partly open cabinet drawer fully open (articulated LIBERO cabinets, free on the table).

Tasks (different fixtures, drawer sizes and handles, not colour swaps):

* ``open_short_cabinet_drawer`` -- LIBERO short_cabinet at 0.5x; its top drawer starts ajar (3.0-3.6 cm) and must be
  pulled out to at least 5 cm.
* ``open_white_cabinet_drawer`` -- LIBERO white_cabinet at 0.45x; its top drawer starts ajar (3.0-3.6 cm) and must be
  pulled out to at least 5 cm.

Why ajar: at a scale the SO-101 can reach, the handles are too small to grasp from closed (short cabinet pull: 4.7 mm
proud, 1 cm long; white cabinet bar: 7.7 mm gap behind an 8 mm bar, while the fixed finger is about 10 mm thick).
The oracle instead pinches the drawer's front wall (panel plus handle, 1.0 / 1.7 cm deep) from above, with one jaw
dropped into the gap behind the front panel, which needs the drawer at least ~3 cm out.

The cabinet is a free body (see ``Articulated``) and the task object of the kit's strict policy (stays put, upright,
still, only the jaws touch it, released at the end). The articulated goal (target drawer at least 5 cm open, every
other drawer closed, robot not touching the cabinet) is the goal's ``check``.

Oracle: top-down pinch of the drawer front with the jaws closing along the drawer axis, straight pull along the axis
to 6.7 cm, release with a small opening, straight lift out of the drawer, rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.val.oracle import FIXED_FACE
from sim.train.variation import GoalRegion
from sim.val.scene import ASSETS, _add_assets, _content, _merge_defaults, _soften, load_mjcf

FAMILY = "drawer_open"
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
    openable=("drawer_top",), sign=1.0, noun="short cabinet")

WHITE_CABINET = CabinetKind(
    # 0.45x (11.4 x 8.6 x 9.8 cm): at 0.5x the top drawer's front edge (10.6 cm) puts the pinch at 9.7 cm, where
    # the IK misses a vertical TCP by 2.6-3.6 mm even at r = 0.20-0.23 m.
    Articulated("cabinet", "articulated_objects/white_cabinet.xml", 0.45, mass=1.8, com=(0.0, 0.007, 0.045),
                size=(0.114, 0.086, 0.098),
                joints={j: dict(range=(-0.065, 0.0045), damping=20.0, frictionloss=0.3)
                        for j in ("top_level", "middle_level", "bottom_level")}),
    drawers={"cabinet_top": "top_level", "cabinet_middle": "middle_level", "cabinet_bottom": "bottom_level"},
    openable=("cabinet_top",), sign=-1.0, noun="cabinet")


class DrawerOpenEnv(TrainEnv):
    instruction = "Pull the top drawer of the short cabinet all the way open."
    task_objects = ("cabinet",)
    kind = SHORT_CABINET
    ajar_range = (0.030, 0.036)
    open_goal = 0.050                                     # success: target drawer at least this far out (half its depth)
    pull_to = 0.067                                       # the pinch slips 5-7 mm over the pull
    closed_tolerance = 0.004
    region = dict(r=(0.18, 0.30), angle=(-55.0, -20.0))
    facing = (25.0, 75.0)
    grip_depth = 0.010                                    # TCP below the drawer front's top edge
    reach_r = (0.13, 0.27)
    frame_xy = dict(x_min=0.09, y_min=-0.22)              # cabinet origin bounds that keep it in the front view

    def scene_objects(self):
        return [self.kind.spec]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._joint = {b: m.joint(f"cabinet_{j}").id for b, j in self.kind.drawers.items()}
        self._drawer_body = {b: m.body(f"cabinet_{b}").id for b in self.kind.drawers}
        self._robot_geoms = {g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], m.body("base").id)}

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
        return any((c.geom1 in mine and c.geom2 in self._robot_geoms) or (c.geom2 in mine and c.geom1 in self._robot_geoms)
                   for c in self.data.contact[:self.data.ncon])

    def opened(self):
        others = [b for b in self.kind.drawers if b != self.target]
        return (self.opening(self.target) >= self.open_goal
                and all(self.opening(b) <= self.closed_tolerance for b in others)
                and not self.robot_touching("cabinet"))

    def task_info(self):
        return {f"{b}_opening": round(self.opening(b), 4) for b in self.kind.drawers}

    # ----- geometry -----------------------------------------------------------------------------------------
    def out_dir(self):
        axis = self.data.xaxis[self._joint[self.target]] * self.kind.sign
        return axis[:2] / np.linalg.norm(axis[:2])

    def front_box(self, drawer):
        return local_box(self, self._drawer_body[drawer], self._body["cabinet"])

    def pinch(self):
        """(centre world xyz at TCP height, depth along the axis) of the drawer front wall incl. handle."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        u = np.r_[self.out_dir(), 0.0]
        body = self._drawer_body[self.target]
        # Oriented extents (cabinet frame, aligned with the drawer): a world-axis AABB of a yawed drawer overstates
        # its front by centimetres.
        R, p = d.xmat[self._body["cabinet"]].reshape(3, 3), d.xpos[self._body["cabinet"]]
        llo, lhi = self.front_box(self.target)
        corners = np.array([(x, y, z) for x in (llo[0], lhi[0]) for y in (llo[1], lhi[1]) for z in (llo[2], lhi[2])])
        world = corners @ R.T + p
        front = float(np.max(world @ u))
        lo, hi = world.min(0), world.max(0)
        # front panel: the widest geom that is thin along the axis, furthest along it
        side = np.array([-u[1], u[0], 0.0])
        best = None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            box = m.geom_aabb[g, 3:]
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            along = float(np.abs(R.T @ u) @ box)            # oriented half extent along the axis
            if float(np.abs(R.T @ side) @ box) > 0.025 and along < 0.006:
                if best is None or c @ u > best[0] @ u:
                    best = (c, along)
        c, along = best
        inner = float(c @ u - along)
        depth = front - inner
        centre = c - (c @ u) * u + 0.5 * (front + inner) * u
        lat = 0.5 * (lo + hi)
        centre = centre - (centre @ side) * side + (lat @ side) * side
        centre[2] = hi[2] - self.grip_depth
        return centre, depth

    def grasp_pose(self, flip):
        centre, depth = self.pinch()
        u = np.r_[self.out_dir(), 0.0]
        x = -u if flip else u
        rot = top_down_mat(float(np.arctan2(x[1], x[0])))
        pos = centre - (FIXED_FACE - depth / 2 - 0.0015) * rot[:, 0]
        return pos, rot, depth

    def plan_poses(self, flip):
        pos, rot, depth = self.grasp_pose(flip)
        u = np.r_[self.out_dir(), 0.0]
        pull = (self.pull_to - self.opening(self.target)) * u
        # Approach and leave 3-5 cm over the drawer front, capped at 12.5 cm (a vertical TCP higher than that over the
        # 11 cm white cabinet is out of IK reach).
        up = np.r_[0, 0, min(0.05, max(0.025, 0.12 - pos[2]))]
        return [(pos + up, 0.015), (pos, 0.0025), (pos + 0.5 * pull, 0.0025), (pos + pull, 0.0025),
                (pos + pull + up, 0.015)], rot, depth

    def reachable(self):
        if not hasattr(self, "_ik"):
            self._ik = TrainOracle(self)
        ik = self._ik
        for flip in (False, True):
            poses, rot, _ = self.plan_poses(flip)
            if any(not self.reach_r[0] <= np.hypot(*p[:2]) <= self.reach_r[1] for p, _ in poses):
                continue
            ik.q = np.radians(np.array(self.rest_deg[:5]))
            ok = True
            for k, (p, tol) in enumerate(poses):
                q, err, tilt = ik.solve(p, rot, seeds=None if k == 0 else [ik.q[4]])
                if err > tol or tilt > np.radians(12 if tol > 0.003 else 8):
                    ok = False
                    break
                ik.q = q
            if ok:
                return flip
        return None

    # ----- layout -------------------------------------------------------------------------------------------
    def layout(self):
        rng = self.np_random
        for b in self.kind.drawers:
            self.set_drawer(b, 0.0)
        self.target = self.kind.openable[int(rng.integers(len(self.kind.openable)))]
        ajar = float(rng.uniform(*self.ajar_range))
        for _ in range(400):
            rr, aa = rng.uniform(*self.region["r"]), np.radians(rng.uniform(*self.region["angle"]))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if xy[0] < self.frame_xy["x_min"] or xy[1] < self.frame_xy["y_min"]:
                continue
            if any(np.linalg.norm(xy - k) < self.footprint("cabinet") + kr + 0.01 for k, kr in self.keepout):
                continue
            yaw = np.radians(rng.uniform(*self.facing)) + np.pi / 2
            self.set_object_pose("cabinet", xy, yaw=yaw)
            self.set_drawer(self.target, ajar)
            mujoco.mj_kinematics(self.model, self.data)
            flip = self.reachable()
            if flip is not None:
                break
        else:
            raise RuntimeError("no reachable cabinet pose")
        self.flip = flip
        poses, _, _ = self.plan_poses(flip)
        placed = [(xy, self.footprint("cabinet"))]
        placed += [(p[:2], 0.05) for p, _ in poses]
        placed.append((poses[1][0][:2] * 0.5, 0.04))
        self.place_distractors(placed)
        self.set_goals(Goal("cabinet", "table", target=tuple(self.object_pos("cabinet")),
                            tolerance=(0.005, 0.005, 0.004), upright_cos=float(np.cos(np.radians(5))),
                            check=lambda env, name: env.opened()))

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


class DrawerOpenOracle(TrainOracle):
    pull_speed = 0.025
    close_seconds = 1.2


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
        poses, rot, depth = env.plan_poses(env.flip)
        above, grasp = poses[0][0], poses[1][0]
        lift = np.r_[0.20 * above[:2] / np.linalg.norm(above[:2]), above[2]]
        yield from self.gripper(self.open_for(depth), 0.4, 0.0)
        yield from self.unfold_transit(lift, rot)
        yield from self.move(above, rot, speed=0.10, tol=0.004, label="above")
        yield from self.move(grasp, rot, speed=0.04, tol=0.002, settle=0.8, label="grasp")
        yield from self.gripper(CLOSED, self.close_seconds, 0.3)
        self.log.append(("grasp", env.is_grasping("cabinet")))
        u = np.r_[env.out_dir(), 0.0]
        end = grasp + (env.pull_to - env.opening(env.target)) * u
        yield from self.move(end, rot, speed=self.pull_speed, tol=0.002, settle=0.6, label="pull")
        yield from self.wait(0.2)
        yield from self.gripper(self.release_for(depth), self.release_seconds, 0.2)
        yield from self.move(poses[4][0] + (end - poses[3][0]), rot, speed=0.05, label="lift")
        yield from self.rest()
        yield from self.wait(1.2)


class WhiteDrawerOpenEnv(DrawerOpenEnv):
    instruction = "Pull the top drawer of the cabinet all the way open."
    kind = WHITE_CABINET
    # Mapped over 300 random poses (IK along the whole pinch-and-pull, keep-outs clear): feasible at -60..-20 deg
    # azimuth with the drawers facing 60-120 deg.
    region = dict(r=(0.17, 0.32), angle=(-58.0, -22.0))
    facing = (60.0, 110.0)


TASKS = [
    define_task(name="open_short_cabinet_drawer", instruction=DrawerOpenEnv.instruction, family=FAMILY,
                env=DrawerOpenEnv, oracle=DrawerOpenOracle, objects=("cabinet",), object_kinds=("drawer",),
                relation="pulled open", goal="short cabinet",
                steps=("Grip the front of the short cabinet's top drawer and pull it all the way out.",)),
    define_task(name="open_white_cabinet_drawer", instruction=WhiteDrawerOpenEnv.instruction, family=FAMILY,
                env=WhiteDrawerOpenEnv, oracle=DrawerOpenOracle, objects=("cabinet",), object_kinds=("drawer",),
                relation="pulled open", goal="cabinet",
                steps=("Grip the front of the cabinet's top drawer and pull it all the way out.",)),
]

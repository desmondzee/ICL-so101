"""bottle_in_rack: put a wine bottle into a rack that holds it in one orientation.

* ``bottle_in_cradle``        -- a wine bottle lying on its side on the table is laid in a wine cradle (two notched
  saddles on a base): the bottle must lie level with its axis along the cradle (within 10 deg), across both
  saddles, held sideways by the notches.
* ``bottle_upright_in_holder`` -- a wine bottle standing on the table is stood upright in a square bottle holder
  (a walled collar on a base).

The bottle is a primitive wine bottle (body, shoulder, neck, cork, label). Its collision is box-only: a square
prism tangent to the visual cylinder for the body and a thin square prism for the neck. A round collision body
(cylinder / capsule / scanned hull) gets a single contact from MuJoCo's general convex collider against boxes
and rocks or rolls (see utensil_to_holder); box-box contacts give full patches, so the lying bottle neither rolls
on the table nor in the cradle, and the standing one does not rock. The flats coincide with the visual surface
wherever the bottle touches the table, the jaws or the rack.

Why the bottle is not turned between lying and standing (measured 2026-10-06): that needs a horizontal gripper
approach, which the 5-joint SO-101 cannot reach near the table (IK misses by 15-29 mm and ~110 deg at
r 0.16-0.24 m). Both tasks keep the top-down pinch; the cradle task needs a yaw turn to align the bottle with
the cradle.

Racks are free primitive bodies (heavy, box geoms only). None of this reproduces the held-out validation
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, COS10, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task
from sim.train.variation import GoalRegion
from sim.val.oracle import FIXED_FACE, top_down_mat

FAMILY = "bottle_in_rack"
GLASS = (0.12, 0.32, 0.16, 1.0)
LABEL = (0.92, 0.88, 0.74, 1.0)
CORK = (0.55, 0.38, 0.22, 1.0)
WOOD = (0.55, 0.36, 0.20, 1.0)
WHITE = (0.86, 0.86, 0.88, 1.0)
MIN_GRASP_Z = 0.0105            # TCP floor above the table: the jaw hull reaches ~8.2 mm below the TCP
REST_ZONE = (np.array([0.16, 0.0]), 0.085)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def _f(v):
    return " ".join(f"{x:.6g}" for x in v)


def _box_geom(body, name, pos, half, rgba, mass, friction=1.0, material="val_fabric"):
    ET.SubElement(body, "geom", name=name, type="box", pos=_f(pos), size=_f(half), rgba=_f(rgba),
                  mass=f"{mass:.6g}", group="1", condim="4", friction=f"{friction} 0.02 0.001", material=material)


# ----- assets ----------------------------------------------------------------------------------------------------

@dataclass
class WineBottle:
    """A free wine bottle. ``lying``: the body frame has the bottle axis along x (neck at +x), else along z (neck
    up). The body origin is the centre of the body prism."""

    name: str
    lying: bool = True
    radius: float = 0.013
    body_len: float = 0.064
    shoulder: float = 0.012
    neck_r: float = 0.005
    neck_len: float = 0.028
    mass: float = 0.06
    rgba: tuple = GLASS
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        # Geometry is written along +z, then the inner body turns it to the requested frame.
        inner = ET.SubElement(body, "body", name=f"{self.name}_axis",
                              quat="0.707107 0 0.707107 0" if self.lying else "1 0 0 0")
        r, L, s, nr, nl = self.radius, self.body_len, self.shoulder, self.neck_r, self.neck_len
        neck_c = L / 2 + (s + nl) / 2
        neck_mass = 0.12 * self.mass
        common = dict(group="3", condim="4", friction=f"{self.friction} 0.02 0.001", rgba="0.5 0.5 0.5 0")
        ET.SubElement(inner, "geom", name=f"{self.name}_body", type="box", pos="0 0 0", size=_f((r, r, L / 2)),
                      mass=f"{self.mass - neck_mass:.6g}", **common)
        ET.SubElement(inner, "geom", name=f"{self.name}_neck", type="box", pos=_f((0, 0, neck_c)),
                      size=_f((nr, nr, (s + nl) / 2)), mass=f"{neck_mass:.6g}", **common)
        vis = dict(contype="0", conaffinity="0", group="1", density="0", material="val_plastic")
        ET.SubElement(inner, "geom", name=f"{self.name}_v_body", type="cylinder", size=_f((r, L / 2)),
                      rgba=_f(self.rgba), **vis)
        ET.SubElement(inner, "geom", name=f"{self.name}_v_label", type="cylinder", pos=_f((0, 0, -0.004)),
                      size=_f((r * 1.015, 0.014)), rgba=_f(LABEL), **vis)
        ET.SubElement(inner, "geom", name=f"{self.name}_v_shoulder", type="ellipsoid", pos=_f((0, 0, L / 2)),
                      size=_f((r, r, s * 1.3)), rgba=_f(self.rgba), **vis)
        ET.SubElement(inner, "geom", name=f"{self.name}_v_neck", type="cylinder", pos=_f((0, 0, neck_c)),
                      size=_f((nr, (s + nl) / 2)), rgba=_f(self.rgba), **vis)
        ET.SubElement(inner, "geom", name=f"{self.name}_v_cork", type="cylinder",
                      pos=_f((0, 0, L / 2 + s + nl - 0.004)), size=_f((nr * 1.08, 0.0045)), rgba=_f(CORK), **vis)


@dataclass
class Cradle:
    """Free wine cradle: a base board and two saddles (cross bar + two posts forming a notch) at x = +-``saddle_x``.
    A bottle lies along the body's x axis across both saddles. Origin: centre of the base underside."""

    name: str
    bottle_r: float = 0.013
    clearance: float = 0.0035    # notch half-gap beyond the bottle flats
    saddle_x: float = 0.023
    bar: float = 0.010           # saddle cross-bar height above the base
    post: float = 0.010          # post height above the cross bar
    thick: float = 0.006         # saddle thickness along x
    base: float = 0.005
    base_half: tuple = (0.040, 0.028)
    rgba: tuple = WOOD
    mass: float = 0.5

    @property
    def floor(self):
        return self.base + self.bar

    def boxes(self):
        b, t, h, p = self.base, self.thick, self.bar, self.post
        half_y = self.base_half[1]
        out = [((0, 0, b / 2), (*self.base_half, b / 2))]
        post_w = half_y - self.bottle_r - self.clearance
        py = self.bottle_r + self.clearance + post_w / 2
        for x in (-self.saddle_x, self.saddle_x):
            out.append(((x, 0, b + h / 2), (t / 2, half_y, h / 2)))
            for y in (-py, py):
                out.append(((x, y, b + h + p / 2), (t / 2, post_w / 2, p / 2)))
        return out

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        boxes = self.boxes()
        vols = [8 * np.prod(h) for _, h in boxes]
        for k, ((pos, half), v) in enumerate(zip(boxes, vols)):
            _box_geom(body, f"{self.name}_geom{k}", pos, half, self.rgba, self.mass * v / sum(vols))


@dataclass
class Holder:
    """Free square bottle holder: a base plate and four walls around an ``inner`` half-width square."""

    name: str
    inner: float = 0.019
    wall: float = 0.004
    height: float = 0.024        # wall height above the base top
    base: float = 0.005
    rgba: tuple = WHITE
    mass: float = 0.5

    @property
    def floor(self):
        return self.base

    def boxes(self):
        i, w, h, b = self.inner, self.wall, self.height, self.base
        zc = b + h / 2
        return [((0, 0, b / 2), (i + w, i + w, b / 2)),
                ((i + w / 2, 0, zc), (w / 2, i + w, h / 2)), ((-i - w / 2, 0, zc), (w / 2, i + w, h / 2)),
                ((0, i + w / 2, zc), (i, w / 2, h / 2)), ((0, -i - w / 2, zc), (i, w / 2, h / 2))]

    build_mjcf = Cradle.build_mjcf


# ----- environment -----------------------------------------------------------------------------------------------

RACK_REGION = dict(r=(0.19, 0.245), angle=(-55.0, 55.0))
OBJECT_REGION = dict(r=(0.18, 0.255), angle=(-62.0, 62.0))


class BottleEnv(TrainEnv):
    obj = "bottle"
    rack_name = "rack"
    task_objects = ("bottle",)
    bottle: WineBottle = None
    rack = None
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("plate", "akita_black_bowl", "white_bowl",
                                                                     "red_bowl"))

    def scene_objects(self):
        return [self.bottle, self.rack]

    def rack_frame(self):
        b = self._body[self.rack_name]
        return self.data.xmat[b].reshape(3, 3), self.data.xpos[b]

    def object_local(self):
        R, p = self.rack_frame()
        return R.T @ (self.object_pos(self.obj) - p)

    def rack_radius(self):
        return float(np.hypot(*self.rack.boxes()[0][1][:2]))

    def object_radius(self):
        return self.footprint(self.obj)

    def rack_yaw(self):
        return self.np_random.uniform(-np.pi, np.pi)

    def object_yaw(self):
        return self.np_random.uniform(-np.pi, np.pi)

    def layout(self):
        placed = []
        rr = self.rack_radius()
        for _ in range(200):
            xy = self.sample_xy(rr, placed, clearance=0.03, **RACK_REGION)
            if np.linalg.norm(xy - REST_ZONE[0]) >= REST_ZONE[1] + rr:
                break
        self.set_object_pose(self.rack_name, xy, yaw=self.rack_yaw())
        placed.append((xy, rr))
        r = self.object_radius()
        for _ in range(200):
            oxy = self.sample_xy(r, placed, clearance=0.035, **OBJECT_REGION)
            if np.linalg.norm(oxy - REST_ZONE[0]) >= REST_ZONE[1] + r:
                break
        self.set_object_pose(self.obj, oxy, yaw=self.object_yaw())
        placed.append((oxy, r))
        self.place_distractors(placed)
        mujoco.mj_forward(self.model, self.data)
        R, p = self.rack_frame()
        c = p + R @ np.array([0.0, 0.0, self.rack.floor + self.origin_height()])
        self.set_goals(Goal(self.obj, self.rack_name, target=tuple(c), reference=self.rack_name,
                            tolerance=self.tolerance, check=self.seated))

    def goal_regions(self):
        R, p = self.rack_frame()
        z = self.rack.floor + 2 * self.bottle.radius
        pts = [tuple(p + R @ np.array([x, y, z])) for x in (-0.012, 0.0, 0.012) for y in (-0.006, 0.0, 0.006)]
        return (GoalRegion(self.rack_name, tuple(pts)),)


class CradleEnv(BottleEnv):
    instruction = "Lay the wine bottle in the wine cradle."
    bottle = WineBottle("bottle", lying=True)
    rack = Cradle("rack")
    tolerance = (0.014, 0.014, 0.004)

    def origin_height(self):
        return self.bottle.radius

    def seated(self, env, name) -> bool:
        R, _ = self.rack_frame()
        local = self.object_local()
        along = self.axis_alignment(self.obj, R[:, 0], local_axis=0) >= COS10
        return bool(along and abs(local[0]) <= 0.010 and abs(local[1]) <= self.rack.clearance + 0.001
                    and local[2] <= self.rack.floor + self.bottle.radius + 0.003)


class CradleNeckRightEnv(CradleEnv):
    """As ``CradleEnv``, but the cradle runs roughly left-right in the front view (its axis within 30 deg of world
    y) and the bottle must lie with its neck toward the viewer's right (world +y)."""

    instruction = "Lay the wine bottle in the wine cradle with its neck pointing to the right."
    neck_heading = np.pi / 2     # world +y: the viewer's right

    def rack_yaw(self):
        return np.pi / 2 + self.np_random.uniform(-np.pi / 6, np.pi / 6)

    def seated(self, env, name) -> bool:
        neck = self.data.xmat[self._body[self.obj]].reshape(3, 3)[:, 0]
        return bool(super().seated(env, name) and neck[1] >= 0.5)


class HolderEnv(BottleEnv):
    instruction = "Stand the wine bottle upright in the bottle holder."
    bottle = WineBottle("bottle", lying=False)
    rack = Holder("rack")
    tolerance = (0.012, 0.012, 0.004)

    def origin_height(self):
        return self.bottle.body_len / 2

    def object_radius(self):
        return float(np.hypot(self.bottle.radius, self.bottle.radius))

    def seated(self, env, name) -> bool:
        local = self.object_local()
        room = self.rack.inner - self.bottle.radius + 0.001
        return bool(abs(local[0]) <= room and abs(local[1]) <= room
                    and local[2] <= self.rack.floor + self.bottle.body_len / 2 + 0.003)


# ----- oracle ----------------------------------------------------------------------------------------------------

class BottleOracle(TrainOracle):
    """Top-down pinch of the bottle body across two flats near its centre of mass (lying) or near the top of the
    body (standing); carry, turn the bottle's axis along the cradle (lying task), level, lower it onto the rack
    floor, ease the grip, open and retreat straight up."""

    close_seconds = 2.0
    release_seconds = 1.5
    open_margin = 0.010
    squeeze = 0.004
    relax_seconds = 0.8
    release_offset = 0.0008
    transit_speed = 0.5
    vmax = np.radians([45.0, 50.0, 50.0, 65.0, 65.0])
    grasp_depth = 0.012          # standing bottle: TCP below the top of the body prism

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def carry_to(self, pos, rot, label="carry"):
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, label=label)
        else:
            yield from self.transit(pos, rot, speed=self.transit_speed, label=label)

    def level(self, name, rot):
        env = self.env
        R_pred = rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        z = R_pred[:, 2]
        axis = np.cross(z, [0.0, 0.0, 1.0])
        s, c = np.linalg.norm(axis), float(z[2])
        if s < 1e-6:
            return rot
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return (np.eye(3) + s * K + (1 - c) * K @ K) @ rot

    def heading(self, name, rot):
        env = self.env
        m = rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def pinch_lateral(self):
        """Offset of the fingertip pads' centre from the TCP along the TCP y axis (measured ~5 mm)."""
        if not hasattr(self, "_pinch_lateral"):
            m, d = self.model, self.ik_data
            d.qpos[self.env._arm_qpos_addrs] = self.q
            mujoco.mj_kinematics(m, d)
            tips = [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
            R = d.site_xmat[self.env._tcp_site_id].reshape(3, 3)
            rel = (d.geom_xpos[tips] - d.site_xpos[self.env._tcp_site_id]) @ R
            self._pinch_lateral = float(rel[:, 1].mean())
        return self._pinch_lateral

    def grasp_plan(self, center, width, angles):
        """Kit plan with the TCP shifted so the fingertip pads (not the TCP site) centre on ``center``."""
        best = None
        for a in angles:
            rot = top_down_mat(a)
            pos = (center - (FIXED_FACE - width / 2 - self.grasp_clearance) * rot[:, 0]
                   - self.pinch_lateral() * rot[:, 1])
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def grasp_point(self):
        """(world pinch centre, closing yaws, TCP z)."""
        env = self.env
        b = env.bottle
        body = env._body[env.obj]
        R, p = env.data.xmat[body].reshape(3, 3), env.data.xpos[body]
        if b.lying:
            com = R.T @ (env.data.subtree_com[body] - p)
            x = float(np.clip(com[0], -b.body_len / 2 + 0.012, b.body_len / 2 - 0.012))
            centre = p + R @ np.array([x, 0.0, 0.0])
            yaw = np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2
            z = max(centre[2], p[2] - b.radius + MIN_GRASP_Z)
            return centre, [yaw, yaw + np.pi], z
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return p, [yaw + k * np.pi / 2 for k in range(4)], p[2] + b.body_len / 2 - self.grasp_depth

    def pick_bottle(self, attempts=2, approach=0.05):
        env = self.env
        name = env.obj
        width = 2 * env.bottle.radius
        self.width = width
        for attempt in range(attempts):
            centre, angles, z = self.grasp_point()
            angles = self.choose_closing(centre, angles, z, width)
            rot, grasp, q = self.grasp_plan(np.r_[centre[:2], z], width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.05, tol=0.003, label="grasp")
            yield from self.gripper(self.gap_angle(width + 0.004), 0.5, 0.1)
            yield from self.gripper(CLOSED, self.close_seconds, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.r_[grasp[:2], self.carry_z], rot, speed=0.08, label="lift")
                if env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def choose_closing(self, centre, angles, z, width):
        """Directed cradle task: of the two closing directions across the bottle, the one whose turned end pose
        (carry and release) the IK reaches (the wrist roll is limited to +-157 deg, so a turn of up to 180 deg is
        reachable from only one of the two grasps)."""
        env = self.env
        heading = getattr(env, "neck_heading", None)
        if heading is None or not env.bottle.lying:
            return angles
        R, _ = env.rack_frame()
        axis_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if np.cos(axis_yaw - heading) < 0:
            axis_yaw += np.pi
        body = env._body[env.obj]
        bottle_yaw = float(np.arctan2(env.data.xmat[body][3], env.data.xmat[body][0]))
        d = wrap(axis_yaw - bottle_yaw)
        target = env.goal_target(env.goals[0])
        best = None
        for a in angles:
            rot, pos, q0 = self.grasp_plan(np.r_[centre[:2], z], width, [a])
            end = rot_z(d) @ rot
            offset = rot_z(d) @ (pos - np.r_[env.object_pos(env.obj)[:2], pos[2]])
            tcp = np.r_[target[:2] + offset[:2], target[2] + (pos[2] - env.object_pos(env.obj)[2])]
            e0 = self.solve(pos, rot)[1]
            (q1, e1, _), (_, e2, _) = self.solve(np.r_[tcp[:2], self.carry_z], end), self.solve(tcp, end)
            score = 200 * max(e0, e1, e2) + 0.3 * abs(q1[4] - q0[4])
            self.log.append(("closing", round(float(a), 2), round(e0 * 1000, 1), round(e1 * 1000, 1),
                             round(e2 * 1000, 1)))
            if best is None or score < best[0]:
                best = (score, a)
        return [best[1]]

    def end_rotation(self, name, target):
        env = self.env
        if not env.bottle.lying:
            return self.level(name, self.carry_rot(target[:2]))
        heading = getattr(env, "neck_heading", None)
        if heading is not None:
            # Directed: the neck must end at ``heading``; only the 2*pi-equivalent turns are candidates.
            R, _ = env.rack_frame()
            axis_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            if np.cos(axis_yaw - heading) < 0:
                axis_yaw += np.pi
            d = wrap(axis_yaw - self.heading(name, self._cmd_rot))
            return self.feasible_rotation(name, target, [self.level(name, rot_z(d + k) @ self._cmd_rot)
                                                          for k in (0.0, -2 * np.pi, 2 * np.pi)])
        R, _ = env.rack_frame()
        axis_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        cur = self.heading(name, self._cmd_rot)
        azimuth = np.arctan2(target[1], target[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        deltas = sorted((wrap(axis_yaw - cur) + k * np.pi for k in (-2, -1, 0, 1, 2)), key=lambda d: abs(d - azimuth))
        rotations = [self.level(name, rot_z(d) @ self._cmd_rot) for d in deltas[:3]]
        return self.feasible_rotation(name, target, rotations)

    def insert(self, name, target, rot):
        target = np.asarray(target, float)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], self.carry_z], rot)
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], self.carry_z], rot, speed=0.04, label="align", smooth=False)
        tcp = target - rot @ self.held(name)
        yield from self.move(tcp + [0, 0, 0.015], rot, speed=0.05, tol=0.002, settle=0.6, label="pre-lower")
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(tcp + [0, 0, self.release_offset], rot, speed=0.03, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.gripper(self.gap_angle(self.width - self.squeeze), self.relax_seconds, 0.2)
        yield from self.gripper(self.open_for(self.width, self.release_margin), self.release_seconds, 0.3)
        # Back the fixed finger off the face slowly before lifting: it rests on the object after the jaw opens
        # and would drag/rock it on the way up.
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.008, tol=0.002,
                             settle=0.3, label="backoff", smooth=False)
        yield from self.move(self._cmd_pos + [0, 0, 0.02], rot, speed=0.03, tol=0.003, settle=0.2,
                             label="slide out", smooth=False)
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], rot, speed=0.08, label="retreat")

    def plan(self):
        env = self.env
        ok = yield from self.pick_bottle()
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        rot = self.end_rotation(env.obj, target)
        yield from self.insert(env.obj, target, rot)
        yield from self.rest()
        yield from self.wait(1.2)


class CradleOracle(BottleOracle):
    carry_z = 0.085
    # Release resting on the saddles: a 0.8 mm fall onto the two narrow saddle bars rocked the bottle at
    # 1070-1900 rad/s^2 (3/4 calibration seeds).
    release_offset = -0.0005


class HolderOracle(BottleOracle):
    carry_z = 0.105              # standing bottle's bottom ~5 cm above the table: clears the holder walls


TASKS = [
    define_task(name="bottle_in_cradle", instruction=CradleEnv.instruction, family=FAMILY, env=CradleEnv,
                oracle=CradleOracle, objects=("bottle",), object_kinds=("wine bottle",),
                relation="lying along in cradle", goal="wine cradle",
                steps=("Pick up the wine bottle, turn it to line up with the cradle and lay it in the wine "
                       "cradle.",)),
    define_task(name="bottle_in_cradle_neck_right", instruction=CradleNeckRightEnv.instruction, family=FAMILY,
                env=CradleNeckRightEnv, oracle=CradleOracle, objects=("bottle",), object_kinds=("wine bottle",),
                relation="lying in cradle with neck pointing right", goal="wine cradle",
                steps=("Pick up the wine bottle, turn it so its neck points to the right and lay it in the wine "
                       "cradle.",)),
    define_task(name="bottle_upright_in_holder", instruction=HolderEnv.instruction, family=FAMILY, env=HolderEnv,
                oracle=HolderOracle, objects=("bottle",), object_kinds=("wine bottle",),
                relation="standing upright in holder", goal="bottle holder",
                steps=("Pick up the wine bottle and stand it upright in the bottle holder.",)),
]

"""utensil_to_holder: put a thin, elongated utensil inside an open holder, lying along it.

Every task starts with the object lying on the table at a random position and heading and ends with it lying
inside the holder: every collision corner inside the walls and below their top, resting on the holder's floor
(positive upward support contact), level, released and settled. The tasks differ in object kind (marker, honey
dipper) and holder kind (pencil box, cutlery tray); both require turning the object so its long axis runs along
the holder.

Object models (measured 2026-10-06). The scanned YCB/GSO convex hulls rest on one or two contact points and rock
forever on a flat support (YCB screwdriver 1.9 rad/s and marker 0.35 rad/s peak after 2 s on the bare table,
because MuJoCo's general convex collider returns one contact per pair). ``ProxyScan`` keeps the scanned visual
mesh and replaces the collision with boxes fitted to slices of the visual mesh along the object's long axis
(box-box/box-plane contacts give full contact patches), with a realistic mass split per slice. Its body frame
is the lying frame: x along the long axis (thick end at -x), z up, origin at the collision bounding-box centre.

Grasping: the open jaw's fixed finger hangs ~8 mm below the fingertips, so a lying object is pinched at its
thick part (marker body 1.9 cm, dipper head 1.6 cm) with the TCP >= 10.5 mm above the table, across the long
axis; the fingertip pads sit ~5 mm to one side of the TCP site, which the grasp plan corrects for.

Holders are heavy free primitive boxes (``Crate``) with low walls (1.6-1.8 cm), every contact box-box.
Dropped (measured): a screwdriver in a 3 cm-walled toolbox (the opening moving jaw hit the wall at 20 N; widened
to 7.2 cm with 2.2 cm walls, 11/24 calibration seeds: tipping on grasp and release spikes), and a honey dipper in
an akita/white bowl (curved floor: the fixed jaw beside the off-centre head pressed the slope at 20-30 N) or a
1.5x LIBERO ramekin "dish" (the gripper housing hit its 5.3 cm wall).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task,
                                  step_text)
from sim.train.tasks.assets import Scanned
from sim.train.variation import GoalRegion
from sim.val.oracle import FIXED_FACE, RELEASE, top_down_mat

FAMILY = "utensil_to_holder"
MIN_GRASP_Z = 0.0105            # TCP floor above the table: the fixed jaw hangs ~8 mm below the fingertips
ROBOT_REACH = 0.315             # every object / holder corner stays within this radius (camera framing, reach)
REST_ZONE = (np.array([0.16, 0.0]), 0.085)   # folded gripper hover and rest sweep: keep holders and objects out


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).flatten())
    return q


# ----- assets -------------------------------------------------------------------------------------------------

@dataclass
class ProxyScan(Scanned):
    """Scanned visual mesh with box collision fitted to slices of it (see module docstring).

    ``segments``: (start, end, mass fraction) along the long axis, as fractions of its length measured from the
    thick end. Each slice of the visual mesh gets one box over its y/z extent.
    """

    segments: tuple = ((0.0, 1.0, 1.0),)
    solref: tuple = (0.03, 1.0)   # object-support contact (jaw geoms have priority, so jaw contacts keep theirs)

    @np.errstate(all="ignore")     # spurious matmul overflow warnings from the vertex transforms
    def build_mjcf(self, mj, asset_el, worldbody):
        from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose
        _, acc = self._accessors()
        acc.ensure_assets(self.model_id)
        parts = acc.collision_parts(self.model_id)
        hull = self._body_verts([(f"{self.name}_probe{k}", p) for k, p in enumerate(parts)], self.scale)
        rest = np.zeros(9)
        mujoco.mju_quat2Mat(rest, np.asarray(get_mujoco_ycb_rest_pose(hull, model_id=self.model_id)[0], float))
        rest = rest.reshape(3, 3)
        visual = acc.visual_mesh(self.model_id)
        verts = self._body_verts([(f"{self.name}_vprobe", SimpleNamespace(path=visual))], self.scale) @ rest.T
        xy = verts[:, :2] - verts[:, :2].mean(0)
        axis = np.linalg.svd(xy, full_matrices=False)[2][0]
        turn = rot_z(-np.arctan2(axis[1], axis[0]))
        lying = verts @ turn.T
        lo, hi = lying.min(0), lying.max(0)
        length = hi[0] - lo[0]

        def area(a, b):
            sl = lying[(lying[:, 0] >= lo[0] + a * length) & (lying[:, 0] <= lo[0] + b * length)]
            return np.ptp(sl[:, 1]) * np.ptp(sl[:, 2]) if len(sl) > 3 else 0.0

        if area(0.8, 1.0) > area(0.0, 0.2):             # thick end at -x
            turn = rot_z(np.pi) @ turn
            lying = verts @ turn.T
            lo, hi = lying.min(0), lying.max(0)
        boxes = []
        for a, b, fraction in self.segments:
            x0, x1 = lo[0] + a * length, lo[0] + b * length
            sl = lying[(lying[:, 0] >= x0) & (lying[:, 0] <= x1)]
            slo, shi = sl.min(0), sl.max(0)
            boxes.append(((x0 + x1) / 2, (slo[1] + shi[1]) / 2, (slo[2] + shi[2]) / 2,
                          (x1 - x0) / 2, (shi[1] - slo[1]) / 2, (shi[2] - slo[2]) / 2, fraction))
        self.profile = [(round(float(x), 4), round(float(np.ptp(lying[np.abs(lying[:, 0] - x) <= length / 80][:, 1])), 4),
                         round(float(np.ptp(lying[np.abs(lying[:, 0] - x) <= length / 80][:, 2])), 4))
                        for x in np.linspace(lo[0], hi[0], 41) if np.sum(np.abs(lying[:, 0] - x) <= length / 80) > 2]
        b = np.array(boxes)
        box_lo = np.min(b[:, :3] - b[:, 3:6], 0)
        box_hi = np.max(b[:, :3] + b[:, 3:6], 0)
        centre = (box_lo + box_hi) / 2
        mass = self.mass if self.mass is not None else 0.02
        prefix = f"{self.name}_scan"
        ET.SubElement(asset_el, "mesh", name=f"{prefix}_vis", file=visual.as_posix(),
                      scale=" ".join([f"{self.scale:.6g}"] * 3))
        material = None
        texture = acc.texture_file(self.model_id)
        if texture.exists():
            ET.SubElement(asset_el, "texture", name=f"{prefix}_tex", type="2d", file=texture.as_posix())
            material = f"{prefix}_mat"
            ET.SubElement(asset_el, "material", name=material, texture=f"{prefix}_tex", texuniform="false",
                          specular="0.1", reflectance="0")
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        for k, (cx, cy, cz, hx, hy, hz, fraction) in enumerate(boxes):
            ET.SubElement(body, "geom", name=f"{prefix}_box{k}", type="box",
                          pos=" ".join(f"{v:.6g}" for v in np.array([cx, cy, cz]) - centre),
                          size=f"{hx:.6g} {hy:.6g} {hz:.6g}", mass=repr(float(mass * fraction)), group="3",
                          condim="4", friction=f"{self.friction} 0.02 0.001",
                          solref=" ".join(map(str, self.solref)), rgba="0 0 0 0")
        R = turn @ rest
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", pos=" ".join(f"{-v:.6g}" for v in centre),
                                quat=" ".join(f"{q:.6g}" for q in _quat(R)))
        vis = dict(name=f"{prefix}_visual", type="mesh", mesh=f"{prefix}_vis", group="1", contype="0",
                   conaffinity="0", mass="0")
        if material:
            vis["material"] = material
        ET.SubElement(upright, "geom", **vis)


@dataclass
class Crate:
    """An open-top rectangular box (free body) made of a floor slab and four walls (after ``gather_two``).

    ``inner``: interior half-extents (x, y); ``wall``: wall height above the floor top. The body origin is the
    centre of the floor's underside; the long axis is the body's x axis."""

    name: str
    inner: tuple = (0.07, 0.03)
    wall: float = 0.02
    thickness: float = 0.006
    floor: float = 0.006
    rgba: tuple = (0.62, 0.43, 0.25, 1.0)
    mass: float = 0.6
    friction: float = 1.0
    material: str = "val_fabric"

    def boxes(self):
        ix, iy = self.inner
        t, f, h = self.thickness, self.floor, self.wall
        zc = (f + h) / 2
        return [((0, 0, f / 2), (ix + t, iy + t, f / 2)),
                ((ix + t / 2, 0, zc), (t / 2, iy + t, zc)), ((-ix - t / 2, 0, zc), (t / 2, iy + t, zc)),
                ((0, iy + t / 2, zc), (ix, t / 2, zc)), ((0, -iy - t / 2, zc), (ix, t / 2, zc))]

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        boxes = self.boxes()
        volumes = [8 * np.prod(h) for _, h in boxes]
        for k, ((pos, half), vol) in enumerate(zip(boxes, volumes)):
            ET.SubElement(body, "geom", name=f"{self.name}_geom{k}", type="box",
                          pos=" ".join(f"{v:.6g}" for v in pos), size=" ".join(f"{v:.6g}" for v in half),
                          rgba=" ".join(map(str, self.rgba)), mass=f"{self.mass * vol / sum(volumes):.6g}", group="1",
                          condim="4", friction=f"{self.friction} 0.02 0.001", material=self.material)


# ----- geometry helpers -----------------------------------------------------------------------------------------

def _rect_circle_gap(centre, yaw, half, point):
    """Distance from ``point`` to an oriented rectangle (0 inside)."""
    local = rot_z(-yaw)[:2, :2] @ (np.asarray(point, float) - centre)
    d = np.maximum(np.abs(local) - np.asarray(half), 0.0)
    return float(np.hypot(*d))


def _rect_cover(centre, yaw, half):
    """Circles (xy, radius) covering an oriented rectangle, for the kit's circle-based ``placed`` list."""
    hx, hy = half
    n = max(1, int(np.ceil(hx / hy)))
    step = 2 * hx / n
    radius = float(np.hypot(step / 2, hy))
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    return [(centre + axis * (-hx + step * (k + 0.5)), radius) for k in range(n)]


class HolderEnv(TrainEnv):
    """A thin object lying on the table and a holder; goal: object wholly inside the holder, lying in it."""

    obj = "utensil"
    holder = "holder"
    holder_region = dict(r=(0.175, 0.225), angle=(-60.0, 60.0))   # release poses stay near-vertical
    object_region = dict(r=(0.15, 0.26), angle=(-65.0, 65.0))     # grasp point (centre of mass) region
    lying_cos = COS10
    inside_margin = 0.0                                             # extra clearance to the walls (m)

    # ----- object geometry -----------------------------------------------------------------------------------
    def boxes(self, name):
        """[(geom id, local centre, half size)] of a free body's collision boxes, in its body frame."""
        m = self.model
        return [(g, m.geom_pos[g].copy(), m.geom_size[g].copy()) for g in self._geoms[name]]

    def local_half(self, name):
        """(half length, half width, bottom offset) of a ProxyScan's collision boxes in its lying frame."""
        pts = self.local_corners(name)
        return (float(np.max(np.abs(pts[:, 0]))), float(np.max(np.abs(pts[:, 1]))), float(-pts[:, 2].min()))

    def local_corners(self, name):
        m = self.model
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        pts = []
        for g in self._geoms[name]:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append((corners * m.geom_size[g]) @ R.reshape(3, 3).T + m.geom_pos[g])
        return np.vstack(pts)

    def world_corners(self, name):
        body = self._body[name]
        return self.local_corners(name) @ self.data.xmat[body].reshape(3, 3).T + self.data.xpos[body]

    def com_local(self, name):
        """Centre of mass of a free body in its body frame (its only massive geoms are on the body itself)."""
        return self.model.body_ipos[self._body[name]].copy()

    # ----- layout --------------------------------------------------------------------------------------------
    def sample_long(self, half, placed, region, clearance=0.03, yaw=None, tries=400, anchor=None):
        """Sample (centre xy, yaw) for an oriented rectangle with half sizes ``half`` whose ``anchor`` point (in its
        frame, default the centre) lies in the polar ``region``; clear of the keep-outs, the rest zone, the
        ``placed`` circles, and within ``ROBOT_REACH``."""
        anchor = np.zeros(2) if anchor is None else np.asarray(anchor, float)
        for _ in range(tries):
            rr = self.np_random.uniform(*region["r"])
            aa = np.radians(self.np_random.uniform(*region["angle"]))
            th = self.np_random.uniform(-np.pi, np.pi) if yaw is None else yaw(np.array([np.cos(aa), np.sin(aa)]))
            point = rr * np.array([np.cos(aa), np.sin(aa)])
            centre = point - rot_z(th)[:2, :2] @ anchor
            corners = [centre + rot_z(th)[:2, :2] @ (np.array(s) * half) for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            if max(np.hypot(*c) for c in corners) > ROBOT_REACH:
                continue
            if max(abs(np.degrees(np.arctan2(c[1], c[0]))) for c in corners) > 78.0:
                continue
            blockers = [*self.keepout, REST_ZONE, *placed]
            if all(_rect_circle_gap(centre, th, half, p) >= q + clearance for p, q in blockers):
                return centre, th
        raise RuntimeError("could not place elongated object")

    max_azimuth_gap = None      # deg: bound on the azimuth swing between the object and the holder

    def place_object(self, placed, holder_xy=None, holder_yaw=None):
        """Put the task object lying on the table with its centre of mass in ``object_region`` (within
        ``max_azimuth_gap`` degrees of the holder's azimuth when set)."""
        hx, hy, _ = self.local_half(self.obj)
        com = self.com_local(self.obj)[:2]
        region = dict(self.object_region)
        if self.max_azimuth_gap is not None:
            az_h = np.degrees(np.arctan2(holder_xy[1], holder_xy[0]))
            lo, hi = region["angle"]
            region["angle"] = (max(lo, az_h - self.max_azimuth_gap), min(hi, az_h + self.max_azimuth_gap))
        centre, yaw = self.sample_long((hx, hy), placed, region, anchor=com)
        self.set_object_pose(self.obj, centre, yaw=yaw)
        placed.extend(_rect_cover(centre, yaw, (hx, hy)))

    # ----- success -------------------------------------------------------------------------------------------
    def lying(self, name, cos=None):
        return self.upright(name, self.lying_cos if cos is None else cos)

    def inside(self, env=None, obj=None) -> bool:
        raise NotImplementedError

    def goal_check(self, env, obj):
        return self.inside() and self.released(obj)


class BoxHolderEnv(HolderEnv):
    """Holder is a ``Crate``: the object must lie with every collision corner inside the walls and below their top."""

    crate: Crate = None

    def scene_objects(self):
        return [self.object_spec(), self.crate]

    def object_spec(self):
        raise NotImplementedError

    def inside(self, env=None, obj=None) -> bool:
        c = self.crate
        body = self._body[self.holder]
        R, p = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        local = (self.world_corners(self.obj) - p) @ R
        m = self.inside_margin
        return bool(np.all(np.abs(local[:, 0]) <= c.inner[0] - m) and np.all(np.abs(local[:, 1]) <= c.inner[1] - m)
                    and np.all(local[:, 2] >= c.floor - 0.003) and np.all(local[:, 2] <= c.floor + c.wall + 0.03))

    def layout(self):
        placed = []
        c = self.crate
        half = (c.inner[0] + c.thickness, c.inner[1] + c.thickness)
        centre, yaw = self.sample_long(half, placed, self.holder_region, clearance=0.03)
        self.set_object_pose(self.holder, centre, yaw=yaw)
        placed.extend(_rect_cover(centre, yaw, half))
        self.place_object(placed, centre, yaw)
        self.place_distractors(placed)
        floor = self.object_pos(self.holder)[2] - self._extent[self.holder]["bottom"] + c.floor
        bottom = self.local_half(self.obj)[2]
        self.set_goals(Goal(self.obj, self.holder, target=(*centre, floor + bottom + 0.0005), reference=self.holder,
                            tolerance=(0.035, 0.035, 0.006), upright_cos=self.lying_cos, check=self.goal_check))

    def goal_regions(self):
        """Samples across the box opening at rim height (the floor is hidden by the near wall)."""
        c = self.crate
        body = self._body[self.holder]
        R, p = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        z = c.floor + c.wall + 0.002
        pts = [tuple(p + R @ np.array([x, y, z])) for x in (-0.6 * c.inner[0], 0, 0.6 * c.inner[0])
               for y in (-0.5 * c.inner[1], 0, 0.5 * c.inner[1])]
        return (GoalRegion(self.holder, tuple(pts)),)

    def place_xy(self):
        return self.goal_target(self.goals[0])[:2]

    def target_yaws(self):
        """Object yaws that lie along the box (either end first)."""
        yaw = self.yaw(self.holder)
        return (yaw, yaw + np.pi)


# ----- oracle ---------------------------------------------------------------------------------------------------

class ThinOracle(TrainOracle):
    """Pinch a lying thin object across its long axis near its centre of mass, carry it level, turn it to the
    holder's axis when the holder has one, lower it until its lowest point is just above the holder surface and
    release.

    * Staged close (from ``relative_left_right``): ease the moving jaw onto the object, then squeeze.
    * Levelling (from ``take_out_of_container``): after the carry the TCP is re-oriented so the held object lies
      level, then re-measured just above the release pose.
    """

    touch_depth = 0.001
    min_grasp_z = MIN_GRASP_Z
    touch_seconds = 0.9
    grasp_offset = 0.0           # grasp point along the object's x axis relative to its centre of mass (m)
    grasp_x_limits = None       # (lo, hi) clamp of the grasp point along x (local frame), e.g. the handle
    pre_release_height = 0.012
    release_clearance = 0.0008  # lowest object point above the holder surface at release
    squeeze = 0.0             # closing target this far inside the object width (fingertip-sphere centres)
    release_seconds = 0.2       # open quickly: a slow opening lets the falling object rattle between the tips

    # ----- grasp ---------------------------------------------------------------------------------------------
    def close_target(self, width):
        """Gripper command closing to ``width - squeeze`` at the fingertip-sphere centres (``None``: fully). The
        full close pressed the tip spheres 1.5-2.5 mm into the marker; on opening it sprang out at 35 rad/s."""
        if self.squeeze is None:
            return CLOSED
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(width - self.squeeze, gaps, angles)))

    def pinch_lateral(self):
        """Offset (m) of the fingertip pads' centre from the TCP along the TCP y axis (the jaw's width direction):
        the three tip spheres per finger sit 1-9 mm to one side of the TCP site (measured ~5 mm)."""
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
            pos = center - (FIXED_FACE - width / 2 - self.grasp_clearance) * rot[:, 0] - self.pinch_lateral() * rot[:, 1]
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def grasp_point(self, name):
        """(world centre, closing yaw, width, TCP z) of the pinch."""
        env = self.env
        body = env._body[name]
        R, p = env.data.xmat[body].reshape(3, 3), env.data.xpos[body]
        x = env.com_local(name)[0] + self.grasp_offset
        if self.grasp_x_limits is not None:
            x = float(np.clip(x, *self.grasp_x_limits))
        best = None
        for g, pos, half in env.boxes(name):
            if pos[0] - half[0] <= x <= pos[0] + half[0]:
                if best is None or half[1] > best[2][1]:
                    best = (g, pos, half)
        _, pos, half = best
        centre = p + R @ np.array([x, pos[1], pos[2]])
        bottom = float(env.world_corners(name)[:, 2].min())
        top = centre[2] + half[2]
        grasp_z = float(min(max(centre[2], bottom + self.min_grasp_z), top - 0.004))
        closing = np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2
        return centre, closing, float(2 * half[1]), grasp_z

    def pick_thin(self, name, attempts=2, approach=0.05):
        centre, closing, width, grasp_z = self.grasp_point(name)
        self.width = width
        for attempt in range(attempts):
            angles = [closing, closing + np.pi]
            rot, grasp, q = self.grasp_plan(np.r_[centre[:2], grasp_z], width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
            yield from self.gripper(self.close_target(width), self.close_seconds, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            self.log_tilt("grasped")
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], self.carry_z]), rot, speed=0.10, label="lift")
                self.log_tilt("lifted")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
            centre, closing, width, grasp_z = self.grasp_point(name)
        return False

    # ----- place ---------------------------------------------------------------------------------------------
    def level_rot(self, name, rot, yaw=None):
        """TCP rotation near ``rot`` at which the held object lies level (body z vertical), optionally with its
        x axis at world heading ``yaw``."""
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        if yaw is None:
            yaw = np.arctan2(m[1, 0], m[0, 0])
        return rot_z(yaw) @ r_obj_tcp.T

    def heading(self, name, rot):
        """World heading of the held object's x axis when the TCP has rotation ``rot``."""
        env = self.env
        m = rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def log_tilt(self, label):
        """Log (label, object pitch, object roll, TCP tilt) in degrees."""
        env = self.env
        R = env.data.xmat[env._body[env.obj]].reshape(3, 3)
        T = self.tcp_rot()
        mine, force = set(env._geoms[env.obj]), 0.0
        jaws = {self.model.body(b).id for b in ("gripper", "moving_jaw_so101_v1")}
        for i, c in enumerate(env.data.contact[:env.data.ncon]):
            if (c.geom1 in mine and self.model.geom_bodyid[c.geom2] in jaws) or (
                    c.geom2 in mine and self.model.geom_bodyid[c.geom1] in jaws):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, env.data, i, f)
                force += f[0]
        self.log.append((label, round(float(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))), 1),
                         round(float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))), 1),
                         round(float(np.degrees(np.arccos(min(1.0, -T[2, 2])))), 1), round(float(force), 1)))

    def achieved_rot(self, pos, rot):
        """TCP rotation the IK reaches for the commanded ``pos``/``rot`` (from the current joint configuration)."""
        q = self.q[:5].copy()
        for _ in range(3):
            q = self.ik(pos, rot, seed=q)
        self._fk(q)
        return self.ik_data.site_xmat[self.env._tcp_site_id].reshape(3, 3).copy()

    def predicted_rot(self, name, rot):
        """World rotation the held object will have when the TCP reaches rotation ``rot`` (rigid hold)."""
        env = self.env
        return rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)

    def rest_z(self, name, xy, R):
        """Body-origin height at which the object, at ``xy`` with world rotation ``R``, has its lowest collision
        corner ``release_clearance`` above the support surface under that corner (tilt-aware: a held object
        pivots a few degrees, and a 12 cm object tilted 3 deg has one end 6 mm lower than the other)."""
        env = self.env
        pts = env.local_corners(name) @ np.asarray(R).T
        need = -np.inf
        for pt in pts:
            s = env.surface_z(np.asarray(xy) + pt[:2], top=0.3, exclude=env.task_objects)
            if s is not None:
                need = max(need, s - pt[2])
        return float(need + self.release_clearance)

    def place_thin(self, name, xy, yaws=None, carry_z=None):
        env = self.env
        carry_z = self.carry_z if carry_z is None else carry_z
        xy = np.asarray(xy, float)
        cur = env.yaw(name)
        if yaws is None:
            rot = self.level_rot(name, self.carry_rot(xy))
        else:
            rots = []
            for y in yaws:
                d = np.angle(np.exp(1j * (y - cur)))
                rots.append((abs(d), self.level_rot(name, rot_z(d) @ self._cmd_rot, yaw=y)))
            rots = [r for _, r in sorted(rots, key=lambda t: t[0])]
            target0 = np.r_[xy, env.object_pos(name)[2]]
            rot = self.feasible_rotation(name, target0, rots, carry_z=carry_z)
        target = np.r_[xy, self.rest_z(name, xy, self.predicted_rot(name, rot))]
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        self.log_tilt("carried")
        rot = self.level_rot(name, rot, yaw=self.heading(name, rot))
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        yield from self.move(tcp + [0, 0, self.pre_release_height], rot, speed=0.06, tol=0.002, settle=0.6,
                             label="pre-lower")
        self.log_tilt("prelowered")
        rot = self.level_rot(name, rot, yaw=self.heading(name, rot))
        # The 5-DoF arm cannot always reach the levelling rotation; predict the object pose from the rotation the
        # IK actually reaches at the release pose, so the lowest corner (not a levelled one) clears the support.
        tcp = self._cmd_pos.copy()
        for _ in range(2):
            achieved = self.achieved_rot(tcp, rot)
            target = np.r_[xy, self.rest_z(name, xy, self.predicted_rot(name, achieved))]
            tcp = target - achieved @ self.held(name)
        yield from self.move(tcp, rot, speed=0.04, tol=self.release_tolerance, settle=0.8, label="lower")
        self.log_tilt("lowered")
        yield from self.release(self.release_for(self.width))
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def release(self, open_to=None):
        """Open quickly, then slide the fixed finger off the object's face (kit back-off).

        Measured alternatives on the marker (24 calibration seeds each, 2026-10-06): opening over 1.5 s let the
        falling object rattle between the tips (16/24 accepted, vs 20/24 at 0.2 s); touching down before opening
        (lowering in 0.5 mm steps until support contact, optionally pressing 1-2.5 mm) 11-13/24; easing the grip to
        ~zero squeeze first 2-12/24; opening while sliding the fixed finger off 8-10/24."""
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        if self.release_backoff:
            yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.03, tol=0.002,
                                 settle=0.2, label="backoff", smooth=False)

    def plan(self):
        env = self.env
        name = env.obj
        ok = yield from self.pick_thin(name)
        if not ok:
            return
        yield from self.place_thin(name, env.place_xy(), env.target_yaws())
        yield from self.rest()
        yield from self.wait(1.2)


# ----- objects --------------------------------------------------------------------------------------------------

def marker_spec(name):
    """YCB large marker, real size (12.1 x 1.9 cm), 15 g (YCB: 15.8 g); one box over its round body."""
    return ProxyScan(name, "ycb", "040_large_marker", scale=1.0, mass=0.015, segments=((0.0, 0.96, 1.0),))


def dipper_spec(name):
    """GSO mini honey dipper at 0.8x (8.9 cm: ridged head 1.6 cm, 0.65 cm handle, end knob), 12 g, most of it in
    the head."""
    return ProxyScan(name, "gso", "Cole_Hardware_Mini_Honey_Dipper", scale=0.8, mass=0.012,
                     segments=((0.0, 0.24, 0.6), (0.26, 0.89, 0.3), (0.9, 1.0, 0.1)))


def _pool(*exclude):
    return tuple(n for n in LOW_DISTRACTORS if n not in exclude)


# ----- 1. marker in the pencil box --------------------------------------------------------------------------------

class MarkerPencilBoxEnv(BoxHolderEnv):
    instruction = "Put the marker in the pencil box."
    task_objects = ("marker",)
    obj, holder = "marker", "pencil_box"
    crate = Crate("pencil_box", inner=(0.07, 0.028), wall=0.018, thickness=0.005, floor=0.005,
                  rgba=(0.15, 0.3, 0.65, 1.0), mass=0.5, material="val_plastic")
    distractor_pool = _pool()

    def object_spec(self):
        return marker_spec("marker")


class MarkerOracle(ThinOracle):
    pass


# ----- 2. honey dipper in the cutlery tray --------------------------------------------------------------------------------

class DipperTrayEnv(BoxHolderEnv):
    """Measured alternatives (2026-10-06): in an akita/white bowl the floor curves up from r = 2.5-3 cm and the
    fixed jaw, beside the off-centre head, pressed the slope at 20-30 N; in a 1.5x LIBERO ramekin ("dish") the
    gripper housing hit its 5.3 cm wall. A low-walled cutlery tray keeps the jaws over a flat floor."""
    instruction = "Put the honey dipper in the cutlery tray."
    task_objects = ("dipper",)
    obj, holder = "dipper", "cutlery_tray"
    max_azimuth_gap = 60.0
    crate = Crate("cutlery_tray", inner=(0.058, 0.03), wall=0.016, thickness=0.005, floor=0.005,
                  rgba=(0.92, 0.92, 0.88, 1.0), mass=0.5, material="val_plastic")
    distractor_pool = _pool()

    def object_spec(self):
        return dipper_spec("dipper")


class DipperOracle(ThinOracle):
    """Pinched on its head, 1.9 cm from its centre of mass: with the zero-squeeze close (11 N) the dipper pivoted
    head-down in the jaws during long carries (grip force fell to 2-5 N; 28/50 in the first qualification); a 0.5 mm
    squeeze (20 N) holds it."""
    grasp_x_limits = (-0.034, -0.033)               # centre of the 2.1 cm head (the thick end)
    squeeze = 0.0005


TASKS = [
    define_task(name="marker_in_pencil_box", instruction=MarkerPencilBoxEnv.instruction, family=FAMILY,
                env=MarkerPencilBoxEnv, oracle=MarkerOracle, objects=("marker",), object_kinds=("marker",),
                relation="inside lying along", goal="pencil box",
                steps=(step_text("lay", "marker", "inside", "pencil box"),)),
    define_task(name="honey_dipper_in_cutlery_tray", instruction=DipperTrayEnv.instruction, family=FAMILY,
                env=DipperTrayEnv, oracle=DipperOracle, objects=("dipper",), object_kinds=("honey dipper",),
                relation="inside lying along", goal="cutlery tray",
                steps=(step_text("lay", "honey dipper", "inside", "cutlery tray"),)),
]

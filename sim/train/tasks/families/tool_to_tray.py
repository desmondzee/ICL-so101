"""tool_to_tray: pick up a thin, elongated utensil and lay it flat on a support (a book, a plate).

Every task starts with the object lying on the table at a random position and heading and ends with it lying flat
on the named support: every collision corner over the support's top surface (within the book's cover, within the
plate's flat centre), resting on the support (positive upward contact), level, released and settled. The tasks
differ in object kind (marker, honey dipper) and support kind (book, plate); on the book the marker must also be
turned to lie along the cover's long side.

Shared machinery is copied from ``utensil_to_holder`` (one module per family): ``ProxyScan`` (scanned visual mesh,
box collision fitted to slices of it, because the scanned convex hulls rock forever on flat supports), the
elongated-footprint layout sampler and ``ThinOracle`` (pinch across the long axis at the thick part, tilt-aware
release height predicted from the rotation the 5-DoF arm actually reaches, fast release).

Dropped (measured 2026-10-06): ``screwdriver_on_wooden_tray`` (YCB screwdriver 0.5x, 20 g, on a low-rimmed tray):
33/50 in qualification (18/24 calibration with every fix here). It rests on its handle with the shaft in the air;
release/landing spikes of 1000-1300 rad/s^2 and occasional drops in long joint-space carries remained.
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
from sim.val.scene import Fixture, Obj

FAMILY = "tool_to_tray"
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

    def place_object(self, placed):
        """Put the task object lying on the table with its centre of mass in ``object_region``."""
        hx, hy, _ = self.local_half(self.obj)
        com = self.com_local(self.obj)[:2]
        centre, yaw = self.sample_long((hx, hy), placed, self.object_region, anchor=com)
        self.set_object_pose(self.obj, centre, yaw=yaw)
        placed.extend(_rect_cover(centre, yaw, (hx, hy)))

    # ----- success -------------------------------------------------------------------------------------------
    def lying(self, name, cos=None):
        return self.upright(name, self.lying_cos if cos is None else cos)

    def inside(self, env=None, obj=None) -> bool:
        raise NotImplementedError

    def goal_check(self, env, obj):
        return self.inside() and self.released(obj)


class BowlHolderEnv(HolderEnv):
    """Holder is a static bowl fixture: every collision corner of the object within ``inner_radius`` of the bowl
    axis and below the rim, the object resting on the bowl."""

    bowl_asset = ("akita_black_bowl", 1.1)
    inner_radius = 0.05
    rim_z = 0.05
    lying_cos = float(np.cos(np.radians(20)))       # the floor curves up toward the wall

    def scene_objects(self):
        return [self.object_spec()]

    def scene_fixtures(self):
        return [Fixture(self.holder, *self.bowl_asset)]

    def object_spec(self):
        raise NotImplementedError

    def inside(self, env=None, obj=None) -> bool:
        pts = self.world_corners(self.obj)
        centre = self.object_pos(self.holder)
        r = np.linalg.norm(pts[:, :2] - centre[:2], axis=1)
        return bool(np.all(r <= self.inner_radius) and np.all(pts[:, 2] <= centre[2] + self.rim_z))

    def layout(self):
        placed = []
        radius = self.fixture_footprint(self.holder) / np.sqrt(2)
        centre = self.sample_xy(radius, [REST_ZONE], clearance=0.03, **self.holder_region)
        self.set_fixture_pose(self.holder, centre, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((centre, radius))
        self.place_object(placed)
        self.place_distractors(placed)
        self.set_goals(Goal(self.obj, self.holder, target=None, upright_cos=self.lying_cos, check=self.goal_check))

    def goal_regions(self):
        c = self.object_pos(self.holder)
        z = c[2] + self.rim_z + 0.002
        s = 0.5 * self.inner_radius
        return (GoalRegion(self.holder, tuple((c[0] + x, c[1] + y, z) for x in (-s, 0, s) for y in (-s, 0, s))),)

    def place_xy(self):
        return self.object_pos(self.holder)[:2]

    def target_yaws(self):
        return None                                  # any heading: the oracle keeps the carried heading


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




# ----- supports specific to this family ------------------------------------------------------------------------

BOOK_FLAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])   # standing LIBERO book laid on its cover


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class BookSupportEnv(HolderEnv):
    """Support is a LIBERO book (free, heavy) lying on its cover; its origin is re-centred on its collision box
    (LIBERO book origins sit 7 cm off the geometry, after ``unstack``). The object must lie with every collision
    corner within the cover, lying flat on it."""

    book_asset = ("black_book", 1.0)
    book_mass = 0.35
    edge_margin = 0.0

    def scene_objects(self):
        return [self.object_spec(), Obj(self.holder, self.book_asset[0], self.book_asset[1], mass=self.book_mass)]

    def object_spec(self):
        raise NotImplementedError

    def _index_bodies(self):
        super()._index_bodies()
        m = self.model
        d = mujoco.MjData(m)
        a = self._qadr[self.holder]
        d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        origin = d.xpos[self._body[self.holder]]
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in self._geoms[self.holder]:
            c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
            h = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        centre = 0.5 * (lo + hi) - origin
        child = m.body(f"{self.holder}_upright").id
        m.body_pos[child] -= centre
        self._extent = self._measure_extents()

    def cover_frame(self):
        """(centre xy, yaw, half extents xy, top z) of the book's cover, from its collision corners."""
        body = self._body[self.holder]
        yaw = self.yaw_now()
        pts = []
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        for g in self._geoms[self.holder]:
            pts.append((corners * self.model.geom_aabb[g, 3:] + self.model.geom_aabb[g, :3])
                       @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g])
        pts = np.vstack(pts)
        local = (pts[:, :2] - self.data.xpos[body][:2]) @ rot_z(yaw)[:2, :2]
        half = (local.max(0) - local.min(0)) / 2
        centre = self.data.xpos[body][:2] + rot_z(yaw)[:2, :2] @ ((local.max(0) + local.min(0)) / 2)
        return centre, yaw, half, float(pts[:, 2].max())

    def inside(self, env=None, obj=None) -> bool:
        centre, yaw, half, top = self.cover_frame()
        pts = self.world_corners(self.obj)
        local = (pts[:, :2] - centre) @ rot_z(yaw)[:2, :2]
        return bool(np.all(np.abs(local) <= half - self.edge_margin) and np.all(pts[:, 2] >= top - 0.003))

    def yaw_now(self):
        """Current heading of the book's long cover axis (its yaw has changed by the body's yaw since layout)."""
        return self.book_yaw + self.book_heading() - self._book_yaw0

    def book_heading(self):
        """Heading of the book body's y axis (horizontal when the book lies on its cover; its x axis is vertical)."""
        R = self.data.xmat[self._body[self.holder]].reshape(3, 3)
        return float(np.arctan2(R[1, 1], R[0, 1]))

    def layout(self):
        placed = []
        quat0 = BOOK_FLAT
        self.set_object_pose(self.holder, (5.0, 5.0), quat=quat0)
        mujoco.mj_kinematics(self.model, self.data)
        # Long cover axis at yaw 0 of the laid-flat book: measure, then sample the book's footprint.
        self.book_yaw = 0.0
        self._book_yaw0 = self.book_heading()
        _, _, half, _ = self.cover_frame()
        if half[1] > half[0]:
            self.book_yaw = np.pi / 2
        hx, hy = max(half), min(half)
        centre, yaw = self.sample_long((hx, hy), placed, self.holder_region, clearance=0.03)
        quat = _qmul(np.array([np.cos((yaw - self.book_yaw) / 2), 0, 0, np.sin((yaw - self.book_yaw) / 2)]), quat0)
        a = self._qadr[self.holder]
        self.data.qpos[a:a + 3] = [centre[0], centre[1], self._extent_flat + 0.0005]
        self.data.qpos[a + 3:a + 7] = quat
        mujoco.mj_kinematics(self.model, self.data)
        self.book_yaw = yaw
        self._book_yaw0 = self.book_heading()
        placed.extend(_rect_cover(centre, yaw, (hx, hy)))
        self.place_object(placed)
        self.place_distractors(placed)
        _, _, _, top = self.cover_frame()
        bottom = self.local_half(self.obj)[2]
        self.set_goals(Goal(self.obj, self.holder, target=(*centre, top + bottom + 0.0005), reference=self.holder,
                            tolerance=(0.035, 0.035, 0.006), upright_cos=self.lying_cos, check=self.goal_check))

    @property
    def _extent_flat(self):
        """Origin height above the table of the book lying on its cover (half its thickness: origin re-centred)."""
        m, d = self.model, mujoco.MjData(self.model)
        a = self._qadr[self.holder]
        d.qpos[a:a + 7] = [0, 0, 1.0, *BOOK_FLAT]
        mujoco.mj_kinematics(m, d)
        low = min(float((d.geom_xpos[g] - np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
                         + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3])[2]) for g in self._geoms[self.holder])
        return 1.0 - low

    def place_xy(self):
        return self.goal_target(self.goals[0])[:2]

    def target_yaws(self):
        yaw = self.yaw_now()
        return (yaw, yaw + np.pi)


class PlateSupportEnv(BowlHolderEnv):
    """Support is a static LIBERO plate; the object must lie within ``inner_radius`` (its flat centre)."""

    bowl_asset = ("plate", 1.2)
    inner_radius = 0.055
    rim_z = 0.03
    lying_cos = COS10

    def goal_regions(self):
        """Samples on the plate's flat centre (at its surface height)."""
        c = self.object_pos(self.holder)
        s = 0.5 * self.inner_radius
        pts = []
        for x in (-s, 0, s):
            for y in (-s, 0, s):
                xy = c[:2] + [x, y]
                pts.append((*xy, (self.surface_z(xy, exclude=self.task_objects) or c[2]) + 0.001))
        return (GoalRegion(self.holder, tuple(pts)),)


# ----- objects --------------------------------------------------------------------------------------------------

def marker_spec(name):
    """YCB large marker, real size (12.1 x 1.9 cm), 15 g (YCB: 15.8 g); one box over its round body."""
    return ProxyScan(name, "ycb", "040_large_marker", scale=1.0, mass=0.015, segments=((0.0, 0.96, 1.0),))


def dipper_spec(name):
    """GSO mini honey dipper at 0.8x (8.9 cm: ridged head 1.6 cm, 0.5 cm handle, end knob), 12 g, most of it in
    the head."""
    return ProxyScan(name, "gso", "Cole_Hardware_Mini_Honey_Dipper", scale=0.8, mass=0.012,
                     segments=((0.0, 0.24, 0.6), (0.26, 0.89, 0.3), (0.9, 1.0, 0.1)))


def _pool(*exclude):
    return tuple(n for n in LOW_DISTRACTORS if n not in exclude)


# ----- 1. marker on the book --------------------------------------------------------------------------------------

class MarkerBookEnv(BookSupportEnv):
    instruction = "Put the marker on the book."
    task_objects = ("marker",)
    obj, holder = "marker", "book"
    distractor_pool = _pool()

    def object_spec(self):
        return marker_spec("marker")


class MarkerOracle(ThinOracle):
    pass


# ----- 2. honey dipper on the plate -------------------------------------------------------------------------------

class DipperPlateEnv(PlateSupportEnv):
    instruction = "Put the honey dipper on the plate."
    task_objects = ("dipper",)
    obj, holder = "dipper", "plate"
    distractor_pool = _pool("plate")

    def object_spec(self):
        return dipper_spec("dipper")


class DipperOracle(ThinOracle):
    """Pinched on its head, 1.9 cm from its centre of mass; a 0.5 mm squeeze (20 N, vs 11 N at zero squeeze) keeps
    it from pivoting head-down in the jaws during long carries."""
    grasp_x_limits = (-0.034, -0.033)               # centre of the 2.1 cm head (the thick end)
    squeeze = 0.0005


TASKS = [
    define_task(name="marker_on_book", instruction=MarkerBookEnv.instruction, family=FAMILY,
                env=MarkerBookEnv, oracle=MarkerOracle, objects=("marker",), object_kinds=("marker",),
                relation="lying flat on along", goal="book",
                steps=(step_text("lay", "marker", "flat on", "book"),)),
    define_task(name="honey_dipper_on_plate", instruction=DipperPlateEnv.instruction, family=FAMILY,
                env=DipperPlateEnv, oracle=DipperOracle, objects=("dipper",), object_kinds=("honey dipper",),
                relation="lying flat on", goal="plate",
                steps=(step_text("lay", "honey dipper", "flat on", "plate"),)),
]

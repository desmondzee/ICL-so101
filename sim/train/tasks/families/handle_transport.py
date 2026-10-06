"""handle_transport: carry an object by its handle and set it down, upright, on a target.

The robot grasps the *handle* (never the body): a top-down grasp whose jaws straddle the handle loop and close
across its thickness. The object hangs off the jaws with its centre of mass several centimetres from the grip,
so it pivots a few degrees about the closing axis. The oracle turns the closing axis tangential to the arm over
the target (the only axis the 5-DOF wrist can tilt about), levels the object by the measured in-jaw tilt, sets it
down upright and releases slowly.

Asset changes (kept here; proposed for the kit): the LIBERO handle collision (3-4 overlapping rotated boxes) is
replaced by one solid box of the same extent (``SolidHandle``), the object origin is recentred on its body, and the
LIBERO book is laid flat and recentred (``FlatLibero``). The pot and mug are 22/25 g (LIBERO density gives 43/48 g;
heavier, they swung out of the 8 mm finger pads at the end of carries).

Success is strict: the object is upright (10 deg), released, settled, carried by the target alone (no table
contact), its whole base footprint lies over the target, and its origin is at the target within tolerance.

Tasks (different object and target kind each):

* ``moka_pot_on_coaster``  -- the LIBERO moka pot, by its handle, onto a cork coaster.
* ``mug_by_handle_on_book`` -- the LIBERO white-yellow mug, by its handle, onto a book lying flat.

Held-out semantics avoided: no mug on a plate and no pan on a stove.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, Goal, TrainEnv, TrainOracle, define_task, mat
from sim.val.oracle import PAN_AXIS, RELEASE
from sim.val.scene import (ASSETS, CATALOG, _add_assets, _collision_geoms, _content, _merge_defaults, _set,
                           _set_free_physics, _soften, _upright_quat, _vec, load_mjcf)

FAMILY = "handle_transport"
REST_SWEEP = [(np.array([0.145, 0.06]), 0.025), (np.array([0.145, -0.06]), 0.025)]
# Distractors no taller than 2.6 cm: the object hangs 5 cm below the TCP while carried.
SHORT_DISTRACTORS = ("cream_cheese", "butter", "chocolate_pudding", "white_bowl", "red_bowl", "plate", "ramekin")


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


# ----- assets ------------------------------------------------------------------------------------------------

@dataclass
class FlatLibero:
    """A free LIBERO catalog object with a fixed rotation baked under its free joint (``quat``), e.g. a book lying
    flat. The scene's extents (``bottom``/``radius``) are measured in that orientation."""

    name: str
    asset: str
    scale: float = 0.5
    quat: tuple = (0.70710678, 0.0, 0.70710678, 0.0)
    mass: float = 0.06
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        root = load_mjcf(ASSETS / CATALOG[self.asset].path, self.scale, self.name)
        _soften(root.find("asset"))
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        # Recentre: the origin goes to the centre of the rotated collision boxes (the LIBERO book origin sits at
        # one edge, which made its footprint radius 9 cm).
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(self.quat, float))
        pts = np.vstack([_box_corners(g) for g in _collision_geoms(root) if g.get("type") == "box"])
        pts = pts @ R.reshape(3, 3).T
        centre = (pts.min(0) + pts.max(0)) / 2
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright",
                                quat=" ".join(f"{v:.8g}" for v in self.quat),
                                pos=" ".join(f"{-v:.6g}" for v in centre))
        upright.extend(_content(root))


def _box_corners(g):
    """Corners of an MJCF box geom element in its parent frame."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, _vec(g, "quat", np.array([1.0, 0, 0, 0])))
    signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return (signs * _vec(g, "size")) @ R.reshape(3, 3).T + _vec(g, "pos", np.zeros(3))


@dataclass
class SolidHandle:
    """A free LIBERO catalog object whose handle collision (the LIBERO boxes lying beyond ``reach`` along the
    horizontal ``axis`` of the object frame) is replaced by one solid box over their extent. The LIBERO handle
    is three or four overlapping, rotated boxes; the jaws closed onto their edges and corners (3-4 mm
    penetration with any grip force). One flat-faced box of the same extent gives the jaw pads a flat face. The
    visual mesh is unchanged; ``mass`` is spread over the remaining collision boxes."""

    name: str
    asset: str
    scale: float = 0.5
    axis: tuple = (0.0, 1.0)
    reach: float = 0.024
    mass: float = 0.03
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        a = CATALOG[self.asset]
        root = load_mjcf(ASSETS / a.path, self.scale, self.name)
        _soften(root.find("asset"))
        axis = np.r_[self.axis, 0.0]
        boxes = [g for g in _collision_geoms(root) if g.get("type") == "box"]
        handle = [g for g in boxes if _vec(g, "pos", np.zeros(3)) @ axis >= self.reach]
        pts = np.vstack([_box_corners(g) for g in handle])
        lo, hi = pts.min(0), pts.max(0)
        rest = np.vstack([_box_corners(g) for g in boxes if not any(g is h for h in handle)])
        centre = (rest.min(0) + rest.max(0)) / 2
        parent = next(b for b in root.iter("body") if any(c is handle[0] for c in b))
        template = dict(handle[0].attrib)
        for g in handle:
            parent.remove(g)
        attrs = {k: v for k, v in template.items() if k in ("solimp", "solref", "rgba")}
        ET.SubElement(parent, "geom", name=f"{self.name}_handle", type="box", group="0",
                      pos=" ".join(f"{v:.6g}" for v in (lo + hi) / 2),
                      size=" ".join(f"{v:.6g}" for v in (hi - lo) / 2), **attrs)
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        # Origin recentred on the body (without the handle) in xy, so the footprint centre does not move when the
        # object turns during the carry (the LIBERO mug origin sits 1.1 cm off its body axis).
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright",
                                pos=f"{-centre[0]:.6g} {-centre[1]:.6g} 0")
        _set(upright, "quat", _upright_quat(a.upright))
        upright.extend(_content(root))


# ----- geometry helpers --------------------------------------------------------------------------------------

def world_points(env, name):
    """Exact collision points of a free body in world coordinates (box corners, mesh vertices)."""
    m, d = env.model, env.data
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    pts = []
    for g in env._geoms[name]:
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            local = corners * m.geom_size[g]
        else:
            local = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
        pts.append(local @ R.T + p)
    return np.vstack(pts)


def local_points(env, name, geoms=None):
    """Collision points of ``name`` (optionally only ``geoms``) in the body frame."""
    m, d = env.model, env.data
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    body = env._body[name]
    R, p = d.xmat[body].reshape(3, 3), d.xpos[body]
    pts = []
    for g in (env._geoms[name] if geoms is None else geoms):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        local = corners * m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else \
            corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
        world = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append((world - p) @ R)
    return np.vstack(pts)


def only_on(env, name, support):
    """``name`` touches nothing but ``support`` (and nothing at all of the robot)."""
    own, sup = set(env._geoms[name]), set(env._geoms[support])
    for c in env.data.contact[:env.data.ncon]:
        if (c.geom1 in own and c.geom2 not in sup) or (c.geom2 in own and c.geom1 not in sup):
            return False
    return True


# ----- environment -------------------------------------------------------------------------------------------

class HandleEnv(TrainEnv):
    """Carry ``obj`` by its handle onto the free support ``target``.

    handle_axis: body-local horizontal unit vector pointing from the body out to the handle.
    grasp_along: grasp point along the handle box, as a fraction from its inner to its outer end.
    grasp_height: TCP height of the handle grasp above the object's lowest point.
    margin: every base point must lie this far inside the target's top face.
    """

    obj = "obj"
    target = "target"
    task_objects = ("obj",)
    handle_axis = np.array([0.0, 1.0])
    grasp_height = 0.05
    grasp_along = 0.5
    margin = 0.004
    tolerance = (0.014, 0.014, 0.006)
    target_region = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
    obj_region = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
    min_travel = 0.10
    distractor_pool = SHORT_DISTRACTORS

    # ----- geometry --------------------------------------------------------------------------------------------
    def handle_geoms(self):
        return [self.model.geom(f"{self.obj}_handle").id]

    def base_height(self):
        """Height of the body origin above the object's lowest collision point (upright)."""
        return float(-local_points(self, self.obj)[:, 2].min())

    def base_points(self):
        """World xy of the collision points within 3 mm of the object's bottom."""
        pts = world_points(self, self.obj)
        return pts[pts[:, 2] <= pts[:, 2].min() + 0.003, :2]

    def target_top(self):
        return float(world_points(self, self.target)[:, 2].max())

    def footprint_inside(self):
        """Every base point inside the target's top face, ``margin`` from its edges (target frame)."""
        body = self._body[self.target]
        R = self.data.xmat[body].reshape(3, 3)
        top = world_points(self, self.target)
        top = top[top[:, 2] >= top[:, 2].max() - 0.002]
        local_top = (top - self.data.xpos[body]) @ R
        lo, hi = local_top[:, :2].min(0), local_top[:, :2].max(0)
        base = np.c_[self.base_points(), np.full(len(self.base_points()), self.data.xpos[body][2])]
        local = (base - self.data.xpos[body]) @ R
        return bool(np.all(local[:, :2] >= lo + self.margin) and np.all(local[:, :2] <= hi - self.margin))

    def debug_state(self):
        g = self.goals[0]
        err = self.object_pos(self.obj) - self.goal_target(g)
        pen = min([c.dist for c in self.data.contact[:self.data.ncon]] or [0])
        return dict(err_mm=np.round(err * 1000, 1).tolist(), inside=self.footprint_inside(),
                    only_on=only_on(self, self.obj, self.target), upright=self.upright(self.obj),
                    settled=self.settled(self.obj), supported=self.supported_by(self.obj, self.target),
                    released=self.released(self.obj), min_dist_mm=round(pen * 1000, 2))

    def goal_check(self, env, name):
        return self.footprint_inside() and only_on(self, name, self.target)

    # ----- layout ----------------------------------------------------------------------------------------------
    def place_target(self, placed):
        return self.place(self.target, placed, region=self.target_region)

    def layout(self):
        placed = list(REST_SWEEP)
        target_xy, _ = self.place_target(placed)
        for _ in range(50):
            trial = list(placed)
            xy, _ = self.place(self.obj, trial, region=self.obj_region)
            if np.linalg.norm(xy - target_xy) >= self.min_travel:
                placed = trial
                break
        else:
            raise RuntimeError("object start too close to the target")
        self.place_distractors(placed)
        centre, top_centre = self.footprint_center_offset(), self.target_top_center()
        target = (top_centre[0] - centre[0], top_centre[1] - centre[1], self.target_top() + self.base_height())
        self.set_goals(Goal(self.obj, self.target, target=target, reference=self.target, tolerance=self.tolerance,
                            check=self.goal_check))

    def target_top_center(self):
        """xy centre of the target's top face (LIBERO origins are not always centred)."""
        body = self._body[self.target]
        R, p = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        top = world_points(self, self.target)
        top = top[top[:, 2] >= top[:, 2].max() - 0.002]
        local = (top - p) @ R
        return (p + R @ np.r_[(local.min(0) + local.max(0))[:2] / 2, 0.0])[:2]

    def footprint_center_offset(self):
        """xy of the base-footprint centre relative to the origin (world axes, current yaw)."""
        pts = self.base_points()
        return (pts.min(0) + pts.max(0)) / 2 - self.object_pos(self.obj)[:2]


class MokaCoasterEnv(HandleEnv):
    instruction = "Pick up the moka pot by its handle and put it on the coaster."
    obj, target = "moka", "coaster"
    task_objects = ("moka",)
    handle_axis = np.array([0.0, 1.0])
    grasp_height = 0.050

    def scene_objects(self):
        return [SolidHandle("moka", "moka_pot", axis=(0.0, 1.0), reach=0.024, mass=0.022, friction=1.5), mat("coaster", half=(0.036, 0.036), thickness=0.006,
                                             rgba=(0.62, 0.45, 0.28, 1.0), mass=0.04)]


class MugBookEnv(HandleEnv):
    instruction = "Pick up the mug by its handle and put it on the book."
    obj, target = "mug", "book"
    task_objects = ("mug",)
    handle_axis = np.array([1.0, 0.0])
    grasp_height = 0.026
    # Along the handle: nearer the body, the fixed jaw's hull (3.6 mm behind its pad, 3.2 cm wide) brushed the
    # mug body on the way down; farther out, the mug crept off the handle's end during long carries.
    grasp_along = 0.72
    target_region = dict(r=(0.18, 0.25), angle=(-55.0, 55.0))

    def scene_objects(self):
        # The black book (0.6x) lying flat: 8.1 x 6.7 x 1.9 cm.
        return [SolidHandle("mug", "white_yellow_mug", axis=(1.0, 0.0), reach=0.017, mass=0.025, friction=1.5),
                FlatLibero("book", "black_book", scale=0.6, mass=0.06)]


# ----- oracle ------------------------------------------------------------------------------------------------

class HandleOracle(TrainOracle):
    carry_z = 0.10
    close_seconds = 2.0         # light objects (22-25 g) spun just over 1000 rad/s^2 under a 1.5 s squeeze
    touch_seconds = 1.3
    release_seconds = 1.5
    open_margin = 0.016
    release_margin = 0.014
    grasp_clearance = 0.002
    touch_depth = 0.001
    backoff_speed = 0.008
    vmax = np.radians([45.0, 50.0, 50.0, 60.0, 60.0])

    grip_squeeze = 0.002   # close just onto the handle: closing further pressed the jaws 3-4 mm into it

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def handle_grasp(self):
        """(world grasp centre at TCP height, closing yaw, handle width along the closing direction)."""
        env = self.env
        name = env.obj
        geoms = env.handle_geoms()
        pts = local_points(env, name, geoms)
        bottom = env.base_height()
        z_local = env.grasp_height - bottom          # TCP height in the body frame
        band = pts
        axis = np.r_[env.handle_axis, 0.0]
        across = np.array([-axis[1], axis[0], 0.0])
        a0, a1 = band @ across, band @ axis
        lo, hi = a0.min(), a0.max()
        # Along the handle: ``grasp_along`` of the way out from its inner to its outer end. Carried objects creep
        # outward along the handle (centripetal load on arcs); a grasp 6 mm from the outer end slid off it.
        along = a1.min() + self.env.grasp_along * (a1.max() - a1.min())
        centre_local = across * (lo + hi) / 2 + axis * along + np.array([0, 0, z_local])
        R = env.data.xmat[env._body[name]].reshape(3, 3)
        centre = env.object_pos(name) + R @ centre_local
        world_across = R @ across
        yaw = float(np.arctan2(world_across[1], world_across[0]))
        return centre, yaw, float(hi - lo)

    def pick_handle(self, attempts=2, approach=0.05):
        env = self.env
        name = env.obj
        for attempt in range(attempts):
            centre, yaw, width = self.handle_grasp()
            self._held_width = width
            rot, grasp, q = self.grasp_plan(centre, width, [yaw, yaw + np.pi])
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.05, tol=0.003, label="grasp")
            yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.2)
            yield from self.gripper(self.gap_angle(width - self.grip_squeeze), self.close_seconds, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok, round(width, 4)))
            if ok:
                yield from self.move(np.r_[grasp[:2], grasp[2] + 0.02], rot, speed=0.03, label="lift1")
                yield from self.move(np.r_[grasp[:2], self.carry_z], rot, speed=0.08, label="lift")
                if env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    carry_speed = 0.08

    def carry_to(self, pos, rot, label="carry"):
        """Slow carries: the object hangs off its handle, and the end of a kit-speed carry swung it out of the
        jaws."""
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(_wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, speed=self.carry_speed, label=label)
        else:
            yield from self.transit(pos, rot, speed=0.4, label=label)

    def level(self, name, rot):
        """``rot`` turned so the held object ends upright (it pivots about the closing axis in the jaws)."""
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

    def release(self, open_to=None):
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.3)
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=self.backoff_speed,
                             tol=0.002, settle=0.3, label="backoff", smooth=False)

    def end_rotation(self, name, target):
        """End TCP rotation over ``target``: the closing axis turned tangential (perpendicular to the radial
        direction from the pan axis). The held object pivots about the closing axis, and the 5-DOF arm can only
        tilt the TCP about a tangential axis (wrist flex), so this is the only orientation in which ``level`` can
        set the object down upright. Of the two tangential senses, the one the IK reaches best, then the one
        closest to the plain azimuth-following carry (least wrist roll)."""
        follow = self.carry_rot(target[:2])
        closing = np.arctan2(follow[1, 0], follow[0, 0])
        radial = np.arctan2(target[1] - PAN_AXIS[1], target[0] - PAN_AXIS[0])
        held, best = self.held(name), None
        for option in (radial + np.pi / 2, radial - np.pi / 2):
            delta = _wrap(option - closing)
            rot = _rz(delta) @ follow
            tcp = target - rot @ held
            err = max(self.solve(np.r_[tcp[:2], self.carry_z], rot)[1], self.solve(tcp, rot)[1])
            score = (round(err, 3), abs(delta))
            if best is None or score < best[0]:
                best = (score, rot)
        return best[1]

    def place_held(self, name, target):
        target = np.asarray(target, float)
        rot = self.end_rotation(name, target)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], self.carry_z], rot)
        yield from self.wait(0.3)
        if not self.env.is_grasping(name):     # dropped in the carry: stop (the episode fails cleanly)
            self.log.append((name, "dropped in carry"))
            return
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], self.carry_z], rot, speed=0.04, label="align", smooth=False)
        yield from self.wait(0.3)
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], tcp[2] + 0.012], rot, speed=0.05, label="descend")
        tcp = target - rot @ self.held(name)
        lift = 0.5 * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + lift + 0.0005], rot, speed=0.02, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(self.open_for(self._held_width, self.release_margin))
        yield from self.move(np.r_[self._cmd_pos[:2], self._cmd_pos[2] + 0.03], rot, speed=0.05, label="retreat1")
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], rot, speed=0.10, label="retreat")

    def plan(self):
        env = self.env
        ok = yield from self.pick_handle()
        if not ok:
            return
        yield from self.place_held(env.obj, env.goal_target(env.goals[0]))
        yield from self.rest()
        yield from self.wait(1.2)


class MugOracle(HandleOracle):
    carry_speed = 0.06          # the mug creeps along its short handle on faster carries


def _task(name, env, kind, goal, obj_text, target_text, oracle=HandleOracle, **kwargs):
    return define_task(name=name, instruction=env.instruction, family=FAMILY, env=env, oracle=oracle,
                       objects=env.task_objects, object_kinds=(kind,), relation="carried by the handle onto",
                       goal=goal, steps=(f"Pick up the {obj_text} by its handle and set it down upright on the "
                                         f"{target_text}.",), **kwargs)


TASKS = [
    _task("moka_pot_on_coaster", MokaCoasterEnv, "moka pot", "coaster", "moka pot", "coaster"),
    _task("mug_by_handle_on_book", MugBookEnv, "mug", "book", "mug", "book", oracle=MugOracle),
]

"""place_on_shelf: put an object onto a raised level of a shelf.

Three tasks that differ in the object, the shelf and the level:

* ``sponge_on_top_of_shelf``   -- a kitchen sponge onto the top board of the LIBERO two-layer wooden shelf
  (0.3x: 9.9 x 5.7 cm, top board 6.3 cm above the table).
* ``cube_on_top_step``         -- a cube onto the top level of a two-step wooden shelf (riser shelf).
* ``block_on_lower_step``      -- a long block onto the lower level of the same kind of step shelf.

Why a step shelf for "top"/"lower" levels (measured 2026-10-06): the SO-101 grasps top-down only, and the wrist
motor starts 10 cm above the TCP, the camera mount 4.5-7.2 cm above it (sticking out 3.5-8.5 cm sideways). The
LIBERO two-layer shelf's lower level is under its top board (6.8 cm clear at 0.5x, 13.8 cm at 1.0x where the
shelf is 32 x 19 cm and 21 cm tall), so it can only be reached from the side, which a top-down gripper cannot do.
The step shelf has its two levels side by side (lower level toward the viewer, upper level toward the robot), so
both are reached top-down; the oracle closes the jaws parallel to the riser and turns the camera mount away from
it (over the lower level), so placing on the lower level never brings the mount near the upper level.

Dropped: the LIBERO soup can on top of the LIBERO shelf (6/12 calibration seeds at best). The placed can's top is
10.1 cm up, above the highest TCP the arm reaches over the shelf (~10.8 cm commanded 11.8), so the open jaws still
straddle it after the retreat and the sweep toward the rest pose knocked it off; sliding out sideways first then
folded the camera mount into the upper arm at that high pose (2/12), and its edge pressed 3 mm into the 1.3 mm
board. The 2.2 cm sponge leaves the jaws 7 mm clear at a 10 cm retreat.

Shelves are free bodies made heavy (1.2-1.5 kg): the strict gate then fails any episode in which the arm or the
object knocks a shelf (<= 5 mm / 0.1 rad). The step shelf's two levels are solid boxes (box/box contacts settle).

Success (kit goal plus ``on_level``): the object is released, settled, upright, supported by the shelf, at the
level's height (z tolerance 6 mm, so resting on the other level or the table fails), with its centre inside the
level's top face inset by ``LEVEL_MARGIN``, and it does not touch the table.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task, step_text, top_down_mat
from sim.val.scene import (ASSETS, Block, _add_assets, _content, _merge_defaults, _set_free_physics, _soften,
                           load_mjcf)

FAMILY = "place_on_shelf"
WOOD = (0.55, 0.38, 0.22, 1.0)
WOOD_DARK = (0.42, 0.28, 0.16, 1.0)
LEVEL_MARGIN = 0.006          # object centre this far inside the level's top face
REST_ZONE = (np.array([0.16, 0.0]), 0.075)   # folded gripper hover / rest sweep: no shelf part inside
HALF = 0.014
BAR = (0.022, 0.010, 0.013)   # 4.4 x 2.0 x 2.6 cm


# ----- assets ------------------------------------------------------------------------------------------------

@dataclass
class StepShelf:
    """A two-step wooden shelf (one heavy free body). Local +x is the front: the lower level spans x in [0, depth],
    the upper level x in [-depth, 0]; both are solid boxes. Thin side panels and a back board are visual only."""

    name: str
    width: float = 0.11
    depth: float = 0.055
    low: float = 0.025
    high: float = 0.055
    mass: float = 1.5

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        w, d = self.width / 2, self.depth
        wood, dark = " ".join(map(str, WOOD)), " ".join(map(str, WOOD_DARK))
        vol_low, vol_high = d * self.low, d * self.high
        for part, cx, h, vol in (("lower", d / 2, self.low, vol_low), ("upper", -d / 2, self.high, vol_high)):
            ET.SubElement(body, "geom", name=f"{self.name}_{part}", type="box", size=f"{d / 2:.6g} {w:.6g} {h / 2:.6g}",
                          pos=f"{cx:.6g} 0 {h / 2:.6g}", rgba=wood,
                          mass=f"{self.mass * vol / (vol_low + vol_high):.6g}", group="1", condim="4",
                          friction="1.0 0.02 0.001", material="val_fabric")
        vis = dict(contype="0", conaffinity="0", group="1", mass="0", rgba=dark, material="val_fabric")
        # Edge trims on the level fronts and side panels: reads as a shelf, not a pair of blocks.
        for part, cx, h in (("lower", d, self.low), ("upper", 0.0, self.high)):
            ET.SubElement(body, "geom", name=f"{self.name}_{part}_trim", type="box", size=f"0.0015 {w + 0.002:.6g} 0.003",
                          pos=f"{cx + 0.0005:.6g} 0 {h - 0.003:.6g}", **vis)
        for side in (-1, 1):
            ET.SubElement(body, "geom", name=f"{self.name}_side{side + 1}", type="box",
                          size=f"{d:.6g} 0.002 {self.low / 2:.6g}", pos=f"0 {side * (w + 0.002):.6g} {self.low / 2:.6g}",
                          **vis)
            ET.SubElement(body, "geom", name=f"{self.name}_sideu{side + 1}", type="box",
                          size=f"{d / 2:.6g} 0.002 {(self.high - self.low) / 2:.6g}",
                          pos=f"{-d / 2:.6g} {side * (w + 0.002):.6g} {(self.high + self.low) / 2:.6g}", **vis)


@dataclass
class LiberoFree:
    """A free LIBERO object loaded from ``ASSETS / path`` (same treatment as ``sim.val.scene`` free objects)."""

    name: str
    path: str
    scale: float = 0.5
    mass: float = 0.2
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        root = load_mjcf(ASSETS / self.path, self.scale, self.name)
        _soften(root.find("asset"))
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat="1 0 0 0")
        upright.extend(_content(root))


SPONGE_HALF = (0.023, 0.015)          # 4.6 x 3.0 cm kitchen sponge (as in place_on_surface)
FOAM_H, PAD_H = 0.017, 0.005


@dataclass
class Sponge:
    """A kitchen sponge: yellow foam box with a green scouring pad (one free body, two box geoms)."""

    name: str
    mass: float = 0.03
    foam: tuple = (0.95, 0.80, 0.25, 1.0)
    pad: tuple = (0.18, 0.50, 0.22, 1.0)
    friction: float = 0.9

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        hx, hy = SPONGE_HALF
        total = FOAM_H + PAD_H
        for part, h, z, rgba in (("foam", FOAM_H, -total / 2 + FOAM_H / 2, self.foam),
                                 ("pad", PAD_H, total / 2 - PAD_H / 2, self.pad)):
            ET.SubElement(body, "geom", name=f"{self.name}_{part}", type="box", size=f"{hx} {hy} {h / 2}",
                          pos=f"0 0 {z:.6g}", rgba=" ".join(map(str, rgba)), mass=repr(self.mass * h / total),
                          group="1", condim="4", friction=f"{self.friction} 0.02 0.001", material="val_fabric")


TWO_LAYER = "turbosquid_objects/wooden_two_layer_shelf/wooden_two_layer_shelf.xml"


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----- environment -------------------------------------------------------------------------------------------

class ShelfEnv(TrainEnv):
    """Object ``obj`` starts on the table; it must end on level ``level`` of free body ``shelf``."""

    task_objects = ("obj",)
    shelf_r = (0.19, 0.215)
    shelf_angle = (32.0, 58.0)        # |azimuth| of the shelf centre (deg), side random
    obj_region = dict(r=(0.16, 0.25), angle=(5.0, 60.0))   # mirrored to the side opposite the shelf
    solid = ()                        # free bodies whose collision becomes one solid box (LIBERO cans)

    # ----- model tweaks ---------------------------------------------------------------------------------------
    def _index_bodies(self):
        super()._index_bodies()
        for name in self.solid:
            self._solidify(name)
        self._extent = self._measure_extents()

    def _body_points(self, name):
        m, d = self.model, mujoco.MjData(self.model)
        a = self._qadr[name]
        d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        origin = d.xpos[self._body[name]]
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        pts = []
        for g in self._geoms[name]:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                v = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(pts) - origin, d

    def _solidify(self, name):
        """One solid box over the collision AABB (copied from unstack): LIBERO cans collide as a ring of 1-2 mm
        plates that the fingertip spheres press into and hook on release."""
        m = self.model
        geoms = self._geoms[name]
        pts, d = self._body_points(name)
        lo, hi = pts.min(0), pts.max(0)
        keep = max(geoms, key=lambda g: float(np.prod(m.geom_size[g])))
        parent = int(m.geom_bodyid[keep])
        R = d.xmat[parent].reshape(3, 3)
        centre = d.xpos[self._body[name]] + 0.5 * (lo + hi)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.T.flatten())
        m.geom_type[keep] = mujoco.mjtGeom.mjGEOM_BOX
        m.geom_size[keep] = 0.5 * (hi - lo)
        m.geom_pos[keep] = R.T @ (centre - d.xpos[parent])
        m.geom_quat[keep] = quat
        m.geom_aabb[keep] = np.r_[np.zeros(3), 0.5 * (hi - lo)]
        m.geom_rbound[keep] = float(np.linalg.norm(0.5 * (hi - lo)))
        for g in geoms:
            if g != keep:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self._geoms[name] = [keep]

    # ----- geometry ---------------------------------------------------------------------------------------------
    def aabb(self, name):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in self._geoms[name]:
            c = self.data.geom_xpos[g] + self.data.geom_xmat[g].reshape(3, 3) @ self.model.geom_aabb[g, :3]
            h = np.abs(self.data.geom_xmat[g].reshape(3, 3)) @ self.model.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        return lo, hi

    def centre(self, name):
        lo, hi = self.aabb(name)
        return (lo + hi) / 2

    def touching_table(self, name):
        mine, table = set(self._geoms[name]), self._owned_geoms("table")
        return any((c.geom1 in mine and c.geom2 in table) or (c.geom2 in mine and c.geom1 in table)
                   for c in self.data.contact[:self.data.ncon])

    def level_face(self):
        """(centre xy in the shelf frame, half extents, top z in the shelf frame) of the goal level."""
        raise NotImplementedError

    def shelf_corners(self, xy, yaw):
        """World xy of the shelf footprint corners (plus edge midpoints) for a pose."""
        raise NotImplementedError

    def on_level(self, env, name):
        b = self._body["shelf"]
        R = self.data.xmat[b].reshape(3, 3)
        local = R.T @ (self.centre(name) - self.data.xpos[b])
        c, half, _ = self.level_face()
        return bool(np.all(np.abs(local[:2] - c) <= np.asarray(half) - LEVEL_MARGIN) and not self.touching_table(name))

    # ----- layout -----------------------------------------------------------------------------------------------
    def shelf_ok(self, xy, yaw):
        pts = self.shelf_corners(xy, yaw)
        ctr, rad = REST_ZONE
        if np.min(np.linalg.norm(pts - ctr, axis=1)) < rad or np.min(np.linalg.norm(pts, axis=1)) < 0.11:
            return False
        # Whole shelf in the reachable fan, and edges clear of the rest zone (sample along the outline).
        outline = np.vstack([pts[i] + t * (pts[(i + 1) % 4] - pts[i]) for i in range(4) for t in np.linspace(0, 1, 9)])
        return bool(np.min(np.linalg.norm(outline - ctr, axis=1)) >= rad
                    and np.all(np.abs(np.degrees(np.arctan2(outline[:, 1], outline[:, 0]))) <= 78))

    def shelf_yaw(self, xy):
        return float(np.arctan2(xy[1], xy[0]) + self.np_random.uniform(-0.3, 0.3))

    def layout(self):
        rng = self.np_random
        side = float(rng.choice((-1.0, 1.0)))
        for _ in range(200):
            r = rng.uniform(*self.shelf_r)
            a = np.radians(side * rng.uniform(*self.shelf_angle))
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            yaw = self.shelf_yaw(xy)
            if self.shelf_ok(xy, yaw):
                break
        else:
            raise RuntimeError("no shelf pose")
        self.set_object_pose("shelf", xy, yaw=yaw)
        radius = float(np.max(np.linalg.norm(self.shelf_corners(xy, yaw) - xy, axis=1)))
        placed = [(xy, radius)]
        lo, hi = self.obj_region["angle"]
        region = dict(r=self.obj_region["r"], angle=tuple(sorted((-side * lo, -side * hi))))
        self.place("obj", placed, region=region, clearance=0.03)
        self.place_distractors(placed)
        c, _, top = self.level_face()
        R = rot_z(yaw)[:2, :2]
        target_xy = xy + R @ c
        z = self.object_pos("shelf")[2] + top + self._extent["obj"]["bottom"]
        self.set_goals(Goal("obj", "shelf", target=(*target_xy, z), reference="shelf",
                            tolerance=(0.03, 0.03, 0.006), check=lambda env, n: env.on_level(env, n)))


class StepShelfEnv(ShelfEnv):
    level = "upper"

    def shelf_spec(self):
        return StepShelf("shelf")

    def level_face(self):
        s = StepShelf("shelf")
        if self.level == "upper":
            return np.array([-s.depth / 2, 0.0]), (s.depth / 2, s.width / 2), s.high
        return np.array([s.depth / 2, 0.0]), (s.depth / 2, s.width / 2), s.low

    def shelf_corners(self, xy, yaw):
        s = StepShelf("shelf")
        R = rot_z(yaw)[:2, :2]
        local = np.array([(x, y) for x in (-s.depth, s.depth) for y in (-s.width / 2 - 0.004, s.width / 2 + 0.004)])
        local = local[[0, 1, 3, 2]]
        return xy + local @ R.T


class CubeTopStepEnv(StepShelfEnv):
    instruction = "Put the red cube on the top shelf."
    level = "upper"
    distractor_pool = LOW_DISTRACTORS

    def scene_objects(self):
        return [Block("obj", half=(HALF,) * 3, rgba=(0.8, 0.1, 0.08, 1.0)), StepShelf("shelf")]


class BlockLowerStepEnv(StepShelfEnv):
    instruction = "Put the blue block on the lower shelf."
    level = "lower"
    distractor_pool = LOW_DISTRACTORS

    def scene_objects(self):
        return [Block("obj", half=BAR, rgba=(0.1, 0.25, 0.8, 1.0)), StepShelf("shelf")]


class SpongeTopShelfEnv(ShelfEnv):
    instruction = "Put the sponge on top of the shelf."
    scale = 0.30
    shelf_r = (0.19, 0.21)
    shelf_angle = (38.0, 58.0)
    distractor_pool = LOW_DISTRACTORS

    def scene_objects(self):
        return [Sponge("obj"), LiberoFree("shelf", TWO_LAYER, self.scale, mass=1.2)]

    board_half = 0.003

    def _index_bodies(self):
        """Thicken the top board's collision downward (top face unchanged): at 0.3x it is 1.3 mm thick and a released
        soup can's edge pressed 3 mm into it (penetration gate)."""
        super()._index_bodies()
        m, d = self.model, mujoco.MjData(self.model)
        a = self._qadr["shelf"]
        d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        top = max(self._geoms["shelf"], key=lambda g: d.geom_xpos[g][2] + (np.abs(d.geom_xmat[g].reshape(3, 3))
                                                                          @ m.geom_size[g])[2])
        R = d.geom_xmat[top].reshape(3, 3)
        k = int(np.argmax(np.abs(R[2])))              # local axis along world z
        grow = self.board_half - m.geom_size[top][k]
        if grow > 0:
            m.geom_size[top][k] = self.board_half
            m.geom_aabb[top, 3 + k] = self.board_half
            # Move the geom down by ``grow`` in its parent body frame (body frames are unrotated at rest).
            m.geom_pos[top] -= np.array([0, 0, grow])
            m.geom_rbound[top] = float(np.linalg.norm(m.geom_size[top]))
        self._extent = self._measure_extents()

    def shelf_yaw(self, xy):
        # Long side (local x) tangential, open front (local -y) toward the viewer side (+x world) or the robot.
        return float(np.arctan2(xy[1], xy[0]) + np.pi / 2 + self.np_random.uniform(-0.3, 0.3))

    def _top_board(self):
        """(centre xy, half extents, top z) of the top board in the shelf frame (from its collision boxes)."""
        if not hasattr(self, "_board"):
            m, d = self.model, mujoco.MjData(self.model)
            a = self._qadr["shelf"]
            d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
            mujoco.mj_kinematics(m, d)
            origin = d.xpos[self._body["shelf"]]
            best = None
            for g in self._geoms["shelf"]:
                R = d.geom_xmat[g].reshape(3, 3)
                h = np.abs(R) @ m.geom_size[g]
                c = d.geom_xpos[g] - origin
                if best is None or c[2] + h[2] > best[2]:
                    best = (c[:2].copy(), h[:2].copy(), float(c[2] + h[2]))
            lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
            for g in self._geoms["shelf"]:
                R = d.geom_xmat[g].reshape(3, 3)
                h = np.abs(R) @ m.geom_size[g]
                c = d.geom_xpos[g] - origin
                lo, hi = np.minimum(lo, c[:2] - h[:2]), np.maximum(hi, c[:2] + h[:2])
            self._board = best
            self._outline = (lo, hi)
        return self._board

    def level_face(self):
        return self._top_board()

    def shelf_corners(self, xy, yaw):
        self._top_board()
        lo, hi = self._outline
        R = rot_z(yaw)[:2, :2]
        local = np.array([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
        return xy + local @ R.T


# ----- oracle ------------------------------------------------------------------------------------------------

class ShelfOracle(TrainOracle):
    """Top-down pick (closing direction chosen so the carry arrives at the place rotation with a small turn), carry
    above the shelf, place with the jaws closing parallel to the levels' riser and the camera mount turned away
    from the upper level."""

    carry_z = 0.092
    width = 2 * HALF
    symmetric = 4
    yaw_offset = 0.0                  # closing direction relative to the object's yaw (local x)
    grasp_up = 0.0                    # TCP above the object's body origin at the grasp

    squeeze = None                    # close to width - squeeze (fingertip-sphere centres); None: fully (kit)

    def close_target(self, width):
        if self.squeeze is None:
            return CLOSED
        self.jaw_gap(0.7)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(width - self.squeeze, gaps, angles)))

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """The kit's ``pick`` with an optional bounded squeeze (identical to it when ``squeeze`` is None)."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.env.object_pos(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.close_target(width), self.close_seconds, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], lift_z]), rot, speed=0.12, label="lift")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    level = False                     # re-level the held object over the target and lower in two stages
    pre_release_height = 0.012

    def level_rot(self, name, rot):
        """TCP rotation near ``rot`` at which the held object's up axis is vertical (it pivots 2-4 deg in the jaws
        while carried; lowered tilted, its low edge hits the level first)."""
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        yaw = np.arctan2(m[1, 0], m[0, 0])
        return rot_z(yaw) @ r_obj_tcp.T

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        if not self.level:
            yield from super().place_object(name, target, rot, drop, open_to, carry_z)
            return
        carry_z = self.carry_z if carry_z is None else carry_z
        target = np.asarray(target, float)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        yield from self.move(tcp + [0, 0, self.pre_release_height], rot, speed=0.06, tol=0.002, settle=0.6,
                             label="pre-lower")
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + lift], rot, speed=0.04, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def place_rot(self):
        """Desired TCP rotation over the target: TCP y (camera-mount side) along the shelf's front (+x)."""
        return top_down_mat(self.env.yaw("shelf") + np.pi / 2)

    def pick_yaw(self, target):
        """Object closing angle (among its symmetric ones) that the azimuth-following carry turns into the place
        rotation's closing direction (or its half turn)."""
        env = self.env
        p = env.object_pos("obj")
        want = self.env.yaw("shelf") + np.pi / 2
        daz = np.arctan2(target[1], target[0]) - np.arctan2(p[1], p[0])
        base = env.yaw("obj") + self.yaw_offset
        cands = [base + k * 2 * np.pi / self.symmetric for k in range(self.symmetric)]
        return min(cands, key=lambda a: abs(np.angle(np.exp(2j * (a + daz - want)))))

    def plan(self):
        env = self.env
        target = env.goal_target(env.goals[0])
        yaw = self.pick_yaw(target)
        grasp_z = env.object_pos("obj")[2] + self.grasp_up
        ok = yield from self.pick("obj", self.width, grasp_z, yaw=yaw, symmetric=2)
        if not ok:
            return
        rot = self.place_rot()
        rot = self.feasible_rotation("obj", target, [rot, rot_z(np.pi) @ rot] if self.flip_ok else [rot])
        yield from self.place_object("obj", target, rot=rot, open_to=self.release_for(self.width))
        yield from self.clear_shelf()
        yield from self.rest()
        yield from self.wait(1.2)

    flip_ok = False

    def clear_shelf(self):
        """Before folding to rest, swing (at carry height, arcing about the pan axis) to straight ahead: folding from
        above the shelf swept the wrist-camera mount into the upper level (2/3 calibration seeds)."""
        rot = top_down_mat(np.pi / 2)
        yield from self.carry_to(np.array([0.20, 0.0, self.carry_z]), rot, label="clear")


class CubeTopStepOracle(ShelfOracle):
    carry_z = 0.098                   # the placed cube's top is 8.3 cm up; the open jaws hang 8 mm below the TCP


class BlockLowerStepOracle(ShelfOracle):
    """Bounded squeeze: closing fully on the 2 cm-wide bar pressed the moving jaw up to 3.8 mm into it, and it
    sprang loose on release (5000-6400 rad/s^2; 11/50 qualification failures). Lowered
    tilted, its low edge struck the level before release (1400-9900 rad/s^2): it is re-levelled over the target."""
    squeeze = 0.003
    level = True
    width = 2 * BAR[1]
    symmetric = 2
    yaw_offset = np.pi / 2


class SpongeTopShelfOracle(ShelfOracle):
    width = 2 * SPONGE_HALF[1]
    symmetric = 2
    yaw_offset = np.pi / 2            # close across the narrow side (local y)
    grasp_up = -0.001
    flip_ok = True
    grasp_clearance = 0.0015          # catch the flat sponge early (it tips about its edge otherwise)
    # The placed sponge's top is 8.5 cm up and the open jaws hang 8 mm below the TCP: at 9.2 cm the sweep to the
    # rest pose dragged it off the shelf (6/12 calibration seeds).
    carry_z = 0.10

    def place_rot(self):
        return self.carry_rot(self.env.goal_target(self.env.goals[0])[:2])

    def pick_yaw(self, target):
        return self.env.yaw("obj") + self.yaw_offset


TASKS = [
    define_task(name="sponge_on_top_of_shelf", instruction=SpongeTopShelfEnv.instruction, family=FAMILY,
                env=SpongeTopShelfEnv, oracle=SpongeTopShelfOracle, objects=("obj",), object_kinds=("sponge",),
                relation="on top of", goal="wooden shelf", steps=(step_text("put", "sponge", "on top of", "shelf"),)),
    define_task(name="cube_on_top_step", instruction=CubeTopStepEnv.instruction, family=FAMILY,
                env=CubeTopStepEnv, oracle=CubeTopStepOracle, objects=("obj",), object_kinds=("cube",),
                relation="on upper level of", goal="step shelf",
                steps=(step_text("put", "red cube", "on", "top shelf"),)),
    define_task(name="block_on_lower_step", instruction=BlockLowerStepEnv.instruction, family=FAMILY,
                env=BlockLowerStepEnv, oracle=BlockLowerStepOracle, objects=("obj",),
                object_kinds=("rectangular block",), relation="on lower level of", goal="step shelf",
                steps=(step_text("put", "blue block", "on", "lower shelf"),)),
]

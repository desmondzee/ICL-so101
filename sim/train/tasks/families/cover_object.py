"""cover_object: put an upside-down container over an object so the object is hidden inside it.

The container starts upside down (opening on the table). The robot grasps it near its closed end (now on top)
across two flat outer faces, carries it over the object, lowers it until its rim rests on the table around the
object, releases and backs away.

Success is strict: the container is still upside down (its axis within 10 deg of vertical), released, settled,
resting on the table by its rim and touching nothing else (in particular not the covered object), its origin is
over the object within tolerance, and every collision point of the covered object lies inside the container's
inner walls with a clear margin and below its inner ceiling. The covered object is a non-task free body, so the
physics gate also requires that covering it never moved it (<= 5 mm, <= 0.1 rad).

Tasks (different container and covered object kind each):

* ``cover_ball_with_cup``     -- an octagonal plastic cup (closed end on top) over a small ball.
* ``cover_pudding_with_box``  -- a small open cardboard box, upside down, over a chocolate-pudding box; the
  box is turned in the carry so its long side lines up with the pudding's.

Both containers are built from primitives: no LIBERO/scanned asset is an open container with straight walls
that fits the 5.6 cm usable jaw opening (the LIBERO bowls have sloped, faceted walls).
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, Goal, TrainEnv, TrainOracle, define_task
from sim.val.oracle import RELEASE
from sim.val.scene import Obj

FAMILY = "cover_object"
# Distractors no taller than 2.6 cm: the carried container's rim passes about 3 cm above the table.
SHORT_DISTRACTORS = ("cream_cheese", "butter", "chocolate_pudding", "popcorn", "white_bowl", "red_bowl", "plate",
                     "ramekin")
REST_SWEEP = [(np.array([0.145, 0.06]), 0.025), (np.array([0.145, -0.06]), 0.025)]


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


# ----- assets ------------------------------------------------------------------------------------------------

@dataclass
class InvertedCup:
    """A free octagonal plastic cup standing upside down: eight wall panels (outer apothem ``radius``) and a
    closed end (cap) on top; the body origin is at the rim's centre, mid-height of the wall.

    Eight sides, because each flat face (1.6 cm wide at a 3.8 cm cup) is wider than the 8 mm jaw pads: on a
    16-sided (or round-looking) cup the pads met 7 mm faces and their edges, the cup crept sideways out of the
    fingertips during carries, tipped in the grasp and wedged against the fixed jaw's hull (calibration, 8 of 24
    seeds). A cup gripped by a foot on its closed end crept out of the fingertips the same way."""

    name: str
    radius: float = 0.019
    height: float = 0.034
    wall: float = 0.0025
    cap: float = 0.003
    sides: int = 8
    rgba: tuple = (0.35, 0.62, 0.85, 1.0)
    mass: float = 0.025
    friction: float = 1.2

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        r, h, w, n = self.radius, self.height, self.wall, self.sides
        half_side = r * np.tan(np.pi / n)           # outer face half-width: the panels meet at the corners
        wall_mass = 0.8 * self.mass / n
        rgba = _fmt(self.rgba)
        for k in range(n):
            a = 2 * np.pi * k / n
            c = (r - w / 2) * np.array([np.cos(a), np.sin(a), 0.0])
            quat = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
            ET.SubElement(body, "geom", name=f"{self.name}_wall{k}", type="box", pos=_fmt(c), quat=_fmt(quat),
                          size=_fmt((w / 2, half_side, h / 2)), group="1", condim="4", mass=f"{wall_mass:.6g}",
                          friction=f"{self.friction} 0.02 0.001", rgba=rgba, material="val_plastic")
        # The cap: an octagonal plate (two crossed boxes cover it) on top of the walls, slightly darker.
        dark = _fmt((*(0.85 * np.asarray(self.rgba[:3])), 1.0))
        inner = r - w
        for k in range(2):
            a = np.pi / 4 * k
            quat = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
            ET.SubElement(body, "geom", name=f"{self.name}_cap{k}", type="box", quat=_fmt(quat),
                          pos=_fmt((0, 0, h / 2 - self.cap / 2)),
                          size=_fmt((inner, inner * np.tan(np.pi / n), self.cap / 2)), group="1", condim="4",
                          mass=f"{0.1 * self.mass:.6g}", friction=f"{self.friction} 0.02 0.001", rgba=dark,
                          material="val_plastic")
            ET.SubElement(body, "geom", name=f"{self.name}_capx{k}", type="box", quat=_fmt(quat),
                          pos=_fmt((0, 0, h / 2 - self.cap / 2)),
                          size=_fmt((inner * np.tan(np.pi / n), inner, self.cap / 2)), group="1", condim="4",
                          mass="0.0001", friction=f"{self.friction} 0.02 0.001", rgba=dark, material="val_plastic")


@dataclass
class InvertedBox:
    """A free open box lying upside down (closed side on top): four walls and a top panel. Outer half-size
    ``half`` (x long, y short), height ``height``; the origin is at the centre of the rim rectangle, mid-height."""

    name: str
    half: tuple = (0.028, 0.020)
    height: float = 0.026
    wall: float = 0.003
    rgba: tuple = (0.70, 0.53, 0.34, 1.0)
    mass: float = 0.03
    friction: float = 1.2

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        hx, hy = self.half
        h, w = self.height, self.wall
        parts = [("top", (0, 0, h / 2 - w / 2), (hx, hy, w / 2), 0.4),
                 ("wall_px", (hx - w / 2, 0, -w / 2), (w / 2, hy, h / 2 - w / 2), 0.1),
                 ("wall_nx", (-hx + w / 2, 0, -w / 2), (w / 2, hy, h / 2 - w / 2), 0.1),
                 ("wall_py", (0, hy - w / 2, -w / 2), (hx - w, w / 2, h / 2 - w / 2), 0.2),
                 ("wall_ny", (0, -hy + w / 2, -w / 2), (hx - w, w / 2, h / 2 - w / 2), 0.2)]
        for label, pos, size, share in parts:
            ET.SubElement(body, "geom", name=f"{self.name}_{label}", type="box", pos=_fmt(pos), size=_fmt(size),
                          rgba=_fmt(self.rgba), mass=f"{self.mass * share:.6g}", group="1", condim="4",
                          friction=f"{self.friction} 0.02 0.001", material="val_fabric")


@dataclass
class Ball:
    """A free solid ball; condim 6 (rolling friction) so it stays put."""

    name: str
    radius: float = 0.012
    rgba: tuple = (0.9, 0.2, 0.15, 1.0)
    mass: float = 0.02
    friction: float = 1.0
    rolling: float = 0.003

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "geom", name=f"{self.name}_geom", type="sphere", size=f"{self.radius:.6g}",
                      rgba=_fmt(self.rgba), mass=repr(self.mass), group="1", condim="6",
                      friction=f"{self.friction} 0.02 {self.rolling}", material="val_plastic")


# ----- geometry helpers --------------------------------------------------------------------------------------

def world_points(env, name):
    """Collision points of a free body in world coordinates (box corners, sphere/cylinder AABB corners)."""
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
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            # Points on the sphere (an AABB corner would overstate its reach by sqrt(3)).
            u = np.vstack([np.eye(3), -np.eye(3), corners / np.sqrt(3)])
            local = u * m.geom_size[g][0]
        else:
            local = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
        pts.append(local @ R.T + p)
    return np.vstack(pts)


def touches_only(env, name, allowed):
    own = set(env._geoms[name])
    for c in env.data.contact[:env.data.ncon]:
        if (c.geom1 in own and c.geom2 not in allowed) or (c.geom2 in own and c.geom1 not in allowed):
            return False
    return True


# ----- environment -------------------------------------------------------------------------------------------

class CoverEnv(TrainEnv):
    """Put the upside-down free container ``cover`` over the free object ``hidden``.

    inner: inner half-extents (x, y) of the container's opening in its frame; ceiling: inner height above the
    rim; margin: clear gap every point of ``hidden`` keeps from the inner walls."""

    cover = "cover"
    hidden = "hidden"
    inner = (0.019, 0.019)
    round_inner = True
    ceiling = 0.037
    margin = 0.002
    tolerance = (0.006, 0.006, 0.004)
    hidden_region = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
    cover_region = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
    min_travel = 0.10
    # Long carries (across the whole front, up to 26 cm) let the hanging container creep sideways along the 8 mm
    # finger pads until it fell out; the container and the object are at most this far apart.
    max_travel = 0.17
    distractor_pool = SHORT_DISTRACTORS

    def covered(self):
        """Every collision point of ``hidden`` inside the container's inner walls and below its ceiling."""
        body = self._body[self.cover]
        R, p = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        rim = p - R[:, 2] * self.rim_offset()
        local = (world_points(self, self.hidden) - rim) @ R
        if np.any(local[:, 2] > self.ceiling - self.margin):
            return False
        if self.round_inner:
            return bool(np.all(np.linalg.norm(local[:, :2], axis=1) <= self.inner[0] - self.margin))
        return bool(np.all(np.abs(local[:, :2]) <= np.asarray(self.inner) - self.margin))

    def rim_offset(self):
        """Height of the origin above the rim plane (container frame)."""
        return float(self._extent[self.cover]["bottom"])

    def goal_check(self, env, name):
        return self.covered() and touches_only(self, name, self._owned_geoms("table"))

    def debug_state(self):
        g = self.goals[0]
        err = self.object_pos(self.cover) - self.goal_target(g)
        return dict(err_mm=np.round(err * 1000, 1).tolist(), covered=self.covered(),
                    only_table=touches_only(self, self.cover, self._owned_geoms("table")),
                    upright=self.upright(self.cover), settled=self.settled(self.cover),
                    released=self.released(self.cover))

    def layout(self):
        placed = list(REST_SWEEP)
        hidden_xy, _ = self.place(self.hidden, placed, region=self.hidden_region, clearance=0.04)
        # The container starts min_travel..max_travel from the object (sampled in that annulus), inside its region
        # and clear of everything placed so far.
        radius, (r0, r1), (a0, a1) = self.footprint(self.cover), self.cover_region["r"], self.cover_region["angle"]
        for _ in range(500):
            d, t = self.np_random.uniform(self.min_travel, self.max_travel), self.np_random.uniform(-np.pi, np.pi)
            xy = hidden_xy + d * np.array([np.cos(t), np.sin(t)])
            if not (r0 <= np.hypot(*xy) <= r1 and a0 <= np.degrees(np.arctan2(xy[1], xy[0])) <= a1):
                continue
            # Keep-outs (robot base, rest pose and its sweep) need only a small clearance; objects 4 cm.
            if (all(np.linalg.norm(xy - p) >= radius + q + 0.01 for p, q in (*self.keepout, *REST_SWEEP))
                    and all(np.linalg.norm(xy - p) >= radius + q + 0.04 for p, q in placed[len(REST_SWEEP):])):
                break
        else:
            raise RuntimeError("no container start near the object")
        self.set_object_pose(self.cover, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((xy, radius))
        self.place_distractors(placed)
        centre = self.hidden_center()
        z = float(self.object_pos(self.cover)[2])
        self.set_goals(Goal(self.cover, "table", target=(centre[0], centre[1], z), reference=self.hidden,
                            tolerance=self.tolerance, check=self.goal_check))

    def hidden_center(self):
        pts = world_points(self, self.hidden)
        return (pts.min(0) + pts.max(0)) / 2


CUP_R = 0.020                       # 4.0 cm across the wall's flats


class BallCupEnv(CoverEnv):
    instruction = "Cover the ball with the cup."
    cover, hidden = "cup", "ball"
    task_objects = ("cup",)
    inner = (CUP_R - 0.0025,) * 2   # inner apothem of the octagonal wall
    ceiling = 0.031

    def scene_objects(self):
        return [InvertedCup("cup", radius=CUP_R), Ball("ball", radius=0.010)]


class PuddingBoxEnv(CoverEnv):
    instruction = "Cover the chocolate pudding with the box."
    cover, hidden = "box", "pudding"
    task_objects = ("box",)
    inner = (0.025, 0.017)
    round_inner = False
    ceiling = 0.023
    distractor_pool = tuple(n for n in SHORT_DISTRACTORS if n != "chocolate_pudding")

    def scene_objects(self):
        # 5.6 x 4.0 x 2.6 cm box over the 0.5x pudding (4.0 x 2.3 x 1.4 cm).
        return [InvertedBox("box", half=(0.028, 0.020), height=0.026), Obj("pudding", "chocolate_pudding")]

    def hidden_long_yaw(self):
        """World yaw of the pudding's long horizontal axis."""
        R = self.data.xmat[self._body[self.hidden]].reshape(3, 3)
        pts = (world_points(self, self.hidden) - self.object_pos(self.hidden)) @ R
        size = pts.max(0) - pts.min(0)
        axis = R[:, 0] if size[0] >= size[1] else R[:, 1]
        return float(np.arctan2(axis[1], axis[0]))

    def goal_check(self, env, name):
        # Long sides lined up within 15 deg (the inner fit leaves no more), besides the generic checks.
        R = self.data.xmat[self._body[self.cover]].reshape(3, 3)
        box_yaw = np.arctan2(R[1, 0], R[0, 0])
        aligned = abs(np.cos(box_yaw - self.hidden_long_yaw())) >= np.cos(np.radians(15))
        return aligned and super().goal_check(env, name)


# ----- oracle ------------------------------------------------------------------------------------------------

class CoverOracle(TrainOracle):
    carry_z = 0.10
    close_seconds = 1.5
    release_seconds = 2.0       # slower: a 1.2 s opening rocked the released container just over 1000 rad/s^2
    open_margin = 0.012
    release_margin = 0.018      # open wide before rising: a 1 cm opening let the rising jaw clip the cup top
    grasp_clearance = 0.002
    touch_depth = 0.001
    grip_squeeze = 0.004
    backoff_speed = 0.01
    grasp_below_top = 0.011        # TCP this far below the top (higher: the fixed jaw hull pressed on the closed end)
    carry_speed = 0.06          # containers hang below the fingertips: faster carries let them creep down
    # Origin height offset at release: the rim just pressing the table. Releasing 0.5 mm above it, a cup hanging
    # 1 deg tilted dropped onto its rim edge and hopped 4 mm when the jaws opened.
    set_down = 0.0     # rim exactly on the table (pressing 0.5 mm stored force that popped the cup up on release)
    vmax = np.radians([50.0, 55.0, 55.0, 65.0, 65.0])

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def grasp_options(self):
        """(closing yaws, width) of the container's flat outer face pairs."""
        raise NotImplementedError

    def pick_cover(self, attempts=2, approach=0.05):
        env = self.env
        name = env.cover
        for attempt in range(attempts):
            yaws, width = self.grasp_options()
            pts = world_points(env, name)
            centre = env.object_pos(name)
            grasp_z = float(pts[:, 2].max()) - self.grasp_below_top
            rot, grasp, q = self.grasp_plan(np.r_[centre[:2], grasp_z], width, yaws)
            self._held_width = width
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.05, tol=0.003, label="grasp")
            yield from self.gripper(self.open_for(width, -self.touch_depth), 0.9, 0.1)
            yield from self.gripper(self.gap_angle(width - self.grip_squeeze), self.close_seconds, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.r_[grasp[:2], grasp[2] + 0.015], rot, speed=0.03, label="lift1")
                yield from self.move(np.r_[grasp[:2], self.carry_z], rot, speed=0.08, label="lift")
                if env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def carry_to(self, pos, rot, label="carry"):
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(_wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, speed=self.carry_speed, label=label)
        else:
            q = getattr(self, "_carry_q", None)
            yield from self.transit(pos, rot, speed=0.45, q=q, label=label)
        self._carry_q = None

    def end_rotation(self, target):
        return self.carry_rot(target[:2])

    def correct_yaw(self, rot):
        return rot

    def release(self, open_to=None):
        """Ease the grip onto the walls first, then open: opening straight from the squeeze released the stored
        jaw force into the light cup (it hopped 3-5 mm, onto the ball)."""
        rot = self._cmd_rot
        yield from self.gripper(self.gap_angle(self._held_width - 0.0005), 1.2, 0.3)
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.3)
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=self.backoff_speed,
                             tol=0.002, settle=0.3, label="backoff", smooth=False)

    def plan(self):
        env = self.env
        name = env.cover
        ok = yield from self.pick_cover()
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        rot = self.end_rotation(target)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], self.carry_z], rot)
        yield from self.wait(0.2)
        if not env.is_grasping(name):          # dropped in the carry: stop (the episode fails cleanly)
            self.log.append((name, "dropped in carry"))
            return
        rot = self.correct_yaw(rot)
        # No levelling: the containers hang with their centre of mass under the grip, and a levelled rotation the
        # 5-DOF arm cannot reach made the IK swing the wrist through a 5-minute "align" (frame budget exceeded).
        target = env.goal_target(env.goals[0])
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], self.carry_z], rot, speed=0.04, label="align", smooth=False)
        # Lower until the rim is 8 mm above the object's top, settle and re-align there (the arm lags a few mm
        # while descending; a 1.5 cm hover put the rim below the ball's top and it clipped the ball), then lower
        # straight down slowly and set the rim on the table.
        hover = float(world_points(env, env.hidden)[:, 2].max()) - (target[2] - env.rim_offset()) + 0.008
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], tcp[2] + max(0.015, hover)], rot, speed=0.05, label="descend")
        yield from self.wait(0.3)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], tcp[2] + max(0.015, hover)], rot, speed=0.03, tol=0.0015, settle=0.6,
                             label="realign", smooth=False)
        tcp = target - rot @ self.held(name)
        lift = 0.5 * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + lift + self.set_down], rot, speed=0.02, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(self.open_for(self._held_width, self.release_margin))
        yield from self.move(np.r_[self._cmd_pos[:2], self._cmd_pos[2] + 0.03], rot, speed=0.05, label="retreat1")
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], rot, speed=0.10, label="retreat")
        yield from self.rest()
        yield from self.wait(1.2)


class CupOracle(CoverOracle):
    # Open only a little before rising: opening wide swings the rotating jaw's lower body onto the closed end.
    release_margin = 0.010

    def grasp_options(self):
        env = self.env
        cup = env.scene_objects()[0]
        yaw = env.yaw(env.cover)
        # Closing along face normals only (opposite faces are parallel).
        return [yaw + k * 2 * np.pi / cup.sides for k in range(cup.sides)], 2 * cup.radius


class BoxOracle(CoverOracle):
    def grasp_options(self):
        env = self.env
        yaw = env.yaw(env.cover)
        return [yaw + np.pi / 2, yaw - np.pi / 2], 0.040

    def end_rotation(self, target):
        """Turn the box so its long side runs along the pudding's: of the turns that do so, the one closest to
        the azimuth change (least wrist roll), among those the IK reaches."""
        env = self.env
        follow = self.carry_rot(target[:2])
        turn_follow = _wrap(np.arctan2(follow[1, 0], follow[0, 0]) - np.arctan2(self._cmd_rot[1, 0],
                                                                               self._cmd_rot[0, 0]))
        need = _wrap(env.hidden_long_yaw() - env.yaw(env.cover))
        held, best = self.held(env.cover), None
        for turn in (need, _wrap(need + np.pi)):
            rot = _rz(turn) @ self._cmd_rot
            tcp = target - rot @ held
            q_low, e_low, _ = self.solve(tcp, rot)
            # Carry pose seeded with the set-down roll: unseeded, the IK settled on a branch pinned at the roll
            # stop and tilted 30 deg, which the following straight-down moves left with a half-turn wrist swing.
            q_carry, e_carry, _ = self.solve(np.r_[tcp[:2], self.carry_z], rot, seeds=[q_low[4]])
            # Keep the wrist roll off its +-157 deg stop: a carry ending pinned at the stop left the IK of the next
            # (straight-down) moves jumping half a turn, a 5-minute wrist swing (frame budget exceeded).
            pinned = max(abs(q_carry[4]), abs(q_low[4])) > np.radians(145)
            score = (pinned, round(max(e_carry, e_low), 3), abs(_wrap(turn - turn_follow)))
            if best is None or score < best[0]:
                best = (score, rot, q_carry)
        self._carry_q = best[2]
        return best[1]


    def correct_yaw(self, rot):
        """After the carry, turn by the remaining yaw error between the box's long side and the pudding's (the
        box can turn a few degrees in the jaws; 7 deg left the pudding's corner outside the walls)."""
        env = self.env
        error = _wrap(env.hidden_long_yaw() - env.yaw(env.cover))
        error = min((error, _wrap(error + np.pi)), key=abs)
        if abs(error) < np.radians(1.0):
            return rot
        self.log.append(("yaw correction", round(float(np.degrees(error)), 1)))
        return _rz(error) @ rot


def _task(name, env, oracle, kinds, goal, steps):
    return define_task(name=name, instruction=env.instruction, family=FAMILY, env=env, oracle=oracle,
                       objects=env.task_objects, object_kinds=kinds, relation="upside down over", goal=goal,
                       steps=steps)


TASKS = [
    _task("cover_ball_with_cup", BallCupEnv, CupOracle, ("cup",), "ball",
          ("Pick up the upside-down cup by its base and put it down over the ball so the ball is covered.",)),
    _task("cover_pudding_with_box", PuddingBoxEnv, BoxOracle, ("box",), "chocolate pudding",
          ("Pick up the upside-down box and put it down over the chocolate pudding so the pudding is covered.",)),
]

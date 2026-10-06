"""stack_two: stack one object on top of another, centred, upright and released.

Each task grasps one free object (the *top*) and stacks it on another free object (the *bottom*) that the
robot never grasps. The bottom object is a non-task free body, so the strict physics gate also requires
that stacking leaves it in place (<= 5 mm, <= 0.1 rad). Success, at every substep of the final suffix:
the top object's origin is at the stacked target (which follows the bottom object, so the top's bounding
box is centred over the bottom's), upright, released, settled, pushed up by the bottom object and touching
*nothing else* (``only_on``: no table, distractor or robot contact), and, for boxes, its long side runs
along the bottom box's long side within 15 deg (``aligned``).

The tasks differ in the object pair and its geometry: a small cube on a big cube (block on block, different
sizes); a cream cheese box lined up on a butter box (flat food box on a box of nearly equal footprint); a cube
on the rim of a can (block on a narrow round support); a can standing on a flat box (cylinder on a box).

Grasps are planned about the collision-geometry centre and heights come from exact mesh vertices (LIBERO
and scanned origins are not always centred, and the kit's AABB extents are inflated for rotated meshes);
for lined-up boxes the target origin follows from the bottom's centre and the top's end yaw, chosen in
``layout()`` (least turn) and followed by the oracle.

Dropped while calibrating (see the family report): the scanned candy box (1.1 cm thin: slides out of the
jaws), the YCB gelatin box (its hull rests 3.8 mm into the table, over the 3 mm penetration gate) and a can
on a can (21 x 24 convex hulls: one substep produced 304 contacts, overflowed MuJoCo's constraint arena
and every body fell through the table).
"""

import mujoco
import numpy as np

from sim.train.tasks.base import (Goal, Obj, TrainEnv, TrainOracle, define_task, step_text)
from sim.val.oracle import RELEASE
from sim.val.scene import Block

GREEN, YELLOW, RED = (0.2, 0.6, 0.25, 1.0), (0.9, 0.75, 0.15, 1.0), (0.8, 0.12, 0.1, 1.0)
SMALL, LARGE = 0.012, 0.018                    # cube half-sizes: 2.4 cm on 3.6 cm
COS15 = float(np.cos(np.radians(15)))
GRASP_TCP_MIN = 0.010                          # jaw hull reaches ~8.2 mm below the TCP: keep it off the table
BOTTOM_REGION = dict(r=(0.17, 0.25), angle=(-60.0, 60.0))
DAIRY_SCALE = 1.0                              # 0.5x dairy boxes are 9 mm thin and slide out of the jaws
PAN_AXIS_X = 0.0388                             # shoulder-pan axis (x, robot base frame)
CAN_CUBE = 0.017                               # cube half-size for the cube-on-can stack
# Grasp region without the closest ring: a close grasp at a large azimuth can fold the camera mount into
# the shoulder on the way from rest (README pitfall; seen once in calibration).
TOP_REGION = dict(r=(0.16, 0.27), angle=(-65.0, 65.0))
# The rest() sweep of the folded gripper/camera mount (x 0.12-0.17, |y| 0.04-0.08): keep it clear.
REST_SWEEP = [(np.array([0.145, 0.06]), 0.025), (np.array([0.145, -0.06]), 0.025)]


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


# ----- success predicates ------------------------------------------------------------------------------------

def only_on(env, name):
    """The object touches no body except its stack support (no table, distractor or robot contact)."""
    own, support = set(env._geoms[name]), set(env._geoms[env.bottom])
    for c in env.data.contact[:env.data.ncon]:
        if (c.geom1 in own and c.geom2 not in support) or (c.geom2 in own and c.geom1 not in support):
            return False
    return True


def aligned(env, name):
    """Long horizontal axis of the top box parallel (either sense) to the bottom box's, within 15 deg."""
    a = env.data.xmat[env._body[name]].reshape(3, 3)[:2, env.long_axis[name]]
    b = env.data.xmat[env._body[env.bottom]].reshape(3, 3)[:2, env.long_axis[env.bottom]]
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) >= COS15


def stack_check(env, name):
    return only_on(env, name) and (not env.align or aligned(env, name))


# ----- environment -------------------------------------------------------------------------------------------

class StackEnv(TrainEnv):
    """``top`` stacked on the free body ``bottom``. Subclasses set the names, specs and options."""

    top = "top"
    bottom = "bottom"
    align = False                     # require the long sides to line up (boxes)
    tolerance = (0.010, 0.010, 0.006)
    bottom_region = BOTTOM_REGION
    # Share of a held object's tilt depth added to the release height. 1.0 = no press (a tilted cube pressing a
    # can rim tips the light can); the kit's 0.5 (slight press, short fall) for flat box-on-box stacks, where a
    # longer fall rocked the box at just over 1000 rad/s^2.
    release_lift_fraction = 1.0
    release_drop = 0.0                # extra release height (m)

    def world_points(self, name):
        """Exact collision points of a free body in world coordinates: mesh vertices, box corners (other
        primitives fall back to their AABB corners). The kit's AABB extents are inflated for rotated scanned
        meshes (by up to ~4 mm), which is too coarse for stacking heights."""
        m, d = self.model, self.data
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        pts = []
        for g in self._geoms[name]:
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

    def local_box(self, name):
        """(centre, size) of a free body's collision geometry in its own frame."""
        body = self._body[name]
        R, p = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        pts = (self.world_points(name) - p) @ R
        return (pts.max(0) + pts.min(0)) / 2, pts.max(0) - pts.min(0)

    def origin_height(self, name):
        """Height of the body origin above its lowest collision point (at the current orientation)."""
        return float(self.object_pos(name)[2] - self.world_points(name)[:, 2].min())

    def box_center(self, name):
        return self.object_pos(name) + self.data.xmat[self._body[name]].reshape(3, 3) @ self.local_box(name)[0]

    @property
    def long_axis(self):
        """Local horizontal axis (0 = x, 1 = y) along the longer side, per stack body."""
        if not hasattr(self, "_long_axis"):
            self._long_axis = {}
            for n in (self.top, self.bottom):
                size = self.local_box(n)[1]
                self._long_axis[n] = 0 if size[0] >= size[1] else 1
        return self._long_axis

    def long_yaw(self, name):
        axis = self.data.xmat[self._body[name]].reshape(3, 3)[:, self.long_axis[name]]
        return float(np.arctan2(axis[1], axis[0]))

    def layout(self):
        placed = list(REST_SWEEP)
        self.place(self.bottom, placed, region=self.bottom_region)
        self.place(self.top, placed, region=TOP_REGION)
        self.place_distractors(placed)
        # End yaw of the top body: boxes turn so the long sides line up, in the sense whose turn is closest to
        # the change in azimuth about the pan axis (the wrist roll then barely moves during the carry, which
        # stays a Cartesian arc; the other sense needs a roll unwind in a joint-space swing that missed its
        # target or swept the box through the arm in calibration). Others keep their yaw (the carry turns
        # them; their origins sit within 2 mm of their box centres).
        self.end_yaw = self.yaw(self.top)
        if self.align:
            pan = np.array([PAN_AXIS_X, 0.0])
            bearing = lambda xy: np.arctan2(xy[1] - pan[1], xy[0] - pan[0])
            azimuth = _wrap(bearing(self.box_center(self.bottom)[:2]) - bearing(self.box_center(self.top)[:2]))
            turn = _wrap(self.long_yaw(self.bottom) - self.long_yaw(self.top))
            turn = min((turn, _wrap(turn + np.pi)), key=lambda t: abs(_wrap(t - azimuth)))
            self.end_yaw = self.yaw(self.top) + turn
        center, size = self.local_box(self.top)
        bottom_center = self.box_center(self.bottom)
        bottom_top = float(self.world_points(self.bottom)[:, 2].max())
        offset = _rz(self.end_yaw) @ center if self.align else np.zeros(3)
        target = (bottom_center[0] - offset[0], bottom_center[1] - offset[1],
                  bottom_top + self.origin_height(self.top))
        self.set_goals(Goal(self.top, self.bottom, target=target, reference=self.bottom, tolerance=self.tolerance,
                            check=stack_check))


class CubeStackEnv(StackEnv):
    instruction = "Stack the small green cube on top of the big yellow cube."
    top, bottom = "small_cube", "big_cube"
    task_objects = ("small_cube",)

    def scene_objects(self):
        return [Block("small_cube", half=(SMALL,) * 3, rgba=GREEN, mass=0.02),
                Block("big_cube", half=(LARGE,) * 3, rgba=YELLOW, mass=0.06)]


class DairyStackEnv(StackEnv):
    instruction = "Put the cream cheese box on top of the butter box."
    top, bottom = "cream_cheese", "butter"
    task_objects = ("cream_cheese",)
    align = True
    release_lift_fraction = 0.5
    # Bottom box kept off the near ring: 5 of 6 qualification failures (first run, 44/50) had the butter at
    # r < 0.195 m and |azimuth| >= 69 deg, where the turned 8 cm box swings close to the arm.
    bottom_region = dict(r=(0.20, 0.25), angle=(-60.0, 60.0))

    def scene_objects(self):
        # 1.0x: 8.2 x 4.2 x 1.8 cm on 7.6 x 4.0 x 1.8 cm. At 0.7-0.9x (1.25-1.6 cm thick) the jaws only reach the
        # box's upper half and it creeps out of the grasp during carries (2 of 6 calibration seeds dropped it).
        return [Obj("cream_cheese", "cream_cheese", DAIRY_SCALE, friction=1.5), Obj("butter", "butter", DAIRY_SCALE)]


class CubeOnCanEnv(StackEnv):
    instruction = "Put the red cube on top of the tomato sauce can."
    top, bottom = "red_cube", "tomato_sauce"
    task_objects = ("red_cube",)
    tolerance = (0.007, 0.007, 0.006)
    # Release 1 mm above the rim: pressing the cube onto the 24-hull can lid while still gripped ejected the
    # light can 6 mm upward once in 20 calibration seeds (181 m/s^2, 3.7 mm penetration).
    release_drop = 0.001

    def scene_objects(self):
        # The cube (3.4 cm) is wider than the can (3.1 cm), so it rests on the can's rim; a smaller cube drops
        # into the recessed lid and rocks on the rim (measured 2.5 deg tilt, 2100 rad/s^2 spikes).
        return [Block("red_cube", half=(CAN_CUBE,) * 3, rgba=RED, mass=0.03), Obj("tomato_sauce", "tomato_sauce")]


class CanOnBoxEnv(StackEnv):
    instruction = "Stand the alphabet soup can on top of the butter box."
    top, bottom = "alphabet_soup", "butter"
    task_objects = ("alphabet_soup",)
    tolerance = (0.010, 0.010, 0.006)

    def scene_objects(self):
        # A 3.1 x 3.8 cm can on a 7.6 x 4.0 x 1.8 cm butter box (1.0x).
        return [Obj("alphabet_soup", "alphabet_soup"), Obj("butter", "butter", DAIRY_SCALE)]


# ----- oracle ------------------------------------------------------------------------------------------------

class StackOracle(TrainOracle):
    """Pick the top object about its box centre across its narrow side, carry it over the bottom one (turning
    a box to the planned end yaw), lower it until it rests on the bottom object, release and retreat.

    Light flat boxes are spun by the closing jaw's impact and pivot in the jaws under carry accelerations, so
    the jaw opens less and closes more slowly, large-turn carries are slower, and the release is gentler."""

    close_seconds = 2.0
    release_seconds = 1.5       # slower opening: release/backoff rocked the object just over 1000 rad/s^2
    open_margin = 0.008
    transit_speed = 0.45        # rad/s peak for joint-space carries (kit default 0.8)

    def carry_to(self, pos, rot, label="carry"):
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(_wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, label=label)
        else:
            yield from self.transit(pos, rot, speed=self.transit_speed, label=label)

    def release(self, open_to=None):
        """As the kit's release, with a slower fixed-finger back-off: a top object that pivoted a few degrees in
        the jaws lands leaning on the fixed finger and settles flat as the finger withdraws (at the kit's
        30 mm/s it slapped flat at just over 1000 rad/s^2)."""
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.008, tol=0.002,
                             settle=0.3, label="backoff", smooth=False)

    def level(self, name, rot):
        """``rot`` turned so the held object ends upright: objects pivot a few degrees in the jaws (measured up
        to 5.4 deg), and an object released tilted onto a stack lands on an edge and slaps flat when the jaws
        open (1000-1650 rad/s^2 measured) or is caught leaning on the fixed finger."""
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

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        """The kit's place_object, plus ``level`` after the carry (when the in-jaw tilt is known): the release
        pose tilts the TCP so the held object is upright over the stack."""
        carry_z = self.carry_z if carry_z is None else carry_z
        target = np.asarray(target, float)
        rot = self.carry_rot(target[:2]) if rot is None else rot
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        tcp = target - rot @ self.held(name)
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + lift], rot, speed=0.06, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def plan(self):
        env = self.env
        top, goal = env.top, env.goals[0]
        center, size = env.local_box(top)
        width = float(min(size[0], size[1]))
        square = abs(size[0] - size[1]) < 0.004
        center_xy = env.box_center(top)[:2]
        floor = env.surface_z(center_xy, exclude=(top,))           # the table top under the object
        grasp_z = floor + float(np.clip(size[2] / 2, GRASP_TCP_MIN, max(GRASP_TCP_MIN, size[2] - 0.003)))
        target = env.goal_target(goal)
        if env.align:
            yaw0 = env.yaw(top)
            turn = _wrap(env.end_yaw - yaw0)
            across = yaw0 + (np.pi / 2 if env.long_axis[top] == 0 else 0.0)
            yaw = self._grasp_for_turn(top, across, width, grasp_z, turn, target)
            ok = yield from self.pick(top, width, grasp_z, yaw=yaw, symmetric=1)
            if not ok:
                return
            rot = _rz(_wrap(env.end_yaw - env.yaw(top))) @ self._cmd_rot
        else:
            ok = yield from self.pick(top, width, grasp_z, yaw=env.yaw(top), symmetric=4 if square else 2)
            if not ok:
                return
            rot = None
        self.release_lift_fraction = env.release_lift_fraction
        yield from self.place_object(top, env.goal_target(goal), rot=rot, open_to=self.release_for(width),
                                     drop=env.release_drop)
        yield from self.rest()
        yield from self.wait(1.2)

    def _grasp_for_turn(self, name, across, width, grasp_z, turn, target):
        """Of the two closing directions across the narrow side, the one whose grasp and turned end pose
        (carry and release) the IK reaches best (wrist roll is limited to +-157 deg)."""
        env, best = self.env, None
        start = np.r_[env.box_center(name)[:2], grasp_z]
        end_center = np.asarray(target, float) + _rz(env.end_yaw) @ env.local_box(name)[0]
        end_center[2] = target[2] + grasp_z - env.object_pos(name)[2]
        for a in (across, across + np.pi):
            rot, pos, _ = self.grasp_plan(start, width, [a])
            end, tcp = _rz(turn) @ rot, end_center + _rz(turn) @ (pos - start)
            (q0, e0, _), (q1, e1, _), (_, e2, _) = (self.solve(pos, rot), self.solve(np.r_[tcp[:2], self.carry_z], end),
                                                     self.solve(tcp, end))
            # Unreachable poses dominate; otherwise prefer the smaller wrist-roll travel during the carry (a
            # near-full unwind of the roll in one transit missed its target by 4 cm in calibration).
            score = 200 * max(e0, e1, e2) + 0.3 * abs(q1[4] - q0[4])
            if best is None or score < best[0]:
                best = (score, a)
        return best[1]

    def grasp_plan(self, center, width, angles):
        """Grasp about the top object's live box centre (scanned origins can sit on an edge)."""
        return super().grasp_plan(np.r_[self.env.box_center(self.env.top)[:2], center[2]], width, angles)


def _task(name, env, kind, goal, obj_text, target_text, **kwargs):
    return define_task(name=name, instruction=env.instruction, family="stack_two", env=env, oracle=StackOracle,
                       objects=env.task_objects, object_kinds=(kind,), relation="stacked on top of", goal=goal,
                       steps=(step_text("stack", obj_text, "on top of", target_text),), **kwargs)


TASKS = [
    _task("stack_small_cube_on_big_cube", CubeStackEnv, "small cube", "big cube", "small green cube",
          "big yellow cube"),
    _task("stack_cream_cheese_on_butter", DairyStackEnv, "cream cheese box", "butter box", "cream cheese box",
          "butter box"),
    # Kitchen only: 6 of the 7 failures of the first qualification run (43/50) were in the living room, where
    # the cube on the can's 24-hull lid kept rocking (settled_speed) or tipped the can.
    _task("stack_cube_on_sauce_can", CubeOnCanEnv, "cube", "tomato sauce can", "red cube", "tomato sauce can",
          arenas=("kitchen",)),
    _task("stack_soup_can_on_butter", CanOnBoxEnv, "soup can", "butter box", "alphabet soup can", "butter box"),
]

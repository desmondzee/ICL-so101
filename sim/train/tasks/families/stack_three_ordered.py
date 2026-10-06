"""stack_three_ordered: build a three-object stack in a stated order (bottom, middle, top).

The bottom object stays where it is (a non-task free body, so the strict gate also requires that the stack
leaves it in place: <= 5 mm, <= 0.1 rad). The robot first stacks the named *middle* object on it, then the
named *top* object on the middle one; the kit's ``OrderedCompletion`` enforces that temporal order (grasping
the top object before the middle one is placed is a permanent failure).

Success, at every substep of the final suffix: both moved objects are upright, released, settled, pushed up by
the object directly below them, centred over the bottom object within ``tolerance`` (and the top one also
within ``tolerance`` of the middle one's box centre), touching nothing but their stack neighbours (no table,
distractor or robot contact), and, for boxes, lined up along the bottom box's long side within 15 deg.

The tasks differ in the objects and their geometry (not colour): three equal cubes whose order is given only by
colour; a big-to-small cube tower; and two food boxes (cream cheese lined up on butter) topped by a cube.

Grasps are planned about the collision-geometry centre and heights come from exact collision points (helpers
copied from ``stack_two``; LIBERO origins are not always centred). The carry height for each placement is
raised above the current stack so the held object clears it.
"""

import mujoco
import numpy as np

from sim.train.tasks.base import Goal, Obj, TrainEnv, TrainOracle, define_task, step_text, then
from sim.val.oracle import RELEASE
from sim.val.scene import Block

FAMILY = "stack_three_ordered"
RED, BLUE, YELLOW = (0.8, 0.12, 0.1, 1.0), (0.12, 0.28, 0.82, 1.0), (0.92, 0.76, 0.12, 1.0)
GREEN, ORANGE, PURPLE = (0.2, 0.6, 0.25, 1.0), (0.95, 0.5, 0.1, 1.0), (0.55, 0.25, 0.7, 1.0)
COS15 = float(np.cos(np.radians(15)))
GRASP_TCP_MIN = 0.010                          # jaw hull reaches ~8.2 mm below the TCP: keep it off the support
PAN_AXIS_X = 0.0388                             # shoulder-pan axis (x, robot base frame)
# The stack is tall, so its base stays inside r <= 0.24 (release poses at 6-8 cm stay near vertical there).
BOTTOM_REGION = dict(r=(0.17, 0.24), angle=(-58.0, 58.0))
# Pick region without the closest ring (a close grasp at a large azimuth can fold the camera mount into the
# shoulder on the way from rest).
PICK_REGION = dict(r=(0.16, 0.27), angle=(-65.0, 65.0))
# The rest() sweep of the folded gripper/camera mount (x 0.12-0.17, |y| 0.04-0.08): keep it clear.
REST_SWEEP = [(np.array([0.145, 0.06]), 0.025), (np.array([0.145, -0.06]), 0.025)]
CARRY_CLEARANCE = 0.025                         # held object's bottom above the stack top while carried
RETREAT_CLEARANCE = 0.035                       # TCP above the released object's top (jaw hull reaches 8.2 mm below)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle):
    return float(np.angle(np.exp(1j * angle)))


# ----- success predicates ------------------------------------------------------------------------------------

def only_touching(env, name, allowed):
    own = set(env._geoms[name])
    ok = set().union(*(env._geoms[n] for n in allowed))
    for c in env.data.contact[:env.data.ncon]:
        if (c.geom1 in own and c.geom2 not in ok) or (c.geom2 in own and c.geom1 not in ok):
            return False
    return True


def aligned(env, a, b):
    """Long horizontal axes of boxes ``a`` and ``b`` parallel (either sense) within 15 deg."""
    va = env.data.xmat[env._body[a]].reshape(3, 3)[:2, env.long_axis(a)]
    vb = env.data.xmat[env._body[b]].reshape(3, 3)[:2, env.long_axis(b)]
    return float(abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)) >= COS15


def middle_check(env, name):
    if not only_touching(env, name, (env.bottom, env.top)):
        return False
    return not env.align_middle or aligned(env, name, env.bottom)


def top_check(env, name):
    if not only_touching(env, name, (env.middle,)):
        return False
    centre = env.box_center(env.middle)[:2]
    return bool(np.linalg.norm(env.box_center(name)[:2] - centre) <= env.tolerance[0])


# ----- environment -------------------------------------------------------------------------------------------

class StackThreeEnv(TrainEnv):
    """``middle`` on the free body ``bottom``, then ``top`` on ``middle``."""

    bottom, middle, top = "bottom", "middle", "top"
    align_middle = False              # the middle box must line up with the bottom box (boxes)
    tolerance = (0.010, 0.010, 0.006)

    # ----- geometry helpers (from stack_two) ---------------------------------------------------------------
    def world_points(self, name):
        """Exact collision points of a free body in world coordinates (mesh vertices, box corners)."""
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

    def top_z(self, name):
        return float(self.world_points(name)[:, 2].max())

    def box_center(self, name):
        return self.object_pos(name) + self.data.xmat[self._body[name]].reshape(3, 3) @ self.local_box(name)[0]

    def long_axis(self, name):
        size = self.local_box(name)[1]
        return 0 if size[0] >= size[1] else 1

    def long_yaw(self, name):
        axis = self.data.xmat[self._body[name]].reshape(3, 3)[:, self.long_axis(name)]
        return float(np.arctan2(axis[1], axis[0]))

    # ----- layout ------------------------------------------------------------------------------------------
    def layout(self):
        placed = list(REST_SWEEP)
        self.place(self.bottom, placed, region=BOTTOM_REGION)
        self.place(self.middle, placed, region=PICK_REGION)
        self.place(self.top, placed, region=PICK_REGION)
        self.place_distractors(placed)
        # Middle end yaw: boxes turn so the long sides line up, in the sense closest to the change in azimuth
        # about the pan axis (the wrist roll then barely moves during the carry). Cubes keep their yaw.
        self.end_yaw = self.yaw(self.middle)
        if self.align_middle:
            pan = np.array([PAN_AXIS_X, 0.0])
            bearing = lambda xy: np.arctan2(xy[1] - pan[1], xy[0] - pan[0])
            azimuth = _wrap(bearing(self.box_center(self.bottom)[:2]) - bearing(self.box_center(self.middle)[:2]))
            turn = _wrap(self.long_yaw(self.bottom) - self.long_yaw(self.middle))
            turn = min((turn, _wrap(turn + np.pi)), key=lambda t: abs(_wrap(t - azimuth)))
            self.end_yaw = self.yaw(self.middle) + turn
        bottom_center = self.box_center(self.bottom)
        bottom_top = self.top_z(self.bottom)
        mid_center, mid_size = self.local_box(self.middle)
        offset = _rz(self.end_yaw) @ mid_center if self.align_middle else np.zeros(3)
        mid_target = (bottom_center[0] - offset[0], bottom_center[1] - offset[1],
                      bottom_top + self.origin_height(self.middle))
        mid_height = self.top_z(self.middle) - float(self.world_points(self.middle)[:, 2].min())
        top_target = (bottom_center[0], bottom_center[1], bottom_top + mid_height + self.origin_height(self.top))
        self.set_goals(
            Goal(self.middle, self.bottom, target=mid_target, reference=self.bottom, tolerance=self.tolerance,
                 check=middle_check),
            Goal(self.top, self.middle, target=top_target, reference=self.bottom, tolerance=self.tolerance,
                 check=top_check))


class ThreeCubesEnv(StackThreeEnv):
    instruction = "Stack the blue cube on the red cube, then put the yellow cube on top."
    bottom, middle, top = "red_cube", "blue_cube", "yellow_cube"
    task_objects = order = ("blue_cube", "yellow_cube")

    def scene_objects(self):
        return [Block("red_cube", half=(0.015,) * 3, rgba=RED, mass=0.05),
                Block("blue_cube", half=(0.015,) * 3, rgba=BLUE, mass=0.03),
                Block("yellow_cube", half=(0.015,) * 3, rgba=YELLOW, mass=0.03)]


class CubeTowerEnv(StackThreeEnv):
    instruction = "Build a tower: put the medium cube on the big cube, then the small cube on top."
    bottom, middle, top = "big_cube", "medium_cube", "small_cube"
    task_objects = order = ("medium_cube", "small_cube")

    def scene_objects(self):
        return [Block("big_cube", half=(0.019,) * 3, rgba=GREEN, mass=0.08),
                Block("medium_cube", half=(0.014,) * 3, rgba=ORANGE, mass=0.03),
                Block("small_cube", half=(0.011,) * 3, rgba=PURPLE, mass=0.015)]


class BoxesAndCubeEnv(StackThreeEnv):
    instruction = "Put the cream cheese box on the butter box, then stack the red cube on top."
    bottom, middle, top = "butter", "cream_cheese", "red_cube"
    task_objects = order = ("cream_cheese", "red_cube")
    align_middle = True
    distractor_pool = tuple(n for n in StackThreeEnv.distractor_pool if n not in ("cream_cheese", "butter"))

    def scene_objects(self):
        # 1.0x boxes: 8.2 x 4.2 x 1.8 cm on 7.6 x 4.0 x 1.8 cm (0.5x boxes are too thin to hold securely).
        return [Obj("butter", "butter", 1.0), Obj("cream_cheese", "cream_cheese", 1.0, friction=1.5),
                Block("red_cube", half=(0.014,) * 3, rgba=RED, mass=0.03)]


# ----- oracle ------------------------------------------------------------------------------------------------

class StackThreeOracle(TrainOracle):
    """Pick each object about its box centre across its narrow side, carry it above the current stack, level it,
    lower it onto the object below, release gently and retreat; middle first, then top."""

    close_seconds = 2.0
    release_seconds = 1.5
    open_margin = 0.008
    transit_speed = 0.45
    release_lift_fraction = 1.0
    _grasping = None

    def carry_to(self, pos, rot, label="carry"):
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(_wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, label=label)
        else:
            yield from self.transit(pos, rot, speed=self.transit_speed, label=label)

    def release(self, open_to=None):
        rot = self._cmd_rot
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.008, tol=0.002,
                             settle=0.3, label="backoff", smooth=False)

    def level(self, name, rot):
        """``rot`` turned so the held object ends upright (it pivots a few degrees in the jaws)."""
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
        # Retreat until the jaws clear the released object's top: at the plain carry height the fingertips still
        # straddled a top cube and the rest swing knocked the three-high stack over (qualification seed 36494401).
        retreat_z = max(carry_z, self.env.top_z(name) + RETREAT_CLEARANCE)
        yield from self.move(np.r_[self._cmd_pos[:2], retreat_z], rot, speed=0.10, label="retreat")

    def grasp_plan(self, center, width, angles):
        """Grasp about the object's live box centre (LIBERO origins can sit off-centre)."""
        return super().grasp_plan(np.r_[self.env.box_center(self._grasping)[:2], center[2]], width, angles)

    def _grasp_params(self, name):
        env = self.env
        center, size = env.local_box(name)
        width = float(min(size[0], size[1]))
        floor = env.object_pos(name)[2] - env.origin_height(name)
        grasp_z = floor + float(np.clip(size[2] / 2, GRASP_TCP_MIN, max(GRASP_TCP_MIN, size[2] - 0.003)))
        return width, size, grasp_z

    def _carry_height(self, name, grasp_z, stack_top):
        """Carry so the held object's lowest point clears the stack (and the default carry height)."""
        below = grasp_z - (self.env.object_pos(name)[2] - self.env.origin_height(name))
        return max(self.carry_z, stack_top + below + CARRY_CLEARANCE)

    def _grasp_for_turn(self, name, across, width, grasp_z, turn, target, end_yaw, carry_z):
        """Of the two closing directions across the narrow side, the one whose grasp and turned end pose the
        IK reaches best, preferring less wrist-roll travel (from stack_two)."""
        env, best = self.env, None
        start = np.r_[env.box_center(name)[:2], grasp_z]
        end_center = np.asarray(target, float) + _rz(end_yaw) @ env.local_box(name)[0]
        end_center[2] = target[2] + grasp_z - env.object_pos(name)[2]
        for a in (across, across + np.pi):
            rot, pos, _ = self.grasp_plan(start, width, [a])
            end, tcp = _rz(turn) @ rot, end_center + _rz(turn) @ (pos - start)
            (q0, e0, _), (q1, e1, _), (_, e2, _) = (self.solve(pos, rot), self.solve(np.r_[tcp[:2], carry_z], end),
                                                     self.solve(tcp, end))
            score = 200 * max(e0, e1, e2) + 0.3 * abs(q1[4] - q0[4])
            if best is None or score < best[0]:
                best = (score, a)
        return best[1]

    def stack(self, name, target, stack_top, end_yaw=None):
        env = self.env
        self._grasping = name
        width, size, grasp_z = self._grasp_params(name)
        carry_z = self._carry_height(name, grasp_z, stack_top)
        if end_yaw is not None:
            yaw0 = env.yaw(name)
            turn = _wrap(end_yaw - yaw0)
            across = yaw0 + (np.pi / 2 if env.long_axis(name) == 0 else 0.0)
            yaw = self._grasp_for_turn(name, across, width, grasp_z, turn, target, end_yaw, carry_z)
            ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=1, lift_z=carry_z)
            if not ok:
                return False
            rot = _rz(_wrap(end_yaw - env.yaw(name))) @ self._cmd_rot
        else:
            square = abs(size[0] - size[1]) < 0.004
            ok = yield from self.pick(name, width, grasp_z, yaw=env.yaw(name), symmetric=4 if square else 2,
                                      lift_z=carry_z)
            if not ok:
                return False
            rot = None
        yield from self.place_object(name, target, rot=rot, open_to=self.release_for(width), carry_z=carry_z)
        return True

    def plan(self):
        env = self.env
        mid_goal, top_goal = env.goals
        ok = yield from self.stack(env.middle, env.goal_target(mid_goal), env.top_z(env.bottom),
                                   end_yaw=env.end_yaw if env.align_middle else None)
        if not ok:
            return
        # The top object goes over where the middle one actually landed (box centre over box centre).
        mid_center = env.box_center(env.middle)
        top_offset = env.box_center(env.top) - env.object_pos(env.top)
        stack_top = env.top_z(env.middle)
        target = np.r_[mid_center[:2] - top_offset[:2], stack_top + env.origin_height(env.top)]
        ok = yield from self.stack(env.top, target, stack_top)
        if not ok:
            return
        yield from self.rest()
        yield from self.wait(1.2)


def _task(name, env, kinds, goal, texts):
    middle_text, bottom_text, top_text = texts
    return define_task(name=name, instruction=env.instruction, family=FAMILY, env=env, oracle=StackThreeOracle,
                       objects=env.task_objects, object_kinds=kinds, order=env.order,
                       relation="stacked three high in order", goal=goal,
                       steps=(step_text("stack", middle_text, "on top of", bottom_text),
                              then(step_text("stack", top_text, "on top of", middle_text))))


TASKS = [
    _task("stack_three_cubes_by_colour", ThreeCubesEnv, ("cube", "cube"), "cube",
          ("blue cube", "red cube", "yellow cube")),
    _task("stack_cube_tower_big_to_small", CubeTowerEnv, ("medium cube", "small cube"), "big cube",
          ("medium cube", "big cube", "small cube")),
    _task("stack_cube_on_cream_cheese_on_butter", BoxesAndCubeEnv, ("cream cheese box", "cube"), "butter box",
          ("cream cheese box", "butter box", "red cube")),
]

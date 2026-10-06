"""Small, bounded training scenes using existing blocks, discs and LIBERO assets.

No validation environment or task predicate is reused. Shared ValEnv/Oracle
provide the robot and motion primitives only. The physics audit remains the
final authority, including contacts during the entire trajectory.
"""

import mujoco
import numpy as np

from sim.train.physics import GoalRule, PhysicsPolicy
from sim.train.variation import GoalRegion
from sim.val.env import ValEnv
from sim.val.oracle import Oracle
from sim.val.scene import Block, Disc, Fixture, SceneSpec
from .schema import OrderedCompletion


HALF = 0.014
BASKET_FLOOR = 0.0109
FINGERS = ("gripper", "moving_jaw_so101_v1")


def crosswise_orientation(matrix):
    return bool(abs(matrix[1, 0]) >= np.cos(np.radians(10)) and matrix[2, 2] >= np.cos(np.radians(10)))


class PilotEnv(ValEnv):
    task_objects = ("block",)
    ordered = False
    orientation = False
    container = False
    removal = False
    beside = False

    def make_scene(self):
        objects = [Block(n, half=(0.024, 0.010, HALF) if self.orientation else (HALF,) * 3,
                         rgba=(0.1, 0.25, 0.8, 1) if n == "blue_block" else (0.8, 0.1, 0.08, 1))
                   for n in self.task_objects]
        fixtures = []
        if self.container:
            fixtures.append(Fixture("basket", "basket", 0.5))
        if self.beside:
            fixtures.append(Fixture("bowl", "akita_black_bowl", 1.0))
        if not self.container or self.removal:
            if not self.beside:
                objects.extend(Disc(n, radius=0.037) for n in self.mats)
        return SceneSpec(arena="kitchen", objects=objects, fixtures=fixtures)

    @property
    def mats(self):
        return ("near_mat", "far_mat") if self.ordered else ("mat",)

    @property
    def table_body(self):
        return "table" if self.scene.arena == "kitchen" else "living_room_table_col"

    def layout(self):
        self.completion = OrderedCompletion(self.task_objects)
        self.active_distractors = []
        jitter = lambda: self.np_random.uniform(-0.008, 0.008, 2)
        self.targets, self.supports = {}, {}
        if self.container:
            basket_xy = np.array([0.24, 0.075]) + jitter()
            self.set_fixture_pose("basket", basket_xy)
            self.set_object_pose("block", basket_xy if self.removal else np.array([0.23, -0.09]) + jitter(),
                                 yaw=self.np_random.uniform(-0.3, 0.3), z=BASKET_FLOOR if self.removal else 0)
            if not self.removal:
                self.targets["block"] = np.r_[basket_xy, BASKET_FLOOR + HALF]
                self.supports["block"] = "basket"
        elif self.ordered:
            for name, xy in zip(self.task_objects, ([0.20, -0.12], [0.28, -0.075])):
                self.set_object_pose(name, np.array(xy) + jitter(), yaw=self.np_random.uniform(-0.4, 0.4))
        else:
            self.set_object_pose(self.task_objects[0], np.array([0.23, -0.09]) + jitter(),
                                 yaw=self.np_random.uniform(-0.3, 0.3))
        if self.beside:
            xy = np.array([0.23, 0.12]) + jitter()
            self.set_fixture_pose("bowl", xy)
            self.targets["block"] = np.r_[xy + [0, -0.085], HALF]
            self.supports["block"] = self.table_body
        elif not self.container or self.removal:
            for name, mat, xy in zip(self.task_objects, self.mats,
                                      ([0.20, 0.065], [0.28, 0.105]) if self.ordered else ([0.23, -0.085] if self.removal else [0.23, 0.085],)):
                xy = np.array(xy) + jitter()
                self.set_object_pose(mat, xy)
                self.targets[name] = np.r_[xy, 0.006 + HALF]
                self.supports[name] = mat

    def at_goal(self, name):
        if name not in getattr(self, "targets", {}):
            return False
        pose = self.object_pose(name)
        if np.any(np.abs(pose[:3] - self.targets[name]) > [0.01, 0.01, 0.006]):
            return False
        if self.orientation:
            # Long local x axis must be parallel to table y, within 10 degrees,
            # and the block must be flat. Both ends are equivalent.
            matrix = self.data.xmat[self._body[name]].reshape(3, 3)
            if not crosswise_orientation(matrix):
                return False
        velocity = self.data.qvel[self._dadr[name]:self._dadr[name] + 6]
        return (not self.is_grasping(name) and np.linalg.norm(velocity[:3]) <= 0.03
                and np.linalg.norm(velocity[3:]) <= 0.5 and self.supported(name))

    def supported(self, name):
        support = self.supports[name]
        geoms = set(self._geoms[name])
        support_geoms = set(self._geoms[support]) if support in self._geoms else {
            g for g in range(self.model.ngeom) if self.model.body(int(self.model.geom_bodyid[g])).name == support}
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            forward = contact.geom1 in support_geoms and contact.geom2 in geoms
            reverse = contact.geom2 in support_geoms and contact.geom1 in geoms
            if (forward or reverse) and contact.dist <= 0.001:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if force[0] > 0 and contact.frame[2] * (1 if forward else -1) >= 0.5:
                    return True
        return False

    def step(self, action, **kwargs):
        result = super().step(action, **kwargs)
        held = tuple(n for n in self.task_objects if self.is_grasping(n))
        self.completion.observe(held=held, placed=tuple(n for n in self.task_objects if self.at_goal(n)))
        return result

    def success(self):
        return (all(self.at_goal(n) for n in self.task_objects)
                and (not self.ordered or self.completion.complete))

    def physics_policy(self):
        allowed = [(self.table_body, n) for n in self.free_names]
        for n in self.task_objects:
            allowed.extend((n, finger) for finger in FINGERS)
            if self.container:
                allowed.append((n, "basket"))
            allowed.append((n, self.supports[n]))
        # Explicit contact exceptions are added only after calibration evidence;
        # unknown robot/table/self contacts deliberately fail closed.
        return PhysicsPolicy(self.task_objects,
            tuple((n, (self.supports[n],)) for n in self.task_objects), tuple(allowed),
            distractors=tuple(n for n in self.free_names if n not in self.task_objects),
            goals=tuple(GoalRule(n, offset=tuple(self.targets[n]), position_tolerance=(0.01, 0.01, 0.006),
                                 upright_cos=float(np.cos(np.radians(10)))) for n in self.task_objects))

    def goal_regions(self):
        return tuple(GoalRegion(self.supports[n] if self.supports[n] != self.table_body else None,
                    tuple(tuple(self.targets[n] + [x, y, -HALF]) for x in (-0.008, 0, 0.008) for y in (-0.008, 0, 0.008)))
                     for n in self.task_objects)


class InsertEnv(PilotEnv):
    container = True
    instruction = "Put the block inside the basket."


class RemoveEnv(PilotEnv):
    container = removal = True
    instruction = "Take the block out of the basket and put it on the mat."


class BesideEnv(PilotEnv):
    beside = True
    instruction = "Place the block to the right of the bowl, leaving a gap."


class OrderedEnv(PilotEnv):
    ordered = True
    task_objects = ("red_block", "blue_block")
    instruction = "Move the red block to the near mat, then the blue block to the far mat."


class OrientationEnv(PilotEnv):
    orientation = True
    task_objects = ("bar",)
    instruction = "Place the long block on the mat with its long side running left to right."


class PilotOracle(Oracle):
    def __init__(self, env, rng=None):
        super().__init__(env)

    def grasp_plan(self, center, width, angles):
        # A rectangular block has two equivalent narrow-side grasps, not four.
        return super().grasp_plan(center, width, angles[::2] if self.env.orientation else angles)

    def plan(self):
        env = self.env
        for name in env.task_objects:
            matrix = env.data.xmat[env._body[name]].reshape(3, 3)
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            grasp_z = env.object_pos(name)[2] - 0.001
            if env.orientation:
                # Narrow side grasp, so the long axis can be commanded explicitly.
                ok = yield from self.pick(name, 0.020, grasp_z, yaw=yaw + np.pi / 2)
            else:
                ok = yield from self.pick(name, 2 * HALF, grasp_z, yaw=yaw)
            if not ok:
                return
            held = self._cmd_rot.T @ (self.tcp() - env.object_pos(name))
            target = env.targets[name]
            rot = None
            if env.orientation:
                current = env.data.xmat[env._body[name]].reshape(3, 3)
                delta = np.pi / 2 - np.arctan2(current[1, 0], current[0, 0])
                c, s = np.cos(delta), np.sin(delta)
                rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self._cmd_rot
            rot = self.carry_rot(target[:2]) if rot is None else rot
            offset = rot @ held
            yield from self.place(target[:2] + offset[:2], target[2] + offset[2] + 0.003, rot=rot)
        yield from self.rest()
        yield from self.wait(1.2)

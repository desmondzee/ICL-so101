"""Pilot family module and worked example of the authoring kit (five distinct manipulation skills).

Each task: a ``TrainEnv`` subclass (scene + randomized layout + goals), a ``TrainOracle`` subclass (plan)
and a ``define_task`` entry in ``TASKS``. Success, physics policy and visibility regions are derived
from the goals declared in ``layout()``.
"""

import numpy as np

from sim.train.tasks.base import (COS10, PLACE_REGION, VIEW, Goal, TrainEnv, TrainOracle, define_task, mat,
                                  step_text, then)
from sim.val.scene import Block, Fixture

HALF = 0.014                      # cube half-size (2.8 cm block)
BAR = (0.024, 0.010, HALF)        # 4.8 x 2.0 x 2.8 cm long block
MAT_HALF = 0.003                  # mats are 7 x 7 cm, 6 mm thick
RED, BLUE = (0.8, 0.1, 0.08, 1), (0.1, 0.25, 0.8, 1)
GREEN_MAT, YELLOW_MAT, GREY_MAT = (0.2, 0.55, 0.25, 1), (0.85, 0.7, 0.15, 1), (0.45, 0.45, 0.5, 1)


def cube(name, rgba=RED):
    return Block(name, half=(HALF,) * 3, rgba=rgba)


def crosswise(env, name):
    """Long (local x) axis along world y (viewer left-right) within 10 deg, and lying flat."""
    return env.axis_alignment(name, (0, 1, 0)) >= COS10 and env.upright(name, COS10)


# ----- container insertion ---------------------------------------------------------------------------------

class InsertEnv(TrainEnv):
    instruction = "Put the block inside the bowl."
    task_objects = ("block",)

    def scene_objects(self):
        return [cube("block")]

    def scene_fixtures(self):
        return [Fixture("bowl", "white_bowl", 1.0)]

    def layout(self):
        placed = []
        bowl_xy, _ = self.place_fixture("bowl", placed, region=dict(r=(0.17, 0.27), angle=(-60, 60)))
        self.place("block", placed)
        self.place_distractors(placed)
        floor = self.surface_z(bowl_xy, exclude=self.task_objects)
        self.set_goals(Goal("block", "bowl", target=(*bowl_xy, floor + HALF), tolerance=(0.018, 0.018, 0.006)))


class InsertOracle(TrainOracle):
    def plan(self):
        env = self.env
        goal = env.goals[0]
        yield from self.pick_and_place("block", env.goal_target(goal) + [0, 0, 0.004], width=2 * HALF)
        yield from self.rest()
        yield from self.wait(1.2)


# ----- container removal -----------------------------------------------------------------------------------

class RemoveEnv(TrainEnv):
    instruction = "Take the block out of the bowl and put it on the mat."
    task_objects = ("block",)
    surfaces = ("bowl",)                      # the block starts on the bowl floor

    def scene_objects(self):
        return [cube("block"), mat("mat", rgba=GREY_MAT)]

    def scene_fixtures(self):
        return [Fixture("bowl", "white_bowl", 1.2)]

    def layout(self):
        placed = []
        bowl_xy, _ = self.place_fixture("bowl", placed, region=dict(r=(0.17, 0.26), angle=(-60, 60)))
        floor = self.surface_z(bowl_xy, exclude=self.task_objects)
        self.set_object_pose("block", bowl_xy + self.np_random.uniform(-0.003, 0.003, 2),
                             yaw=self.np_random.uniform(-np.pi, np.pi), z=floor)
        mat_xy, _ = self.place("mat", placed, region=PLACE_REGION)
        self.place_distractors(placed)
        self.set_goals(Goal("block", "mat", target=(*mat_xy, 2 * MAT_HALF + HALF), reference="mat",
                            tolerance=(0.02, 0.02, 0.006)))


class RemoveOracle(TrainOracle):
    open_margin = 0.008                       # open just enough inside the bowl

    def plan(self):
        env = self.env
        yield from self.pick_and_place("block", env.goal_target(env.goals[0]), width=2 * HALF)
        yield from self.rest()
        yield from self.wait(1.2)


# ----- spatial arrangement ---------------------------------------------------------------------------------

class BesideEnv(TrainEnv):
    instruction = "Place the block to the right of the bowl, leaving a gap."
    task_objects = ("block",)
    gap = 0.105           # bowl centre to block centre along the viewer's right (world +y)

    def scene_objects(self):
        return [cube("block")]

    def scene_fixtures(self):
        return [Fixture("bowl", "akita_black_bowl", 1.0)]

    def layout(self):
        placed = []
        for _ in range(20):
            bowl_xy = self.sample_xy(self.fixture_footprint("bowl"), placed, r=(0.17, 0.27), angle=(-70, 35))
            target = bowl_xy + self.gap * VIEW["right"]
            if 0.14 <= np.hypot(*target) <= 0.27 and abs(np.degrees(np.arctan2(target[1], target[0]))) <= 65:
                break
        else:
            raise RuntimeError("no reachable beside target")
        self.set_fixture_pose("bowl", bowl_xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((bowl_xy, self.fixture_footprint("bowl")))
        placed.append((target, 0.03))                      # keep the goal spot free
        self.place("block", placed)
        self.place_distractors(placed)
        self.set_goals(Goal("block", "table", target=(*target, HALF), tolerance=(0.015, 0.015, 0.006)))


class BesideOracle(TrainOracle):
    def plan(self):
        env = self.env
        yield from self.pick_and_place("block", env.goal_target(env.goals[0]), width=2 * HALF)
        yield from self.rest()
        yield from self.wait(1.2)


# ----- ordered relocation ----------------------------------------------------------------------------------

class OrderedEnv(TrainEnv):
    instruction = "Move the red block onto the green mat first, then the blue block onto the yellow mat."
    task_objects = ("red_block", "blue_block")
    order = ("red_block", "blue_block")

    def scene_objects(self):
        return [cube("red_block", RED), cube("blue_block", BLUE),
                mat("green_mat", rgba=GREEN_MAT),
                mat("yellow_mat", rgba=YELLOW_MAT)]

    def layout(self):
        placed = []
        for name in ("green_mat", "yellow_mat"):
            self.place(name, placed, region=PLACE_REGION)
        for name in self.task_objects:
            self.place(name, placed)
        self.place_distractors(placed)
        self.set_goals(*(Goal(block, mat, target=(*self.object_pos(mat)[:2], 2 * MAT_HALF + HALF), reference=mat,
                              tolerance=(0.02, 0.02, 0.006))
                         for block, mat in (("red_block", "green_mat"), ("blue_block", "yellow_mat"))))


class OrderedOracle(TrainOracle):
    def plan(self):
        env = self.env
        for goal in env.goals:
            ok = yield from self.pick_and_place(goal.obj, env.goal_target(goal), width=2 * HALF)
            if not ok:
                return
        yield from self.rest()
        yield from self.wait(1.2)


# ----- orientation-sensitive placement ---------------------------------------------------------------------

class OrientationEnv(TrainEnv):
    instruction = "Place the long block on the mat with its long side running left to right."
    task_objects = ("bar",)

    def scene_objects(self):
        return [Block("bar", half=BAR, rgba=RED), mat("mat", rgba=GREY_MAT)]

    def layout(self):
        placed = []
        mat_xy, _ = self.place("mat", placed, region=PLACE_REGION)
        # Lengthwise start (long axis within 50 deg of radial), so the turn to crosswise is real.
        _, yaw = self.place("bar", placed, yaw=0.0)
        bar_xy = self.object_pos("bar")[:2]
        yaw = np.arctan2(bar_xy[1], bar_xy[0]) + self.np_random.uniform(-0.9, 0.9)
        self.set_object_pose("bar", bar_xy, yaw=yaw)
        self.place_distractors(placed)
        self.set_goals(Goal("bar", "mat", target=(*mat_xy, 2 * MAT_HALF + HALF), reference="mat",
                            tolerance=(0.02, 0.02, 0.006), check=crosswise))


class OrientationOracle(TrainOracle):
    def plan(self):
        env = self.env
        yaw = env.yaw("bar")
        # Grasp across the narrow side: closing direction along the bar's short (local y) axis.
        ok = yield from self.pick("bar", 2 * BAR[1], env.object_pos("bar")[2] - 0.001, yaw=yaw + np.pi / 2,
                                  symmetric=2)
        if not ok:
            return
        # Turn the held bar so its long axis ends along world y. Both ends are equivalent: try the
        # smaller turn first, keep whichever end rotation the IK can reach (wrist roll is limited).
        current = env.yaw("bar")
        deltas = sorted((np.pi / 2 - current + k * np.pi for k in (-1, 0, 1)), key=abs)
        rotations = [np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]]) @ self._cmd_rot
                     for d in deltas]
        target = env.goal_target(env.goals[0])
        rot = self.feasible_rotation("bar", target, rotations)
        yield from self.place_object("bar", target, rot=rot,
                                     open_to=self.release_for(2 * BAR[1]))
        yield from self.rest()
        yield from self.wait(1.2)


TASKS = [
    define_task(name="block_in_bowl", instruction=InsertEnv.instruction, family="container_insertion",
                env=InsertEnv, oracle=InsertOracle, objects=("block",), relation="inside", goal="bowl",
                steps=(step_text("put", "block", "inside", "bowl"),)),
    define_task(name="block_out_of_bowl", instruction=RemoveEnv.instruction, family="container_removal",
                env=RemoveEnv, oracle=RemoveOracle, objects=("block",), relation="out of bowl onto", goal="mat",
                steps=("Lift the block out of the bowl and put it on the mat.",)),
    define_task(name="block_beside_bowl", instruction=BesideEnv.instruction, family="spatial_arrangement",
                env=BesideEnv, oracle=BesideOracle, objects=("block",), relation="right of separated", goal="bowl",
                steps=("Pick up the block and put it down to the right of the bowl, leaving a gap.",)),
    define_task(name="blocks_onto_mats_in_order", instruction=OrderedEnv.instruction, family="ordered_relocation",
                env=OrderedEnv, oracle=OrderedOracle, objects=("red_block", "blue_block"),
                object_kinds=("block", "block"), relation="on in temporal order", goal="matching mats",
                steps=(step_text("put", "red block", "on", "green mat"),
                       then(step_text("put", "blue block", "on", "yellow mat")))),
    define_task(name="bar_crosswise_on_mat", instruction=OrientationEnv.instruction,
                family="orientation_sensitive_placement", env=OrientationEnv, oracle=OrientationOracle,
                objects=("bar",), object_kinds=("rectangular block",), relation="on crosswise",
                goal="mat with long axis along viewer left-right",
                steps=("Pick up the long block, turn it so it runs left to right, and lay it on the mat.",)),
]

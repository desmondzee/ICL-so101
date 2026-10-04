"""Task 1: put the red block on the red plate and the blue block on the blue plate."""

from __future__ import annotations

import mujoco
import numpy as np

from sim.val.env import ValEnv
from sim.val.oracle import Oracle
from sim.val.scene import Block, Obj, SceneSpec

COLORS = {"red": (0.78, 0.07, 0.06, 1.0), "blue": (0.06, 0.22, 0.78, 1.0)}
PLATE_TINT = {"red": (1.0, 0.32, 0.28, 1.0), "blue": (0.35, 0.5, 1.0, 1.0)}
BLOCK_HALF = 0.014
GRASP_Z = 0.013
FINGER_CLEARANCE = 0.03
PLATE_SCALE = 0.65
DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn", "ramekin", "white_yellow_mug")
BLOCK_REGION = dict(r=(0.15, 0.28), angle=(-60, 60))
PLATE_REGION = dict(r=(0.15, 0.30), angle=(-60, 60))


class SortBlocksEnv(ValEnv):
    instruction = "Put the red block on the red plate and the blue block on the blue plate."
    pairs = tuple((f"{c}_block", f"{c}_plate") for c in COLORS)

    def make_scene(self):
        objects = []
        for c, rgba in COLORS.items():
            objects.append(Block(f"{c}_block", half=(BLOCK_HALF,) * 3, rgba=rgba))
            objects.append(Obj(f"{c}_plate", "plate", PLATE_SCALE, rgba=PLATE_TINT[c], mass=0.25))
        return SceneSpec(arena="kitchen", objects=objects, distractors=[Obj(n, n) for n in DISTRACTORS])

    def layout(self):
        placed = []
        order = [n for pair in self.pairs for n in pair]
        for i in self.np_random.permutation(len(order)):
            n = order[i]
            block = n.endswith("block")
            radius = self.footprint(n) + (FINGER_CLEARANCE if block else 0.0)
            xy = self.sample_xy(radius, placed, **(BLOCK_REGION if block else PLATE_REGION), clearance=0.02)
            placed.append((xy, radius))
            self.set_object_pose(n, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        self.place_distractors(placed)

    def on_plate(self, block, plate):
        b, p = self.object_pos(block), self.object_pos(plate)
        inside = np.linalg.norm(b[:2] - p[:2]) < 0.8 * self.footprint(plate)
        resting = b[2] - BLOCK_HALF > p[2] - 0.002 and b[2] < p[2] + 0.06
        still = np.linalg.norm(self.data.qvel[self._dadr[block] : self._dadr[block] + 3]) < 0.05
        return bool(inside and resting and still and not self.is_grasping(block))

    def success(self):
        return all(self.on_plate(b, p) for b, p in self.pairs)

    def task_info(self):
        return {f"{b}_on_plate": self.on_plate(b, p) for b, p in self.pairs}


class SortBlocksOracle(Oracle):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng or np.random.default_rng()

    def plan(self):
        env = self.env
        pairs = list(env.pairs)
        if self.rng.random() < 0.5:
            pairs.reverse()
        for block, plate in pairs:
            R = quat_mat(env.object_pose(block)[3:])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            ok = yield from self.pick(block, 2 * BLOCK_HALF, GRASP_Z, yaw=yaw)
            if not ok:
                return
            hold = self.tcp()[2] - env.object_pos(block)[2]
            target = env.object_pos(plate)[:2] + self.held_offset(block)
            top = env.object_pos(plate)[2] - env._extent[plate]["bottom"] + env.height(plate)
            yield from self.place(target, top + BLOCK_HALF + 0.008 + hold)
        yield from self.rest()


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)

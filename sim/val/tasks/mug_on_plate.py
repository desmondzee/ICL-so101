"""Mug on saucer: pick up the red mug and set it upright on the named small plate (two plates, two mugs).

Scene: the red coffee mug (target), a white-and-yellow mug (confounder, never named), two small plates (0.6x LIBERO
plate, one untinted white, one tinted blue) and 2-4 short distractors. The instruction names one plate per seed:
"Put the red mug on the blue plate." / "Put the red mug on the white plate."

The red mug is the narrowest LIBERO mug (about 3.6-4.2 cm across at 0.5x, 6.6 cm tall), so the open gripper
(6.7 cm jaw gap at 0.70 rad) closes around its body with room to spare. The oracle grasps it top-down by the body,
closing perpendicular to the handle, about 3 cm below the rim, and carries it at 11 cm so the hanging mug clears the
confounder mug and every distractor in the pool.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim.val.env import ValEnv
from sim.val.oracle import CLOSED, OPEN, RELEASE, Oracle
from sim.val.scene import Obj, SceneSpec

MUG, OTHER_MUG = "red_mug", "yellow_mug"
PLATE_SCALE = 0.6
PLATE_TINT = {"white": None, "blue": (0.42, 0.58, 1.0, 1.0)}
PLATES = {c: f"{c}_plate" for c in PLATE_TINT}
# Red coffee mug at 0.5x, in its body frame: the cylinder axis sits here (the handle points along +x).
MUG_AXIS = np.array([-0.012, -0.001])
MUG_WIDTH = 0.041          # body diameter across the handle axis at the grasp height
GRASP_Z = 0.036            # TCP height above the mug bottom at the grasp (rim at 0.066)
LIFT_Z = 0.11              # carry height of the TCP
PRE_REST = (0.16, 0.0)     # TCP xy to return through before the rest move
RELEASE_GAP = 0.005        # mug bottom above the plate surface at release
FINGER_CLEARANCE = 0.02   # around the mug, for the open moving jaw
UPRIGHT_COS = np.cos(np.radians(12))
DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn", "ramekin",
               "akita_black_bowl", "white_bowl")
MUG_REGION = dict(r=(0.14, 0.26), angle=(-60, 60))
PLATE_REGION = dict(r=(0.14, 0.26), angle=(-62, 62))
OTHER_REGION = dict(r=(0.13, 0.30), angle=(-65, 65))


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


class MugOnPlateEnv(ValEnv):
    instruction = "Put the red mug on the blue plate."
    n_distractors = (2, 4)

    def make_scene(self):
        objects = [Obj(MUG, "red_coffee_mug"), Obj(OTHER_MUG, "white_yellow_mug")]
        for c, tint in PLATE_TINT.items():
            objects.append(Obj(PLATES[c], "plate", PLATE_SCALE, rgba=tint, mass=0.12))
        return SceneSpec(arena="kitchen", objects=objects, distractors=[Obj(n, n) for n in DISTRACTORS])

    def layout(self):
        colours = list(PLATE_TINT)
        self.target_colour = colours[int(self.np_random.integers(len(colours)))]
        self.target_plate = PLATES[self.target_colour]
        self.instruction = f"Put the red mug on the {self.target_colour} plate."
        placed = []
        names = [MUG, *PLATES.values(), OTHER_MUG]
        regions = {MUG: MUG_REGION, OTHER_MUG: OTHER_REGION}
        for i in self.np_random.permutation(len(names)):
            n = names[i]
            if n == MUG:
                radius = self.footprint(n) + FINGER_CLEARANCE
            elif n in PLATES.values():
                radius = self.plate_radius(n)
            else:
                radius = self.footprint(n)
            xy = self.sample_xy(radius, placed, **regions.get(n, PLATE_REGION), clearance=0.015)
            placed.append((xy, radius))
            yaw = self.np_random.uniform(-np.pi, np.pi)
            if n == MUG:  # `xy` is where the cylinder axis goes; shift the body origin accordingly
                c, s = np.cos(yaw), np.sin(yaw)
                xy = xy - np.array([[c, -s], [s, c]]) @ MUG_AXIS
            self.set_object_pose(n, xy, yaw=yaw)
        self.place_distractors(placed)

    # ----- geometry helpers -----------------------------------------------------------------------------------
    def mug_axis(self):
        """World xy of the red mug's cylinder axis and its body rotation."""
        p = self.object_pose(MUG)
        R = quat_mat(p[3:])
        return p[:2] + R[:2, :2] @ MUG_AXIS, R

    def mug_bottom(self):
        return float(self.object_pos(MUG)[2] - self._extent[MUG]["bottom"])

    def plate_radius(self, plate):
        return self.footprint(plate) / np.sqrt(2)  # footprint is the AABB corner of a round plate

    def plate_surface(self, plate):
        """Height of the plate's top surface at its centre."""
        if not hasattr(self, "_surface"):
            self._surface = {n: self._measure_surface(n) for n in PLATES.values()}
        p = self.object_pose(plate)
        return float(p[2] + quat_mat(p[3:])[2, 2] * self._surface[plate])

    def _measure_surface(self, plate):
        """Plate-frame z of the collision surface at the plate centre, by a ray cast on a scratch copy with the
        plate lifted clear of everything else."""
        d = mujoco.MjData(self.model)
        a = self._qadr[plate]
        d.qpos[a : a + 7] = [0, 0, 5.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(self.model, d)
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, d, np.array([0, 0, 6.0]), np.array([0, 0, -1.0]), np.array([0, 0, 0, 1, 0, 0], np.uint8), 1, -1, gid)
        assert dist > 0 and gid[0] in self._geoms[plate], "plate surface ray missed"
        return float(6.0 - dist - 5.0)

    # ----- success --------------------------------------------------------------------------------------------
    def mug_on(self, plate):
        axis, R = self.mug_axis()
        centre = self.object_pos(plate)[:2]
        inside = np.linalg.norm(axis - centre) < 0.7 * self.plate_radius(plate)
        upright = R[2, 2] > UPRIGHT_COS
        resting = self.touching(MUG, plate) and abs(self.mug_bottom() - self.plate_surface(plate)) < 0.006
        v = self.data.qvel[self._dadr[MUG] : self._dadr[MUG] + 6]
        still = np.linalg.norm(v[:3]) < 0.03 and np.linalg.norm(v[3:]) < 0.5
        return bool(inside and upright and resting and still and not self.is_grasping(MUG))

    def success(self):
        return self.mug_on(self.target_plate)

    def task_info(self):
        return {"mug_upright": bool(self.mug_axis()[1][2, 2] > UPRIGHT_COS),
                **{f"mug_on_{c}_plate": self.mug_on(p) for c, p in PLATES.items()}}


class MugOnPlateOracle(Oracle):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng or np.random.default_rng()

    def pick_mug(self, attempts=3, approach=0.05):
        """Top-down body grasp of the red mug, closing perpendicular to the handle, lifted to LIFT_Z.
        Same as `Oracle.pick` but with the closing directions restricted to the two that avoid the handle and the
        grasp centred on the cylinder axis rather than the body origin."""
        env = self.env
        for attempt in range(attempts):
            axis, R = env.mug_axis()
            handle = np.arctan2(R[1, 0], R[0, 0])
            centre = np.array([axis[0], axis[1], env.mug_bottom() + GRASP_Z])
            rot, grasp, q = self.grasp_plan(centre, MUG_WIDTH, [handle + np.pi / 2, handle - np.pi / 2])
            yield from self.gripper(OPEN, 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.08, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, 0.5, 0.3)
            ok = env.is_grasping(MUG)
            self.log.append((MUG, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], LIFT_Z]), rot, speed=0.12, label="lift")
                if env.is_grasping(MUG):
                    return True
                self.log.append((MUG, "dropped", attempt))
            yield from self.gripper(OPEN, 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def place_mug(self, plate):
        env = self.env
        axis, _ = env.mug_axis()
        offset = self.tcp()[:2] - axis
        hold = self.tcp()[2] - env.mug_bottom()
        xy = env.object_pos(plate)[:2] + offset
        release_z = env.plate_surface(plate) + RELEASE_GAP + hold
        rot = self.carry_rot(xy)
        yield from self.move(np.array([xy[0], xy[1], LIFT_Z]), rot, label="carry")
        yield from self.move(np.array([xy[0], xy[1], release_z + 0.03]), rot, speed=0.08, label="lower")
        yield from self.move(np.array([xy[0], xy[1], release_z]), rot, speed=0.03, label="set down")
        yield from self.gripper(RELEASE, 0.5, 0.3)
        yield from self.move(np.array([xy[0], xy[1], LIFT_Z]), rot, speed=0.08, label="retreat")

    def plan(self):
        ok = yield from self.pick_mug()
        if not ok:
            return
        yield from self.place_mug(self.env.target_plate)
        # Swing back to the front of the base while high, so the joint-space rest move does not sweep the arm
        # through a tall mug standing on a plate near the base.
        pre_rest = np.array([PRE_REST[0], PRE_REST[1], LIFT_Z + 0.01])
        yield from self.transit(pre_rest, self.carry_rot(pre_rest[:2]), speed=1.2, label="pre-rest")
        yield from self.rest()

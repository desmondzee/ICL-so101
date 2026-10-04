"""Stack bowls: put the white bowl in the black bowl.

The white LIBERO bowl (1.0x scale, 7.2 cm across, 3.4 cm tall) is picked by a top-down rim grasp on the side of
the bowl nearest the robot, with the closing direction radial so the jaws pinch the wall. It is carried over the
black akita bowl (1.0x, 9.4 cm across, 5.1 cm tall) and released with the fingertips just above the black rim, so it
drops the last 2-3 cm and nests. Both bowls are 2x the project's 0.5x LIBERO convention so the stacking reads clearly
from the front camera and the gripper has a whole wall to pinch.

Success: the white bowl rests centred and seated inside the black bowl (origin height within 8 mm of the measured
nested height), both upright, touching, still, and not grasped. `task_info` also reports whether any distractor moved.
"""

from __future__ import annotations

import numpy as np

from sim.val.env import ValEnv
from sim.val.oracle import CARRY_Z, OPEN, Oracle
from sim.val.scene import Obj, SceneSpec

INNER, OUTER = "white_bowl", "black_bowl"
INNER_SCALE, OUTER_SCALE = 1.0, 1.0
INNER_WALL_R = 0.0357 * INNER_SCALE     # wall-centre radius of the white bowl at mid-wall height (collision boxes)
OUTER_RIM_R = 0.047 * OUTER_SCALE       # black bowl wall-centre radius near the rim
NEST_DZ = 0.0063                        # white-bowl origin above black-bowl origin when nested (measured)
WALL = 0.004                            # width passed to the grasp planner (wall plus margin)
GRASP_Z = 0.021                         # TCP height for the rim grasp; fingertips about 1.6 cm below the rim
RIM_CLEAR = 0.012                       # TCP height above the black rim at release
FINGER_CLEARANCE = 0.035                # room for the outer finger beside the white bowl
DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn")   # all <= 4 cm tall: the hanging bowl clears them
INNER_REGION = dict(r=(0.17, 0.27), angle=(-55, 55))
OUTER_REGION = dict(r=(0.15, 0.27), angle=(-55, 55))


class StackBowlsEnv(ValEnv):
    instruction = "Put the white bowl in the black bowl."
    n_distractors = (2, 4)
    # The base keepouts plus the wrist-camera mount, which at the rest pose hangs 4-8 cm above the table at
    # x 0.12-0.17, y 0.04-0.08 and would rest on a 5 cm tall bowl rim.
    keepout = (*ValEnv.keepout, (np.array([0.142, 0.06]), 0.032))

    def make_scene(self):
        objects = [Obj(INNER, "white_bowl", INNER_SCALE, mass=0.04), Obj(OUTER, "akita_black_bowl", OUTER_SCALE, mass=0.12)]
        return SceneSpec(arena="kitchen", objects=objects, distractors=[Obj(n, n) for n in DISTRACTORS])

    def bowl_radius(self, name):
        """True xy radius of a bowl (the base footprint() is the bounding-box corner radius, sqrt(2) too large)."""
        return self.footprint(name) / np.sqrt(2)

    def layout(self):
        placed = []
        spots = [(INNER, INNER_REGION, FINGER_CLEARANCE), (OUTER, OUTER_REGION, 0.0)]
        if self.np_random.random() < 0.5:
            spots.reverse()
        for n, region, extra in spots:
            radius = self.bowl_radius(n) + extra
            xy = self.sample_xy(radius, placed, **region, clearance=0.02)
            placed.append((xy, radius))
            self.set_object_pose(n, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        self.place_distractors(placed)

    def reset(self, *args, **kwargs):
        self._start = {}
        out = super().reset(*args, **kwargs)
        self._start = {n: self.object_pos(n) for n in self.active_distractors}
        return out

    def upright(self, name):
        q = self.object_pose(name)[3:]
        return 1 - 2 * (q[1] ** 2 + q[2] ** 2)          # z-axis . world z

    def speed(self, name):
        return float(np.linalg.norm(self.data.qvel[self._dadr[name] : self._dadr[name] + 3]))

    def nested(self):
        i, o = self.object_pos(INNER), self.object_pos(OUTER)
        centred = np.linalg.norm(i[:2] - o[:2]) < 0.4 * OUTER_RIM_R
        seated = abs((i[2] - o[2]) - NEST_DZ) < 0.008
        upright = self.upright(INNER) > 0.97 and self.upright(OUTER) > 0.97
        still = self.speed(INNER) < 0.03 and self.speed(OUTER) < 0.03
        return bool(centred and seated and upright and still and self.touching(INNER, OUTER) and not self.is_grasping(INNER))

    def success(self):
        return self.nested()

    def distractors_moved(self):
        """Largest xy displacement of an active distractor since reset (m)."""
        start = getattr(self, "_start", {})
        return max([float(np.linalg.norm(self.object_pos(n)[:2] - p[:2])) for n, p in start.items()] or [0.0])

    def task_info(self):
        return {"white_bowl_in_black_bowl": self.nested(), "distractors_undisturbed": self.distractors_moved() < 0.01}


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class StackBowlsOracle(Oracle):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng or np.random.default_rng()

    def pick_rim(self, name, attempts=3, approach=0.05):
        """Top-down pinch of the bowl wall at the point nearest the robot (closing direction radial), lifted to
        CARRY_Z. Both jaw orientations (moving jaw inside or outside the bowl) are considered."""
        env = self.env
        for attempt in range(attempts):
            c = env.object_pos(name)
            u = c[:2] / np.linalg.norm(c[:2])
            jitter = self.rng.uniform(-0.25, 0.25)                     # grasp a little left/right of the nearest point
            u = _rotz(jitter)[:2, :2] @ u
            p = c[:2] - INNER_WALL_R * u
            az = np.arctan2(u[1], u[0])
            rot, grasp, q = self.grasp_plan(np.array([p[0], p[1], GRASP_Z]), WALL, [az, az + np.pi])
            yield from self.gripper(OPEN, 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.08, tol=0.003, label="grasp")
            yield from self.gripper(-0.17, 0.5, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], CARRY_Z]), rot, speed=0.10, label="lift")
                yield from self.wait(0.5)                                  # let the bowl settle in the pinch
                if env.is_grasping(name):
                    # Bowl origin in the TCP frame. The pinched bowl moves rigidly with the gripper, which tilts by up to
                    # ~25 deg where vertical is infeasible (ROT_WEIGHT), so the place target is solved through the IK.
                    self.rel = self.tcp_rot().T @ (env.object_pos(name) - self.tcp())
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(OPEN, 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def arc_move(self, pos, speed=0.16, settle=0.6, label="arc"):
        """Move of the TCP to `pos` along an arc around the shoulder-pan axis (radius, azimuth and height
        interpolated, `Oracle.move(arc=True)`) with the orientation turned with the azimuth. A straight Cartesian carry
        between the left and right of the table passes close to the base, where the arm sags and the hanging bowl hits
        whatever is below."""
        yield from self.move(pos, self.carry_rot(pos[:2]), speed=speed, settle=settle, label=label, arc=True)

    def predict(self, pos, rot):
        """Where the held object's origin ends up if the TCP is commanded to `pos`/`rot`: the IK compromise tilts
        the gripper (ROT_WEIGHT), and the pinched bowl tilts rigidly with it."""
        q = self.ik(pos, rot)
        tcp = self._fk(q)
        r = self.ik_data.site_xmat[self.env._tcp_site_id].reshape(3, 3)
        return tcp + r @ self.rel

    def place_target(self, obj_xy, z):
        """TCP position and orientation that put the held object's origin at (`obj_xy`, `z`) after the
        azimuth-following carry, iterating through the arm's actual (IK) pose."""
        goal = np.array([obj_xy[0], obj_xy[1], z])
        pos = goal.copy()
        for _ in range(8):
            pos = pos + goal - self.predict(pos, self.carry_rot(pos[:2]))
        return pos, self.carry_rot(pos[:2])

    def plan(self):
        env = self.env
        for _ in range(2):                                           # one re-pick if the bowl slips out in the carry
            ok = yield from self.pick_rim(INNER)
            if not ok:
                return
            target = env.object_pos(OUTER)
            outer_top = target[2] - env._extent[OUTER]["bottom"] + env.height(OUTER)
            # The finger outside the white bowl cannot fit between the two walls, so release with the fingertips
            # just above the black bowl's rim and let the white bowl drop the last 2-3 cm into it.
            pos, rot = self.place_target(target[:2], target[2] + NEST_DZ + 0.010)
            if pos[2] < outer_top + RIM_CLEAR:
                pos, rot = self.place_target(target[:2], target[2] + NEST_DZ + 0.010 + outer_top + RIM_CLEAR - pos[2])
            yield from self.arc_move(np.array([pos[0], pos[1], CARRY_Z]), label="carry")
            if env.is_grasping(INNER):
                break
            self.log.append((INNER, "dropped in carry"))
            yield from self.gripper(OPEN, 0.4, 0.2)
        else:
            return
        yield from self.move(pos, rot, speed=0.06, label="lower")
        yield from self.gripper(0.55, 0.4, 0.4)
        yield from self.move(np.array([pos[0], pos[1], CARRY_Z]), rot, speed=0.10, label="retreat")
        yield from self.rest()

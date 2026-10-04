"""Task: put two mugs, one after the other, into an open microwave.

Geometry (all in the robot base frame):
- The LIBERO microwave is a static fixture at 1.0x (cavity 21 x 18 x 14 cm, 35 cm wide outside), on the robot's
  left at (0.19, 0.27) with yaw 0, so its opening faces -y: the front camera (x = 0.48, y = -0.13) looks into the
  cavity about 40 deg off its axis, and the arm reaches in about 15-40 deg off its axis. The door hinge is on the
  side nearer the robot, so the door is swung almost flat (172 deg) to lie behind the robot; its joint range is
  pinned there. A top-down grasp can't work inside a 14 cm cavity, so the mugs are carried with a side grasp.
- The mugs are small (red mug 0.32x, 4.2 cm tall; white/yellow mug 0.4x, 4.1 cm tall) and are set down side by
  side on the cavity floor, 4.5 cm inside the front, 10 cm apart, inside the `microwave_heating_region` site.

Oracle (side grasp): the jaws close horizontally and the approach axis points 35 deg below horizontal along the
arm's plane. That plane turns about the shoulder-pan axis (x = 3.9 cm), and the TCP sits 1.35 cm to the side of
it, so the approach azimuth is computed from the pan axis (`approach_azimuth`); a plain radial approach costs
5-10 mm of IK error. Each mug is picked from behind with the handle pointing away from the robot (yaw spread
+-15 deg), lifted, swung to a point 6 cm in front of its slot (an arc around the pan axis), slid in along the
microwave's axis on a slight slope (clearing the front lip), pressed onto the cavity floor, released by opening the
jaw a little, backed out horizontally along the approach axis (so the fixed finger clears the rim before
any sideways motion), and finally pulled out along the microwave's axis. At 35 deg pitch the wrist-camera mount
clears the cavity's top edge by about 1 cm. Which mug goes first is random per seed (the oracle's RNG), but each mug
has its own slot: the red one on the panel side, the white one on the hinge side.
"""

from __future__ import annotations

import numpy as np

from sim.val.env import ValEnv
from sim.val.oracle import CLOSED, Oracle
from sim.val.scene import Fixture, Obj, SceneSpec

MW = "microwave"
MW_SCALE = 1.0
MW_XY = (0.19, 0.27)
MW_YAW = 0.0  # the opening faces -y
DOOR_Q = -3.0  # rad; LIBERO's range is [-2.094, 0], widened and pinned here so the door lies behind the robot
FLOOR_Z = 0.024  # top of the cavity floor at scale 1 (asset units)
# name: (asset, scale, body radius in m). The red mug is shrunk so both mugs are ~4 cm tall: a taller mug wedges
# its rim under the gripper body.
MUGS = {"red_mug": ("red_coffee_mug", 0.32, 0.012), "white_mug": ("white_yellow_mug", 0.4, 0.015)}
SLOTS = ((0.025, -0.065), (-0.075, -0.065))  # mug centres in the microwave frame (asset units); [0] is the panel side
DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn", "ramekin")
MUG_REGION = dict(r=(0.23, 0.30), angle=(-50, 20))
HANDLE_SPREAD = np.radians(15)  # handle yaw around "away from the robot along the approach"
MUG_TORSION = 0.1  # torsional friction on the mugs (soft-pad grip), so a mug doesn't swing about the jaw axis

PITCH = np.radians(35)
PAN_XY = (0.0388, 0.0)  # shoulder-pan axis in the base frame
LATERAL = 0.0135  # TCP offset from the arm plane when the jaws close horizontally
OPEN_SIDE = 0.55  # rad, about a 6 cm jaw gap for a 3 cm mug
FINGER_FACE = 0.0193  # inner face of the fixed finger along the TCP x axis
FINGER_GAP = 0.0035  # finger-to-mug gap on arrival, so closing barely slides (and tips) the mug
GRASP_H = 0.022  # TCP height above the mug bottom
OUT_DIST = 0.06  # pull-out distance along the microwave axis
RELEASE_DELTA = 0.3  # rad opened from the measured grip on release
LIFT_OFF = 0.006
BACK_DIST = 0.035
LIP_CLEAR = 0.006  # extra TCP height at the start of the slide-in
SET_PRESS = 0.004  # how far the TCP goes on down once the mug touches the floor (as the original fixed set height did)
UPRIGHT_COS = np.cos(np.radians(15))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def approach_azimuth(p):
    """Azimuth of the side approach that reaches `p`: the arm plane turns about the shoulder-pan axis, and the TCP
    sits LATERAL to the side of it when the jaws close horizontally."""
    dx, dy = p[0] - PAN_XY[0], p[1] - PAN_XY[1]
    return np.arctan2(dy, dx) + np.arcsin(min(LATERAL / np.hypot(dx, dy), 0.9))


def side_rot(a, pitch=PITCH):
    """TCP rotation: approach (z) along azimuth `a` tilted down by `pitch`; jaws close horizontally (x), and the
    wrist-camera mount (+y) points up."""
    z = np.array([np.cos(a) * np.cos(pitch), np.sin(a) * np.cos(pitch), -np.sin(pitch)])
    x = np.array([-np.sin(a), np.cos(a), 0.0])
    return np.stack([x, np.cross(z, x), z], 1)


def side_frame(p):
    return side_rot(approach_azimuth(p))


class MugsInMicrowaveEnv(ValEnv):
    instruction = "Put both mugs in the microwave."
    mugs = tuple(MUGS)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        m = self.model
        self._door_j = m.joint(f"{MW}_microjoint").id
        m.jnt_range[self._door_j] = [DOOR_Q - 0.02, DOOR_Q + 0.02]
        self._heat = m.site(f"{MW}_heating_region").id
        for n in self.mugs:
            m.geom_friction[self._geoms[n], 1] = MUG_TORSION

    def make_scene(self):
        return SceneSpec(arena="kitchen", objects=[Obj(n, a, s) for n, (a, s, _) in MUGS.items()],
                         fixtures=[Fixture(MW, "microwave", MW_SCALE)], distractors=[Obj(n, n) for n in DISTRACTORS])

    # ----- microwave geometry ------------------------------------------------------------------------------
    def mw_to_world(self, local_xy):
        return np.asarray(MW_XY) + rot_z(MW_YAW)[:2, :2] @ (np.asarray(local_xy, float) * MW_SCALE)

    def slot_xy(self, i):
        return self.mw_to_world(SLOTS[i])

    def into_cavity(self):
        """Unit vector from the opening into the cavity (the microwave's local +y)."""
        return np.array([-np.sin(MW_YAW), np.cos(MW_YAW), 0.0])

    def floor_z(self):
        return FLOOR_Z * MW_SCALE

    # ----- layout and success ------------------------------------------------------------------------------
    def layout(self):
        self.set_fixture_pose(MW, MW_XY, yaw=MW_YAW)
        self.data.qpos[self.model.jnt_qposadr[self._door_j]] = DOOR_Q
        # keep-outs: the microwave body, and the strip in front of the opening that the arm sweeps
        placed = [(self.mw_to_world((x, 0.0)), 0.095) for x in (-0.11, 0.0, 0.11)]
        placed += [(self.mw_to_world((x, -0.2)), 0.06) for x in (-0.09, 0.01)]
        for n in self.mugs:
            radius = self.footprint(n) + 0.03
            xy = self.sample_xy(radius, placed, **MUG_REGION, clearance=0.02)
            placed.append((xy, radius))
            # handle (body +x) points away from the robot along the approach, +-HANDLE_SPREAD
            self.set_object_pose(n, xy, yaw=approach_azimuth(xy) + self.np_random.uniform(-HANDLE_SPREAD, HANDLE_SPREAD))
        self.place_distractors(placed)

    def in_microwave(self, n):
        """Mug origin inside the heating region (xy), resting upright on the cavity floor, still, not held."""
        p = self.object_pos(n)
        c, R, h = self.data.site_xpos[self._heat], self.data.site_xmat[self._heat].reshape(3, 3), self.model.site_size[self._heat]
        local = R.T @ (p - c)
        inside = bool(np.all(np.abs(local[:2]) <= h[:2]))
        on_floor = self.floor_z() - 0.005 < p[2] < self.floor_z() + 0.01
        upright = self.data.xmat[self._body[n]][8] > UPRIGHT_COS
        still = np.linalg.norm(self.data.qvel[self._dadr[n]: self._dadr[n] + 3]) < 0.05
        return bool(inside and on_floor and upright and still and not self.is_grasping(n))

    def success(self):
        return all(self.in_microwave(n) for n in self.mugs)

    def task_info(self):
        return {f"{n}_in_microwave": self.in_microwave(n) for n in self.mugs}


class MugsInMicrowaveOracle(Oracle):
    rest_unwind = False  # the side-grasp fingers end 6 cm in front of the mugs: unwinding the roll there drags them out

    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng or np.random.default_rng()

    def side_pick(self, name, back=0.045, attempts=2):
        """Side grasp from behind the mug at GRASP_H, then lift 3 cm. Returns True if held."""
        env = self.env
        for attempt in range(attempts):
            obj = env.object_pos(name)
            c = np.array([obj[0], obj[1], obj[2] - env._extent[name]["bottom"] + GRASP_H])
            rot = side_frame(c)
            g = c - (FINGER_FACE - MUGS[name][2] - FINGER_GAP) * rot[:, 0] + 0.004 * rot[:, 2]
            rot = side_frame(g)
            pre = g - back * rot[:, 2]
            yield from self.gripper(OPEN_SIDE, 0.4, 0.0)
            yield from self.transit(pre + [0, 0, 0.06], rot, label="above")
            yield from self.move(pre, rot, speed=0.1, label="pregrasp")
            yield from self.move(g, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, 0.5, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(g + [0, 0, 0.03], rot, label="lift")
                if env.is_grasping(name):
                    return True
            yield from self.gripper(OPEN_SIDE, 0.4, 0.2)
            yield from self.move(pre + [0, 0, 0.05], rot, label="back off")
        return False

    def insert(self, name, slot_xy):
        """Carry the held mug in front of `slot_xy`, slide it in along the microwave axis, set it down, back out."""
        env = self.env
        held = self._cmd_rot.T @ (self.tcp() - env.object_pos(name))  # TCP in the mug frame, kept while turning
        mug = np.array([slot_xy[0], slot_xy[1], env.floor_z() + env._extent[name]["bottom"] + 0.005])
        rot = side_frame(mug)
        for _ in range(2):
            rot = side_frame(mug + rot @ held)
        tgt = mug + rot @ held
        out = tgt - OUT_DIST * env.into_cavity()
        yield from self.move(self._cmd_pos + [0, 0, 0.05], self._cmd_rot, label="raise")
        yield from self.move(out + [0, 0, 0.035], rot, label="swing")
        # The mug slides a few mm down in the jaws on the way. Slide in on a slope, LIP_CLEAR higher at the front, so
        # its bottom clears the cavity's front lip; the end height is unchanged (the wrist-camera mount clears the
        # cavity's top edge by only about 1 cm there).
        yield from self.move(out + [0, 0, 0.004 + LIP_CLEAR], rot, speed=0.08, label="front")
        yield from self.move(tgt + [0, 0, 0.004], rot, speed=0.06, label="in")
        # Set the mug down on the cavity floor and press it SET_PRESS further, wherever it hangs in the jaws: a mug
        # released above the floor, slightly tilted, tips or is pushed by the opening jaw.
        gap = max(env.object_pos(name)[2] - env._extent[name]["bottom"] - env.floor_z(), 0.0)
        tgt = np.array([tgt[0], tgt[1], min(tgt[2], self._cmd_pos[2] - gap - SET_PRESS)])
        yield from self.move(tgt, rot, speed=0.04, label="set")
        grip = float(env._get_current_qpos()[5])
        yield from self.gripper(grip + RELEASE_DELTA, 0.4, 0.3)
        yield from self.move(tgt + [0, 0, LIFT_OFF], rot, speed=0.03, label="lift off")
        flat = np.r_[rot[:2, 2], 0.0] / np.linalg.norm(rot[:2, 2])
        back = tgt - BACK_DIST * flat + 0.004 * rot[:, 0] + [0, 0, LIFT_OFF]
        yield from self.move(back, rot, speed=0.05, label="back")
        yield from self.move(np.r_[out[:2], back[2]], rot, speed=0.08, label="out")
        yield from self.move(out + [0, 0, 0.06], rot, label="up")

    def plan(self):
        mugs = list(self.env.mugs)
        if self.rng.random() < 0.5:
            mugs.reverse()
        for n in mugs:  # the red mug always takes the panel-side slot, the white one the hinge-side slot
            ok = yield from self.side_pick(n)
            if not ok:
                return
            yield from self.insert(n, self.env.slot_xy(self.env.mugs.index(n)))
        yield from self.rest()

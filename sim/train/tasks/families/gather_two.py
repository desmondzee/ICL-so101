"""gather_two: put two named objects into the same container (or onto the same surface), in a stated order.

Every task places object A first and object B second into two side-by-side slots of one receptacle. The
slots lie along the receptacle's long axis (roughly tangential to the robot, so both stay inside the
reach band), and each object is released with the jaws closing across that axis, so the fingers never
reach toward the object already placed. The strict success predicate is the kit's: both objects settled,
released and supported by the receptacle at their slots, plus ``OrderedCompletion`` (A is grasped and
placed before B is first grasped).

Receptacles: the LIBERO plate (1.0x, static), and two open wooden receptacles built here from boxes
(``Crate``: a low-rimmed tray and a deeper box), because the LIBERO bowls/plate at a reachable scale
have flat floors only ~5-6 cm across and the 0.5x basket is 7 cm deep (measured with ``surface_z``).
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from sim.train.tasks.base import COS10, TrainEnv, TrainOracle, define_task, step_text, then
from sim.train.tasks.base import CLOSED
from sim.val.scene import Block, Fixture, Obj

WOOD = (0.62, 0.43, 0.25, 1.0)
DARK_WOOD = (0.42, 0.28, 0.17, 1.0)


@dataclass
class Crate:
    """An open-top rectangular receptacle (free body) made of a floor slab and four walls.

    ``inner``: interior half-extents (x, y) in m; ``wall``: wall height above the floor top. The body
    origin is at the centre of the floor's underside, so the interior floor is at ``floor`` above it.
    Built through the scene's ``build_mjcf`` extension point (the same hook ``Scanned`` uses)."""

    name: str
    inner: tuple = (0.05, 0.035)
    wall: float = 0.02
    thickness: float = 0.006
    floor: float = 0.006
    rgba: tuple = WOOD
    mass: float = 0.4
    friction: float = 1.0

    def boxes(self):
        ix, iy = self.inner
        t, f, h = self.thickness, self.floor, self.wall
        zc, zh = (f + h) / 2, (f + h) / 2
        return [((0, 0, f / 2), (ix + t, iy + t, f / 2)),
                ((ix + t / 2, 0, zc), (t / 2, iy + t, zh)), ((-ix - t / 2, 0, zc), (t / 2, iy + t, zh)),
                ((0, iy + t / 2, zc), (ix, t / 2, zh)), ((0, -iy - t / 2, zc), (ix, t / 2, zh))]

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        boxes = self.boxes()
        volumes = [8 * np.prod(h) for _, h in boxes]
        for k, ((pos, half), vol) in enumerate(zip(boxes, volumes)):
            ET.SubElement(body, "geom", name=f"{self.name}_geom{k}", type="box", pos=" ".join(f"{v:.6g}" for v in pos),
                          size=" ".join(f"{v:.6g}" for v in half), rgba=" ".join(map(str, self.rgba)),
                          mass=f"{self.mass * vol / sum(volumes):.6g}", group="1", condim="4",
                          friction=f"{self.friction} 0.02 0.001", material="val_fabric")


@dataclass
class SmoothCan:
    """A LIBERO can (``alphabet_soup``/``tomato_sauce``, 0.5x) with its visual mesh but one smooth cylinder for
    collision. The LIBERO collision model is a ring of 21 thin boxes (plates ~1.8 mm thick); the jaw pads
    lodge between plates (3 mm penetration while carried) and the can pivots 4-5 deg in the jaws, so it
    touched down on an edge and snapped flat (1200-7600 rad/s^2, 5/10 calibration seeds). Built through the
    scene's ``build_mjcf`` extension point; mass and friction match the LIBERO object."""

    name: str
    asset: str
    radius: float = 0.0155
    half_height: float = 0.019
    mass: float = 0.032
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        from sim.val import scene
        scene._free_body(mj, asset_el, worldbody, Obj(self.name, self.asset))
        body = next(b for b in worldbody.iter("body") if b.get("name") == self.name)
        for parent in list(body.iter()):
            for geom in list(parent):
                if geom.tag == "geom" and geom.get("group") == "3":
                    parent.remove(geom)
        ET.SubElement(body, "geom", name=f"{self.name}_cyl", type="cylinder",
                      size=f"{self.radius:.6g} {self.half_height:.6g}", mass=f"{self.mass:.6g}", group="3",
                      condim="4", friction=f"{self.friction} 0.02 0.001", rgba="0 0 0 0")


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----- shared environment and oracle -------------------------------------------------------------------------

class GatherEnv(TrainEnv):
    """Two task objects (``task_objects`` in the required order) into slots of one receptacle.

    Subclasses set ``receptacle`` (body name), ``slot_offset`` (slot centre distance from the receptacle
    centre along its long axis), ``receptacle_region`` and ``tolerance``; ``receptacle_is_fixture`` for a
    static LIBERO receptacle. ``axis_yaw`` is the receptacle's long-axis yaw, ``slots`` the slot xy (in
    the order of ``task_objects``)."""

    receptacle = ""
    receptacle_is_fixture = False
    slot_offset = 0.03
    receptacle_region = dict(r=(0.18, 0.225), angle=(-50.0, 50.0))
    tolerance = (0.02, 0.02, 0.006)
    yaw_jitter = np.radians(15)
    slot_reach = (0.15, 0.262)          # every slot centre: radius band (m) about the robot base
    slot_angle = 66.0                   # and |azimuth| (deg)

    def _place_receptacle(self, placed):
        name = self.receptacle
        radius = self.receptacle_radius()
        for _ in range(100):
            xy = self.sample_xy(radius, placed, clearance=0.02, **self.receptacle_region)
            yaw = np.arctan2(xy[1], xy[0]) + np.pi / 2 + self.np_random.uniform(-self.yaw_jitter, self.yaw_jitter)
            axis = np.array([np.cos(yaw), np.sin(yaw)])
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            slots = [xy + sign * self.slot_offset * axis, xy - sign * self.slot_offset * axis]
            if all(self.slot_reach[0] <= np.hypot(*s) <= self.slot_reach[1]
                   and abs(np.degrees(np.arctan2(s[1], s[0]))) <= self.slot_angle for s in slots):
                break
        else:
            raise RuntimeError("no reachable receptacle pose")
        if self.receptacle_is_fixture:
            self.set_fixture_pose(name, xy, yaw=yaw)
        else:
            self.set_object_pose(name, xy, yaw=yaw)
        placed.append((xy, radius))
        self.axis_yaw = float(yaw)
        self.slots = tuple(np.asarray(s) for s in slots)
        return xy

    def receptacle_radius(self):
        if self.receptacle_is_fixture:
            return self.fixture_footprint(self.receptacle)
        return self.footprint(self.receptacle)

    # Elongated objects: start yaw of the long (local x) axis relative to tangential, so the turn from the
    # grasp (across the narrow side) to the release (jaws closing across the slot axis) stays within
    # ``max_cartesian_turn`` and the carry is Cartesian; a joint-space transit with a 90 deg wrist turn let
    # the pudding box slip out of the jaws (measured, seed 9001).
    long_axis_jitter = np.radians(40)
    elongated: tuple[str, ...] = ()

    def object_yaw(self, name):
        return self.np_random.uniform(-np.pi, np.pi)

    def orient(self, name):
        if name in self.elongated:
            xy = self.object_pos(name)[:2]
            yaw = (np.arctan2(xy[1], xy[0]) + np.pi / 2 + self.np_random.uniform(-1, 1) * self.long_axis_jitter
                   + np.pi * self.np_random.integers(2))
            self.set_object_pose(name, xy, yaw=yaw)

    def layout(self):
        placed = []
        self._place_receptacle(placed)
        for name in self.task_objects:
            self.place(name, placed, yaw=self.object_yaw(name))
            self.orient(name)
        self.place_distractors(placed)
        goals = []
        reference = None if self.receptacle_is_fixture else self.receptacle
        for name, slot in zip(self.task_objects, self.slots):
            floor = self.surface_z(slot, exclude=self.task_objects)
            goals.append(self.goal_for(name, (*slot, floor + self._extent[name]["bottom"]), reference))
        self.set_goals(*goals)

    def goal_for(self, name, target, reference):
        from sim.train.tasks.base import Goal
        return Goal(name, self.receptacle, target=target, reference=reference, tolerance=self.tolerance)


class GatherOracle(TrainOracle):
    """Pick each object in order and release it in its slot, jaws closing across the receptacle's long axis."""

    # Per object: (width across the jaws, symmetric, grasp height above the object's bottom or None for
    # the centre, yaw offset of the closing direction from the object's local x).
    grips = {}
    # Optional gentleness (per task): ``squeeze`` eases the jaw to that much below the object width before
    # opening (the stored grip force otherwise springs a light object off the fixed finger -- the remedy the
    # put_in_container family found), ``hold_squeeze`` closes only that far below the width instead of to
    # CLOSED, so the PD grip force (and the jaw's penetration into a can's hull) stays bounded.
    squeeze = None
    relax_seconds = 0.6
    hold_squeeze = None
    lift_fractions = {}         # per-object override of the kit's ``release_lift_fraction``
    release_drop = 0.0          # release height above the goal (negative: settle into contact first)
    release_margin = 0.016      # wider than the kit's 8 mm: after a low grasp the pivoting moving jaw (and
                                # the 4 mm back-off toward it) clipped the released object (release spikes)

    def grasp_spec(self, name):
        return self.grips[name]

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """The kit's top-down pick, closing to ``hold_squeeze`` below ``width`` when set."""
        self._held_width = width
        close = CLOSED if self.hold_squeeze is None else self.gap_angle(width - self.hold_squeeze)
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
            yield from self.gripper(close, self.close_seconds, 0.3)
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

    def release(self, open_to=None):
        width = getattr(self, "_held_width", None)
        if width is not None and self.squeeze is not None:
            yield from self.gripper(self.gap_angle(width - self.squeeze), self.relax_seconds, 0.2)
        yield from super().release(open_to)

    def plan(self):
        env = self.env
        for goal in env.goals:
            name = goal.obj
            width, symmetric, height, yaw_offset = self.grasp_spec(name)
            pos = env.object_pos(name)
            grasp_z = pos[2] - 0.001 if height is None else pos[2] - env._extent[name]["bottom"] + height
            # yaw_offset None: round object, grasp with the jaws closing radially (the least wrist turn).
            yaw = None if yaw_offset is None else env.yaw(name) + yaw_offset
            ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=symmetric)
            if not ok:
                return
            target = env.goal_target(goal)
            current = np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
            closing = [env.axis_yaw + np.pi / 2, env.axis_yaw - np.pi / 2]
            # Rotate the held object with the TCP: the command rotation turned about z so the closing
            # direction (TCP x) ends perpendicular to the slot axis. Least turn first.
            deltas = sorted((np.angle(np.exp(1j * (c - current))) for c in closing), key=abs)
            rotations = [_rot_z(d) @ self._cmd_rot for d in deltas]
            rot = self.feasible_rotation(name, target, rotations)
            if name in self.lift_fractions:
                self.release_lift_fraction = self.lift_fractions[name]
            yield from self.place_object(name, target, rot=rot, drop=self.release_drop,
                                         open_to=self.release_for(width))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- task 1: two cans into a wooden box --------------------------------------------------------------------

class CansInBoxEnv(GatherEnv):
    instruction = "Put the alphabet soup can in the wooden box first, then the tomato sauce can."
    task_objects = ("alphabet_soup", "tomato_sauce")
    order = ("alphabet_soup", "tomato_sauce")
    receptacle = "wooden_box"
    slot_offset = 0.026

    def scene_objects(self):
        return [SmoothCan("alphabet_soup", "alphabet_soup"), SmoothCan("tomato_sauce", "tomato_sauce"),
                Crate("wooden_box", inner=(0.056, 0.045), wall=0.022, rgba=DARK_WOOD, mass=0.5)]


class CansInBoxOracle(GatherOracle):
    squeeze = 0.006
    # A low-grasped can rides ~3.5 deg tilted in the jaws; lift the release by the whole tilt depth so the
    # low edge only touches the floor (the kit's half-lift pressed it 0.5-0.8 mm in and it snapped flat).
    release_lift_fraction = 1.0
    grips = {"alphabet_soup": (0.031, 4, 0.012, None), "tomato_sauce": (0.031, 4, 0.012, None)}


# ----- task 2: a can and a long block onto a wooden tray --------------------------------------------------------------

BAR = (0.024, 0.010, 0.014)      # 4.8 x 2.0 x 2.8 cm long block
RED = (0.8, 0.1, 0.08, 1.0)


class CanBlockOnTrayEnv(GatherEnv):
    instruction = "Put the tomato sauce can on the wooden tray first, then the long red block."
    task_objects = ("tomato_sauce", "red_block")
    order = ("tomato_sauce", "red_block")
    receptacle = "wooden_tray"
    slot_offset = 0.029
    elongated = ("red_block",)

    def scene_objects(self):
        return [SmoothCan("tomato_sauce", "tomato_sauce"), Block("red_block", half=BAR, rgba=RED),
                Crate("wooden_tray", inner=(0.062, 0.040), wall=0.008, rgba=WOOD, mass=0.4)]


class CanBlockOnTrayOracle(GatherOracle):
    # Both grasped low (TCP 1.2 / 1.0 cm above the bottom): the pivoting moving jaw drags its contact
    # point up as it opens, and a low contact keeps that lever short (a centre grasp tipped cans 17 deg
    # on release). The block is grasped across its narrow side (local y) and lands lengthwise along the
    # tray. A chocolate-pudding variant was dropped: the 1.4 cm box slipped out of the jaws in 8/10
    # calibration seeds; the 0.9 cm cream cheese/butter boxes were squeezed up and out of the jaws.
    squeeze = 0.006
    # Full tilt-depth lift for the can (as in the box); the kit's half lift for the flat-bottomed block,
    # which slapped flat when released from the full lift (1000-2200 rad/s^2, 3/14 qualification seeds).
    lift_fractions = {"tomato_sauce": 1.0, "red_block": 0.5}
    grips = {"tomato_sauce": (0.031, 4, 0.012, None), "red_block": (2 * BAR[1], 2, 0.010, np.pi / 2)}


# ----- task 3: two cubes onto the plate ---------------------------------------------------------------------

CUBE = 0.012
GREEN, YELLOW = (0.2, 0.6, 0.25, 1.0), (0.9, 0.75, 0.1, 1.0)


class CubesOnPlateEnv(GatherEnv):
    instruction = "Put the green cube on the plate first, then the yellow cube."
    task_objects = ("green_cube", "yellow_cube")
    order = ("green_cube", "yellow_cube")
    receptacle = "plate"
    receptacle_is_fixture = True
    slot_offset = 0.0175
    tolerance = (0.018, 0.018, 0.006)

    def scene_objects(self):
        return [Block("green_cube", half=(CUBE,) * 3, rgba=GREEN), Block("yellow_cube", half=(CUBE,) * 3, rgba=YELLOW)]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 1.0)]

    def receptacle_radius(self):
        return self.fixture_footprint("plate") / np.sqrt(2)     # round: footprint is the bbox corner radius


class CubesOnPlateOracle(GatherOracle):
    # The cubes came to rest ~1 mm below the ray-cast plate height at the slot and fell that far when
    # released (1010-1410 rad/s^2 on landing); release 1 mm lower, slower close against grasp nudges.
    squeeze = 0.004
    close_seconds = 1.8
    release_drop = -0.001
    grips = {"green_cube": (2 * CUBE, 4, 0.010, 0.0), "yellow_cube": (2 * CUBE, 4, 0.010, 0.0)}


FAMILY = "gather_two"

TASKS = [
    define_task(name="soup_then_sauce_in_wooden_box", instruction=CansInBoxEnv.instruction, family=FAMILY,
                env=CansInBoxEnv, oracle=CansInBoxOracle, objects=CansInBoxEnv.task_objects,
                object_kinds=("soup can", "sauce can"), relation="inside one after the other", goal="wooden box",
                steps=(step_text("put", "alphabet soup can", "in", "wooden box"),
                       then(step_text("put", "tomato sauce can", "in", "wooden box")))),
    define_task(name="sauce_then_block_on_tray", instruction=CanBlockOnTrayEnv.instruction, family=FAMILY,
                env=CanBlockOnTrayEnv, oracle=CanBlockOnTrayOracle, objects=CanBlockOnTrayEnv.task_objects,
                object_kinds=("sauce can", "long block"), relation="on one after the other",
                goal="wooden tray",
                steps=(step_text("put", "tomato sauce can", "on", "wooden tray"),
                       then(step_text("put", "long red block", "on", "wooden tray")))),
    define_task(name="two_cubes_on_plate_in_order", instruction=CubesOnPlateEnv.instruction, family=FAMILY,
                env=CubesOnPlateEnv, oracle=CubesOnPlateOracle, objects=CubesOnPlateEnv.task_objects,
                object_kinds=("cube", "cube"), relation="on one after the other", goal="plate",
                steps=(step_text("put", "green cube", "on", "plate"),
                       then(step_text("put", "yellow cube", "on", "plate")))),
]

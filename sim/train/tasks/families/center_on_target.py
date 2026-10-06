"""center_on_target: put an object down precisely centred on a small flat marker.

Tighter than ordinary placement: the object's centre must end within a few millimetres of the marker's
centre (a disc of radius ``CENTER_TOL`` about it, on top of the per-axis physics tolerance), its whole
footprint must lie on (or inside) the marker, and it must be upright, released and settled. Every scene
also has a second marker of a different shape and a random different colour as a confounder, plus 2-4
ordinary distractors. Marker colours are drawn per episode (the instructions name the marker by shape).

Local kit extensions (no shared file is changed):

* ``RoundPad``: a round coaster whose *collision* is two overlapping boxes rotated by 0/45 deg (a regular
  octagon inscribed in the visual disc) under a visual-only cylinder. Three boxes (a 12-gon) overflowed the
  constraint arena under a 25-piece mesh object (LIBERO ramekin). MuJoCo's box-on-thin-cylinder
  contact lets objects sink into a 6 mm ``Disc`` (see ``base.mat``); box-box contact is robust.
* ``TapeSquare``: a square outline of four thin tape strips (one free body), for "centre it inside the
  taped square" where the object rests on the table and must not touch the tape.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, Block, Goal, Obj, TrainEnv, TrainOracle, define_task, mat)

FAMILY = "center_on_target"
CENTER_TOL = 0.008                       # max xy distance object centre -> marker centre (m)
TOL = (CENTER_TOL, CENTER_TOL, 0.005)    # per-axis physics/goal tolerance (the radius check is stricter)
TARGET_REGION = dict(r=(0.16, 0.26), angle=(-62.0, 62.0))   # place targets: vertical release, clear of rest
OBJECT_REGION = dict(r=(0.15, 0.26), angle=(-62.0, 62.0))
PAD_THICK = 0.006
CUBE_HALF = 0.014

# Distinct, saturated marker colours (target and confounder never share one).
PALETTE = ((0.85, 0.2, 0.15, 1), (0.15, 0.35, 0.85, 1), (0.2, 0.6, 0.25, 1), (0.9, 0.72, 0.1, 1),
           (0.55, 0.25, 0.7, 1), (0.95, 0.5, 0.1, 1), (0.1, 0.6, 0.65, 1), (0.92, 0.92, 0.9, 1),
           (0.15, 0.15, 0.17, 1))


# ----- marker primitives -------------------------------------------------------------------------------------

@dataclass
class RoundPad:
    """Free round coaster: visual cylinder, collision = two boxes at 0/45 deg (an inscribed octagon)."""

    name: str
    radius: float = 0.035
    thickness: float = PAD_THICK
    rgba: tuple = (0.2, 0.3, 0.8, 1.0)
    mass: float = 0.05
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        h = self.thickness / 2
        a = self.radius * np.cos(np.radians(22.5))   # octagon with its vertices on the visual rim
        for k, deg in enumerate((0, 45)):
            th = np.radians(deg) / 2
            ET.SubElement(body, "geom", name=f"{self.name}_coll_{k}", type="box", size=f"{a:.6g} {a:.6g} {h:.6g}",
                          quat=f"{np.cos(th):.6g} 0 0 {np.sin(th):.6g}", mass=repr(self.mass / 2), group="3",
                          condim="4", friction=f"{self.friction} 0.02 0.001", rgba="0 0 0 0")
        ET.SubElement(body, "geom", name=f"{self.name}_visual", type="cylinder", size=f"{self.radius:.6g} {h:.6g}",
                      rgba=" ".join(map(str, self.rgba)), group="1", contype="0", conaffinity="0", mass="0",
                      material="val_fabric")


@dataclass
class TapeSquare:
    """Free square outline of tape strips; ``inner`` is the half-size of the open square inside the tape."""

    name: str
    inner: float = 0.026
    width: float = 0.006
    thickness: float = 0.0015
    rgba: tuple = (0.9, 0.72, 0.1, 1.0)
    mass: float = 0.02

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        h, w, s = self.thickness / 2, self.width / 2, self.inner
        long = s + self.width
        strips = (((s + w, 0), (w, long)), ((-(s + w), 0), (w, long)), ((0, s + w), (s, w)), ((0, -(s + w)), (s, w)))
        for k, ((x, y), (hx, hy)) in enumerate(strips):
            ET.SubElement(body, "geom", name=f"{self.name}_strip_{k}", type="box", pos=f"{x:.6g} {y:.6g} 0",
                          size=f"{hx:.6g} {hy:.6g} {h:.6g}", rgba=" ".join(map(str, self.rgba)), group="1",
                          condim="4", mass=repr(self.mass / 4), friction="1.0 0.02 0.001", material="val_fabric")


# ----- shared env helpers ------------------------------------------------------------------------------------

def footprint_xy(env, name):
    """World xy of the object's collision-AABB corners (in its own frame), i.e. its footprint polygon."""
    m, d = env.model, env.data
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in env._geoms[name]:
        pts.append((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T
                   + d.geom_xpos[g])
    return np.vstack(pts)[:, :2]


def in_marker_frame(env, marker, pts):
    yaw = env.yaw(marker)
    c, s = np.cos(yaw), np.sin(yaw)
    rel = pts - env.object_pos(marker)[:2]
    return rel @ np.array([[c, -s], [s, c]])


def centred_on_round(marker, radius):
    """Centre within CENTER_TOL of the round marker's centre and the whole footprint on it."""
    def check(env, obj):
        centre = env.object_pos(marker)[:2]
        if not env.within_radius(obj, centre, CENTER_TOL):
            return False
        return bool(np.linalg.norm(footprint_xy(env, obj) - centre, axis=1).max() <= radius)
    return check


def centred_on_square(marker, half):
    def check(env, obj):
        if not env.within_radius(obj, env.object_pos(marker), CENTER_TOL):
            return False
        return bool(np.abs(in_marker_frame(env, marker, footprint_xy(env, obj))).max() <= half)
    return check


def centred_inside_tape(marker, inner, margin=0.003):
    """Centred in the tape square, footprint inside the open square with ``margin`` to the tape, and not
    touching the tape."""
    def check(env, obj):
        if not env.within_radius(obj, env.object_pos(marker), CENTER_TOL):
            return False
        if np.abs(in_marker_frame(env, marker, footprint_xy(env, obj))).max() > inner - margin:
            return False
        return not env.touching(obj, marker)
    return check


class CenterEnv(TrainEnv):
    """Common layout: target marker, confounder marker, the object, then 2-4 distractors."""

    target_marker = "target"
    confounder = "confounder"
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n != "plate")   # a plate reads as a round target

    def marker_colors(self):
        i, j = self.np_random.choice(len(PALETTE), size=2, replace=False)
        for name, rgba in ((self.target_marker, PALETTE[i]), (self.confounder, PALETTE[j])):
            for g in range(self.model.ngeom):
                if self.model.body(int(self.model.geom_bodyid[g])).name == name and self.model.geom_group[g] == 1:
                    self.model.geom_rgba[g] = rgba

    def place_markers(self, placed):
        self.marker_colors()
        target_xy, _ = self.place(self.target_marker, placed, region=TARGET_REGION, clearance=0.025)
        self.place(self.confounder, placed, region=dict(r=(0.14, 0.27), angle=(-68, 68)), clearance=0.025)
        return target_xy

    def place_object(self, placed):
        name = self.task_objects[0]
        self.place(name, placed, region=OBJECT_REGION, clearance=0.03)


class CenterOracle(TrainOracle):
    width = 0.028
    symmetric = 4
    grasp_offset = None          # TCP height above the object origin at the grasp (None: kit default)
    narrow_axis_yaw = 0.0        # added to the object yaw to close across its narrow side

    level_held = False           # cancel the object's in-hand pivot before release (see ``leveled_rot``)
    hover = 0.004                # re-level this far above the seated height (``level_held`` only)
    press = 0.0                  # release this far below the seated height (the support takes the load first)
    squeeze = None               # close to a fingertip gap this much below the width (None: fully closed)

    def gripper(self, value, seconds=0.5, hold=0.2):
        # pick() closes with gripper(CLOSED, ...); with ``squeeze`` set, close only to ``width - squeeze``.
        if self.squeeze is not None and value == CLOSED:
            value = self.open_for(self.width, -self.squeeze)
        return (yield from super().gripper(value, seconds, hold))

    def leveled_rot(self, name, rot):
        """TCP orientation near ``rot`` at which the held object (rigid in the jaws) would sit level.

        Flat objects pivot up to 7 deg about the closing axis while carried (measured, 0.7x pudding box);
        lowered at a vertical TCP they touch down on one edge (2.5 mm penetration) and slap flat on release
        (>1000 rad/s^2). Tilting the TCP by the inverse pivot lands them flat."""
        R_rel = self.tcp_rot().T @ self.env.data.xmat[self.env._body[name]].reshape(3, 3)
        pred = rot @ R_rel
        yaw = np.arctan2(pred[1, 0], pred[0, 0])
        level = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        return level @ R_rel.T

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        if not self.level_held:
            return (yield from super().place_object(name, target, rot=rot, drop=drop, open_to=open_to,
                                                    carry_z=carry_z))
        # As TrainOracle.place_object, with the TCP orientation re-levelled after the carry.
        carry_z = self.carry_z if carry_z is None else carry_z
        target = np.asarray(target, float)
        rot = self.leveled_rot(name, self.carry_rot(target[:2]) if rot is None else rot)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        rot = self.leveled_rot(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        # The in-hand tilt drifts while lowering (measured 1.4 -> 1.9 deg): hover, re-level, then seat.
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + self.hover], rot, speed=0.06, tol=self.release_tolerance,
                             settle=0.6, label="hover")
        rot = self.leveled_rot(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + self.hover], rot, speed=0.03, tol=self.release_tolerance,
                             settle=0.6, label="relevel", smooth=False)
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + lift], rot, speed=0.06, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def plan(self):
        env = self.env
        goal = env.goals[0]
        name = goal.obj
        grasp_z = None if self.grasp_offset is None else env.object_pos(name)[2] + self.grasp_offset
        ok = yield from self.pick_and_place(name, env.goal_target(goal), width=self.width, grasp_z=grasp_z,
                                            yaw=env.yaw(name) + self.narrow_axis_yaw, symmetric=self.symmetric,
                                            drop=-self.press)
        if ok:
            yield from self.rest()
            yield from self.wait(1.2)


# ----- task 1: pudding box on the round coaster ---------------------------------------------------------------

class PuddingOnCoasterEnv(CenterEnv):
    instruction = "Put the pudding box down right in the middle of the round coaster."
    task_objects = ("pudding",)
    radius = 0.040
    scale = 0.7
    # Coated-carton friction (0.5-0.7). At the LIBERO default 1.0 the opening jaw dragged the box edge
    # (1086-2064 rad/s^2 on release in 2 of 30 calibration seeds); at 0.7 the same seeds stay < 500.
    friction = 0.7
    distractor_pool = tuple(n for n in CenterEnv.distractor_pool if n != "chocolate_pudding")

    def scene_objects(self):
        return [Obj("pudding", "chocolate_pudding", scale=self.scale, friction=self.friction), RoundPad("target", radius=self.radius),
                mat("confounder", half=(0.032, 0.032))]

    def layout(self):
        placed = []
        xy = self.place_markers(placed)
        self.place_object(placed)
        self.place_distractors(placed)
        z = PAD_THICK + self._extent["pudding"]["bottom"]
        self.set_goals(Goal("pudding", "target", target=(*xy, z), reference="target", tolerance=TOL,
                            check=centred_on_round("target", self.radius)))


class PuddingOracle(CenterOracle):
    width = 0.0325
    symmetric = 2
    narrow_axis_yaw = np.pi / 2         # the box's long axis is its body x
    # The closing moving jaw pitched the light box (1056 rad/s^2) when it met it higher up: grasp with the
    # TCP at the box's mid-height (jaw hull 1.4 mm above the table) and close more slowly.
    grasp_offset = 0.0
    close_seconds = 1.6
    level_held = True
    # Fully closed, the jaws squeezed the box at ~26 N; on release the opening moving jaw's tip dragged one
    # edge up 16 deg by friction and it slapped back (2064 rad/s^2). A limited squeeze avoids the drag.
    squeeze = 0.007


# ----- task 2: soup can on the square mat --------------------------------------------------------------------

class CanOnMatEnv(CenterEnv):
    instruction = "Set the soup can in the exact centre of the square mat."
    task_objects = ("can",)
    half = 0.031
    distractor_pool = tuple(n for n in CenterEnv.distractor_pool if n not in ("alphabet_soup", "tomato_sauce"))  # look alike

    def scene_objects(self):
        return [Obj("can", "alphabet_soup"), mat("target", half=(self.half, self.half)),
                RoundPad("confounder", radius=0.032)]

    def layout(self):
        placed = []
        xy = self.place_markers(placed)
        self.place_object(placed)
        self.place_distractors(placed)
        z = PAD_THICK + self._extent["can"]["bottom"]
        self.set_goals(Goal("can", "target", target=(*xy, z), reference="target", tolerance=TOL,
                            check=centred_on_square("target", self.half)))


class CanOracle(CenterOracle):
    # Collision diameter 3.13 cm, but its centre sits 1.9 mm off the body origin (more than the 1.5 mm grasp
    # clearance), so the fixed finger landed on the rim for some yaws: plan as if 3.65 cm wide.
    width = 0.0365
    # Grasped at mid-height the can's top sat ~2 cm above the TCP, near the palm; a can that turned in the
    # jaws during a carry wedged against the fixed-jaw body (176 N) and rode up on retreat. Grasp 4 mm higher.
    grasp_offset = 0.004
    # The can pivots in the jaws and rocked onto its rim when released at a vertical TCP (2317 rad/s^2,
    # 3.2 mm penetration in qualification): level it before release and open more slowly.
    level_held = True
    release_seconds = 1.6


# ----- task 3: cube inside the taped square ------------------------------------------------------------------

class CubeInTapeEnv(CenterEnv):
    instruction = "Place the cube in the middle of the taped square on the table without touching the tape."
    task_objects = ("cube",)
    inner = 0.026

    def scene_objects(self):
        return [Block("cube", half=(CUBE_HALF,) * 3, rgba=(0.85, 0.85, 0.82, 1)), TapeSquare("target", inner=self.inner),
                RoundPad("confounder", radius=0.03)]

    def marker_colors(self):
        super().marker_colors()
        # The cube takes a colour different from both markers.
        rest = [c for c in PALETTE if not any(np.allclose(c, self.model.geom_rgba[g]) for g in
                                              (self.model.geom("target_strip_0").id, self.model.geom("confounder_visual").id))]
        self.model.geom_rgba[self.model.geom("cube_geom").id] = rest[int(self.np_random.integers(len(rest)))]

    def layout(self):
        placed = []
        xy = self.place_markers(placed)
        self.place_object(placed)
        self.place_distractors(placed)
        self.set_goals(Goal("cube", "table", target=(*xy, CUBE_HALF), reference="target", tolerance=TOL,
                            check=centred_inside_tape("target", self.inner)))


class CubeOracle(CenterOracle):
    width = 2 * CUBE_HALF


TASKS = [
    define_task(name="pudding_centered_on_round_coaster", instruction=PuddingOnCoasterEnv.instruction, family=FAMILY,
                env=PuddingOnCoasterEnv, oracle=PuddingOracle, objects=("pudding",), object_kinds=("flat food box",),
                relation="centred on", goal="round coaster",
                steps=("Pick up the pudding box and set it down in the middle of the round coaster.",)),
    define_task(name="soup_can_centered_on_square_mat", instruction=CanOnMatEnv.instruction, family=FAMILY,
                env=CanOnMatEnv, oracle=CanOracle, objects=("can",), object_kinds=("can",),
                relation="centred on", goal="square mat",
                steps=("Pick up the soup can and stand it in the centre of the square mat.",)),
    define_task(name="cube_centered_in_tape_square", instruction=CubeInTapeEnv.instruction, family=FAMILY,
                env=CubeInTapeEnv, oracle=CubeOracle, objects=("cube",), object_kinds=("cube",),
                relation="centred inside without touching", goal="taped square outline",
                steps=("Pick up the cube and set it down in the middle of the taped square, clear of the tape.",)),
]

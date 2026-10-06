"""place_on_surface: put one object onto a flat receptacle surface (plate, wooden tray, coaster).

Three tasks that differ in object kind, receptacle kind and grasp:

* ``golf_ball_on_plate``   -- a YCB golf ball (sphere grasp at the equator) set on a plate.
* ``sauce_on_wooden_tray`` -- the tomato-sauce can carried into the low-walled LIBERO wooden tray, released with
  the jaws closing along the tray's long axis so they clear its side walls.
* ``sponge_on_coaster``    -- a flat kitchen sponge (yellow foam, green scouring pad) centred on a cork coaster.

Asset helpers kept here instead of in the shared kit (proposed kit additions, see the family report):

* ``SphereScan`` -- a scanned ball with an exact sphere collision (the convex hull of the YCB golf ball rocks
  between facets and never settles).
* ``LiberoFree`` -- a free LIBERO object from any asset path (the wooden tray is not in ``sim.val.scene.CATALOG``).

Dropped: ``banana_on_plate``. The YCB banana (convex-sliced collision so the jaws meet the visible fruit) is only
1.7-1.9 cm wide at 0.45-0.5x; the clipped gripper command gives little grip force at that width and the
fingertips meet the rounded cross-section below its equator, so the banana is squeezed down out of the jaws
during the carry, and at 0.5-0.6x its ends land on the plate rim. 41/48 prescribed seeds before the run was cut,
and 0-2/7 of the failing seeds after five variants (scale 0.45-0.6, plate 1.2-1.3x, COM grasp, friction,
rim-aware release height).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task, mat,
                                  step_text, top_down_mat)
from sim.train.tasks.assets import Scanned
from sim.val.scene import (ASSETS, Fixture, Obj, _add_assets, _content, _merge_defaults, _set_free_physics, _soften,
                           load_mjcf)

FAMILY = "place_on_surface"


# ----- asset helpers ---------------------------------------------------------------------------------------------

@dataclass
class SphereScan(Scanned):
    """Scanned ball (visual mesh) with an exact sphere collision. The YCB golf ball's convex hull is a polyhedron
    that rocks between facets on a flat support (it never settled within 3 cm/s in 1 of 4 seeds); a sphere with
    rolling friction (condim 6) comes to rest."""

    rolling: float = 0.002
    torsional: float = 0.02

    @np.errstate(all="ignore")
    def build_mjcf(self, mj, asset_el, worldbody):
        _, acc = self._accessors()
        acc.ensure_assets(self.model_id)
        visual = acc.visual_mesh(self.model_id)
        verts = self._body_verts([(f"{self.name}_probe", SimpleNamespace(path=visual))], self.scale)
        centre = 0.5 * (verts.min(0) + verts.max(0))
        radius = float(np.mean(verts.max(0) - verts.min(0)) / 2)
        prefix = f"{self.name}_scan"
        ET.SubElement(asset_el, "mesh", name=f"{prefix}_vis", file=visual.as_posix(),
                      scale=" ".join([f"{self.scale:.6g}"] * 3))
        material = None
        texture = acc.texture_file(self.model_id)
        if texture.exists():
            ET.SubElement(asset_el, "texture", name=f"{prefix}_tex", type="2d", file=texture.as_posix())
            material = f"{prefix}_mat"
            ET.SubElement(asset_el, "material", name=material, texture=f"{prefix}_tex", texuniform="false",
                          specular="0.1", reflectance="0")
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "geom", name=f"{prefix}_coll", type="sphere", size=f"{radius:.6g}",
                      mass=repr(float(self.mass if self.mass is not None else 0.025)), group="3", condim="6",
                      friction=f"{self.friction} {self.torsional} {self.rolling}")
        vis = dict(name=f"{prefix}_visual", type="mesh", mesh=f"{prefix}_vis", group="1", contype="0",
                   conaffinity="0", mass="0", pos=" ".join(f"{-c:.6g}" for c in centre))
        if material:
            vis["material"] = material
        ET.SubElement(body, "geom", **vis)


@dataclass
class LiberoFree:
    """A free LIBERO object loaded from ``ASSETS / path`` (same treatment as ``sim.val.scene`` free objects)."""

    name: str
    path: str
    scale: float = 0.5
    mass: float = 0.2
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        root = load_mjcf(ASSETS / self.path, self.scale, self.name)
        _soften(root.find("asset"))
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat="1 0 0 0")
        upright.extend(_content(root))


# ----- shared geometry / oracle helpers -------------------------------------------------------------------------

def touching_table(env, name):
    mine, table = set(env._geoms[name]), env._owned_geoms("table")
    return any((c.geom1 in mine and c.geom2 in table) or (c.geom2 in mine and c.geom1 in table)
               for c in env.data.contact[:env.data.ncon])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SurfaceOracle(TrainOracle):
    """``TrainOracle`` with a grasp at an explicit centre (not the body origin).

    Grasps are gentler than the kit default: light, round or curved objects (sauce can, banana) were nudged into
    the fixed finger while the jaw closed, with 1000-1200 rad/s^2 spikes (5/24 calibration seeds).
    """

    grasp_clearance = 0.003
    close_seconds = 1.5

    def pick_at(self, name, centre_fn, width, grasp_z, yaw, symmetric=2, attempts=2, approach=0.05):
        for attempt in range(attempts):
            centre = np.asarray(centre_fn(), float)
            angles = [yaw + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([centre[0], centre[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, self.close_seconds, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], self.carry_z]), rot, speed=0.12, label="lift")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def yaw_rotations(self, rot):
        """``rot`` and its half-turn twin (two equivalent closing directions for a two-finger grasp)."""
        return [rot, rot_z(np.pi) @ rot]


# ----- golf ball on the plate ----------------------------------------------------------------------------------

BALL_W = 0.0215                      # hull diameter at 0.5x (measured; the inventory box is conservative)


def ball_on_plate(env, name):
    """Ball resting on the plate floor near its centre, clear of the table."""
    return bool(env.within_radius(name, env.object_pos("plate"), 0.025) and not touching_table(env, name))


class BallPlateEnv(TrainEnv):
    instruction = "Put the golf ball on the plate."
    task_objects = ("ball",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n != "plate")

    def scene_objects(self):
        return [SphereScan("ball", "ycb", "058_golf_ball", 0.5, mass=0.025)]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 1.0)]

    def layout(self):
        placed = []
        rng = self.np_random
        side = rng.choice((-1.0, 1.0))
        plate_r = self.fixture_footprint("plate") / np.sqrt(2)
        plate_xy = self.sample_xy(plate_r, placed, clearance=0.02, r=(0.18, 0.26),
                                  angle=tuple(sorted((side * 5.0, side * 60.0))))
        self.set_fixture_pose("plate", plate_xy, yaw=rng.uniform(-np.pi, np.pi))
        placed.append((plate_xy, plate_r))
        self.place("ball", placed, region=dict(r=(0.15, 0.26), angle=tuple(sorted((-side * 0.0, -side * 62.0)))))
        self.place_distractors(placed)
        self.set_goals(Goal("ball", "plate", target=None, upright_cos=None, check=ball_on_plate))


class BallPlateOracle(SurfaceOracle):
    def plan(self):
        env = self.env
        # TCP at the ball centre: the jaw pads then meet it at the equator (measured), and the jaw hull stays
        # >= 2 mm above the table.
        ok = yield from self.pick_at("ball", lambda: env.object_pos("ball"), BALL_W,
                                     env.object_pos("ball")[2],
                                     np.arctan2(*env.object_pos("ball")[1::-1]), symmetric=4)
        if not ok:
            return
        plate = env.object_pos("plate")[:2]
        floor = env.surface_z(plate, exclude=env.task_objects)
        target = np.r_[plate, floor + env._extent["ball"]["bottom"] + 0.001]
        yield from self.place_object("ball", target, open_to=self.release_for(BALL_W))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- tomato sauce on the wooden tray --------------------------------------------------------------------------

TRAY_PATH = "turbosquid_objects/wooden_tray/wooden_tray.xml"
CAN_W = 0.031


class SauceTrayEnv(TrainEnv):
    instruction = "Put the tomato sauce on the wooden tray."
    task_objects = ("sauce",)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("tomato_sauce", "alphabet_soup"))

    def scene_objects(self):
        return [Obj("sauce", "tomato_sauce"), LiberoFree("tray", TRAY_PATH, 0.5, mass=0.25)]

    def layout(self):
        placed = []
        rng = self.np_random
        side = rng.choice((-1.0, 1.0))
        tray_r = 0.075
        tray_xy = self.sample_xy(tray_r, placed, clearance=0.02, r=(0.20, 0.25),
                                 angle=tuple(sorted((side * 15.0, side * 60.0))))
        # Long side roughly tangential (across the reach), so both ends stay inside the workspace.
        yaw = np.arctan2(tray_xy[1], tray_xy[0]) + np.pi / 2 + rng.uniform(-0.35, 0.35)
        self.set_object_pose("tray", tray_xy, yaw=yaw)
        placed.append((tray_xy, tray_r))
        self.place("sauce", placed, region=dict(r=(0.16, 0.25), angle=tuple(sorted((-side * 5.0, -side * 60.0)))))
        self.place_distractors(placed)
        floor = self.surface_z(tray_xy, exclude=self.task_objects)
        z = floor + self._extent["sauce"]["bottom"]
        self.set_goals(Goal("sauce", "tray", target=(*tray_xy, z), reference="tray", tolerance=(0.02, 0.02, 0.006)))


class SauceTrayOracle(SurfaceOracle):
    def plan(self):
        env = self.env
        ok = yield from self.pick("sauce", CAN_W, env.object_pos("sauce")[2] + 0.004, yaw=0.0, symmetric=8)
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        tray_yaw = env.yaw("tray")
        # Close along the tray's long axis: the opening jaws then stay clear of the long side walls.
        rots = [top_down_mat(tray_yaw), top_down_mat(tray_yaw + np.pi)]
        rot = self.feasible_rotation("sauce", target, rots)
        yield from self.place_object("sauce", target, rot=rot, open_to=self.release_for(CAN_W))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- sponge on the coaster -----------------------------------------------------------------------------------

SPONGE_HALF = (0.023, 0.015)          # 4.6 x 3.0 cm kitchen sponge (half-size scale, like the LIBERO assets)
FOAM_H, PAD_H = 0.017, 0.005           # yellow foam with a green scouring pad on top
SPONGE_W = 2 * SPONGE_HALF[1]
COASTER_HALF, COASTER_T = 0.038, 0.005
CORK = (0.72, 0.55, 0.36, 1.0)


@dataclass
class Sponge:
    """A kitchen sponge: yellow foam box with a green scouring pad (one free body, two box geoms)."""

    name: str
    mass: float = 0.03
    foam: tuple = (0.95, 0.80, 0.25, 1.0)
    pad: tuple = (0.18, 0.50, 0.22, 1.0)
    friction: float = 0.9

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        hx, hy = SPONGE_HALF
        total = FOAM_H + PAD_H
        for part, h, z, rgba, material in (("foam", FOAM_H, -total / 2 + FOAM_H / 2, self.foam, "val_fabric"),
                                           ("pad", PAD_H, total / 2 - PAD_H / 2, self.pad, "val_fabric")):
            ET.SubElement(body, "geom", name=f"{self.name}_{part}", type="box", size=f"{hx} {hy} {h / 2}",
                          pos=f"0 0 {z:.6g}", rgba=" ".join(map(str, rgba)), mass=repr(self.mass * h / total),
                          group="1", condim="4", friction=f"{self.friction} 0.02 0.001", material=material)


class SpongeCoasterEnv(TrainEnv):
    instruction = "Put the sponge on the coaster."
    task_objects = ("sponge",)

    def scene_objects(self):
        return [Sponge("sponge"),
                mat("coaster", half=(COASTER_HALF, COASTER_HALF), thickness=COASTER_T, rgba=CORK, mass=0.04)]

    def layout(self):
        placed = []
        rng = self.np_random
        side = rng.choice((-1.0, 1.0))
        self.place("coaster", placed, region=dict(r=(0.17, 0.26), angle=tuple(sorted((side * 5.0, side * 65.0)))))
        self.place("sponge", placed, region=dict(r=(0.16, 0.26), angle=tuple(sorted((-side * 0.0, -side * 62.0)))))
        self.place_distractors(placed)
        cxy = self.object_pos("coaster")[:2]
        z = COASTER_T + self._extent["sponge"]["bottom"]
        self.set_goals(Goal("sponge", "coaster", target=(*cxy, z), reference="coaster", tolerance=(0.015, 0.015, 0.006)))


class SpongeCoasterOracle(SurfaceOracle):
    # The flat sponge tips about its bottom edge when the angled moving jaw meets it far from the fixed finger
    # (1050-1530 rad/s^2 in 6/25 seeds at 3 mm clearance): catch it early with the kit's 1.5 mm clearance.
    grasp_clearance = 0.0015

    def plan(self):
        env = self.env
        yaw = env.yaw("sponge") + np.pi / 2                    # close across the narrow side (local y)
        ok = yield from self.pick("sponge", SPONGE_W, env.object_pos("sponge")[2] - 0.001, yaw=yaw, symmetric=2)
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        carry = self.carry_rot(target[:2])
        rot = self.feasible_rotation("sponge", target, self.yaw_rotations(carry))
        yield from self.place_object("sponge", target, rot=rot, open_to=self.release_for(SPONGE_W))
        yield from self.rest()
        yield from self.wait(1.2)


TASKS = [
    define_task(name="golf_ball_on_plate", instruction=BallPlateEnv.instruction, family=FAMILY,
                env=BallPlateEnv, oracle=BallPlateOracle, objects=("ball",), object_kinds=("golf ball",),
                relation="on", goal="plate", steps=(step_text("put", "golf ball", "on", "plate"),)),
    define_task(name="sauce_on_wooden_tray", instruction=SauceTrayEnv.instruction, family=FAMILY,
                env=SauceTrayEnv, oracle=SauceTrayOracle, objects=("sauce",), object_kinds=("tomato sauce can",),
                relation="on", goal="wooden tray", steps=(step_text("put", "tomato sauce", "on", "wooden tray"),)),
    define_task(name="sponge_on_coaster", instruction=SpongeCoasterEnv.instruction, family=FAMILY,
                env=SpongeCoasterEnv, oracle=SpongeCoasterOracle, objects=("sponge",), relation="on",
                goal="coaster", steps=(step_text("put", "sponge", "on", "coaster"),)),
]

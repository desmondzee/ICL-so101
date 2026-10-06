"""sort_by_category: two kinds of objects, two containers; put each object into its category's container in a
stated order.

The instruction states the sorting rule by category (food into one container, toy blocks into the other) and
which object to start with. Every scene holds exactly the objects the rule applies to; distractors are drawn
from kitchenware that belongs to neither category (plate, ramekin, mug, and a bowl only when no bowl is a
target), so the rule is unambiguous. The strict success predicate is the kit's: each object released, settled,
upright, supported by its category's container and inside its walls (``inside`` check), plus
``OrderedCompletion`` (the stated object is grasped and placed first, and the later placement never disturbs
the earlier one). This is not the held-out "sort the red/blue blocks onto matching plates": the rule is by
object kind (food vs toy blocks), the receptacles are containers, never plates, and colour plays no role.

Containers: the LIBERO basket rescaled wider and shallower (0.7, 0.7, 0.4; put_in_container qualified a can
into it 50/50), a static LIBERO white bowl (pilot block_in_bowl 49/50), and open wooden/cardboard boxes built
from primitives (``Crate``). Cans use ``SolidLibero`` (one solid box collision; the LIBERO 21-plate collision
catches the jaw pads). Visibility goal regions sit in each container's opening, because the near wall hides a
container floor from the low front camera.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import CLOSED, GRASP_REGION, Goal, TrainEnv, TrainOracle, define_task, then
from sim.train.variation import GoalRegion
from sim.val.scene import (ASSETS, CATALOG, Block, Fixture, _add_assets, _collision_geoms, _content, _hull_volume,
                           _merge_defaults, _set, _set_free_physics, _soften, _upright_quat, _vec, load_mjcf)

FAMILY = "sort_by_category"
CUBE = 0.014                              # 2.8 cm cube half-size
BAR = (0.024, 0.010, 0.014)               # 4.8 x 2.0 x 2.8 cm long block
CAN_W = 0.0312                            # SolidLibero can collision box width (square)
RED, BLUE, YELLOW = (0.8, 0.1, 0.08, 1.0), (0.1, 0.25, 0.8, 1.0), (0.9, 0.75, 0.1, 1.0)
WOOD, CARDBOARD = (0.50, 0.32, 0.18, 1.0), (0.84, 0.70, 0.48, 1.0)
CONTAINER_REGION = dict(r=(0.17, 0.25), angle=(-62.0, 62.0))
OBJECT_REGION = dict(r=(0.17, 0.26), angle=(-62.0, 62.0))     # r >= 0.17: no camera-mount fold-in on pickup
REST_POINT, REST_CLEAR = np.array([0.16, 0.0]), 0.06


# ----- local primitives (copied from sibling families; shared files are not edited) --------------------------

@dataclass
class Crate:
    """An open-top rectangular box (free body): floor slab + four walls; body origin at the floor's underside."""

    name: str
    inner: tuple = (0.05, 0.042)
    wall: float = 0.022
    thickness: float = 0.006
    floor: float = 0.006
    rgba: tuple = WOOD
    mass: float = 1.0
    friction: float = 1.0
    feet: float = 0.003                # radius of four sphere feet under the floor corners (0: none)

    def boxes(self):
        ix, iy = self.inner
        t, f, h = self.thickness, self.floor, self.wall
        zc = (f + h) / 2
        return [((0, 0, f / 2), (ix + t, iy + t, f / 2)),
                ((ix + t / 2, 0, zc), (t / 2, iy + t, zc)), ((-ix - t / 2, 0, zc), (t / 2, iy + t, zc)),
                ((0, iy + t / 2, zc), (ix, t / 2, zc)), ((0, -iy - t / 2, zc), (ix, t / 2, zc))]

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
        if self.feet:
            # Sphere feet: on the living-room table (thin 12 mm slabs) the slab-on-slab box contact of a heavy
            # flat-bottomed receptacle went unstable (objects inside spun at 1000-5000 rad/s^2); sphere-box
            # contacts are robust. The floor slab rides ``feet`` above the table.
            fx, fy = self.inner[0] + self.thickness - 0.006, self.inner[1] + self.thickness - 0.006
            for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
                ET.SubElement(body, "geom", name=f"{self.name}_foot{k}", type="sphere", size=f"{self.feet:.6g}",
                              pos=f"{sx * fx:.6g} {sy * fy:.6g} 0", rgba=" ".join(map(str, self.rgba)),
                              mass="0.001", group="1", condim="4", friction=f"{self.friction} 0.02 0.001")


@dataclass
class SolidLibero:
    """A free LIBERO object whose collision boxes are replaced by one solid box over their extent (visual mesh
    and mass unchanged). Copied from put_in_container."""

    name: str
    asset: str
    scale: float = 0.5
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        a = CATALOG[self.asset]
        root = load_mjcf(ASSETS / a.path, self.scale, self.name)
        _soften(root.find("asset"))
        mass = _hull_volume(root) * a.density
        boxes = [g for g in _collision_geoms(root) if g.get("type") == "box"]
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in boxes:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, _vec(g, "quat", np.array([1.0, 0, 0, 0])))
            half = np.abs(R.reshape(3, 3)) @ _vec(g, "size")
            pos = _vec(g, "pos", np.zeros(3))
            lo, hi = np.minimum(lo, pos - half), np.maximum(hi, pos + half)
        parent = next(b for b in root.iter("body") if any(c is boxes[0] for c in b))
        for g in boxes:
            parent.remove(g)
        ET.SubElement(parent, "geom", name=f"{self.name}_solid", type="box", group="3",
                      pos=" ".join(f"{v:.6g}" for v in (lo + hi) / 2),
                      size=" ".join(f"{v:.6g}" for v in (hi - lo) / 2))
        _set_free_physics(root, mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright")
        _set(upright, "quat", _upright_quat(a.upright))
        upright.extend(_content(root))


@dataclass
class ScaledAsset:
    """A free LIBERO catalog object scaled per axis (copied from put_in_container). The basket at (0.7, 0.7,
    0.4) is wide enough inside for the jaws and shallow enough for the arm to reach over its rim."""

    name: str
    asset: str
    scale: tuple = (0.5, 0.5, 0.5)
    mass: float = 0.15
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        S = np.asarray(self.scale, float)
        root = load_mjcf(ASSETS / CATALOG[self.asset].path, 1.0, self.name)
        for el in root.iter():
            if el.tag in ("body", "geom", "site") and "pos" in el.attrib:
                _set(el, "pos", _vec(el, "pos") * S)
            if el.tag == "mesh":
                _set(el, "scale", _vec(el, "scale", np.ones(3)) * S)
            if el.tag in ("geom", "site") and el.get("type") == "box" and "size" in el.attrib:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, _vec(el, "quat", np.array([1.0, 0, 0, 0])))
                axes = np.abs(R.reshape(3, 3)).argmax(axis=0)
                _set(el, "size", _vec(el, "size") * S[axes])
        _soften(root.find("asset"))
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright")
        upright.extend(_content(root))


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _local_xy(env, name, ref):
    """xy of body ``name`` in the frame of body ``ref`` (about its vertical axis)."""
    m = env.data.xmat[env._body[ref]].reshape(3, 3)
    return (m.T @ (env.object_pos(name) - env.object_pos(ref)))[:2]


def world_aabb(env, name):
    m, d = env.model, env.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in env._geoms[name]:
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        half = np.abs(R) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    return lo, hi


# ----- shared environment ------------------------------------------------------------------------------------

class SortEnv(TrainEnv):
    """``targets``: task object -> container body. ``boxes``: container -> inner half extents for the
    ``inside`` check (None: a round container checked by radius ``radii``). Containers are placed first, then the
    objects on the table, then 2-4 neutral distractors."""

    targets: dict = {}
    inside_margin: dict = {}           # object -> clearance of its centre from the inner walls (m)
    radii: dict = {}                   # round container -> max object-centre offset (m)
    floor_height: dict = {}            # free box container -> floor top above its origin (slab underside)
    slot: dict = {}                    # object -> offset (container frame) of its goal (two in one box)
    fixtures: tuple = ()

    def container_names(self):
        return tuple(dict.fromkeys(self.targets.values()))

    def place_container(self, name, placed):
        if name in self.fixtures:
            radius = self.fixture_footprint(name) / np.sqrt(2)
            xy = self.sample_xy(radius, placed, clearance=0.03, **CONTAINER_REGION)
            self.set_fixture_pose(name, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        else:
            radius = self.footprint(name)
            xy = self.sample_xy(radius, placed, clearance=0.03, **CONTAINER_REGION)
            yaw = np.arctan2(xy[1], xy[0]) + np.pi / 2 + self.np_random.uniform(-0.3, 0.3)
            self.set_object_pose(name, xy, yaw=yaw)
        placed.append((xy, radius))
        return xy

    def layout(self):
        placed = []
        for name in self.container_names():
            self.place_container(name, placed)
        for name in self.task_objects:
            self.place(name, placed, region=OBJECT_REGION, clearance=0.025)
        self.place_distractors(placed)
        self.set_goals(*(self.goal_for(n) for n in self.task_objects))

    def goal_for(self, name):
        cont = self.targets[name]
        c = self.object_pos(cont)
        off = self.slot.get(name, (0.0, 0.0))
        if cont in self.fixtures:
            xy = c[:2]
            # A bowl floor is curved: a cube rests on its corners, higher than the floor under its centre.
            # Rest height = highest floor point under the object's footprint ring (and centre).
            r = self.footprint(name) * 0.95
            ring = [xy] + [xy + r * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
            floor = max(self.surface_z(p, exclude=self.task_objects) for p in ring)
            return Goal(name, cont, target=(*xy, floor + self._extent[name]["bottom"]), tolerance=(0.018, 0.018, 0.006),
                        check=self.inside_round(cont))
        R = self.data.xmat[self._body[cont]].reshape(3, 3)
        xy = c[:2] + (R[:2, :2] @ np.asarray(off))
        if cont in self.floor_height:
            floor = c[2] + self.floor_height[cont]    # origin: floor slab underside
        else:
            floor = self.surface_z(xy, exclude=self.task_objects)
        return Goal(name, cont, target=(*xy, floor + self._extent[name]["bottom"]), reference=cont,
                    tolerance=(0.025, 0.025, 0.006), check=self.inside_box(cont))

    def inside_box(self, cont):
        def check(env, name):
            half = np.asarray(env.box_inner[cont]) - env.inside_margin.get(name, 0.012)
            return bool(np.all(np.abs(_local_xy(env, name, cont)) <= half))
        check.__name__ = f"inside_{cont}"
        return check

    def inside_round(self, cont):
        def check(env, name):
            return env.within_radius(name, env.object_pos(cont), env.radii.get(cont, 0.018))
        check.__name__ = f"inside_{cont}"
        return check

    def goal_regions(self):
        """Samples across each container's opening at rim height (the near wall hides the floor)."""
        regions = []
        for goal in self.goals:
            cont = goal.support
            lo, hi = world_aabb(self, cont)
            c = self.goal_target(goal)
            s = 0.012
            regions.append(GoalRegion(cont, tuple((c[0] + x, c[1] + y, hi[2] + 0.002)
                                                  for x in (-s, 0, s) for y in (-s, 0, s))))
        return tuple(regions)


# ----- shared oracle -----------------------------------------------------------------------------------------

class SortOracle(TrainOracle):
    """Pick each object off the table in order and release it resting on its container's floor.

    ``grips``: name -> (width across the jaws, grasp TCP height above the object's bottom or None for its origin,
    yaw offset of a closing face from the object's local x, equivalent closing directions)."""

    grips: dict = {}
    hold_squeeze: dict = {}
    tall: dict = {}                    # name -> (open_margin, grasp_clearance)
    squeeze = 0.006                    # ease the jaw to this far below the width before opening
    relax_seconds = 0.6
    release_drop = -0.0005             # release resting on the floor: a 2 mm fall slaps a small object
    touch_depth = 0.001
    touch_seconds = 0.9
    grasp_settle = 0.3
    pre_release_height = 0.010
    release_margin = 0.012
    release_drops: dict = {}           # per-object overrides of release_drop / release_margin
    release_margins: dict = {}

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def grasp_center(self, name):
        env = self.env
        local = self._local_points(name)
        mid = (local.min(0) + local.max(0)) / 2
        body = env._body[name]
        return env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ np.r_[mid[:2], 0.0]

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Kit pick with a staged close (ease onto the object and stop, then squeeze from rest) and an optional
        bounded squeeze; tall objects get a wider opening (the pivoting jaw narrows above the tips)."""
        self._held_width = width
        hs = self.hold_squeeze.get(name)
        close = CLOSED if hs is None else self.gap_angle(width - hs)
        lift_z = self.carry_z if lift_z is None else lift_z
        defaults = (self.open_margin, self.grasp_clearance)
        self.open_margin, self.grasp_clearance = self.tall.get(name, defaults)
        try:
            for attempt in range(attempts):
                obj = self.grasp_center(name)
                base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
                angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
                rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
                yield from self.gripper(self.open_for(width), 0.4, 0.0)
                pre = grasp + [0, 0, approach]
                yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
                yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
                yield from self.wait(self.grasp_settle)
                yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
                yield from self.gripper(close, self.close_seconds, 0.3)
                ok = self.env.is_grasping(name)
                self.log.append((name, "grasp", attempt, ok))
                if ok:
                    yield from self.move(np.array([grasp[0], grasp[1], lift_z]), rot, speed=0.10, label="lift")
                    if self.env.is_grasping(name):
                        return True
                    self.log.append((name, "dropped", attempt))
                yield from self.gripper(self.open_for(width), 0.4, 0.2)
                yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
            return False
        finally:
            self.open_margin, self.grasp_clearance = defaults

    def release(self, open_to=None):
        width = getattr(self, "_held_width", None)
        if width is not None and self.squeeze is not None:
            yield from self.gripper(self.gap_angle(width - self.squeeze), self.relax_seconds, 0.2)
        yield from super().release(open_to)

    def level_rot(self, name, rot):
        """TCP rotation near ``rot`` at which the held object sits level (cancels its in-hand pivot)."""
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        return _rot_z(np.arctan2(m[1, 0], m[0, 0])) @ r_obj_tcp.T

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        carry_z = self.carry_z if carry_z is None else carry_z
        target = np.asarray(target, float)
        rot = self.carry_rot(target[:2]) if rot is None else rot
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], carry_z], rot)
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], carry_z], rot, speed=0.05, label="align", smooth=False)
        yield from self.move(tcp + [0, 0, self.pre_release_height], rot, speed=0.06, tol=0.002, settle=0.6,
                             label="pre-lower")
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + drop + lift], rot, speed=0.04, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(open_to)
        yield from self.move(np.r_[self._cmd_pos[:2], carry_z], rot, speed=0.10, label="retreat")

    def plan(self):
        env = self.env
        for name in env.order:
            goal = next(g for g in env.goals if g.obj == name)
            width, height, offset, sym = self.grips[name]
            pos = env.object_pos(name)
            grasp_z = pos[2] - 0.001 if height is None else pos[2] - env._extent[name]["bottom"] + height
            ok = yield from self.pick(name, width, grasp_z, yaw=env.yaw(name) + offset, symmetric=sym)
            if not ok:
                return
            yield from self.place_object(name, env.goal_target(goal), drop=self.release_drops.get(name, self.release_drop),
                                         open_to=self.open_for(width, self.release_margins.get(name, self.release_margin)))
        yield from self.rest()
        yield from self.wait(1.2)


NEUTRAL = ("plate", "ramekin", "porcelain_mug")


# ----- task 1: food (soup can) into the basket, toy (cube) into the bowl; can first --------------------------

class CanBasketCubeBowlEnv(SortEnv):
    instruction = ("Sort these by kind: food goes in the basket and toy blocks go in the bowl. "
                   "Start with the soup can.")
    task_objects = ("soup_can", "cube")
    order = ("soup_can", "cube")
    targets = {"soup_can": "basket", "cube": "bowl"}
    fixtures = ("bowl",)
    box_inner = {"basket": (0.043, 0.043)}
    inside_margin = {"soup_can": 0.0155 + 0.006}
    radii = {"bowl": 0.018}
    distractor_pool = NEUTRAL

    def scene_objects(self):
        return [SolidLibero("soup_can", "alphabet_soup"), Block("cube", half=(CUBE,) * 3, rgba=RED),
                ScaledAsset("basket", "basket", scale=(0.7, 0.7, 0.4), mass=0.15)]

    def scene_fixtures(self):
        return [Fixture("bowl", "white_bowl", 1.0)]


class CanBasketCubeBowlOracle(SortOracle):
    carry_z = 0.09              # the held can's bottom ~1.5 cm above the 6 cm basket rim
    # As put_in_container's basket oracle: a slower swing keeps the 32 g can from turning in the jaws.
    vmax = np.radians([40.0, 50.0, 50.0, 60.0, 60.0])
    grips = {"soup_can": (CAN_W, 0.015, 0.0, 4), "cube": (2 * CUBE, None, 0.0, 4)}
    hold_squeeze = {"soup_can": 0.010}
    tall = {"soup_can": (0.014, 0.0015)}
    release_margin = 0.016
    # The cube rests on the bowl's curved floor: released at floor contact with a wide jaw, the opening moving
    # jaw pressed the bowl wall at 6-17 N (all 6 calibration seeds); released 4 mm above the centre floor it
    # landed on a corner (1200-1800 rad/s^2, 10/50 qualification seeds). The goal height is now the cube's true
    # resting height on the curved floor (``goal_for``); release 1 mm above it with a narrow opening.
    release_drops = {"cube": 0.001}
    release_margins = {"cube": 0.008}
    release_backoff = 0.006
    close_seconds = 1.6


# ----- task 2: toy (long block) into the cardboard box, food (tomato sauce) into the wooden crate; block first -

class BlockBoxSauceCrateEnv(SortEnv):
    instruction = ("Put the toy blocks in the cardboard box and the food in the wooden crate. "
                   "Begin with the long block.")
    task_objects = ("long_block", "tomato_sauce")
    order = ("long_block", "tomato_sauce")
    targets = {"long_block": "cardboard_box", "tomato_sauce": "crate"}
    box_inner = {"cardboard_box": (0.05, 0.042), "crate": (0.046, 0.042)}
    floor_height = {"cardboard_box": 0.006, "crate": 0.006}
    inside_margin = {"long_block": 0.008, "tomato_sauce": 0.0155 + 0.004}
    distractor_pool = NEUTRAL + ("akita_black_bowl", "white_bowl")

    def scene_objects(self):
        return [Block("long_block", half=BAR, rgba=BLUE), SolidLibero("tomato_sauce", "tomato_sauce"),
                Crate("cardboard_box", inner=(0.05, 0.042), wall=0.026, thickness=0.004, rgba=CARDBOARD, mass=1.0),
                Crate("crate", inner=(0.046, 0.042), wall=0.018, rgba=WOOD, mass=1.0)]

    def inside_box(self, cont):
        check = super().inside_box(cont)
        if cont != "cardboard_box":
            return check

        def block_check(env, name):
            # The long block: its long half-length must clear the walls along whichever axis it lies.
            local = np.abs(_local_xy(env, name, cont))
            inner = np.asarray(env.box_inner[cont])
            return bool(np.all(local <= inner - 0.004) and min(inner - local) >= 0.004)
        return block_check


class BlockBoxSauceCrateOracle(SortOracle):
    carry_z = 0.085
    grips = {"long_block": (2 * BAR[1], 0.012, np.pi / 2, 2), "tomato_sauce": (CAN_W, 0.015, 0.0, 4)}
    hold_squeeze = {"tomato_sauce": 0.010}
    tall = {"tomato_sauce": (0.014, 0.0015)}
    close_seconds = 1.6

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        if name == "long_block" and rot is None:
            # Square the block with the box: least wrist turn beyond following the arm's azimuth.
            env = self.env
            box_yaw = env.yaw("cardboard_box")
            azimuth = np.arctan2(target[1], target[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
            base = box_yaw - env.yaw(name)
            deltas = sorted((base + k * np.pi / 2 for k in range(-4, 5)),
                            key=lambda d: abs(np.angle(np.exp(1j * (d - azimuth)))))
            rotations = [_rot_z(d) @ self._cmd_rot for d in deltas[:2]]
            rot = self.feasible_rotation(name, target, rotations)
        return (yield from super().place_object(name, target, rot=rot, drop=drop, open_to=open_to, carry_z=carry_z))


TASKS = [
    define_task(name="sort_can_to_basket_cube_to_bowl", instruction=CanBasketCubeBowlEnv.instruction,
                family=FAMILY, env=CanBasketCubeBowlEnv, oracle=CanBasketCubeBowlOracle,
                objects=("soup_can", "cube"), object_kinds=("food can", "toy block"),
                relation="sorted by kind into, food first", goal="basket for food and bowl for toy blocks",
                steps=("Pick up the soup can and put it in the basket, because it is food.",
                       then("Pick up the cube and put it in the bowl, because it is a toy block."))),
    define_task(name="sort_block_to_box_sauce_to_crate", instruction=BlockBoxSauceCrateEnv.instruction,
                family=FAMILY, env=BlockBoxSauceCrateEnv, oracle=BlockBoxSauceCrateOracle,
                objects=("long_block", "tomato_sauce"), object_kinds=("toy block", "food can"),
                relation="sorted by kind into, toy first", goal="cardboard box for toy blocks and wooden crate for food",
                steps=("Pick up the long block and put it in the cardboard box, because it is a toy block.",
                       then("Pick up the tomato sauce and put it in the wooden crate, because it is food."))),
]

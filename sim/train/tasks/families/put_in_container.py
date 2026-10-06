"""put_in_container: pick one object and put it inside a container.

Three distinct pick-and-insert tasks (different object kind and receptacle kind each):

* ``soup_can_in_basket``      -- a soup can into the LIBERO basket, rescaled wider and shallower (0.7, 0.7,
  0.4) so the jaws fit inside; carried over its 6 cm rim.
* ``golf_ball_in_ramekin``    -- a YCB golf ball (recentred mesh) into a 1.15x LIBERO ramekin.
* ``pudding_in_cardboard_box`` -- a chocolate-pudding box (0.7x) into an open cardboard box built from
  primitives, squared with the box walls; containment checked in the box's own frame.

Containment is strict: the object must be released, settled, upright where that is meaningful, carried by the
container's floor (positive upward contact force from the container body) and inside its walls. None of these
reproduce the held-out validation semantics (blocks on plates, bowl in bowl, mug on plate, mugs in microwave,
pan on stove).
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, Goal, Scanned, TrainEnv, TrainOracle, define_task,
                                  step_text)
from sim.val.scene import (ASSETS, CATALOG, Fixture, Obj, _add_assets, _collision_geoms, _content, _hull_volume,
                           _merge_defaults, _set, _set_free_physics, _soften, _upright_quat, _vec, load_mjcf)

FAMILY = "put_in_container"


# ----- an open cardboard box built from primitives (no open-top box exists in the LIBERO catalog) -----------

@dataclass
class OpenBox:
    """A free, open-topped rectangular box (floor + four walls). ``inner`` is the inner half-size (x, y),
    ``depth`` the outer height. Uses the scene builder's ``build_mjcf`` extension point (as ``Scanned`` does)."""

    name: str
    inner: tuple = (0.05, 0.045)
    depth: float = 0.035
    wall: float = 0.004
    floor: float = 0.004
    rgba: tuple = (0.70, 0.53, 0.34, 1.0)
    mass: float = 0.25
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ix, iy = self.inner
        w, h, f = self.wall, self.depth, self.floor
        rgba = " ".join(map(str, self.rgba))
        wall_h = (h - f) / 2
        parts = [("floor", (0, 0, f / 2), (ix + w, iy + w, f / 2), 0.5),
                 ("wall_px", (ix + w / 2, 0, f + wall_h), (w / 2, iy + w, wall_h), 0.125),
                 ("wall_nx", (-ix - w / 2, 0, f + wall_h), (w / 2, iy + w, wall_h), 0.125),
                 ("wall_py", (0, iy + w / 2, f + wall_h), (ix, w / 2, wall_h), 0.125),
                 ("wall_ny", (0, -iy - w / 2, f + wall_h), (ix, w / 2, wall_h), 0.125)]
        for label, pos, size, share in parts:
            ET.SubElement(body, "geom", name=f"{self.name}_{label}", type="box",
                          pos=" ".join(f"{v:.6g}" for v in pos), size=" ".join(f"{v:.6g}" for v in size),
                          rgba=rgba, mass=f"{self.mass * share:.6g}", group="1", condim="4",
                          friction=f"{self.friction} 0.02 0.001", material="val_fabric")


@dataclass
class ScaledAsset:
    """A free LIBERO catalog object scaled per axis (x, y, z). The LIBERO basket is 7 cm deep at 0.5x and
    narrower inside (6.2-6.8 cm) than the SO-101 jaws (about 6.1 cm along the closing direction near the
    tips), so a top-down insertion cannot fit; a wider, shallower basket (0.7, 0.7, 0.4) leaves room for the
    jaws and a carry height the arm can reach. Built through the scene builder's ``build_mjcf`` extension
    point; box collision sizes are scaled along the world axis each local axis maps to."""

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
                axes = np.abs(R.reshape(3, 3)).argmax(axis=0)      # world axis each local axis maps to
                _set(el, "size", _vec(el, "size") * S[axes])
        _soften(root.find("asset"))
        _set_free_physics(root, self.mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright")
        upright.extend(_content(root))


@dataclass
class SolidLibero:
    """A free LIBERO catalog object whose collision boxes are replaced by one solid box over their extent.
    LIBERO cans collide as a ring of 1-2 mm plates around a core box; the jaws caught plate edges and corners,
    so a closing jaw spun the can (1000-3500 rad/s^2) and a grip pressed a corner 3 mm into a finger. A single
    box of the same 3.1 x 3.1 x 3.8 cm extent, gripped across two faces, closes and releases cleanly. The
    visual mesh and the mass (hull volume x catalog density) are unchanged."""

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
class CenteredScan(Scanned):
    """A scanned object whose body origin is moved to the centre of its collision geometry. The YCB golf-ball
    mesh origin sits ~1.5 cm off the ball centre, which would make grasp/goal targets depend on how it rolled.
    """

    def build_mjcf(self, mj, asset_el, worldbody):
        from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose
        super().build_mjcf(mj, asset_el, worldbody)
        _, acc = self._accessors()
        coll = [(f"{self.name}_scan_coll_{k}", part) for k, part in enumerate(acc.collision_parts(self.model_id))]
        verts = self._body_verts(coll, self.scale)
        quat = get_mujoco_ycb_rest_pose(verts, model_id=self.model_id)[0]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        centre = R.reshape(3, 3) @ ((verts.min(0) + verts.max(0)) / 2)
        upright = next(b for b in worldbody.iter("body") if b.get("name") == f"{self.name}_upright")
        _set(upright, "pos", -centre)


# Pick region: r >= 0.18 because a close pick at a large azimuth (r 0.168, 51 deg) folded the wrist-camera
# mount into the shoulder on the pregrasp transit.
OBJECT_REGION = dict(r=(0.18, 0.26), angle=(-65, 65))


def _without(*names):
    return tuple(n for n in LOW_DISTRACTORS if n not in names)


def _local_xy(env, name, ref):
    """xy of body ``name`` in the frame of body ``ref`` (about its vertical axis)."""
    m = env.data.xmat[env._body[ref]].reshape(3, 3)
    return (m.T @ (env.object_pos(name) - env.object_pos(ref)))[:2]


# ----- oracle with a bounded grip ------------------------------------------------------------------------

class GentleOracle(TrainOracle):
    """``TrainOracle`` that relaxes the grip before releasing. The grasp still closes fully (a secure carry),
    but once the object rests on the container floor the jaw first eases to a gap ``squeeze`` below the
    object width, so the PD grip force -- and the contact force stored in the fixed finger -- is small when
    the jaw opens. With a direct open from full close, small light objects (thin food box) sprang off the
    fixed finger and chattered on the floor (1000-2500 rad/s^2 measured)."""

    squeeze = 0.006
    relax_seconds = 1.0
    release_seconds = 1.5
    release_offset = -0.0005    # release resting on the floor: even a 2 mm fall slaps a small object (>1000 rad/s^2)

    grip_squeeze = None         # close to a gap this far below the width (None: fully closed, the kit default)

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """As ``TrainOracle.pick``; with ``grip_squeeze`` set, the jaw closes only to ``width - grip_squeeze``
        (a full close pressed the jaws up to 3.2 mm into a can or ball while it was carried)."""
        self._held_width = width
        close = CLOSED if self.grip_squeeze is None else self.gap_angle(width - self.grip_squeeze)
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.grasp_center(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            # Two-stage close: quickly to just short of contact, then slowly onto the object, so the moving
            # jaw meets it at low speed (a single min-jerk close hit a can at mid-stroke speed and spun it).
            yield from self.gripper(self.gap_angle(width + 0.005), 0.5, 0.1)
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

    def grasp_center(self, name):
        """Centre of the object's collision AABB in its own frame, in world coordinates. LIBERO body origins
        can sit off the collision centre (1.8 mm for the soup can), which put the fixed finger onto the can's
        rim during the descent."""
        env = self.env
        local = self._local_points(name)
        mid = (local.min(0) + local.max(0)) / 2
        body = env._body[name]
        return env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ np.r_[mid[:2], 0.0]

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def relaxed_angle(self, width):
        return self.gap_angle(width - self.squeeze)

    def release(self, open_to=None):
        width = getattr(self, "_held_width", None)
        if width is not None and self.squeeze is not None:
            yield from self.gripper(self.relaxed_angle(width), self.relax_seconds, 0.2)
        yield from super().release(open_to)


# ----- soup can into the basket ----------------------------------------------------------------------------

BASKET = ScaledAsset("basket", "basket", scale=(0.7, 0.7, 0.4), mass=0.15)
BASKET_INNER = 0.043            # inner half-width of the scaled basket (x 4.6-4.8 cm, y 4.3 cm)


class CanBasketEnv(TrainEnv):
    instruction = "Put the soup can in the basket."
    task_objects = ("soup_can",)
    distractor_pool = _without("alphabet_soup", "tomato_sauce")

    def scene_objects(self):
        return [SolidLibero("soup_can", "alphabet_soup"), BASKET]

    def layout(self):
        placed = []
        basket_xy, _ = self.place("basket", placed, region=dict(r=(0.19, 0.25), angle=(-60, 60)))
        self.place("soup_can", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        floor = self.surface_z(basket_xy, exclude=self.task_objects)
        bottom = self._extent["soup_can"]["bottom"]
        self.set_goals(Goal("soup_can", "basket", target=(*basket_xy, floor + bottom), reference="basket",
                            tolerance=(0.03, 0.03, 0.006), check=self.inside))

    def inside(self, env, name):
        # Can centre at least its radius plus 6 mm inside every basket wall.
        return bool(np.all(np.abs(_local_xy(env, name, "basket")) <= BASKET_INNER - 0.0155 - 0.006))


CAN_W = 0.0312                  # the can's solid collision box (see SolidLibero), gripped across two faces


class CanBasketOracle(GentleOracle):
    carry_z = 0.09              # held can bottom ~1.5 cm above the 6 cm basket rim
    grip_squeeze = 0.010
    # Slower than the kit (60/60/60/75/75 deg/s): swinging the 32 g can across the front at full pan speed
    # rotated it in the jaws until a corner of its collision polygon pressed 3 mm into a finger.
    vmax = np.radians([40.0, 50.0, 50.0, 60.0, 60.0])
    release_margin = 0.020      # open wide in the roomy basket: a tight opening left the can wedged on a finger
    release_backoff = 0.006
    close_seconds = 2.0
    grasp_height = 0.015        # TCP above the can bottom: a low grasp keeps the can high while carried

    def plan(self):
        env = self.env
        can = env.object_pos("soup_can")
        bottom = can[2] - env._extent["soup_can"]["bottom"]
        ok = yield from self.pick("soup_can", CAN_W, bottom + self.grasp_height, yaw=env.yaw("soup_can"),
                                  symmetric=4)
        if not ok:
            return
        yield from self.place_object("soup_can", env.goal_target(env.goals[0]) + [0, 0, self.release_offset],
                                     open_to=self.release_for(CAN_W))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- golf ball into the ramekin --------------------------------------------------------------------------

BALL_D = 0.0256                 # YCB golf ball at 0.6x (collision sphere radius ~12.8 mm)


class BallRamekinEnv(TrainEnv):
    instruction = "Put the golf ball in the ramekin."
    task_objects = ("golf_ball",)
    distractor_pool = _without("ramekin")

    def scene_objects(self):
        return [CenteredScan("golf_ball", "ycb", "058_golf_ball", scale=0.6, mass=0.012)]

    def scene_fixtures(self):
        return [Fixture("ramekin", "ramekin", 1.15)]

    def layout(self):
        placed = []
        ramekin_xy, _ = self.place_fixture("ramekin", placed, region=dict(r=(0.16, 0.26), angle=(-65, 65)))
        self.place("golf_ball", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        floor = self.surface_z(ramekin_xy, exclude=self.task_objects)
        self.set_goals(Goal("golf_ball", "ramekin", target=(*ramekin_xy, floor + BALL_D / 2),
                            tolerance=(0.012, 0.012, 0.005), upright_cos=None))


class BallRamekinOracle(GentleOracle):
    grip_squeeze = 0.006        # 12 mm pressed the fingertips 3.2 mm into the ball; 4 mm let it slip out
    # Slower than the kit: long carries across the front (ball and ramekin on opposite sides) at full pan
    # speed rolled the ball in the jaws until they pressed 3.0-3.3 mm into its hull (5/50 qualification seeds).
    vmax = np.radians([40.0, 50.0, 50.0, 60.0, 60.0])
    def plan(self):
        env = self.env
        ball = env.object_pos("golf_ball")
        ok = yield from self.pick("golf_ball", BALL_D, ball[2] - 0.001, yaw=None, symmetric=4)
        if not ok:
            return
        yield from self.place_object("golf_ball", env.goal_target(env.goals[0]) + [0, 0, self.release_offset],
                                     open_to=self.release_for(BALL_D))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- chocolate pudding into the cardboard box ------------------------------------------------------------

PUDDING_SCALE = 0.7
PUDDING_W = 0.023 * PUDDING_SCALE / 0.5     # short side (local y) of the pudding box
PUDDING_HALF_L = 0.020 * PUDDING_SCALE / 0.5
BOX = OpenBox("box")


class PuddingBoxEnv(TrainEnv):
    instruction = "Put the chocolate pudding in the cardboard box."
    task_objects = ("pudding",)
    distractor_pool = _without("chocolate_pudding")

    def scene_objects(self):
        # 0.7x (5.6 x 3.2 x 2.0 cm): at 0.5x the 1.4 cm box overlaps the jaw pads by only ~8 mm and slid out
        # of the fingertips during the carry in 3/10 calibration seeds.
        return [Obj("pudding", "chocolate_pudding", scale=PUDDING_SCALE), BOX]

    def layout(self):
        placed = []
        box_xy, _ = self.place("box", placed, region=dict(r=(0.18, 0.25), angle=(-60, 60)))
        self.place("pudding", placed, region=OBJECT_REGION)
        self.place_distractors(placed)
        floor = self.object_pos("box")[2] - self._extent["box"]["bottom"] + BOX.floor
        bottom = self._extent["pudding"]["bottom"]
        self.set_goals(Goal("pudding", "box", target=(*box_xy, floor + bottom), reference="box",
                            tolerance=(0.03, 0.03, 0.005), check=self.inside))

    def inside(self, env, name):
        # Pudding centre at least its half-length plus 2 mm inside each inner wall (either orientation).
        local = np.abs(_local_xy(env, name, "box"))
        return bool(np.all(local <= np.asarray(BOX.inner) - PUDDING_HALF_L - 0.002))


class PuddingBoxOracle(GentleOracle):
    grasp_z = 0.011             # TCP height (box 2.0 cm tall); the jaw hull reaches ~8.2 mm below the TCP
    close_seconds = 2.0         # the thin box tips (>1000 rad/s^2) under the default 1.2 s close
    release_offset = -0.0005    # release resting on the floor: even a 2 mm fall slaps the thin box (>1000 rad/s^2)

    def plan(self):
        env = self.env
        ok = yield from self.pick("pudding", PUDDING_W, self.grasp_z, yaw=env.yaw("pudding") + np.pi / 2,
                                  symmetric=2)
        if not ok:
            return
        # Square the pudding with the box walls (either axis: it fits both ways). Prefer the end yaw that needs
        # the least wrist turn beyond following the arm's azimuth, so the carry stays a smooth Cartesian arc
        # (a joint-space swing flung the low-held box out of the jaws).
        box_xy = env.object_pos("box")[:2]
        azimuth = np.arctan2(box_xy[1], box_xy[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        base = env.yaw("box") - env.yaw("pudding")
        deltas = sorted((base + k * np.pi / 2 for k in range(-4, 5)),
                        key=lambda d: abs(np.angle(np.exp(1j * (d - azimuth)))))
        rotations = [np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]]) @ self._cmd_rot
                     for d in deltas[:2]]
        target = env.goal_target(env.goals[0]) + [0, 0, self.release_offset]
        rot = self.feasible_rotation("pudding", target, rotations)
        yield from self.place_object("pudding", target, rot=rot, open_to=self.release_for(PUDDING_W))
        yield from self.rest()
        yield from self.wait(1.2)


TASKS = [
    define_task(name="soup_can_in_basket", instruction=CanBasketEnv.instruction, family=FAMILY,
                env=CanBasketEnv, oracle=CanBasketOracle, objects=("soup_can",), object_kinds=("can",),
                relation="inside", goal="basket",
                steps=(step_text("put", "soup can", "in", "basket"),)),
    define_task(name="golf_ball_in_ramekin", instruction=BallRamekinEnv.instruction, family=FAMILY,
                env=BallRamekinEnv, oracle=BallRamekinOracle, objects=("golf_ball",), object_kinds=("ball",),
                relation="inside", goal="ramekin",
                steps=(step_text("put", "golf ball", "in", "ramekin"),)),
    define_task(name="pudding_in_cardboard_box", instruction=PuddingBoxEnv.instruction, family=FAMILY,
                env=PuddingBoxEnv, oracle=PuddingBoxOracle, objects=("pudding",),
                object_kinds=("food box",), relation="inside", goal="cardboard box",
                steps=(step_text("put", "chocolate pudding", "in", "cardboard box"),)),
]

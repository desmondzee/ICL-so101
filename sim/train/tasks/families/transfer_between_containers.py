"""Transfer between containers: lift an object out of one container and put it inside another.

Both containers are free bodies (not static fixtures), so "the source container stays where it was" is a real
physical condition: the strict gate already limits every non-task free body to <= 5 mm / 0.1 rad of motion, and
the success predicate additionally requires the source container to be upright, within 5 mm of its start, and
no longer touching the object. The object must end inside the destination: on its floor (z within tolerance,
so resting on the rim fails), near its centre, released and settled, and supported by the destination.

Visibility goal regions are placed in the destination's opening (rim plane), not on its floor, because the
near wall hides a container floor from the low front camera.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import COS10, Block, Goal, Obj, TrainEnv, TrainOracle, define_task
from sim.train.variation import GoalRegion
from sim.val.oracle import FIXED_FACE, top_down_mat

HALF = 0.014                                   # 2.8 cm cube
CONTAINER_REGION = dict(r=(0.17, 0.255), angle=(-62.0, 62.0))
SOURCE_SHIFT = 0.005                           # max source-container displacement still counted as undisturbed
# Low, non-container distractors: a bowl or ramekin distractor would make "the bowl" ambiguous.
PLAIN_DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn",
                     "plate")


@dataclass
class Primitive:
    """A free primitive solid (``"cylinder"`` standing on its base, or ``"sphere"``), built through the scene's
    ``build_mjcf`` extension point. ``size`` is MuJoCo's: (radius, half-height) or (radius,)."""

    name: str
    shape: str
    size: tuple
    rgba: tuple = (0.8, 0.1, 0.1, 1.0)
    mass: float = 0.03
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "geom", name=f"{self.name}_geom", type=self.shape, size=" ".join(map(str, self.size)),
                      rgba=" ".join(map(str, self.rgba)), mass=str(self.mass), group="1", condim="4",
                      friction=f"{self.friction} 0.02 0.001", material="val_plastic")


class TransferEnv(TrainEnv):
    """Object ``item`` starts on the floor of container ``source`` and must end inside ``dest``."""

    task_objects = ("item",)
    surfaces = ("source",)                     # the item starts on the source floor; jaws may brush it
    distractor_pool = PLAIN_DISTRACTORS
    item_upright = True                        # False for items with no upright (e.g. a ball)
    goal_xy_tolerance = 0.02
    goal_z_tolerance = 0.008
    start_jitter = 0.003

    def item_spec(self):
        raise NotImplementedError

    def container_specs(self):                 # (source, dest) free-body specs named "source" and "dest"
        raise NotImplementedError

    def scene_objects(self):
        return [self.item_spec(), *self.container_specs()]

    # ----- geometry --------------------------------------------------------------------------------------------
    def aabb(self, name):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in self._geoms[name]:
            c = self.data.geom_xpos[g] + self.data.geom_xmat[g].reshape(3, 3) @ self.model.geom_aabb[g, :3]
            h = np.abs(self.data.geom_xmat[g].reshape(3, 3)) @ self.model.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        return lo, hi

    def centre(self, name):
        """xy of a free body's collision-AABB centre (bowl/ramekin origins sit within ~1 mm of it)."""
        lo, hi = self.aabb(name)
        return (lo[:2] + hi[:2]) / 2

    def floor_z(self, name):
        xy = self.centre(name)
        return min(self.surface_z(xy + d, exclude=self.task_objects)
                   for d in (np.zeros(2), [0.004, 0], [-0.004, 0], [0, 0.004], [0, -0.004]))

    # ----- layout ----------------------------------------------------------------------------------------------
    def layout(self):
        placed = []
        self.place("source", placed, region=CONTAINER_REGION)
        self.place("dest", placed, region=CONTAINER_REGION)
        src = self.centre("source")
        self._source_start = src.copy()
        self.set_object_pose("item", src + self.np_random.uniform(-self.start_jitter, self.start_jitter, 2),
                             yaw=self.np_random.uniform(-np.pi, np.pi), z=self.floor_z("source"))
        self.place_distractors(placed)
        target = (*self.centre("dest"), self.floor_z("dest") + self._extent["item"]["bottom"] + 0.001)
        self.set_goals(Goal("item", "dest", target=target, reference="dest",
                            tolerance=(self.goal_xy_tolerance, self.goal_xy_tolerance, self.goal_z_tolerance),
                            upright_cos=COS10 if self.item_upright else None, check=source_undisturbed))

    def source_ok(self) -> bool:
        """Source container upright, within 5 mm of where it started, and no longer touching the item."""
        return bool(np.linalg.norm(self.centre("source") - self._source_start) <= SOURCE_SHIFT
                    and self.upright("source", COS10) and not self.touching("item", "source"))

    # ----- visibility ------------------------------------------------------------------------------------------
    def goal_regions(self):
        """Samples across the destination's opening at rim height (its floor is hidden by the near wall)."""
        c = self.centre("dest")
        z = self.aabb("dest")[1][2] + 0.002
        s = 0.35 * self.footprint("dest") / np.sqrt(2)
        return (GoalRegion("dest", tuple((c[0] + x, c[1] + y, z) for x in (-s, 0, s) for y in (-s, 0, s))),)


def source_undisturbed(env, obj):
    return env.source_ok()


def item_points(env, name="item"):
    """World points on the item's collision geometry (mesh vertices, box corners, cylinder/sphere rims)."""
    m, d = env.model, env.data
    pts = []
    ring = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for g in env._geoms[name]:
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        kind, size = m.geom_type[g], m.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
            r = size[0]
            h = size[1] if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else 0.0
            local = np.array([(r * np.cos(a), r * np.sin(a), z) for a in ring for z in (-h, h)])
        else:
            corners = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
            local = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
        pts.append(local @ R.T + p)
    return np.vstack(pts)


class TransferOracle(TrainOracle):
    open_margin = 0.008                        # open just enough inside the source container
    width = 2 * HALF
    symmetric = 4
    radial = False                             # grasp round objects along the radial direction
    grasp_height = None                        # TCP height above the item's base (default: its middle)
    yaw_offset = 0.0                           # closing direction relative to the item's yaw
    drop = 0.0                                 # release height above the goal target (1 mm above the floor)

    def grasp_plan(self, center, width, angles):
        """As the kit, but the fixed finger is placed ``grasp_clearance`` from the item's *actual* collision
        surface along each closing direction, not from ``origin + width/2`` (origins of scanned meshes are off
        their axes, and round items roll across any gap the closing jaw pushes them over)."""
        pts = item_points(self.env)
        best = None
        for a in angles:
            rot = top_down_mat(a)
            x, y = rot[:, 0], rot[:, 1]
            px, py = pts @ x, pts @ y
            mid = x * (px.min() + px.max()) / 2 + y * (py.min() + py.max()) / 2
            pos = np.r_[mid[:2], center[2]] - (FIXED_FACE - np.ptp(px) / 2 - self.grasp_clearance) * x
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def plan(self):
        env = self.env
        goal = env.goals[0]
        p = env.object_pos("item")
        yaw = float(np.arctan2(p[1], p[0])) if self.radial else env.yaw("item") + self.yaw_offset
        lo, hi = env.aabb("item")
        grasp_z = (lo[2] + hi[2]) / 2 - 0.001 if self.grasp_height is None else lo[2] + self.grasp_height
        ok = yield from self.pick_and_place("item", env.goal_target(goal), width=self.width, yaw=yaw,
                                            grasp_z=float(grasp_z), symmetric=self.symmetric,
                                            drop=self.drop)
        if not ok:
            return
        yield from self.rest()
        yield from self.wait(1.2)


# ----- block: white bowl -> ramekin --------------------------------------------------------------------------

class BlockBowlToRamekinEnv(TransferEnv):
    instruction = "Move the block from the bowl into the ramekin."

    def item_spec(self):
        return Block("item", half=(HALF,) * 3, rgba=(0.85, 0.45, 0.1, 1.0))

    def container_specs(self):
        return [Obj("source", "white_bowl", 1.2), Obj("dest", "ramekin", 1.0)]


class BlockBowlToRamekinOracle(TransferOracle):
    pass


# ----- short cylinder: white bowl -> black bowl ----------------------------------------------------------------

CYL_R, CYL_H = 0.015, 0.013                    # 3.0 cm across, 2.6 cm tall


class CylinderBowlToBowlEnv(TransferEnv):
    instruction = "Move the cylinder from the white bowl into the black bowl."

    def item_spec(self):
        # Glazed bowl, smooth plastic (friction 0.5; MuJoCo uses the larger of a pair, so the bowl is set too):
        # the closing jaw then slides the cylinder the last 0.8 mm instead of rocking it on its base rim.
        return Primitive("item", "cylinder", (CYL_R, CYL_H), rgba=(0.15, 0.55, 0.3, 1.0), mass=0.04, friction=0.5)

    def container_specs(self):
        return [Obj("source", "white_bowl", 1.2, friction=0.5), Obj("dest", "akita_black_bowl", 1.0)]


class CylinderBowlToBowlOracle(TransferOracle):
    width = 2 * CYL_R
    radial = True
    grasp_clearance = 0.0008
    release_margin = 0.014                     # else the opened jaw still pinches the round side on backoff
    close_seconds = 1.6
    release_seconds = 1.5
    # The jaw-gap table measures fingertip-sphere centres, so a 30 mm item at the 8 mm margin leaves ~1 mm for
    # the descending moving jaw, whose tips then land on the round top rim. The flat floors leave room for more.
    open_margin = 0.014


# ----- long block: black bowl -> ramekin ---------------------------------------------------------------------

BAR = (0.022, 0.010, 0.013)                    # 4.4 x 2.0 x 2.6 cm


class BarBowlToRamekinEnv(TransferEnv):
    instruction = "Move the long block from the black bowl into the ramekin."

    def item_spec(self):
        return Block("item", half=BAR, rgba=(0.2, 0.35, 0.8, 1.0))

    def container_specs(self):
        return [Obj("source", "akita_black_bowl", 1.0), Obj("dest", "ramekin", 1.0)]


class BarBowlToRamekinOracle(TransferOracle):
    width = 2 * BAR[1]                         # close across the bar's narrow (local y) side
    symmetric = 2
    yaw_offset = np.pi / 2
    grasp_clearance = 0.0006                   # less sideways push on close, so the bar is not clamped tilted


FAMILY = "transfer_between_containers"

TASKS = [
    define_task(name="block_bowl_to_ramekin", instruction=BlockBowlToRamekinEnv.instruction, family=FAMILY,
                env=BlockBowlToRamekinEnv, oracle=BlockBowlToRamekinOracle, objects=("item",),
                object_kinds=("block",), relation="from bowl into", goal="ramekin",
                steps=("Pick up the block from inside the bowl and put it inside the ramekin.",)),
    define_task(name="cylinder_bowl_to_bowl", instruction=CylinderBowlToBowlEnv.instruction, family=FAMILY,
                env=CylinderBowlToBowlEnv, oracle=CylinderBowlToBowlOracle, objects=("item",),
                object_kinds=("cylinder",), relation="from bowl into", goal="bowl",
                steps=("Pick up the cylinder from inside the white bowl and put it inside the black bowl.",)),
    define_task(name="long_block_bowl_to_ramekin", instruction=BarBowlToRamekinEnv.instruction, family=FAMILY,
                env=BarBowlToRamekinEnv, oracle=BarBowlToRamekinOracle, objects=("item",),
                object_kinds=("rectangular block",), relation="from bowl into", goal="ramekin",
                steps=("Pick up the long block from inside the black bowl and put it inside the ramekin.",)),
]

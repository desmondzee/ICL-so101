"""nest_containers: put a small container into a larger one.

Two tasks that differ in the nested container and the receiving container:

* ``ramekin_in_bowl``         -- a LIBERO ramekin (0.4x) set on the floor of a large black bowl (static fixture).
* ``cup_in_bin``              -- a small plastic cup set on the floor of an open-top storage bin.

Held out: validation's "put the white bowl in the black bowl" (bowl inside bowl). No task here nests a bowl in a
bowl, and bowls/ramekins are never distractors (so "the bowl" names exactly one object).

Collision of the nested containers (measured 2026-10-06): the LIBERO ramekin and bowls collide as rings of
0.5-2.6 mm thin wall plates, and a closing fingertip sphere (0.75 mm radius) presses into a plate past its half
thickness (angular-acceleration spikes on the first jaw touch, see take_out_of_container / center_on_target). Nothing
is put *into* the nested container in these tasks, so its collision is replaced by a solid octagonal prism (four
overlapping boxes over the visual mesh's extent): box/box contacts grip, release and rest cleanly and the jaws close on
two parallel faces. The visual mesh is unchanged.

Supports: the bowl is a static fixture; the storage bin is a free body made heavy (a free receiving container that is
knocked is caught by the strict gate, <= 5 mm / 0.1 rad).

Dropped: ``bowl_in_wooden_tray`` (a small LIBERO bowl, 0.5x, into the LIBERO wooden tray). Best variant 19/24
calibration seeds (79 %, below the 86 % admission bar). The tray's three floor boards (tilted ~0.3 deg, tops 0.1 mm
apart) were merged into one flat board (fixed the settle jitter), and the jaws were turned so the carry stayed a
Cartesian move (a joint-space transit dipped the bowl into the tray wall). What remained: the 1.8 cm-tall bowl
creeps 1-2 cm down/across the jaw pads during the carry (grip held mostly by the fingertip spheres; 2 seeds failed
in every one of 9 variants of squeeze, mass 0.02-0.06 kg, friction, grasp height, scale 0.6x and contact
stiffness), and lands tilted on release (1000-1300 rad/s^2).

Success (kit goal plus ``inside``): the nested container is released, settled, upright, supported by the receiving
container, its base within ``tolerance`` of the receiving container's floor, its centre within the receiving
container's inner floor area, and it does not touch the table.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task,
                                  step_text, top_down_mat)
from sim.train.variation import GoalRegion
from sim.val.oracle import FIXED_FACE
from sim.val.scene import (ASSETS, CATALOG, Fixture, _add_assets, _content, _merge_defaults, _soften, load_mjcf)

FAMILY = "nest_containers"
CONTAINERS = ("white_bowl", "red_bowl", "akita_black_bowl", "ramekin", "plate")
PLAIN_DISTRACTORS = tuple(n for n in LOW_DISTRACTORS if n not in CONTAINERS)
WOOD = (0.62, 0.45, 0.28, 1.0)


# ----- asset helpers ---------------------------------------------------------------------------------------------

@np.errstate(all="ignore")
def _visual_extent(root):
    """(lo, hi) of the visual mesh geoms of a loaded LIBERO object, in its ``object`` body frame."""
    mj = ET.Element("mujoco")
    mj.append(copy.deepcopy(root.find("asset")))
    if root.find("default") is not None:
        mj.append(copy.deepcopy(root.find("default")))
    wb = ET.SubElement(mj, "worldbody")
    body = ET.SubElement(wb, "body", name="probe")
    for el in _content(copy.deepcopy(root)):
        if el.tag == "geom" and el.get("type") == "mesh":
            el.attrib.pop("class", None)
            body.append(el)
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pts = []
    for g in range(model.ngeom):
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    pts = np.vstack(pts)
    return pts.min(0), pts.max(0)


def octagon_boxes(parent, name, centre, apothem, half_h, mass, friction, solref=None):
    """A solid octagonal prism at ``centre`` (parent frame): the union of four boxes of half size
    (apothem, apothem * tan 22.5 deg) at 0/45/90/135 deg is exactly the regular octagon (each box's ends are two
    opposite octagon faces; its long sides lie inside its neighbours)."""
    side = apothem * np.tan(np.pi / 8)
    for k in range(4):
        yaw = k * np.pi / 4
        attrs = dict(name=f"{name}_oct{k}", type="box", size=f"{apothem:.6g} {side:.6g} {half_h:.6g}",
                     pos=" ".join(f"{c:.6g}" for c in centre), quat=f"{np.cos(yaw / 2):.6g} 0 0 {np.sin(yaw / 2):.6g}",
                     mass=f"{mass / 4:.6g}", condim="4", friction=f"{friction} 0.02 0.001")
        if solref is not None:
            attrs["solref"] = " ".join(map(str, solref))
        attrs.update(group="3", rgba="0.8 0.8 0.8 0.3")
        ET.SubElement(parent, "geom", **attrs)


@dataclass
class OctoLibero:
    """A free LIBERO object whose collision is a solid octagonal prism over its visual mesh's extent (the
    original thin-plate collision geoms are dropped; the visual mesh is kept)."""

    name: str
    asset: str
    scale: float = 0.5
    mass: float = 0.03
    friction: float = 1.0
    fill: float = 0.97                 # octagon apothem / visual radius (octagon corners reach 1.05 R)
    solref: tuple | None = (0.01, 1.0)

    def build_mjcf(self, mj, asset_el, worldbody):
        root = load_mjcf(ASSETS / CATALOG[self.asset].path, self.scale, self.name)
        _soften(root.find("asset"))
        lo, hi = _visual_extent(root)
        inner = next(b for b in root.find("worldbody/body").iter("body") if b.get("name", "").endswith("object"))
        for g in list(inner):
            if g.tag == "geom" and (g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0"):
                inner.remove(g)
            elif g.tag == "geom":
                g.set("density", "0")
                g.attrib.pop("mass", None)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        upright = ET.SubElement(body, "body", name=f"{self.name}_upright", quat="1 0 0 0")
        upright.extend(_content(root))
        half = (hi - lo) / 2
        octagon_boxes(upright, self.name, (lo + hi) / 2, self.fill * min(half[0], half[1]), half[2], self.mass,
                      self.friction, solref=self.solref)


@dataclass
class Cup:
    """A small plastic cup: visual open cylinder (outer wall, dark inside, rim), solid octagonal collision."""

    name: str
    radius: float = 0.017
    height: float = 0.034
    rgba: tuple = (0.15, 0.55, 0.75, 1.0)
    mass: float = 0.03
    friction: float = 1.0
    solref: tuple = (0.01, 1.0)

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        r, h = self.radius, self.height / 2
        rgba = " ".join(map(str, self.rgba))
        inner = " ".join(map(str, (*(0.55 * np.asarray(self.rgba[:3])), 1.0)))
        vis = dict(contype="0", conaffinity="0", group="1", mass="0", material="val_plastic")
        ET.SubElement(body, "geom", name=f"{self.name}_wall", type="cylinder", size=f"{r} {h}", rgba=rgba, **vis)
        ET.SubElement(body, "geom", name=f"{self.name}_inside", type="cylinder", size=f"{r - 0.0025} 0.0006",
                      pos=f"0 0 {h - 0.0006 + 1e-4}", rgba=inner, **vis)
        octagon_boxes(body, self.name, (0.0, 0.0, 0.0), 0.97 * r, h, self.mass, self.friction, solref=self.solref)


@dataclass
class Bin:
    """An open-top storage bin (floor + four walls, one heavy free body)."""

    name: str
    inner: tuple = (0.040, 0.034)      # inner half extents (x, y)
    wall_h: float = 0.028              # wall height above the table
    wall_t: float = 0.004
    floor_t: float = 0.004
    rgba: tuple = (0.92, 0.92, 0.88, 1.0)
    mass: float = 0.8
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ix, iy = self.inner
        t, ft, h = self.wall_t, self.floor_t, self.wall_h
        parts = [("floor", (ix + t, iy + t, ft / 2), (0, 0, ft / 2)),
                 ("wall_px", (t / 2, iy + t, h / 2), (ix + t / 2, 0, h / 2)),
                 ("wall_nx", (t / 2, iy + t, h / 2), (-ix - t / 2, 0, h / 2)),
                 ("wall_py", (ix, t / 2, h / 2), (0, iy + t / 2, h / 2)),
                 ("wall_ny", (ix, t / 2, h / 2), (0, -iy - t / 2, h / 2))]
        vol = sum(8 * np.prod(s) for _, s, _ in parts)
        for part, size, pos in parts:
            ET.SubElement(body, "geom", name=f"{self.name}_{part}", type="box", size=" ".join(f"{v:.6g}" for v in size),
                          pos=" ".join(f"{v:.6g}" for v in pos), rgba=" ".join(map(str, self.rgba)),
                          mass=f"{self.mass * 8 * np.prod(size) / vol:.6g}", group="1", condim="4",
                          friction=f"{self.friction} 0.02 0.001", material="val_plastic")


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----- environment -----------------------------------------------------------------------------------------------

class NestEnv(TrainEnv):
    """``item`` (the small container) starts on the table and must end inside ``dest``."""

    task_objects = ("item",)
    distractor_pool = PLAIN_DISTRACTORS
    dest_is_fixture = False
    dest_region = dict(r=(0.19, 0.25), angle=(-60.0, 60.0))
    item_region = dict(r=(0.16, 0.26), angle=(-62.0, 62.0))
    inner_margin = 0.004             # the item's centre must lie this far inside the inner floor (beyond its radius)
    z_tolerance = 0.005
    min_travel = 0.10
    item_mass = 0.03
    # MuJoCo's default contact time constant: with 0.01 the item landed on the bowl/tray floor with 1000-1200
    # rad/s^2 single-substep spikes on release (5/6 calibration seeds); with 0.02 and the bounded squeeze 0/12.
    item_solref = (0.02, 1.0)
    item_friction = 1.0

    # geometry ------------------------------------------------------------------------------------------------
    def aabb(self, name):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in self._geoms[name]:
            c = self.data.geom_xpos[g] + self.data.geom_xmat[g].reshape(3, 3) @ self.model.geom_aabb[g, :3]
            h = np.abs(self.data.geom_xmat[g].reshape(3, 3)) @ self.model.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        return lo, hi

    def centre(self, name):
        lo, hi = self.aabb(name)
        return (lo + hi) / 2

    def item_radius(self):
        """Circumradius of the item's octagonal collision in xy."""
        m = self.model
        g = self._geoms["item"][0]
        return float(m.geom_size[g][0] / np.cos(np.pi / 8))

    def item_bottom(self):
        return float(self.aabb("item")[0][2])

    def dest_floor(self):
        xy = self.dest_centre()
        return min(self.surface_z(xy + d, exclude=self.task_objects)
                   for d in (np.zeros(2), [0.004, 0], [-0.004, 0], [0, 0.004], [0, -0.004]))

    def dest_centre(self):
        return self.centre("dest")[:2]

    def dest_place(self, placed):
        fp = self.fixture_footprint("dest") if self.dest_is_fixture else self.footprint("dest")
        xy = self.sample_xy(fp, placed, clearance=0.02, **self.dest_region)
        yaw = self.dest_yaw(xy)
        if self.dest_is_fixture:
            self.set_fixture_pose("dest", xy, yaw=yaw)
        else:
            self.set_object_pose("dest", xy, yaw=yaw)
        placed.append((xy, fp))
        return xy

    def dest_yaw(self, xy):
        return self.np_random.uniform(-np.pi, np.pi)

    def inside(self, env, name):
        """Centre over the inner floor and not touching the table."""
        raise NotImplementedError

    def touching_table(self, name):
        mine, table = set(self._geoms[name]), self._owned_geoms("table")
        return any((c.geom1 in mine and c.geom2 in table) or (c.geom2 in mine and c.geom1 in table)
                   for c in self.data.contact[:self.data.ncon])

    def layout(self):
        placed = []
        dest_xy = self.dest_place(placed)
        for _ in range(100):
            try:
                item_xy = self.sample_xy(self.footprint("item"), placed, clearance=0.025, **self.item_region)
            except RuntimeError:
                continue
            if np.linalg.norm(item_xy - dest_xy) >= self.min_travel:
                break
        else:
            raise RuntimeError("no item start")
        self.set_object_pose("item", item_xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((item_xy, self.footprint("item")))
        self.place_distractors(placed)
        c = self.dest_centre()
        target = (*c, self.dest_floor() + self._extent["item"]["bottom"])
        self.set_goals(Goal("item", "dest", target=target, reference=None if self.dest_is_fixture else "dest",
                            tolerance=(0.03, 0.03, self.z_tolerance), check=lambda env, n: env.inside(env, n)))

    def goal_regions(self):
        """Samples across the receiving container's opening just above its floor (front view sees into it)."""
        c = self.dest_centre()
        z = self.dest_floor() + 0.004
        s = 0.012
        body = "dest"
        return (GoalRegion(body, tuple((c[0] + x, c[1] + y, z) for x in (-s, 0, s) for y in (-s, 0, s))),)


def _radial_inside(radius):
    def inside(env, name):
        off = np.linalg.norm(env.centre(name)[:2] - env.dest_centre())
        return bool(off <= radius and not env.touching_table(name))
    return inside


# ----- 1. ramekin in the bowl --------------------------------------------------------------------------------------

class RamekinBowlEnv(NestEnv):
    instruction = "Put the ramekin in the bowl."
    dest_is_fixture = True
    item_mass = 0.03

    def scene_objects(self):
        return [OctoLibero("item", "ramekin", 0.4, mass=self.item_mass, solref=self.item_solref,
                           friction=self.item_friction)]

    def scene_fixtures(self):
        return [Fixture("dest", "akita_black_bowl", 1.0)]

    def dest_centre(self):
        return self.object_pos("dest")[:2]

    def inside(self, env, name):
        return _radial_inside(0.012)(env, name)


# ----- 2. cup in the bin -------------------------------------------------------------------------------------------

class CupBinEnv(NestEnv):
    instruction = "Put the blue cup in the storage bin."

    def scene_objects(self):
        return [Cup("item", solref=self.item_solref), Bin("dest")]

    def dest_yaw(self, xy):
        return np.arctan2(xy[1], xy[0]) + self.np_random.uniform(-0.6, 0.6)

    def inside(self, env, name):
        m, d = self.model, self.data
        b = self._body["dest"]
        R = d.xmat[b].reshape(3, 3)
        local = R.T @ (self.centre(name) - d.xpos[b])
        r = self.item_radius()
        inner = np.asarray(Bin.inner) - r
        return bool(np.all(np.abs(local[:2]) <= inner) and not self.touching_table(name))


# ----- oracle ------------------------------------------------------------------------------------------------------

class NestOracle(TrainOracle):
    """Grasp the small container across two parallel faces of its octagonal collision, carry it above the
    receiving container's walls, level it, lower it onto the floor and release with a small opening."""

    grasp_frac = 0.6                   # TCP height above the item's base, as a fraction of its height
    min_grasp = 0.0095                 # the jaw hull hangs ~8.2 mm below the TCP
    carry_z = 0.09
    pre_release_height = 0.012
    drop = 0.001

    squeeze = 0.004                    # close to width - squeeze (fingertip-sphere centres); None: fully

    def close_target(self, width):
        """Bounded squeeze: closing fully stores the PD's grip force in the contact, and on release it springs the
        light container (1000-1100 rad/s^2 on the 20 g small bowl, 2/3 calibration seeds)."""
        if self.squeeze is None:
            return CLOSED
        self.jaw_gap(0.7)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(width - self.squeeze, gaps, angles)))

    def item_width(self):
        g = self.env._geoms["item"][0]
        return 2 * float(self.model.geom_size[g][0])

    def item_yaw(self):
        g = self.env._geoms["item"][0]
        m = self.env.data.geom_xmat[g].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def grasp_plan(self, center, width, angles):
        """As the kit, but around the item's collision centre (not the body origin)."""
        best = None
        for a in angles:
            rot = top_down_mat(a)
            pos = center - (FIXED_FACE - width / 2 - self.grasp_clearance) * rot[:, 0]
            q, err, tilt = self.solve(pos, rot)
            score = 200 * err + 2 * tilt + 0.3 * abs(q[4] - self.q[4])
            if best is None or score < best[0]:
                best = (score, rot, pos, q)
        return best[1:]

    def pick_item(self, attempts=2, approach=0.05):
        env = self.env
        width = self.item_width()
        for attempt in range(attempts):
            c = env.centre("item")
            lo, hi = env.aabb("item")
            grasp_z = lo[2] + max(self.min_grasp, self.grasp_frac * (hi[2] - lo[2]))
            yaw = self.item_yaw()
            angles = self.grasp_angles(c, [yaw + k * np.pi / 4 for k in range(8)])
            rot, grasp, q = self.grasp_plan(np.array([c[0], c[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.close_target(width), self.close_seconds, 0.3)
            ok = env.is_grasping("item")
            self.log.append(("item", "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], self.carry_z]), rot, speed=0.10, label="lift")
                if env.is_grasping("item"):
                    return True
                self.log.append(("item", "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def grasp_angles(self, centre, angles):
        """Candidate closing directions (all eight faces by default)."""
        return angles

    def level_rot(self, name, rot):
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        yaw = np.arctan2(m[1, 0], m[0, 0])
        r_des = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        return r_des @ r_obj_tcp.T

    def place_rotations(self, target):
        """Candidate TCP rotations over the target (the carry rotation and its half turn)."""
        carry = self.carry_rot(target[:2])
        c, s = -1.0, 0.0
        half = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return [carry, half @ carry]

    def place_item(self, target):
        name = "item"
        target = np.asarray(target, float)
        rot = self.feasible_rotation(name, target, self.place_rotations(target))
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], self.carry_z], rot)
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], self.carry_z], rot, speed=0.05, label="align", smooth=False)
        yield from self.move(tcp + [0, 0, self.pre_release_height], rot, speed=0.06, tol=0.002, settle=0.6,
                             label="pre-lower")
        rot = self.level_rot(name, rot)
        tcp = target - rot @ self.held(name)
        lift = self.release_lift_fraction * self.release_lift(name, rot)
        yield from self.move(np.r_[tcp[:2], tcp[2] + self.drop + lift], rot, speed=0.04, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.release(self.release_for(self.item_width()))
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], rot, speed=0.10, label="retreat")

    def plan(self):
        env = self.env
        ok = yield from self.pick_item()
        if not ok:
            return
        yield from self.place_item(env.goal_target(env.goals[0]))
        yield from self.rest()
        yield from self.wait(1.2)


class RamekinBowlOracle(NestOracle):
    """Full close: with the bounded squeeze the 30 g ramekin slid down in the jaws and met the bowl floor while
    still held (2/12 calibration seeds)."""
    carry_z = 0.095
    drop = -0.0005
    squeeze = None


class CupBinOracle(NestOracle):
    carry_z = 0.09

    def place_rotations(self, target):
        """The octagonal cup looks the same every 45 deg, so turn at most 45 deg from the carry rotation. Offering
        the half turn let the IK-error tie-break pick it; the kit then carries a > 60 deg turn as a joint-space
        transit, whose path dipped the cup into the bin wall (qualification seed 42658183)."""
        carry = self.carry_rot(target[:2])
        return [carry, rot_z(np.pi / 4) @ carry, rot_z(-np.pi / 4) @ carry]


TASKS = [
    define_task(name="ramekin_in_bowl", instruction=RamekinBowlEnv.instruction, family=FAMILY,
                env=RamekinBowlEnv, oracle=RamekinBowlOracle, objects=("item",), object_kinds=("ramekin",),
                relation="inside", goal="bowl", steps=(step_text("put", "ramekin", "inside", "bowl"),)),
    define_task(name="cup_in_bin", instruction=CupBinEnv.instruction, family=FAMILY,
                env=CupBinEnv, oracle=CupBinOracle, objects=("item",), object_kinds=("cup",),
                relation="inside", goal="storage bin", steps=(step_text("put", "blue cup", "inside", "storage bin"),)),
]

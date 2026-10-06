"""empty_container: take two objects out of one container, one after the other in a stated order, and put each
on a stated spot outside it.

Both objects start resting on the floor of an open wooden receptacle (a free body built from boxes, see
``Crate``), side by side along its long axis. The instruction names the order and where each object goes. The
strict success predicate is the kit's: each object released, settled, upright and supported by its stated
destination at its target, the receptacle undisturbed (the physics gate limits every non-task free body to
<= 5 mm / 0.1 rad), plus ``OrderedCompletion``: the first object is grasped and placed before the second is
first grasped, and placing the second never disturbs the first (both goals must hold at the end).

Tasks differ in the receptacle (a deep wooden box, a shallow tray), the objects (cube + soup can, tomato
sauce + long block, ...), the order and the destinations (a mat and a plate; the table to the viewer's left
and right of the tray).

Objects: LIBERO cans collide as a ring of 21 thin plates whose edges catch the jaw pads (1000-3500 rad/s^2
spins measured by other families), so cans use ``SolidLibero`` (LIBERO visual, one solid box collision of
the same 3.1 x 3.1 x 3.8 cm extent, which qualified 50/50 for soup_can_in_basket). Thin food boxes are not
used: they spin or slip when grasped off a container floor (take_out_of_container notes).

Grasps: the jaws close across the receptacle's long axis, so the fingers descend between the object and the
receptacle's long walls, never between the two objects. Objects sit squared with the receptacle (+-10 deg).
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, VIEW, Goal, TrainEnv, TrainOracle, define_task, mat,
                                  step_text, then)
from sim.val.scene import (ASSETS, CATALOG, Block, Fixture, _add_assets, _collision_geoms, _content, _hull_volume,
                           _merge_defaults, _set, _set_free_physics, _soften, _upright_quat, _vec, load_mjcf)

FAMILY = "empty_container"
WOOD, DARK_WOOD = (0.62, 0.43, 0.25, 1.0), (0.42, 0.28, 0.17, 1.0)
MAT_THICK = 0.006
CUBE = 0.014                              # 2.8 cm cube half-size
BAR = (0.024, 0.010, 0.014)               # 4.8 x 2.0 x 2.8 cm long block
CAN_W = 0.0312                            # SolidLibero can collision box width (square)
RED, BLUE, GREEN = (0.8, 0.1, 0.08, 1.0), (0.1, 0.25, 0.8, 1.0), (0.2, 0.55, 0.25, 1.0)
GREY_MAT = (0.45, 0.45, 0.5, 1.0)
DEST_REGION = dict(r=(0.165, 0.26), angle=(-65.0, 65.0))     # destination centres (r >= 0.16: no fold-in)
# A 0.5 kg receptacle rocked on its own soft table contact while the jaws closed on an object inside it (can
# spins 1100-9000 rad/s^2, 3-16 mm penetration in 5/5 seeds); at 2 kg the same grasps measured <= 200 rad/s^2.
CRATE_MASS = 2.0
CRATE_SOLREF = None
# Living-room arena: its table top is thin 12 mm slabs, and the slab-on-slab contact of the flat-bottomed
# receptacle went unstable while the jaws closed on an object inside (7 mm jaw/can penetration, the receptacle
# jumping; 5 of 6 living-room calibration seeds at any mass 0.5-2 kg). Sphere feet (``Crate.feet``) fixed it.
ARENAS = ("living_room", "kitchen")
REST_POINT, REST_CLEAR = np.array([0.16, 0.0]), 0.07          # folded gripper hover/sweep at rest


# ----- local primitives (copied from sibling families; shared files are not edited) --------------------------

@dataclass
class Crate:
    """An open-top rectangular receptacle (free body): a floor slab and four walls. ``inner``: interior half
    extents (x, y); ``wall``: wall height above the floor top. The body origin is at the centre of the floor's
    underside, so the interior floor is ``floor`` above it."""

    name: str
    inner: tuple = (0.058, 0.045)
    wall: float = 0.022
    thickness: float = 0.006
    floor: float = 0.006
    rgba: tuple = WOOD
    mass: float = 0.5
    friction: float = 1.0
    solref: tuple | None = None        # contact time constant/damping (None: MuJoCo default 0.02 1)
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
                          friction=f"{self.friction} 0.02 0.001", material="val_fabric",
                          **({} if self.solref is None else {"solref": " ".join(map(str, self.solref))}))
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
    and mass unchanged). Copied from put_in_container: the 21-plate LIBERO can collision catches jaw pads."""

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


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return float(np.angle(np.exp(1j * a)))


def _pool(*exclude):
    return tuple(n for n in LOW_DISTRACTORS if n not in exclude)


def world_extent(env, name):
    """(lo, hi) world xyz of a body's collision-geometry AABB."""
    m, d = env.model, env.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in env._geoms[name]:
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        half = np.abs(R) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    return lo, hi


# ----- shared environment ------------------------------------------------------------------------------------

class EmptyEnv(TrainEnv):
    """Two task objects (``task_objects``, in the required order) start in the slots of ``container``.

    ``slot_offset``: slot distance from the receptacle centre along its long (local x) axis.
    ``axis_mode``: "tangential" (long axis across the robot's reach, at the receptacle's azimuth) or
    "world_y" (long axis along the viewer's left-right).
    Subclasses implement ``destinations(placed)`` -> {name: Goal}."""

    container = "box"
    slot_offset = 0.028
    container_region = dict(r=(0.19, 0.225), angle=(-50.0, 50.0))
    axis_mode = "tangential"
    yaw_jitter = np.radians(12)
    object_jitter = np.radians(8)
    slot_reach = (0.15, 0.262)
    slot_angle = 62.0
    symmetric: dict = {}               # name -> equivalent yaw count about the vertical (4 square, 2 elongated)

    def container_half(self):
        """Outer half extents (x, y) of the receptacle in its own frame."""
        spec = next(o for o in self.scene_objects() if o.name == self.container)
        return np.asarray(spec.inner) + spec.thickness

    def rect_distance(self, point, xy, yaw, half):
        """Distance from ``point`` to the receptacle rectangle (0 inside)."""
        local = _rot_z(-yaw)[:2, :2] @ (np.asarray(point) - xy)
        return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))

    def container_pose(self, placed):
        """Rejection-sample the receptacle pose with its true rectangle against the keep-outs (the bounding
        circle of a 13 cm receptacle would exclude the whole front of the table around the rest pose)."""
        half = self.container_half()
        for _ in range(400):
            r = self.np_random.uniform(*self.container_region["r"])
            a = np.radians(self.np_random.uniform(*self.container_region["angle"]))
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            base = np.arctan2(xy[1], xy[0]) + np.pi / 2 if self.axis_mode == "tangential" else np.pi / 2
            yaw = base + self.np_random.uniform(-self.yaw_jitter, self.yaw_jitter)
            axis = np.array([np.cos(yaw), np.sin(yaw)])
            slots = [xy + self.slot_offset * axis, xy - self.slot_offset * axis]
            if not all(self.slot_reach[0] <= np.hypot(*s) <= self.slot_reach[1]
                       and abs(np.degrees(np.arctan2(s[1], s[0]))) <= self.slot_angle for s in slots):
                continue
            if any(self.rect_distance(c, xy, yaw, half) < rad + 0.012 for c, rad in self.keepout):
                continue
            if any(self.rect_distance(c, xy, yaw, half) < rad + 0.03 for c, rad in placed):
                continue
            if self.container_ok(xy, yaw):
                return xy, yaw, slots
        raise RuntimeError("no reachable container pose")

    def container_ok(self, xy, yaw):
        return True

    def fill(self, placed):
        xy, yaw, slots = self.container_pose(placed)
        self.set_object_pose(self.container, xy, yaw=yaw)
        self.container_rect = (xy.copy(), float(yaw), self.container_half())
        self.axis_yaw = float(yaw)
        floor = self.object_pos(self.container)[2] + self.crate_floor()    # origin: floor slab underside
        order = self.np_random.permutation(2)
        self.start_slots = {}
        for name, k in zip(self.task_objects, order):
            sym = self.symmetric.get(name, 4)
            obj_yaw = (yaw + self.np_random.integers(sym) * 2 * np.pi / sym
                       + self.np_random.uniform(-self.object_jitter, self.object_jitter))
            self.set_object_pose(name, slots[k], yaw=obj_yaw, z=floor)
            self.start_slots[name] = slots[k]
        return xy

    def crate_floor(self):
        return 0.006

    def sample_xy(self, radius, placed, r=(0.12, 0.36), angle=(-80, 80), clearance=0.02, tries=500):
        """As ValEnv, and clear of the receptacle's true rectangle (not its bounding circle)."""
        rect = getattr(self, "container_rect", None)
        for _ in range(tries):
            xy = super().sample_xy(radius, placed, r=r, angle=angle, clearance=clearance, tries=100)
            if rect is None or self.rect_distance(xy, *rect) >= radius + clearance:
                return xy
        raise RuntimeError("could not place object")

    def place_pad(self, name, placed, region=None, object_radius=0.03, clearance=0.02, fixture=False, round_pad=False):
        """Place a flat destination (mat, board, coaster, plate). Its centre keeps the placed object clear of
        the hovering rest gripper (``REST_CLEAR``); the flat pad itself may pass under the rest pose (>= 9 mm
        table clearance), so the kit's whole-footprint keep-out is not applied to it."""
        region = DEST_REGION if region is None else region
        radius = self.fixture_footprint(name) if fixture else self.footprint(name)
        if round_pad or fixture:
            radius /= np.sqrt(2)
        rect = self.container_rect
        for _ in range(2000):
            rr = self.np_random.uniform(*region["r"])
            aa = np.radians(self.np_random.uniform(*region["angle"]))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if np.linalg.norm(xy - REST_POINT) < REST_CLEAR + object_radius:
                continue
            if np.linalg.norm(xy) < 0.10 + radius + 0.01:
                continue
            if any(np.linalg.norm(xy - p) < radius + q + clearance for p, q in placed):
                continue
            if rect is not None and self.rect_distance(xy, *rect) < radius + clearance:
                continue
            break
        else:
            raise RuntimeError("could not place destination")
        yaw = self.np_random.uniform(-np.pi, np.pi)
        if fixture:
            self.set_fixture_pose(name, xy, yaw=yaw)
        else:
            self.set_object_pose(name, xy, yaw=yaw)
        placed.append((xy, radius))
        return xy

    def layout(self):
        placed = []
        self.container_rect = None
        self.fill(placed)
        goals = self.destinations(placed)
        self.place_distractors(placed)
        self.set_goals(*(goals[n] for n in self.task_objects))

    def destinations(self, placed):
        raise NotImplementedError

    def closing_yaw(self, name):
        """Preferred jaw-closing direction for the grasp: across the receptacle's long axis."""
        return self.axis_yaw + np.pi / 2


# ----- shared oracle -----------------------------------------------------------------------------------------

class EmptyOracle(TrainOracle):
    """Pick each object in order out of the receptacle (jaws closing across its long axis) and place it.

    ``grips``: name -> (width across the jaws, grasp TCP height above the object's bottom or None for its
    origin, yaw offset of a closing face from the object's local x, equivalent closing directions).
    ``hold_squeeze``: close to that far below the width (None: fully closed); ``squeeze``: ease the jaw to
    that far below the width before opening (a light object otherwise springs off the fixed finger)."""

    grips: dict = {}
    hold_squeeze: dict = {}
    squeeze = 0.006
    relax_seconds = 0.6
    open_margin = 0.010
    release_drop = -0.0005
    level = True
    pre_release_height = 0.010

    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    tall: dict = {}                    # name -> (open_margin, grasp_clearance) for objects standing above the jaws
    touch_depth = 0.001                # staged close: ease the jaw to this far inside the width (ending at rest) ...
    touch_seconds = 0.9                # ... over this long, then squeeze over close_seconds
    grasp_settle = 0.3                 # let the arm settle at the grasp pose before the jaw moves
    staged = "touch"                   # or "approach": fast to 5 mm short of contact, then one slow close

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Kit pick with a staged close (relative_left_right): the moving jaw eases onto the object and stops,
        then squeezes from rest, so it never meets the object at speed; an optional bounded squeeze; and, for
        tall objects, a wider opening and fixed-finger clearance (the pivoting jaw narrows above the tips and
        the fixed jaw body grazed a 3.8 cm can top on the way down)."""
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
                if self.staged == "touch":
                    yield from self.gripper(self.open_for(width, -self.touch_depth), self.touch_seconds, 0.1)
                else:   # quickly to just short of contact, then slowly onto the object (put_in_container)
                    yield from self.gripper(self.gap_angle(width + 0.005), 0.5, 0.1)
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

    def grasp_center(self, name):
        """Centre of the object's collision box in its own frame, in world coordinates (LIBERO can origins sit
        1.8 mm off the collision centre, which shifts the fixed finger's clearance by as much)."""
        env = self.env
        local = self._local_points(name)
        mid = (local.min(0) + local.max(0)) / 2
        body = env._body[name]
        return env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ np.r_[mid[:2], 0.0]

    def release(self, open_to=None):
        width = getattr(self, "_held_width", None)
        if width is not None and self.squeeze is not None:
            yield from self.gripper(self.gap_angle(width - self.squeeze), self.relax_seconds, 0.2)
        yield from super().release(open_to)

    def level_rot(self, name, rot):
        """TCP rotation near ``rot`` at which the held object's up axis is vertical (cancels the few-degree pivot
        in the jaws picked up while carrying, so it lands flat instead of slapping down on an edge)."""
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        yaw = np.arctan2(m[1, 0], m[0, 0])
        return _rot_z(yaw) @ r_obj_tcp.T

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        if not self.level:
            yield from super().place_object(name, target, rot, drop, open_to, carry_z)
            return
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

    def grasp_yaw(self, name):
        """The object's closing face direction nearest the receptacle's preferred closing direction."""
        env = self.env
        width, height, offset, sym = self.grips[name]
        face = env.yaw(name) + offset
        want = env.closing_yaw(name)
        k = np.round(_wrap(want - face) / (2 * np.pi / sym))
        return face + k * 2 * np.pi / sym

    def plan(self):
        env = self.env
        for goal in env.goals:
            name = goal.obj
            width, height, _, _ = self.grips[name]
            pos = env.object_pos(name)
            grasp_z = pos[2] - 0.001 if height is None else pos[2] - env._extent[name]["bottom"] + height
            ok = yield from self.pick(name, width, grasp_z, yaw=self.grasp_yaw(name), symmetric=2)
            if not ok:
                return
            yield from self.place_object(name, env.goal_target(goal), drop=self.release_drop,
                                         open_to=self.release_for(width))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- task 1: cube then soup can out of the wooden box, onto the mat and the plate ---------------------------

class BoxToMatPlateEnv(EmptyEnv):
    instruction = ("Take the cube out of the wooden box and put it on the mat, then take the soup can out and "
                   "put it on the plate.")
    task_objects = ("cube", "soup_can")
    order = ("cube", "soup_can")
    surfaces = ("box",)
    container = "box"
    distractor_pool = _pool("alphabet_soup", "tomato_sauce", "plate")

    def scene_objects(self):
        return [Block("cube", half=(CUBE,) * 3, rgba=RED), SolidLibero("soup_can", "alphabet_soup"),
                Crate("box", inner=(0.058, 0.045), wall=0.022, rgba=DARK_WOOD, mass=CRATE_MASS, solref=CRATE_SOLREF), mat("mat", rgba=GREY_MAT)]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.65)]

    def destinations(self, placed):
        mat_xy = self.place_pad("mat", placed, object_radius=0.02)
        plate_xy = self.place_pad("plate", placed, object_radius=0.022, fixture=True)
        top = self.surface_z(plate_xy, exclude=self.task_objects)
        return {"cube": Goal("cube", "mat", target=(*mat_xy, MAT_THICK + CUBE), reference="mat",
                             tolerance=(0.02, 0.02, 0.006)),
                "soup_can": Goal("soup_can", "plate", target=(*plate_xy, top + self._extent["soup_can"]["bottom"]),
                                 tolerance=(0.025, 0.025, 0.006))}


class BoxToMatPlateOracle(EmptyOracle):
    carry_z = 0.085
    grips = {"cube": (2 * CUBE, 0.012, 0.0, 4), "soup_can": (CAN_W, 0.015, 0.0, 4)}
    hold_squeeze = {"soup_can": 0.010}
    tall = {"soup_can": (0.018, 0.003)}
    close_seconds = 1.6


# ----- task 2: long block then tomato sauce out of the tray, onto the cutting board and the round coaster ------

@dataclass
class RoundPad:
    """Free round coaster (copied from center_on_target): visual cylinder, collision two boxes at 0/45 deg (an
    inscribed octagon); MuJoCo box-on-thin-cylinder contact lets objects sink into a thin ``Disc``."""

    name: str
    radius: float = 0.035
    thickness: float = MAT_THICK
    rgba: tuple = (0.2, 0.3, 0.8, 1.0)
    mass: float = 0.05
    friction: float = 1.0

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        h = self.thickness / 2
        a = self.radius * np.cos(np.radians(22.5))
        for k, deg in enumerate((0, 45)):
            th = np.radians(deg) / 2
            ET.SubElement(body, "geom", name=f"{self.name}_coll_{k}", type="box", size=f"{a:.6g} {a:.6g} {h:.6g}",
                          quat=f"{np.cos(th):.6g} 0 0 {np.sin(th):.6g}", mass=repr(self.mass / 2), group="3",
                          condim="4", friction=f"{self.friction} 0.02 0.001", rgba="0 0 0 0")
        ET.SubElement(body, "geom", name=f"{self.name}_visual", type="cylinder", size=f"{self.radius:.6g} {h:.6g}",
                      rgba=" ".join(map(str, self.rgba)), group="1", contype="0", conaffinity="0", mass="0",
                      material="val_fabric")


BOARD = (0.05, 0.035)           # cutting board half extents
BOARD_RGBA = (0.78, 0.6, 0.38, 1.0)
CORK = (0.25, 0.35, 0.75, 1.0)


class TrayToBoardCoasterEnv(EmptyEnv):
    instruction = ("Take the long block out of the tray and put it on the cutting board, then take the tomato "
                   "sauce out and put it on the round coaster.")
    task_objects = ("long_block", "tomato_sauce")
    order = ("long_block", "tomato_sauce")
    surfaces = ("tray",)
    container = "tray"
    slot_offset = 0.03
    symmetric = {"long_block": 2}
    distractor_pool = _pool("alphabet_soup", "tomato_sauce")

    def scene_objects(self):
        return [Block("long_block", half=BAR, rgba=BLUE), SolidLibero("tomato_sauce", "tomato_sauce"),
                Crate("tray", inner=(0.064, 0.044), wall=0.012, rgba=WOOD, mass=CRATE_MASS, solref=CRATE_SOLREF),
                mat("board", half=BOARD, rgba=BOARD_RGBA, mass=0.08), RoundPad("coaster", radius=0.035, rgba=CORK)]

    def destinations(self, placed):
        board_xy = self.place_pad("board", placed, object_radius=0.027)
        coaster_xy = self.place_pad("coaster", placed, object_radius=0.022, round_pad=True)
        return {"long_block": Goal("long_block", "board", target=(*board_xy, MAT_THICK + BAR[2]), reference="board",
                                   tolerance=(0.02, 0.02, 0.006)),
                "tomato_sauce": Goal("tomato_sauce", "coaster",
                                     target=(*coaster_xy, MAT_THICK + self._extent["tomato_sauce"]["bottom"]),
                                     reference="coaster", tolerance=(0.018, 0.018, 0.006))}


class TrayToBoardCoasterOracle(EmptyOracle):
    carry_z = 0.085
    grips = {"tomato_sauce": (CAN_W, 0.015, 0.0, 4), "long_block": (2 * BAR[1], 0.012, np.pi / 2, 2)}
    hold_squeeze = {"tomato_sauce": 0.010}
    tall = {"tomato_sauce": (0.018, 0.003)}
    close_seconds = 1.6


TASKS = [
    define_task(name="cube_then_soup_can_out_of_wooden_box", instruction=BoxToMatPlateEnv.instruction,
                family=FAMILY, arenas=ARENAS, env=BoxToMatPlateEnv, oracle=BoxToMatPlateOracle, objects=("cube", "soup_can"),
                object_kinds=("cube", "soup can"), relation="out of wooden box one after the other onto",
                goal="mat and plate",
                steps=("Lift the cube out of the wooden box and put it on the mat.",
                       then("Lift the soup can out of the wooden box and put it on the plate."))),
    define_task(name="long_block_then_sauce_out_of_tray", instruction=TrayToBoardCoasterEnv.instruction,
                family=FAMILY, arenas=ARENAS, env=TrayToBoardCoasterEnv, oracle=TrayToBoardCoasterOracle,
                objects=("long_block", "tomato_sauce"), object_kinds=("long block", "sauce can"),
                relation="out of tray one after the other onto", goal="cutting board and round coaster",
                steps=("Lift the long block out of the tray and put it on the cutting board.",
                       then("Lift the tomato sauce out of the tray and put it on the round coaster."))),
]

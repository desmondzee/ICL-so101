"""place_in_rack: stand a flat object upright in a slot of a slotted rack.

Two orientation-sensitive insertion tasks. The flat object starts standing on the table at a random position and
heading; the robot pinches it across its thickness near its top edge (top-down, so its plane stays vertical),
turns it in the air until its faces are parallel to the rack's dividers, lowers it into the slot and lets go:

* ``book_in_rack_middle_slot`` -- the LIBERO black book into the *middle* slot of a three-slot wooden book rack
  (solid dividers).
* ``board_in_dish_rack``       -- a thin wooden cutting board into *any* slot of a white dish rack whose slots
  are formed by two rows of pegs (the board passes between the pegs of both rows).

Success is strict: the object is released, settled, upright within 10 deg, its faces parallel to the slot
(thin axis within 10 deg of the slot normal), its centre inside the slot (between the dividers / pegs and
within the rack's length) and it is carried by the rack's base (positive upward contact force).

Why the object starts standing (measured 2026-10-06): turning a flat-lying object upright needs the gripper's
approach horizontal (90 deg pitch). With the 5-joint SO-101 the IK misses such poses by 15-29 mm and ~110 deg
at r 0.16-0.24 m, z 0.04-0.10 m; even a 60 deg pitch misses by 5-23 mm (45 deg is reachable at r >= 0.2 m).
So a top-down pinch across the thickness, which keeps the object's plane vertical and only needs a yaw turn,
is the only reliable way to slot a flat object with this arm.

Racks are free primitive bodies (box geoms only, heavy so a brushing jaw cannot move them). The book's body
origin is moved to the centre of its single collision box, so its goal does not depend on which way round it
is slotted. None of this reproduces the held-out validation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import (CLOSED, COS10, LOW_DISTRACTORS, Goal, TrainEnv, TrainOracle, define_task)
from sim.train.variation import GoalRegion
from sim.val.oracle import RELEASE
from sim.val.scene import (ASSETS, CATALOG, Block, _add_assets, _collision_geoms, _content, _hull_volume,
                           _merge_defaults, _set, _set_free_physics, _soften, _vec, load_mjcf)

FAMILY = "place_in_rack"
WOOD = (0.62, 0.43, 0.25, 1.0)
LIGHT_WOOD = (0.80, 0.63, 0.42, 1.0)
WHITE = (0.88, 0.88, 0.90, 1.0)
REST_ZONE = (np.array([0.16, 0.0]), 0.085)     # folded-gripper hover and rest sweep: keep tall things out


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def _box_geom(body, name, pos, half, rgba, mass, friction=1.0, material="val_fabric"):
    ET.SubElement(body, "geom", name=name, type="box", pos=" ".join(f"{v:.6g}" for v in pos),
                  size=" ".join(f"{v:.6g}" for v in half), rgba=" ".join(map(str, rgba)), mass=f"{mass:.6g}",
                  group="1", condim="4", friction=f"{friction} 0.02 0.001", material=material)


# ----- assets ----------------------------------------------------------------------------------------------------

@dataclass
class CenteredLibero:
    """A free LIBERO object with box collision whose body origin is moved to the centre of its collision boxes
    (the LIBERO book's box sits 6 mm off its origin). Mass = visual hull volume x catalog density."""

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
        _set_free_physics(root, mass, self.friction)
        _merge_defaults(mj, root)
        _add_assets(asset_el, root)
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        inner = ET.SubElement(body, "body", name=f"{self.name}_upright")
        _set(inner, "pos", -(lo + hi) / 2)
        inner.extend(_content(root))


@dataclass
class SlotRack:
    """Free slotted rack: a base slab and ``n_slots + 1`` solid dividers. Slots run along the body's x axis and
    are stacked along y; the body origin is the centre of the base's underside."""

    name: str
    n_slots: int = 3
    gap: float = 0.022           # slot width (between dividers)
    divider: float = 0.004       # divider thickness
    length: float = 0.072        # slot length (x)
    height: float = 0.040        # divider height above the base top
    base: float = 0.006
    rgba: tuple = WOOD
    mass: float = 0.5

    @property
    def width(self):
        return self.n_slots * self.gap + (self.n_slots + 1) * self.divider

    def slot_y(self, k):
        return -self.width / 2 + self.divider + self.gap / 2 + k * (self.gap + self.divider)

    def boxes(self):
        L, W, b, h = self.length, self.width, self.base, self.height
        out = [((0, 0, b / 2), (L / 2, W / 2, b / 2))]
        for k in range(self.n_slots + 1):
            y = -W / 2 + self.divider / 2 + k * (self.gap + self.divider)
            out.append(((0, y, b + h / 2), (L / 2, self.divider / 2, h / 2)))
        return out

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        boxes = self.boxes()
        vols = [8 * np.prod(h) for _, h in boxes]
        for k, ((pos, half), v) in enumerate(zip(boxes, vols)):
            _box_geom(body, f"{self.name}_geom{k}", pos, half, self.rgba, self.mass * v / sum(vols))


@dataclass
class PegRack(SlotRack):
    """Free dish rack: a base tray and two rows of square pegs (at x = +-``row``). Consecutive pegs in each row
    bound a slot; a board standing in a slot passes between the pegs of both rows."""

    peg: float = 0.005           # peg side
    row: float = 0.022           # peg rows at x = +-row
    length: float = 0.070
    height: float = 0.036
    rgba: tuple = WHITE
    rim: float = 0.004           # low rim around the tray

    def boxes(self):
        L, W, b, h = self.length, self.width, self.base, self.height
        out = [((0, 0, b / 2), (L / 2, W / 2, b / 2))]
        for x in (-self.row, self.row):
            for k in range(self.n_slots + 1):
                y = -W / 2 + self.divider / 2 + k * (self.gap + self.divider)
                out.append(((x, y, b + h / 2), (self.peg / 2, self.divider / 2, h / 2)))
        r = self.rim
        out += [((sx * (L / 2 - r / 2), 0, b + r / 2), (r / 2, W / 2, r / 2)) for sx in (-1, 1)]
        return out


# ----- environment -----------------------------------------------------------------------------------------------

RACK_REGION = dict(r=(0.19, 0.245), angle=(-55.0, 55.0))
OBJECT_REGION = dict(r=(0.18, 0.255), angle=(-62.0, 62.0))


def _without(*names):
    return tuple(n for n in LOW_DISTRACTORS if n not in names)


class RackEnv(TrainEnv):
    """A flat object standing on the table and a slotted rack. Subclasses set ``rack`` (spec), ``obj``
    (object name), ``slots`` (allowed slot indices) and implement ``object_spec``."""

    obj = "flat"
    rack_name = "rack"
    rack: SlotRack = None
    slots: tuple = (1,)
    distractor_pool = _without("plate", "akita_black_bowl", "white_bowl", "red_bowl")   # keep them low
    yaw_cos = COS10
    grasp_depth = 0.013          # TCP below the object's top edge

    def scene_objects(self):
        return [self.object_spec(), self.rack]

    def object_spec(self):
        raise NotImplementedError

    # ----- geometry ------------------------------------------------------------------------------------------
    def half(self, name=None):
        """Half extents of the object's collision box (its origin is the box centre)."""
        g = self._geoms[name or self.obj][0]
        return self.model.geom_size[g].copy()

    def rack_frame(self):
        b = self._body[self.rack_name]
        return self.data.xmat[b].reshape(3, 3), self.data.xpos[b]

    def slot_center(self, k):
        """World position of slot ``k``'s centre at the base top."""
        R, p = self.rack_frame()
        return p + R @ np.array([0.0, self.rack.slot_y(k), self.rack.base])

    def object_local(self):
        R, p = self.rack_frame()
        return R.T @ (self.object_pos(self.obj) - p)

    def in_slot(self, k) -> bool:
        local = self.object_local()
        hx = self.half()[1]                 # half width along the slot
        return bool(abs(local[1] - self.rack.slot_y(k)) <= self.rack.gap / 2 - self.half()[0] + 0.0015
                    and abs(local[0]) <= self.rack.length / 2 - hx + 0.004
                    and local[2] <= self.rack.base + self.half()[2] + 0.004)

    def faces_parallel(self) -> bool:
        R, _ = self.rack_frame()
        return self.axis_alignment(self.obj, R[:, 1], local_axis=0) >= self.yaw_cos

    def seated(self, env, name) -> bool:
        return self.faces_parallel() and any(self.in_slot(k) for k in self.slots)

    # ----- layout --------------------------------------------------------------------------------------------
    def layout(self):
        placed = []
        rack_r = float(np.hypot(self.rack.length / 2, self.rack.width / 2))
        for _ in range(200):
            xy = self.sample_xy(rack_r, placed, clearance=0.03, **RACK_REGION)
            if np.linalg.norm(xy - REST_ZONE[0]) >= REST_ZONE[1] + rack_r:
                break
        yaw = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self.set_object_pose(self.rack_name, xy, yaw=yaw)
        placed.append((xy, rack_r))
        r = self.footprint(self.obj)
        for _ in range(200):
            oxy = self.sample_xy(r, placed, clearance=0.035, **OBJECT_REGION)
            if np.linalg.norm(oxy - REST_ZONE[0]) >= REST_ZONE[1] + r:
                break
        self.set_object_pose(self.obj, oxy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((oxy, r))
        self.place_distractors(placed)
        mujoco.mj_forward(self.model, self.data)
        h = self.half()[2]
        if len(self.slots) == 1:
            c = self.slot_center(self.slots[0])
            self.set_goals(Goal(self.obj, self.rack_name, target=(c[0], c[1], c[2] + h), reference=self.rack_name,
                                tolerance=(0.012, 0.012, 0.004), check=self.seated))
        else:
            self.set_goals(Goal(self.obj, self.rack_name, check=self.seated))

    def goal_regions(self):
        """Points along the target slot(s) at the rack's divider-top height."""
        R, p = self.rack_frame()
        z = self.rack.base + self.rack.height
        pts = [tuple(p + R @ np.array([x, self.rack.slot_y(k), z])) for k in self.slots
               for x in (-0.25 * self.rack.length, 0.0, 0.25 * self.rack.length)]
        return (GoalRegion(self.rack_name, tuple(pts)),)


# ----- oracle ----------------------------------------------------------------------------------------------------

class RackOracle(TrainOracle):
    """Pinch the standing flat object across its thickness near its top edge, lift it above the rack, turn it so
    its faces are parallel to the slot (least wrist turn of the two equivalent headings), level it upright,
    lower it into the slot until it rests on the base, ease the grip, open and retreat straight up."""

    close_seconds = 2.0
    release_seconds = 1.5
    open_margin = 0.010
    carry_z = 0.12               # held object's bottom ~6 cm above the table: clears the rack and distractors
    squeeze = 0.004              # relaxed grip (below the thickness) before opening
    relax_seconds = 0.8
    release_offset = 0.0008      # object bottom above the base top at release
    transit_speed = 0.5
    vmax = np.radians([45.0, 50.0, 50.0, 65.0, 65.0])

    # ----- helpers -------------------------------------------------------------------------------------------
    def gap_angle(self, gap):
        self.jaw_gap(CLOSED)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(gap, gaps, angles)))

    def carry_to(self, pos, rot, label="carry"):
        delta = np.arctan2(rot[1, 0], rot[0, 0]) - np.arctan2(self._cmd_rot[1, 0], self._cmd_rot[0, 0])
        azimuth = np.arctan2(pos[1], pos[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        if abs(wrap(delta - azimuth)) <= self.max_cartesian_turn:
            yield from self.move(pos, rot, label=label)
        else:
            yield from self.transit(pos, rot, speed=self.transit_speed, label=label)

    def level(self, name, rot):
        """``rot`` turned so the held object ends upright (it pivots a few degrees in the jaws)."""
        env = self.env
        R_pred = rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        z = R_pred[:, 2]
        axis = np.cross(z, [0.0, 0.0, 1.0])
        s, c = np.linalg.norm(axis), float(z[2])
        if s < 1e-6:
            return rot
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return (np.eye(3) + s * K + (1 - c) * K @ K) @ rot

    def heading(self, name, rot):
        """World heading of the held object's x axis (its thin axis) when the TCP has rotation ``rot``."""
        env = self.env
        m = rot @ self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    # ----- skills --------------------------------------------------------------------------------------------
    def pick_flat(self, name, attempts=2, approach=0.05):
        env = self.env
        half = env.half(name)
        width = 2 * half[0]
        self.width = width
        for attempt in range(attempts):
            centre = env.object_pos(name)
            top = centre[2] + half[2]
            yaw = env.yaw(name)
            rot, grasp, q = self.grasp_plan(np.r_[centre[:2], top - env.grasp_depth], width, [yaw, yaw + np.pi])
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.05, tol=0.003, label="grasp")
            # Two-stage close: quickly to just short of contact, then slowly onto the object.
            yield from self.gripper(self.gap_angle(width + 0.004), 0.5, 0.1)
            yield from self.gripper(CLOSED, self.close_seconds, 0.3)
            ok = env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.r_[grasp[:2], self.carry_z], rot, speed=0.08, label="lift")
                if env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def slot_rotation(self, name, target, slot_normal_yaw):
        """Of the end rotations putting the held object's thin axis along the slot normal (either sense), the
        reachable one needing the least wrist turn beyond following the arm's azimuth."""
        cur = self.heading(name, self._cmd_rot)
        azimuth = np.arctan2(target[1], target[0]) - np.arctan2(self._cmd_pos[1], self._cmd_pos[0])
        deltas = sorted((wrap(slot_normal_yaw - cur) + k * np.pi for k in (-2, -1, 0, 1, 2)),
                        key=lambda d: abs(d - azimuth))
        rotations = [self.level(name, rot_z(d) @ self._cmd_rot) for d in deltas[:3]]
        return self.feasible_rotation(name, target, rotations)

    def insert(self, name, target, rot):
        env = self.env
        target = np.asarray(target, float)
        tcp = target - rot @ self.held(name)
        yield from self.carry_to(np.r_[tcp[:2], self.carry_z], rot)
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(np.r_[tcp[:2], self.carry_z], rot, speed=0.04, label="align", smooth=False)
        # Into the slot: stop 1.5 cm above the seat, re-level and re-measure, then the final descent.
        tcp = target - rot @ self.held(name)
        yield from self.move(tcp + [0, 0, 0.015], rot, speed=0.05, tol=0.002, settle=0.6, label="pre-lower")
        rot = self.level(name, rot)
        tcp = target - rot @ self.held(name)
        yield from self.move(tcp + [0, 0, self.release_offset], rot, speed=0.03, tol=self.release_tolerance,
                             settle=0.8, label="lower")
        yield from self.gripper(self.gap_angle(self.width - self.squeeze), self.relax_seconds, 0.2)
        yield from self.gripper(self.open_for(self.width, self.release_margin), self.release_seconds, 0.3)
        # Back the fixed finger off the face slowly before lifting: it rests on the object after the jaw opens
        # and would drag/rock it on the way up.
        yield from self.move(self._cmd_pos + self.release_backoff * rot[:, 0], rot, speed=0.008, tol=0.002,
                             settle=0.3, label="backoff", smooth=False)
        yield from self.move(self._cmd_pos + [0, 0, 0.02], rot, speed=0.03, tol=0.003, settle=0.2,
                             label="slide out", smooth=False)
        yield from self.move(np.r_[self._cmd_pos[:2], self.carry_z], rot, speed=0.08, label="retreat")

    def target_slot(self):
        env = self.env
        goal = env.goals[0]
        if goal.target is not None:
            return env.goal_target(goal)
        # Any slot: the one whose centre is nearest the robot's current reach (prefer the middle on ties).
        h = env.half()[2]
        best = min(env.slots, key=lambda k: (abs(k - (env.rack.n_slots - 1) / 2), k))
        c = env.slot_center(best)
        return c + [0, 0, h]

    def plan(self):
        env = self.env
        name = env.obj
        ok = yield from self.pick_flat(name)
        if not ok:
            return
        target = self.target_slot()
        R, _ = env.rack_frame()
        normal = float(np.arctan2(R[1, 1], R[0, 1]))
        rot = self.slot_rotation(name, target, normal)
        yield from self.insert(name, target, rot)
        yield from self.rest()
        yield from self.wait(1.2)


# ----- tasks -----------------------------------------------------------------------------------------------------

class BookRackEnv(RackEnv):
    instruction = "Stand the book upright in the middle slot of the book rack."
    obj = "book"
    task_objects = ("book",)
    rack = SlotRack("rack", n_slots=3, gap=0.020, length=0.072, height=0.030)
    slots = (1,)
    grasp_depth = 0.020          # TCP 2 cm below the top edge: nearer the centre of mass (see RackOracle)

    def object_spec(self):
        return CenteredLibero("book", "black_book")


class BoardDishRackEnv(RackEnv):
    instruction = "Put the cutting board in the dish rack."
    obj = "board"
    task_objects = ("board",)
    rack = PegRack("rack", n_slots=3, gap=0.019, divider=0.005, peg=0.005, row=0.022, length=0.070, height=0.036)
    slots = (0, 1, 2)

    def object_spec(self):
        # A thin wooden board standing on its long edge: 0.9 x 5.0 x 6.2 cm (x = thickness, z = up).
        return Block("board", half=(0.0045, 0.025, 0.031), rgba=LIGHT_WOOD, mass=0.03, friction=1.0)


class BookSlotMoveEnv(BookRackEnv):
    """The book starts standing in the rack's right slot (viewer's right, world +y); goal: the left slot. The rack
    is turned so its slots are stacked roughly along the viewer's left-right (rack y within 30 deg of world y)."""

    instruction = "Move the book from the right slot of the book rack to the left slot."

    @property
    def slots(self):
        return (self.end_slot(-1),)

    def end_slot(self, side):
        """Index of the end slot on the viewer's ``side`` (-1 left = world -y, +1 right = world +y)."""
        R, _ = self.rack_frame()
        ends = (0, self.rack.n_slots - 1)
        return max(ends, key=lambda k: side * self.rack.slot_y(k) * R[1, 1])

    def layout(self):
        placed = []
        rack_r = float(np.hypot(self.rack.length / 2, self.rack.width / 2))
        for _ in range(200):
            xy = self.sample_xy(rack_r, placed, clearance=0.03, r=(0.20, 0.245), angle=(-50.0, 50.0))
            if np.linalg.norm(xy - REST_ZONE[0]) >= REST_ZONE[1] + rack_r:
                break
        self.set_object_pose(self.rack_name, xy, yaw=self.np_random.uniform(-np.pi / 6, np.pi / 6))
        placed.append((xy, rack_r))
        mujoco.mj_forward(self.model, self.data)
        R, _ = self.rack_frame()
        start = self.slot_center(self.end_slot(+1))
        yaw = float(np.arctan2(R[1, 1], R[0, 1])) + self.np_random.uniform(-0.05, 0.05)
        self.set_object_pose(self.obj, start[:2], yaw=yaw, z=start[2])
        self.place_distractors(placed)
        mujoco.mj_forward(self.model, self.data)
        c = self.slot_center(self.slots[0])
        self.set_goals(Goal(self.obj, self.rack_name, target=(c[0], c[1], c[2] + self.half()[2]),
                            reference=self.rack_name, tolerance=(0.012, 0.012, 0.004), check=self.seated))


TASKS = [
    define_task(name="book_in_rack_middle_slot", instruction=BookRackEnv.instruction, family=FAMILY,
                env=BookRackEnv, oracle=RackOracle, objects=("book",), object_kinds=("book",),
                relation="upright in middle slot", goal="book rack",
                steps=("Pick up the book, turn it parallel to the slots and stand it upright in the middle slot of "
                       "the book rack.",)),
    define_task(name="board_in_dish_rack", instruction=BoardDishRackEnv.instruction, family=FAMILY,
                env=BoardDishRackEnv, oracle=RackOracle, objects=("board",), object_kinds=("cutting board",),
                relation="upright in slot", goal="dish rack",
                steps=("Pick up the cutting board, turn it parallel to the slots and stand it upright in a slot of "
                       "the dish rack.",)),
    define_task(name="book_rack_right_to_left_slot", instruction=BookSlotMoveEnv.instruction, family=FAMILY,
                env=BookSlotMoveEnv, oracle=RackOracle, objects=("book",), object_kinds=("book",),
                relation="from right slot to left slot", goal="book rack",
                steps=("Lift the book out of the right slot of the book rack and stand it upright in the left "
                       "slot.",)),
]

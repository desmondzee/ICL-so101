"""Unstack family: take the top object off a two-object stack and put it at a named spot.

The stack is part of the initial layout: the top object is set on the bottom object's upper surface during
``layout()`` and settles during the reset frames, before the first recorded frame. Each task differs in the
stacked object kinds and in where the top object goes (onto a plate, onto a mat, into a bowl, onto the table
beside the bottom object).

Only box-collision objects are stacked (primitive blocks, LIBERO cans and books): scanned GSO/YCB meshes resting
on a free body chatter indefinitely (measured 3-5 rad/s for the gelatin box and candy box on a book or block), so
they never settle.

Success (derived from the declared goal) needs the top object at its target, upright, released, settled and
supported by the goal support, and the goal ``check`` (``bottom_check``) requires the bottom object to have stayed
put. The bottom object is not a task object, so the strict physics gate also treats it as a non-task free body
(<= 5 mm / 0.1 rad motion over the whole episode).

Grasping the top object: the jaw collision hull reaches ~8.2 mm below the TCP, so the oracle grasps with the TCP
at least ``JAW_CLEARANCE`` above the bottom object's upper surface (never at the bottom object).
"""

import mujoco
import numpy as np

from sim.train.tasks.base import LOW_DISTRACTORS, VIEW, Goal, TrainEnv, TrainOracle, define_task, mat
from sim.val.oracle import RELEASE
from sim.val.scene import Block, Fixture, Obj

MAT_T = 0.006                     # mat thickness
JAW_CLEARANCE = 0.0115            # min TCP height above the bottom object's top surface while grasping the top
BOTTOM_TOL = 0.004                # bottom object may move at most this (m) in success(); gate allows 5 mm
STACK_REGION = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
TARGET_REGION = dict(r=(0.16, 0.26), angle=(-55.0, 55.0))

RED, BLUE, GREEN = (0.8, 0.1, 0.08, 1), (0.1, 0.25, 0.8, 1), (0.2, 0.55, 0.25, 1)
GREY_MAT, ORANGE_MAT = (0.45, 0.45, 0.5, 1), (0.85, 0.45, 0.15, 1)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class UnstackEnv(TrainEnv):
    """Shared layout logic: a stack (``top`` on ``bottom``) and a goal spot, both randomized."""

    top = "top"
    bottom = "bottom"
    stack_offset = 0.003           # random xy offset of the top object on the bottom one (m)
    bottom_quat = None             # optional base orientation of the bottom object (before yaw)

    solid = ()                     # free bodies whose collision is replaced by one solid box (see _solidify)

    def _index_bodies(self):
        super()._index_bodies()
        for name in self.solid:
            self._solidify(name)
        # Scanned meshes and LIBERO books have their body origin at a corner/edge (measured up to 4.5 cm off the
        # geometry). Shift their content so the origin is the collision-geometry centre: grasps, stacking and goal
        # targets all use the body origin. Model-only change, done once per env before extents are measured.
        for name in (self.top, self.bottom):
            pts = self._body_points(name, (1.0, 0.0, 0.0, 0.0))
            centre = 0.5 * (pts.min(0) + pts.max(0))
            child = self.model.body(f"{name}_upright").id if f"{name}_upright" in self._body_names() else None
            if child is not None and np.linalg.norm(centre) > 1e-4:
                self.model.body_pos[child] -= centre
        self._extent = self._measure_extents()

    def _solidify(self, name):
        """Replace the collision of ``name`` by one solid box over its collision AABB (model-only, at init).

        LIBERO cans collide as a ring of 1-2 mm thin plates around a core box. The SO-101 fingertip spheres
        (0.75 mm radius) press 1-1.5 mm into a gripped plate, past its half thickness, and stay hooked in it after
        the jaw opens: the retreat then lifts the can off its target or springs it loose (angular-acceleration
        spikes). A single solid box of the same 3.1 x 3.1 x 3.8 cm extent grips and releases cleanly. Mass and
        inertia are unchanged (compiled from the original geoms)."""
        m = self.model
        geoms = self._geoms[name]
        pts = self._body_points(name, (1.0, 0.0, 0.0, 0.0))
        lo, hi = pts.min(0), pts.max(0)
        keep = max(geoms, key=lambda g: float(np.prod(m.geom_size[g])))
        parent = int(m.geom_bodyid[keep])
        # Parent (upright child body) pose relative to the free body, at the model's reference configuration.
        d = mujoco.MjData(m)
        a = self._qadr[name]
        d.qpos[a:a + 7] = [0, 0, 1.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        R = d.xmat[parent].reshape(3, 3)
        origin = d.xpos[self._body[name]]
        centre = origin + 0.5 * (lo + hi)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.T.flatten())          # geom axes = free-body axes
        m.geom_type[keep] = mujoco.mjtGeom.mjGEOM_BOX
        m.geom_size[keep] = 0.5 * (hi - lo)
        m.geom_pos[keep] = R.T @ (centre - d.xpos[parent])
        m.geom_quat[keep] = quat
        m.geom_aabb[keep] = np.r_[np.zeros(3), 0.5 * (hi - lo)]
        m.geom_rbound[keep] = float(np.linalg.norm(0.5 * (hi - lo)))
        for g in geoms:
            if g != keep:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self._geoms[name] = [keep]

    def _body_names(self):
        if not hasattr(self, "_names_cache"):
            self._names_cache = {self.model.body(i).name for i in range(self.model.nbody)}
        return self._names_cache

    def _stack_speed(self):
        out = 0.0
        for name in (self.top, self.bottom):
            v = self.data.qvel[self._dadr[name]:self._dadr[name] + 6]
            out = max(out, float(np.linalg.norm(v[:3])) / 0.002, float(np.linalg.norm(v[3:])) / 0.03)
        return out

    def _settle_after_reset(self):
        """The kit's 20 reset frames, then more (robot held at rest) until the stack is still: <= 2 mm/s and
        <= 0.03 rad/s for 10 consecutive frames, at most 4 s. Runs before the first observation."""
        super()._settle_after_reset()
        calm = 0
        self.settle_frames = 20
        for _ in range(120):
            if calm >= 10:
                break
            mujoco.mj_step(self.model, self.data, nstep=self._N_SUBSTEPS)
            self.settle_frames += 1
            calm = calm + 1 if self._stack_speed() <= 1.0 else 0

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        # After the reset settle frames: this is the first recorded frame's pose.
        self._bottom_start = self.object_pose(self.bottom).copy()
        return result

    # ----- stack construction ----------------------------------------------------------------------------------
    def _body_points(self, name, quat):
        """Collision points (mesh vertices / box corners) of free body ``name`` relative to its origin, with the
        body at orientation ``quat``."""
        m, d = self.model, mujoco.MjData(self.model)
        a = self._qadr[name]
        d.qpos[a:a + 7] = [0, 0, 1.0, *quat]
        mujoco.mj_kinematics(m, d)
        origin = d.xpos[self._body[name]]
        pts = []
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for g in self._geoms[name]:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                v = corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(pts) - origin

    def _rotated_extent(self, name, quat):
        """(xy radius, bottom offset) of free body ``name`` when its body quat is ``quat``."""
        pts = self._body_points(name, quat)
        return float(np.max(np.linalg.norm(pts[:, :2], axis=1))), float(-pts[:, 2].min())

    def bottom_footprint(self):
        if self.bottom_quat is None:
            return self.footprint(self.bottom)
        return self._rotated_extent(self.bottom, self.bottom_quat)[0]

    def set_bottom(self, xy, yaw):
        if self.bottom_quat is None:
            self.set_object_pose(self.bottom, xy, yaw=yaw)
            return
        quat = _qmul(_yaw_quat(yaw), self.bottom_quat)
        _, below = self._rotated_extent(self.bottom, quat)
        a = self._qadr[self.bottom]
        self.data.qpos[a:a + 3] = [xy[0], xy[1], below + 0.0005]
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[self._dadr[self.bottom]:self._dadr[self.bottom] + 6] = 0
        mujoco.mj_kinematics(self.model, self.data)

    def bottom_top_z(self):
        """Height of the bottom object's upper surface under the top object (ray cast, top excluded)."""
        return self.surface_z(self.object_pos(self.bottom)[:2], exclude=(self.top,))

    def place_stack(self, placed, xy=None, top_yaw=None):
        """Put the bottom object at ``xy`` (default: sampled in ``STACK_REGION``) and the top object resting on
        it; returns the stack xy."""
        radius = max(self.bottom_footprint(), self.footprint(self.top))
        if xy is None:
            xy = self.sample_xy(radius, placed, clearance=0.03, **STACK_REGION)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.set_bottom(xy, yaw)
        surface = self.bottom_top_z()
        top_xy = xy + self.np_random.uniform(-self.stack_offset, self.stack_offset, 2)
        top_yaw = yaw + self.np_random.uniform(-0.25, 0.25) if top_yaw is None else top_yaw
        self.set_object_pose(self.top, top_xy, yaw=top_yaw, z=surface)
        placed.append((xy, radius))
        return xy

    # ----- success ---------------------------------------------------------------------------------------------
    def bottom_check(self, env=None, obj=None) -> bool:
        """Goal ``check``: the bottom object is within ``BOTTOM_TOL`` of its first-frame pose, unrotated (0.1 rad),
        resting on the table, and no longer touched by the top object."""
        start = getattr(self, "_bottom_start", None)
        if start is None:
            return False
        pose = self.object_pose(self.bottom)
        if np.linalg.norm(pose[:3] - start[:3]) > BOTTOM_TOL:
            return False
        dot = abs(float(np.dot(pose[3:], start[3:])))
        if 2 * np.arccos(min(1.0, dot)) > 0.1:
            return False
        return self.supported_by(self.bottom, "table") and not self.touching(self.top, self.bottom)


class UnstackOracle(TrainOracle):
    """Pick the top object off the stack (TCP kept clear of the bottom object) and place it at the goal."""

    width = 0.028                  # top object's width along the closing direction
    symmetric = 4
    yaw_offset = 0.0               # closing direction relative to the top object's yaw
    drop = 0.0

    def grasp_z(self):
        env = self.env
        surface = env.bottom_top_z()
        centre = env.object_pos(env.top)[2]
        top_of_top = surface + env.height(env.top)
        return float(min(max(centre - 0.001, surface + JAW_CLEARANCE), top_of_top - 0.005))

    def plan(self):
        env = self.env
        top = env.top
        ok = yield from self.pick(top, self.width, self.grasp_z(), yaw=env.yaw(top) + self.yaw_offset,
                                  symmetric=self.symmetric)
        if not ok:
            return
        target = env.goal_target(env.goals[0])
        yield from self.place_object(top, target, drop=self.drop, open_to=self.release_for(self.width))
        yield from self.rest()
        yield from self.wait(1.2)


# ----- 1. block off block, onto the plate ---------------------------------------------------------------------

HALF_TOP, HALF_BOTTOM = 0.014, 0.016


class BlockToPlateEnv(UnstackEnv):
    instruction = "Take the red block off the blue block and put it on the plate."
    top, bottom = "red_block", "blue_block"
    task_objects = ("red_block",)
    surfaces = ("blue_block",)
    extra_contacts = (("red_block", "blue_block"),)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n != "plate")

    def scene_objects(self):
        return [Block("red_block", half=(HALF_TOP,) * 3, rgba=RED),
                Block("blue_block", half=(HALF_BOTTOM,) * 3, rgba=BLUE, mass=0.05)]

    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.5)]

    def layout(self):
        placed = []
        self.place_stack(placed)
        plate_xy, _ = self.place_fixture("plate", placed, region=TARGET_REGION)
        self.place_distractors(placed)
        surface = self.surface_z(plate_xy, exclude=self.task_objects)
        self.set_goals(Goal("red_block", "plate", target=(*plate_xy, surface + HALF_TOP),
                            tolerance=(0.02, 0.02, 0.006), check=self.bottom_check))


class BlockToPlateOracle(UnstackOracle):
    width = 2 * HALF_TOP


# ----- 2. soup can off the book, onto the mat -----------------------------------------------------------------

BOOK_FLAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])   # standing LIBERO book laid on its cover


class CanToMatEnv(UnstackEnv):
    instruction = "Take the soup can off the book and put it on the mat."
    top, bottom = "soup_can", "book"
    task_objects = ("soup_can",)
    solid = ("soup_can",)
    surfaces = ("book",)
    extra_contacts = (("soup_can", "book"),)
    bottom_quat = BOOK_FLAT
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce"))

    def scene_objects(self):
        return [Obj("soup_can", "alphabet_soup"), Obj("book", "black_book", mass=0.06),
                mat("mat", half=(0.035, 0.035), rgba=ORANGE_MAT)]

    def layout(self):
        placed = []
        self.place_stack(placed)
        mat_xy, _ = self.place("mat", placed, region=TARGET_REGION)
        self.place_distractors(placed)
        self.set_goals(Goal("soup_can", "mat", target=(*mat_xy, MAT_T + self._extent["soup_can"]["bottom"]),
                            reference="mat", tolerance=(0.02, 0.02, 0.006), check=self.bottom_check))


class CanOracle(UnstackOracle):
    """Can pick-and-place (cans use the solid-box collision from ``UnstackEnv._solidify``). Measured fixes:
    a wide release (the can turns a few degrees in the jaws while carried; 3.1 cm box, 4.4 cm diagonal), lowering
    until contact before opening, a slow back-off along the measured can-to-finger direction."""
    width = 0.031
    symmetric = 2
    release_margin = 0.020
    release_backoff = 0.006
    release_seconds = 1.5
    backoff_speed = 0.012          # slow: the can, pressed tilted against the fixed finger, rights itself gently

    def release(self, open_to=None):
        """Touch down, open, then slide the fixed finger off the can along the *measured* can-to-finger direction:
        the can slips several mm in the jaws while carried, so the commanded closing axis (the kit's back-off
        direction) can point partly into the can and drag it along (measured, calibration seed 9104)."""
        env, rot = self.env, self._cmd_rot
        # Lower until the can touches its support (<= 3 mm, 0.5 mm steps): a can held a few degrees tilted is
        # released ~2 mm high by the kit's tilt-aware release height and slaps flat onto the rigid table
        # (measured 1000-1200 rad/s^2 against the 1000 limit).
        support = env._owned_geoms(env.goals[0].support)
        held = set(env._geoms[env.top])
        for _ in range(6):
            if any((c.geom1 in held and c.geom2 in support) or (c.geom2 in held and c.geom1 in support)
                   for c in env.data.contact[:env.data.ncon]):
                break
            yield from self.move(self._cmd_pos - [0, 0, 0.0005], rot, speed=0.01, tol=0.0005, settle=0.15,
                                 label="touch", smooth=False)
        yield from self.gripper(RELEASE if open_to is None else open_to, self.release_seconds, 0.2)
        tips = [self.model.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
        finger = env.data.geom_xpos[tips].mean(0)[:2]
        away = finger - env.object_pos(env.top)[:2]
        away = away / max(np.linalg.norm(away), 1e-6)
        yield from self.move(self._cmd_pos + self.release_backoff * np.r_[away, 0.0], rot, speed=self.backoff_speed,
                             tol=0.002, settle=0.2, label="backoff", smooth=False)


class CanToMatOracle(CanOracle):
    pass


# ----- 3. block off the book, into the bowl -------------------------------------------------------------------

class BlockIntoBowlEnv(UnstackEnv):
    instruction = "Take the green block off the book and put it in the bowl."
    top, bottom = "green_block", "book"
    task_objects = ("green_block",)
    surfaces = ("book",)
    extra_contacts = (("green_block", "book"),)
    bottom_quat = BOOK_FLAT
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if "bowl" not in n and n != "ramekin")

    def scene_objects(self):
        return [Block("green_block", half=(HALF_TOP,) * 3, rgba=GREEN), Obj("book", "yellow_book", mass=0.06)]

    def scene_fixtures(self):
        return [Fixture("bowl", "white_bowl", 1.2)]

    def layout(self):
        placed = []
        self.place_stack(placed)
        bowl_xy, _ = self.place_fixture("bowl", placed, region=dict(r=(0.17, 0.26), angle=(-60, 60)))
        self.place_distractors(placed)
        floor = self.surface_z(bowl_xy, exclude=self.task_objects)
        self.set_goals(Goal("green_block", "bowl", target=(*bowl_xy, floor + HALF_TOP),
                            tolerance=(0.018, 0.018, 0.006), check=self.bottom_check))


class BlockIntoBowlOracle(UnstackOracle):
    width = 2 * HALF_TOP
    drop = 0.001


# ----- 4. tomato sauce can off the book, onto the table left of the book --------------------------------------

class SauceLeftOfBookEnv(UnstackEnv):
    instruction = "Take the tomato sauce can off the book and set it on the table to the left of the book."
    top, bottom = "sauce_can", "book"
    task_objects = ("sauce_can",)
    solid = ("sauce_can",)
    surfaces = ("book",)
    extra_contacts = (("sauce_can", "book"),)
    bottom_quat = BOOK_FLAT
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n not in ("alphabet_soup", "tomato_sauce"))
    gap = 0.085            # book centre to can centre along the viewer's left (world -y)

    def scene_objects(self):
        return [Obj("sauce_can", "tomato_sauce"), Obj("book", "black_book", mass=0.06)]

    def layout(self):
        placed = []
        for _ in range(200):
            stack_xy = self.sample_xy(self.bottom_footprint(), [], clearance=0.03, r=STACK_REGION["r"],
                                      angle=(-20.0, 55.0))
            target = stack_xy + self.gap * VIEW["left"]
            r, a = np.hypot(*target), np.degrees(np.arctan2(target[1], target[0]))
            in_sweep = target[0] < 0.21 and abs(target[1]) < 0.11     # rest sweep of the folded gripper
            if 0.18 <= r <= TARGET_REGION["r"][1] and abs(a) <= 55 and not in_sweep:
                break
        else:
            raise RuntimeError("no reachable target left of the book")
        self.place_stack(placed, xy=stack_xy)
        placed.append((target, self.footprint("sauce_can")))
        self.place_distractors(placed)
        self.set_goals(Goal("sauce_can", "table", target=(*target, self._extent["sauce_can"]["bottom"]),
                            tolerance=(0.02, 0.02, 0.006), check=self.bottom_check))


class SauceLeftOfBookOracle(CanOracle):
    pass


TASKS = [
    define_task(name="unstack_block_onto_plate", instruction=BlockToPlateEnv.instruction, family="unstack",
                env=BlockToPlateEnv, oracle=BlockToPlateOracle, objects=("red_block",), object_kinds=("block",),
                relation="off block onto", goal="plate",
                steps=("Pick up the red block from the top of the blue block and put it on the plate.",)),
    define_task(name="unstack_can_onto_mat", instruction=CanToMatEnv.instruction, family="unstack",
                env=CanToMatEnv, oracle=CanToMatOracle, objects=("soup_can",), object_kinds=("can",),
                relation="off book onto", goal="mat",
                steps=("Pick up the soup can from the top of the book and put it on the mat.",)),
    define_task(name="unstack_block_into_bowl", instruction=BlockIntoBowlEnv.instruction, family="unstack",
                env=BlockIntoBowlEnv, oracle=BlockIntoBowlOracle, objects=("green_block",),
                object_kinds=("block",), relation="off book into", goal="bowl",
                steps=("Pick up the green block from the top of the book and put it in the bowl.",)),
    define_task(name="unstack_can_left_of_book", instruction=SauceLeftOfBookEnv.instruction, family="unstack",
                env=SauceLeftOfBookEnv, oracle=SauceLeftOfBookOracle, objects=("sauce_can",),
                object_kinds=("can",), relation="off book onto table left of", goal="book",
                steps=("Pick up the tomato sauce can from the top of the book and set it on the table to the left "
                       "of the book.",)),
]

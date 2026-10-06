"""take_out_of_container: lift an object out of a container and set it down at a named spot outside it.

Every task starts with the object resting inside a static container (a fixture, so it never moves) and
ends with the object released, settled and upright on a destination outside that container: a mat, a
plate, a coaster, or a named side of the container on the bare table. The tasks differ in the object
kind (tomato sauce jar, soup can, golf ball, pudding cup), the container kind (black bowl, ramekin, red bowl) and the
destination/relation (on a mat, on a plate, on a coaster, on the table to the left of the container).

Container choice (measured 2026-10-06): the SO-101 wrist-camera mount sits 4.5-7 cm above the TCP and
sticks out 8 cm sideways, and the motor housing starts 3 cm above the TCP, so the LIBERO basket (7 cm
walls at 0.5x, 5.6 cm at 0.4x) cannot be entered top-down for a flat object. Shallow containers with a
flat floor are used instead: akita black bowl 1.0x (flat floor r <= 2.5 cm, rim 5 cm at r = 5 cm),
ramekin 1.0x (flat floor r <= 3 cm, wall 4.2 cm at r = 4 cm), red bowl 1.5x (flat r <= 2.5 cm, rim 5.2 cm
at r = 6 cm).

Thin objects are not used: the open jaw meshes hang 8.1 mm below the TCP, so a 9 mm LIBERO cream cheese /
butter box on a container floor can only be pinched by the fingertip spheres; it either slipped out during
the carry or (squeezed) sank > 3 mm into the jaws (5 of 12 calibration seeds failed). The 1.6 cm YCB gelatin
box was tried next and dropped too: its convex-hull mesh rocks on the 6 mm box mat (4 mm penetration, never
settles) and it slipped in the jaws during carries.
"""

import mujoco

import numpy as np

from sim.train.tasks.base import (CLOSED, LOW_DISTRACTORS, OPEN, VIEW, Goal, TrainEnv, TrainOracle, define_task,
                                  mat)
from sim.train.tasks.assets import Scanned
from sim.val.scene import Fixture, Obj

FAMILY = "take_out_of_container"
MAT_THICK = 0.006
GREY_MAT, CORK = (0.45, 0.45, 0.5, 1.0), (0.62, 0.45, 0.28, 1.0)
CONTAINER_REGION = dict(r=(0.16, 0.25), angle=(-60.0, 60.0))   # container centre (object grasped there)
DEST_REGION = dict(r=(0.16, 0.26), angle=(-65.0, 65.0))        # destination centre (r >= 0.16: no fold-in)


# Bowl-like distractors are excluded everywhere so "the bowl"/"the ramekin" names exactly one container (the
# akita "black" bowl renders speckled grey and the LIBERO red bowl renders white, so colour words are avoided).
CONTAINERS = ("white_bowl", "red_bowl", "akita_black_bowl", "ramekin")


def _pool(*exclude):
    return tuple(n for n in LOW_DISTRACTORS if n not in exclude)


class OutOfContainerEnv(TrainEnv):
    """Shared layout: container fixture, object inside it, a destination outside it, distractors."""

    container = "bowl"
    obj = "item"
    jitter = 0.004                     # object offset from the container centre (m)

    def place_container(self, placed):
        xy, _ = self.place_fixture(self.container, placed, region=CONTAINER_REGION)
        return xy

    def put_inside(self, container_xy):
        """Rest the object on the container floor with its geometric centre near the container centre."""
        floor = self.surface_z(container_xy, exclude=self.task_objects)
        xy = container_xy + self.np_random.uniform(-self.jitter, self.jitter, 2)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.set_object_pose(self.obj, xy, yaw=yaw, z=floor)
        shift = xy - self.center(self.obj)[:2]          # scanned meshes: origin is off the centre
        self.set_object_pose(self.obj, xy + shift, yaw=yaw, z=floor)
        # Centre height above the object's lowest point, for centre-based placement.
        self.rest_height = float(self.center(self.obj)[2] - self.bottom_z(self.obj))
        return floor

    def local_points(self, name):
        """Collision-geometry AABB corners of a free body in its own frame."""
        m, d = self.model, self.data
        body = self._body[name]
        R, p = d.xmat[body].reshape(3, 3), d.xpos[body]
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        pts = [((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ d.geom_xmat[g].reshape(3, 3).T
                + d.geom_xpos[g] - p) @ R for g in self._geoms[name]]
        return np.vstack(pts)

    def hull_points(self, name):
        """World-frame collision vertices (mesh vertices, box corners; other primitives by their AABB)."""
        m, d = self.model, self.data
        pts = []
        for g in self._geoms[name]:
            R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                half = m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_aabb[g, 3:]
                local = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
            pts.append(np.einsum("ij,nj->ni", R, np.asarray(local, np.float64)) + p)
        return np.vstack(pts)

    def bottom_z(self, name):
        return float(self.hull_points(name)[:, 2].min())

    def center(self, name):
        """World position of the centre of the body's collision bounding box."""
        pts = self.local_points(name)
        return self.data.xpos[self._body[name]] + self.data.xmat[self._body[name]].reshape(3, 3) @ (
            (pts.max(0) + pts.min(0)) / 2)

    def origin_height(self, name):
        """Body-origin height above the object's lowest collision point."""
        return self._extent[name]["bottom"]


class HardScanned(Scanned):
    """Scanned object with a stiffer contact (``solref`` time constant; MuJoCo default 0.02). With the default,
    the 10 g scanned golf ball squeezed 3.0-3.4 mm into the closed jaws; real golf balls and card boxes are
    stiff, and 0.01 matches the SO-101 jaw geoms."""

    solref = (0.01, 1.0)

    def build_mjcf(self, mj, asset_el, worldbody):
        super().build_mjcf(mj, asset_el, worldbody)
        for body in worldbody.iter("body"):
            if body.get("name") == self.name:
                for geom in body.iter("geom"):
                    if geom.get("contype") != "0":
                        geom.set("solref", " ".join(map(str, self.solref)))


# ----- 1. tomato sauce: black bowl -> mat ------------------------------------------------------------------

ON_MAT = 0.02                      # object centre within 2 cm of the mat/coaster centre


def centred_on(support, radius=ON_MAT):
    def check(env, name):
        return bool(np.linalg.norm(env.center(name)[:2] - env.object_pos(support)[:2]) <= radius)
    check.__name__ = f"centred_on_{support}"
    return check


class TomatoSauceEnv(OutOfContainerEnv):
    instruction = "Take the tomato sauce out of the bowl and put it on the grey mat."
    task_objects = ("tomato_sauce",)
    obj, container = "tomato_sauce", "bowl"
    surfaces = ("bowl",)
    distractor_pool = _pool("tomato_sauce", "alphabet_soup", *CONTAINERS)

    def scene_objects(self):
        return [Obj("tomato_sauce", "tomato_sauce"), mat("mat", rgba=GREY_MAT)]

    def scene_fixtures(self):
        return [Fixture("bowl", "akita_black_bowl", 1.0)]

    def layout(self):
        placed = []
        bowl_xy = self.place_container(placed)
        self.put_inside(bowl_xy)
        mat_xy, _ = self.place("mat", placed, region=DEST_REGION, clearance=0.04)
        self.place_distractors(placed)
        self.set_goals(Goal("tomato_sauce", "mat", target=(*mat_xy, MAT_THICK + self.origin_height("tomato_sauce")),
                            reference="mat", tolerance=(0.02, 0.02, 0.006)))


# ----- 2. soup can: ramekin -> plate -----------------------------------------------------------------------

class SoupCanEnv(OutOfContainerEnv):
    instruction = "Take the soup can out of the ramekin and put it on the plate."
    task_objects = ("alphabet_soup",)
    obj, container = "alphabet_soup", "ramekin"
    surfaces = ("ramekin",)
    distractor_pool = _pool("alphabet_soup", "tomato_sauce", "plate", *CONTAINERS)

    def scene_objects(self):
        return [Obj("alphabet_soup", "alphabet_soup")]

    def scene_fixtures(self):
        return [Fixture("ramekin", "ramekin", 1.0), Fixture("plate", "plate", 0.65)]

    def layout(self):
        placed = []
        ramekin_xy = self.place_container(placed)
        self.put_inside(ramekin_xy)
        plate_xy, _ = self.place_fixture("plate", placed, region=DEST_REGION, clearance=0.03)
        self.place_distractors(placed)
        top = self.surface_z(plate_xy, exclude=self.task_objects)
        self.set_goals(Goal("alphabet_soup", "plate", target=(*plate_xy, top + self.origin_height("alphabet_soup")),
                            tolerance=(0.025, 0.025, 0.006)))


# ----- 3. golf ball: ramekin -> coaster --------------------------------------------------------------------

BALL_MASS = 0.03                  # a real golf ball is 46 g; 10 g squeezed 3 mm into the jaws (light contact mass)




class BallEnv(OutOfContainerEnv):
    instruction = "Take the golf ball out of the ramekin and put it on the brown coaster."
    task_objects = ("golf_ball",)
    obj, container = "golf_ball", "ramekin"
    surfaces = ("ramekin",)
    distractor_pool = _pool(*CONTAINERS)

    def scene_objects(self):
        return [HardScanned("golf_ball", "ycb", "058_golf_ball", scale=0.5, mass=BALL_MASS),
                mat("coaster", half=(0.03, 0.03), rgba=CORK)]

    def scene_fixtures(self):
        return [Fixture("ramekin", "ramekin", 1.0)]

    def layout(self):
        placed = []
        ramekin_xy = self.place_container(placed)
        self.put_inside(ramekin_xy)
        coaster_xy, _ = self.place("coaster", placed, region=DEST_REGION, clearance=0.04)
        self.place_distractors(placed)
        # The scanned ball's body origin is 1.5 cm off its centre and its final spin is free, so the goal is
        # "resting on the coaster" (support contact, released, settled) with its centre over the coaster.
        self.set_goals(Goal("golf_ball", "coaster", target=None, upright_cos=None, check=centred_on("coaster")))


# ----- 4. pudding: red bowl -> table left of the bowl ------------------------------------------------------

class PuddingLeftEnv(OutOfContainerEnv):
    instruction = "Take the chocolate pudding out of the bowl and set it on the table to the left of the bowl."
    task_objects = ("chocolate_pudding",)
    obj, container = "chocolate_pudding", "bowl"
    surfaces = ("bowl",)
    distractor_pool = _pool("chocolate_pudding", "popcorn", *CONTAINERS)
    gap = 0.11                         # bowl centre to goal centre along the viewer's left (world -y)

    def scene_objects(self):
        return [Obj("chocolate_pudding", "chocolate_pudding")]

    def scene_fixtures(self):
        return [Fixture("bowl", "red_bowl", 1.5)]

    def layout(self):
        placed = []
        for _ in range(50):
            bowl_xy = self.sample_xy(self.fixture_footprint("bowl"), placed, r=(0.17, 0.25), angle=(-30.0, 60.0))
            target = bowl_xy + self.gap * VIEW["left"]
            r, a = np.hypot(*target), np.degrees(np.arctan2(target[1], target[0]))
            if 0.16 <= r <= 0.26 and abs(a) <= 65:
                break
        else:
            raise RuntimeError("no reachable left-of-bowl target")
        self.set_fixture_pose("bowl", bowl_xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        placed.append((bowl_xy, self.fixture_footprint("bowl")))
        placed.append((target, 0.035))                     # keep the goal spot free
        self.put_inside(bowl_xy)
        self.place_distractors(placed)

        def left_of_bowl(env, name):
            p = env.object_pos(name)[:2]
            return bool(p[1] < bowl_xy[1] - 0.08 and abs(p[0] - bowl_xy[0]) < 0.03)

        self.set_goals(Goal("chocolate_pudding", "table",
                            target=(*target, self.origin_height("chocolate_pudding")),
                            tolerance=(0.02, 0.02, 0.006), check=left_of_bowl))


# ----- oracle ----------------------------------------------------------------------------------------------

class OutOfContainerOracle(TrainOracle):
    """Grasp the object inside its container across its narrow side and place it on the goal.

    The jaw opens only ``open_margin`` beyond the object (the container walls are close) and the TCP stays
    at least ``floor_clearance`` above the container floor (the jaw hull reaches ~8 mm below the TCP)."""

    open_margin = 0.008
    floor_clearance = 0.0095
    symmetric_object = False           # round objects: any closing direction
    squeeze = 0.0                      # close to width - squeeze at the fingertip-sphere centres (None: fully)

    def close_target(self, width):
        if self.squeeze is None:
            return CLOSED
        self.jaw_gap(OPEN)
        angles, gaps = self._gap_table
        return float(max(CLOSED, np.interp(width - self.squeeze, gaps, angles)))

    def pick(self, name, width, grasp_z, yaw=None, attempts=2, approach=0.05, symmetric=4, lift_z=None):
        """Kit ``pick`` with a bounded squeeze: closing fully on a light, soft-contact box (LIBERO food boxes
        have solref 0.02) pressed the fingertip spheres 3 mm into it and could pop it out of the jaws."""
        lift_z = self.carry_z if lift_z is None else lift_z
        for attempt in range(attempts):
            obj = self.env.center(name)
            base = np.arctan2(obj[1], obj[0]) if yaw is None else yaw
            angles = [base + k * 2 * np.pi / symmetric for k in range(symmetric)]
            rot, grasp, q = self.grasp_plan(np.array([obj[0], obj[1], grasp_z]), width, angles)
            yield from self.gripper(self.open_for(width), 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(self.close_target(width), self.close_seconds, 0.3)
            ok = self.env.is_grasping(name)
            self.log.append((name, "grasp", attempt, ok))
            if ok:
                yield from self.move(np.array([grasp[0], grasp[1], lift_z]), rot, speed=self.lift_speed, label="lift")
                if self.env.is_grasping(name):
                    return True
                self.log.append((name, "dropped", attempt))
            yield from self.gripper(self.open_for(width), 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    lift_speed = 0.12
    level = True                       # re-level the held object before lowering it
    pre_release_height = 0.012         # pause this far above the release pose to re-measure the hold

    def level_rot(self, name, rot):
        """TCP rotation near ``rot`` at which the held object's up axis is vertical. A lightly squeezed
        object pivots 2-4 deg in the jaws while carried; released tilted, it drops onto its low edge and
        slaps flat (measured 1000-1400 rad/s^2 on the pudding cup)."""
        env = self.env
        r_obj_tcp = self.tcp_rot().T @ env.data.xmat[env._body[name]].reshape(3, 3)
        m = rot @ r_obj_tcp
        yaw = np.arctan2(m[1, 0], m[0, 0])
        r_des = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        return r_des @ r_obj_tcp.T

    def place_object(self, name, target, rot=None, drop=0.0, open_to=None, carry_z=None):
        """Kit ``place_object`` plus levelling: after the carry the TCP is re-oriented so the held object
        sits level, lowered to just above the release pose, re-levelled and re-measured, then lowered and
        released."""
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

    by_center = False                  # place the geometric centre (scanned meshes) instead of the origin

    def held(self, name):
        if self.by_center:
            return self.tcp_rot().T @ (self.env.center(name) - self.tcp())
        return super().held(name)

    def narrow_grasp(self, name):
        """(closing yaw, width) across the object's narrow horizontal side (rotating calipers, 1 deg)."""
        xy = self.env.hull_points(name)[:, :2]
        angles = np.radians(np.arange(0.0, 180.0, 1.0))
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        proj = np.einsum("nk,ak->na", xy, dirs)
        widths = proj.max(0) - proj.min(0)
        k = int(np.argmin(widths))
        return float(angles[k]), float(widths[k])

    def goal_point(self, goal):
        """Where the placed object's reference point (origin, or centre when ``by_center``) must go."""
        env = self.env
        target = env.goal_target(goal)
        if target is not None:
            return target
        support = env.object_pos(goal.support)[:2]
        return np.r_[support, env.surface_z(support, exclude=env.task_objects) + env.rest_height]

    def plan(self):
        env = self.env
        name = env.task_objects[0]
        goal = env.goals[0]
        yaw, width = self.narrow_grasp(name)
        bottom = env.bottom_z(name)
        grasp_z = max(env.center(name)[2], bottom + self.floor_clearance)
        ok = yield from self.pick(name, width, grasp_z, yaw=yaw, symmetric=4 if self.symmetric_object else 2)
        if not ok:
            return
        yield from self.place_object(name, self.goal_point(goal), open_to=self.release_for(width))
        yield from self.rest()
        yield from self.wait(1.2)


class CanOracle(OutOfContainerOracle):
    """The 32 g LIBERO soup can pivoted in the jaws with a zero squeeze (4/25 calibration seeds failed by an
    angular spike on release or a slip in the carry); a 3 mm squeeze holds them (7/8 on those seeds)."""
    squeeze = 0.003


class TomatoSauceOracle(OutOfContainerOracle):
    """The tomato sauce jar slipped out of a 3 mm squeeze while being levelled over the mat (10/50 qualification
    failures); with the kit's full close it held on 12 of 13 of those and neighbouring seeds."""
    squeeze = None


class BallOracle(OutOfContainerOracle):
    """The ball is placed by its geometric centre (its body origin is off-centre) and not levelled; the hard
    ball needs the full kit close (a bounded squeeze does not reach the grasp-detection force)."""
    symmetric_object = True
    by_center = True
    squeeze = None
    level = False
    release_lift_fraction = 0.0


TASKS = [
    define_task(name="tomato_sauce_out_of_bowl_onto_mat", instruction=TomatoSauceEnv.instruction, family=FAMILY,
                env=TomatoSauceEnv, oracle=TomatoSauceOracle, objects=("tomato_sauce",),
                object_kinds=("tomato sauce jar",), relation="out of bowl onto", goal="mat",
                steps=("Lift the tomato sauce out of the bowl and put it on the grey mat.",)),
    define_task(name="soup_can_out_of_ramekin_onto_plate", instruction=SoupCanEnv.instruction, family=FAMILY,
                env=SoupCanEnv, oracle=CanOracle, objects=("alphabet_soup",),
                object_kinds=("soup can",), relation="out of ramekin onto", goal="plate",
                steps=("Lift the soup can out of the ramekin and put it on the plate.",)),
    define_task(name="golf_ball_out_of_ramekin_onto_coaster", instruction=BallEnv.instruction, family=FAMILY,
                env=BallEnv, oracle=BallOracle, objects=("golf_ball",), object_kinds=("golf ball",),
                relation="out of ramekin onto", goal="coaster",
                steps=("Lift the golf ball out of the ramekin and put it on the brown coaster.",)),
    define_task(name="pudding_out_of_bowl_left_of_bowl", instruction=PuddingLeftEnv.instruction, family=FAMILY,
                env=PuddingLeftEnv, oracle=OutOfContainerOracle, objects=("chocolate_pudding",),
                object_kinds=("pudding box",), relation="out of bowl onto table left of", goal="bowl",
                steps=("Lift the chocolate pudding out of the bowl and set it on the table to the left of "
                       "the bowl.",)),
]

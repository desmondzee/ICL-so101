"""push_together: push one object along the table (closed jaws, never grasped) until it touches another.

Non-prehensile, as in ``push_to_target`` (whose pusher model, slide-friction and contact settings this module
copies; see that module's docstring for the measurements). The pushed object starts 5-9 cm from the reference,
on the outward normal of one of the reference's faces; the oracle pushes it along that normal, stops a few mm short,
measures the remaining gap and closes it with a final slow stroke of gap + ``overlap``.

Strict success, on top of the kit's released/settled/upright(10 deg)/supported-by-the-table goal:

* the pushed object's collision geometry is within ``TOUCH`` of the reference's (they touch; a resting contact
  with no load flickers in and out of MuJoCo's contact list, so a distance is used);
* it sits against a face, not a corner: its centre is within the reference face's half width (along the face) of
  the reference centre;
* it was never lifted or grasped, and the jaws no longer touch it.

The reference must stay put: a free reference is a non-task body for the strict physics gate (moved <= 5 mm,
turned <= 0.1 rad); a standing book is a static fixture (free standing books topple on their own). Only the
object/reference contact is added to the allowed contacts; any robot contact with the reference fails.
"""

import mujoco
import numpy as np

from sim.train.tasks.base import LOW_DISTRACTORS, TRAIN_KEEPOUT, Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.val.scene import Block, Fixture, Obj

FAMILY = "push_together"
TOUCH = 0.0015             # max collision-geometry distance between the two objects (m)
LIFT_TOL = 0.004           # max rise of the pushed object above its resting height at any frame (m)


# ----- geometry helpers (as push_to_target) ------------------------------------------------------------------

def collision_points(env, name):
    """World xyz of the collision-geometry AABB corners of a free body."""
    m, d = env.model, env.data
    corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = []
    for g in env._geoms[name]:
        rot = d.geom_xmat[g].reshape(3, 3)
        pts.append((corners * m.geom_aabb[g, 3:] + m.geom_aabb[g, :3]) @ rot.T + d.geom_xpos[g])
    return np.vstack(pts)


def extent_along(env, name, u):
    proj = collision_points(env, name)[:, :2] @ np.asarray(u, float)[:2]
    return float(proj.min()), float(proj.max())


def center(env, name):
    """World xy of the centre of the collision bounding box (mesh body origins can sit off centre)."""
    return np.array([np.mean(extent_along(env, name, u)) for u in ((1.0, 0.0), (0.0, 1.0))])



def gap(env, a, b):
    """Minimum distance (m) between the collision geometry of bodies ``a`` and ``b`` (negative: penetrating)."""
    m, d = env.model, env.data
    return min(mujoco.mj_geomDistance(m, d, g, h, 0.1, None) for g in env._geoms[a] for h in env._geoms[b])


def body_axes(env, name):
    """World xy unit vectors of the body's local x and y axes."""
    R = env.data.xmat[env._body[name]].reshape(3, 3)
    return [R[:2, k] / np.linalg.norm(R[:2, k]) for k in (0, 1)]


def against(env, obj):
    """Touching the reference, against one of its faces, never lifted/grasped, jaws away."""
    ref = env.ref
    if gap(env, obj, ref) > TOUCH:
        return False
    rel = center(env, obj) - center(env, ref)
    normal = env.face_normal()
    tangent = np.array([-normal[1], normal[0]])
    lo, hi = extent_along(env, ref, tangent)
    if abs(rel @ tangent) > (hi - lo) / 2 or rel @ normal <= 0:
        return False
    return obj not in env.lifted and not env.jaws_touching(obj)


# ----- environment -------------------------------------------------------------------------------------------

class TogetherEnv(TrainEnv):
    """Reference ``ref`` and pushed object ``obj``; subclasses add ``scene_objects``/``scene_fixtures``."""

    obj = ""
    ref = ""
    ref_region = dict(r=(0.17, 0.25), angle=(-55.0, 55.0))
    start_region = dict(r=(0.145, 0.26), angle=(-62.0, 62.0))
    pusher_region = dict(r=(0.125, 0.27), angle=(-70.0, 70.0))
    start_gap = (0.05, 0.09)        # initial surface gap along the push (m)
    faces = (0, 1, 2, 3)            # reference faces the object may start on: +x, +y, -x, -y (local)
    obj_yaw_along = None            # if set: the object's local axis (0/1) to align with the push, +-20 deg
    standoff = 0.03
    corridor = 0.055
    slide_friction = 0.4
    contact_timeconst = 0.025
    contact_damping = 2.0
    distractor_pool = LOW_DISTRACTORS
    probe_oracle = None             # oracle class used to screen layouts for a feasible push (set below)

    @property
    def extra_contacts(self):
        return ((self.obj, self.ref),)

    def reset(self, *args, **kwargs):
        self.lifted = set()
        self._rest_z = {}
        return super().reset(*args, **kwargs)

    def step(self, action, **kwargs):
        result = super().step(action, **kwargs)
        for name in self.task_objects:
            if name in self._rest_z and (self.object_pos(name)[2] > self._rest_z[name] + LIFT_TOL
                                         or self.is_grasping(name)):
                self.lifted.add(name)
        return result

    def jaws_touching(self, name):
        geoms = set(self._geoms[name])
        jaws = {self.model.body(b).id for b in ("gripper", "moving_jaw_so101_v1")}
        for c in self.data.contact[:self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if pair & geoms and any(self.model.geom_bodyid[g] in jaws for g in pair - geoms):
                return True
        return False

    def set_slide_friction(self):
        """As push_to_target: sliding friction 0.4 (with contact priority) so the finger-side push does not tip
        the object, and a slightly soft, overdamped contact against stick-slip chatter."""
        for g in self._geoms[self.obj]:
            self.model.geom_priority[g] = 1
            self.model.geom_friction[g] = (self.slide_friction, 0.005, 0.0001)
            self.model.geom_solref[g] = (self.contact_timeconst, self.contact_damping)

    def polar_ok(self, xy, region):
        r, a = np.hypot(*xy), np.degrees(np.arctan2(xy[1], xy[0]))
        return region["r"][0] <= r <= region["r"][1] and region["angle"][0] <= a <= region["angle"][1]

    def clear_of_keepout(self, xy, radius):
        return all(np.linalg.norm(xy - c) >= q + radius for c, q in TRAIN_KEEPOUT)

    def put(self, name, xy, yaw):
        """Put ``name`` (free body or fixture) with its bounding-box centre at ``xy``."""
        setter = self.set_fixture_pose if name in self.fixture_names else self.set_object_pose
        setter(name, xy, yaw=yaw)
        setter(name, np.asarray(xy) + self.object_pos(name)[:2] - center(self, name), yaw=yaw)

    def ref_radius(self):
        return self.fixture_footprint(self.ref) if self.ref in self.fixture_names else self.footprint(self.ref)

    def push_feasible(self, xy, goal_xy, u, half):
        """The oracle's own check: a reachable push orientation whose hover/start/end poses and transit from
        rest keep the camera mount clear of the arm (some pushes are only clear with the mount leading)."""
        if self.probe_oracle is None:
            return True
        probe = self.probe_oracle(self)
        probe.start()
        start = xy - u * (half + probe.standoff)
        return probe.choose(start, goal_xy - u * half, u)[2]

    def face_normal(self):
        """Outward normal (world xy) of the reference face the object is pushed against (frozen at layout)."""
        axes = body_axes(self, self.ref)
        k = self._face
        return (axes[k % 2] if k < 2 else -axes[k % 2])

    def layout(self):
        rng = self.np_random
        self.set_slide_friction()
        for _ in range(800):
            rr, aa = rng.uniform(*self.ref_region["r"]), np.radians(rng.uniform(*self.ref_region["angle"]))
            ref_xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if not self.clear_of_keepout(ref_xy, self.ref_radius() + 0.01):
                continue
            self.put(self.ref, ref_xy, float(rng.uniform(-np.pi, np.pi)))
            self._face = int(rng.choice(self.faces))
            n = self.face_normal()
            lo, hi = extent_along(self, self.ref, n)
            face_xy = center(self, self.ref) + n * (hi - lo) / 2
            if self.obj_yaw_along is None:
                yaw = float(rng.uniform(-np.pi, np.pi))
            else:
                yaw = float(np.arctan2(n[1], n[0]) + np.radians(rng.uniform(-20, 20)) + np.pi * rng.integers(2)
                            - self.obj_yaw_along * np.pi / 2)
            self.put(self.obj, face_xy, yaw)
            olo, ohi = extent_along(self, self.obj, n)
            tangent = np.array([-n[1], n[0]])
            xy = face_xy + n * ((ohi - olo) / 2 + rng.uniform(*self.start_gap)) + tangent * rng.uniform(-0.004, 0.004)
            if not self.polar_ok(xy, self.start_region) or not self.clear_of_keepout(xy, self.footprint(self.obj)):
                continue
            self.put(self.obj, xy, yaw)
            u = -n
            pusher = xy - u * ((ohi - olo) / 2 + self.standoff)
            side = tangent * 0.022
            if not all(self.polar_ok(p, self.pusher_region) for p in (pusher + side, pusher - side)):
                continue
            goal_xy = face_xy + n * (ohi - olo) / 2
            if not self.clear_of_keepout(goal_xy, self.footprint(self.obj)):
                continue
            if not self.push_feasible(xy, goal_xy, u, (ohi - olo) / 2):
                continue
            break
        else:
            raise RuntimeError("no push-together layout")
        placed = [(ref_xy, self.ref_radius())]
        for a, b in ((pusher, xy), (xy, goal_xy)):
            k = max(int(np.ceil(np.linalg.norm(b - a) / 0.02)), 1)
            placed += [(a + (b - a) * i / k, self.corridor) for i in range(k + 1)]
        self.place_distractors(placed)
        mujoco.mj_forward(self.model, self.data)
        self._rest_z = {self.obj: float(self.object_pos(self.obj)[2])}
        origin = goal_xy + self.object_pos(self.obj)[:2] - center(self, self.obj)
        self.set_goals(Goal(self.obj, "table", target=(*origin, self._rest_z[self.obj]),
                            tolerance=(0.03, 0.03, 0.004), check=against))


# ----- oracle (as push_to_target, plus a measured final gap-closing stroke) ----------------------------------

class PushOracle(TrainOracle):
    """Approach from behind along the push line, push with the side of the closed fingers, re-measure, correct.

    Positions are planned for the pusher *face point*: the middle of the closed fingers' side, ``FACE_X`` along
    the TCP x axis and ``face`` ahead of the TCP along the push. The push is split: a first stroke to
    ``first_stroke`` of the way, then the actual face-to-centre offset (the object turns flush against the
    fingers) is measured and the stroke finished toward the target; after a short back-off the residual error is
    measured and up to ``corrections`` short corrective pushes follow."""

    push_z = 0.013              # TCP height while pushing (finger hull bottom ~4.8 mm above the table)
    hover_z = 0.06
    face_x = 0.022              # face centre along the TCP x axis (fingers span x 12-32 mm)
    face = 0.0085               # nominal face offset ahead of the TCP along the push (measured 5-9.7 mm)
    standoff = 0.02             # face gap behind the object before the stroke
    contact_gap = 0.003
    push_speed = 0.015          # slow: at 2.5 cm/s the object stick-slipped on the table (1000+ rad/s^2)
    first_stroke = 0.6
    retreat = 0.015
    corrections = 3
    correct_above = 0.004       # residual error (m) worth a corrective push
    max_steer = 25.0            # deg: re-aim the second stroke only within this turn
    finish_z = 0.08
    min_self_clearance = 0.006   # at the hover, start and end poses
    min_path_clearance = 0.003   # along the joint-space transit (the rest pose itself has 5-12 mm)
    allow_lead = True            # may push with the camera mount leading (over low objects only)

    def push_rot(self, u, lead=False):
        """Vertical TCP whose -y axis is the push direction ``u`` (mount trailing), or +y (``lead``)."""
        angle = np.arctan2(-u[0], u[1]) + (np.pi if lead else 0.0)
        return top_down_mat(angle)

    def tcp_xy(self, face_xy, rot, u):
        """TCP xy putting the pusher face point at ``face_xy`` while pushing along ``u``."""
        return np.asarray(face_xy) - self.face_x * rot[:2, 0] - self.face * np.asarray(u)

    def face_now(self, u):
        """Actual face point (from the measured TCP pose) for a push along ``u``."""
        return self.tcp()[:2] + self.face_x * self.tcp_rot()[:2, 0] + self.face * np.asarray(u)

    def self_clearance(self, q):
        """Minimum distance (m) between the gripper/camera mount and the upper arm links at arm joints ``q`` (the
        wrist-camera mount folds into the shoulder/upper arm in some near-base poses that the IK reaches)."""
        env, m = self.env, self.model
        if not hasattr(self, "_self_pairs"):
            geoms = lambda body: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id
                                  and (m.geom_contype[g] or m.geom_conaffinity[g])]
            hand = [g for b in ("camera_mount", "gripper", "moving_jaw_so101_v1") for g in geoms(b)]
            arm = [g for b in ("shoulder", "upper_arm", "lower_arm") for g in geoms(b)]
            self._self_pairs = [(g, h) for g in hand for h in arm]
            self._probe = mujoco.MjData(m)
        d = self._probe
        d.qpos[:] = env.data.qpos
        d.qpos[env._arm_qpos_addrs] = q
        d.qpos[env._qpos_addrs[5]] = self.grip
        mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, g, h, 0.2, None) for g, h in self._self_pairs)

    def path_clearance(self, q0, q1, samples=10):
        return min(self.self_clearance(q0 + (q1 - q0) * k / samples) for k in range(1, samples + 1))

    def choose(self, start_face, end_face, u):
        """Mount-trailing orientation unless the other one reaches clearly better or keeps the camera mount clear
        of the arm where the first does not. Returns (rot, lead, ok)."""
        options = []
        for lead in ((False, True) if self.allow_lead else (False,)):
            rot = self.push_rot(u, lead)
            hover = np.r_[self.tcp_xy(start_face, rot, u), self.hover_z]
            q_hover, err, _ = self.solve(hover, rot)
            errs, clear, staged = [err], self.path_clearance(self.q, q_hover), False
            if clear < self.min_path_clearance:
                # Unfold first with the wrist roll held, then turn the roll in place.
                mid = q_hover.copy()
                mid[4] = self.q[4]
                staged_clear = min(self.path_clearance(self.q, mid), self.path_clearance(mid, q_hover))
                if staged_clear > clear:
                    clear, staged = staged_clear, True
            path_bad = clear < self.min_path_clearance
            clear = self.self_clearance(q_hover)
            for face in (start_face, end_face):
                q, e, _ = self.solve(np.r_[self.tcp_xy(face, rot, u), self.push_z], rot, seeds=[q_hover[4]])
                errs.append(e)
                clear = min(clear, self.self_clearance(q))
            bad = path_bad or max(errs) > 0.003 or clear < self.min_self_clearance
            options.append((bad, max(errs) if max(errs) > 0.003 else 0.0, lead, rot, q_hover, staged))
        bad, _, lead, rot, q_hover, staged = min(options, key=lambda o: o[:3])
        return rot, lead, not bad, q_hover, staged

    def stroke(self, goal_xy, first):
        env, name = self.env, self.env.obj
        c = center(env, name)
        delta = goal_xy - c
        dist = float(np.linalg.norm(delta))
        u = delta / dist
        lo, hi = extent_along(env, name, u)
        half = (hi - lo) / 2
        start = c - u * (half + self.standoff)
        end = goal_xy - u * half
        rot, lead, ok, q_hover, staged = self.choose(start, end, u)
        if not ok:
            self.log.append(("no clear push pose", round(float(dist) * 1000, 1)))
            if not first:
                return False
        tcp = lambda face_xy, z=self.push_z: np.r_[self.tcp_xy(face_xy, rot, u), z]
        if not first:
            yield from self.move(np.r_[self._cmd_pos[:2], self.hover_z], self._cmd_rot, speed=0.08, label="up")
        pre = tcp(start, self.hover_z)
        if staged:
            mid = q_hover.copy()
            mid[4] = self.q[4]
            yield from self.joint_move(mid, max(0.6, float(np.max(np.abs(mid - self.q))) / 0.8))
        yield from self.transit(pre, rot, speed=0.8, q=q_hover, label="above start")
        yield from self.move(tcp(start), rot, speed=0.05, tol=0.002, label="down")
        near = c - u * (half + self.contact_gap)
        yield from self.move(tcp(near), rot, speed=0.03, tol=0.002, label="approach")
        if first and dist > 0.03:
            mid = near + u * (self.first_stroke * (dist + self.contact_gap))
            yield from self.move(tcp(mid), rot, speed=self.push_speed, tol=0.002, settle=0.3, label="stroke1")
            # Re-aim: the object has turned flush against the fingers; measure the actual face-to-centre offset
            # along the push and steer the rest of the stroke at the target.
            c = center(env, name)
            delta = goal_xy - c
            u2 = delta / max(np.linalg.norm(delta), 1e-6)
            back = float((c - self.face_now(u)) @ u)
            lo, hi = extent_along(env, name, u)
            self.face += back - (hi - lo) / 2                  # the fingers' actual reach for this object
            turn = np.degrees(np.arccos(np.clip(u2 @ u, -1, 1)))
            if turn <= self.max_steer and back > 0:
                rot = self.push_rot(u2, lead)
                u = u2
            lo, hi = extent_along(env, name, u)
            end = goal_xy - u * (hi - lo) / 2
            yield from self.move(tcp(end), rot, speed=self.push_speed, tol=0.002, settle=0.3, label="stroke2")
        else:
            yield from self.move(tcp(end), rot, speed=self.push_speed, tol=0.002, settle=0.3, label="stroke")
        yield from self.finish_stroke(u)
        c = center(env, name)
        lo, hi = extent_along(env, name, u)
        self.log.append(("reach", round((float((c - self.face_now(u)) @ u) - (hi - lo) / 2) * 1000, 1)))
        # Back off along the push line before lifting, so the fingers leave the face without dragging it.
        yield from self.move(np.r_[self._cmd_pos[:2] - u * self.retreat, self.push_z], self._cmd_rot, speed=0.03,
                             tol=0.003, settle=0.2, label="back off")
        yield from self.move(np.r_[self._cmd_pos[:2], self.hover_z], self._cmd_rot, speed=0.06, label="lift")
        return True

    def finish_stroke(self, u):
        """Hook run at the end of each stroke while the fingers still touch the object."""
        return
        yield

    def plan(self):
        env = self.env
        yield from self.gripper(-0.17, 0.4, 0.0)
        yield from self.stroke(self.target_xy(), first=True)
        for k in range(self.corrections):
            goal_xy = self.target_xy()
            err = float(np.linalg.norm(goal_xy - center(env, env.obj)))
            self.log.append(("residual", k, round(err * 1000, 1)))
            if err <= self.correct_above:
                break
            if not (yield from self.stroke(goal_xy, first=False)):
                break
        yield from self.move(np.r_[self._cmd_pos[:2], self.finish_z], self._cmd_rot, speed=0.08, label="up")
        yield from self.rest()
        yield from self.wait(1.2)


class TogetherOracle(PushOracle):
    """Push toward the reference face, stop ``stop_gap`` short, then close the measured gap + ``overlap``."""

    stop_gap = 0.006
    overlap = 0.0008
    corrections = 2

    def target_xy(self):
        """Centre aim: ``stop_gap`` short of the reference face along its (current) normal."""
        env = self.env
        n = env.face_normal()
        lo, hi = extent_along(env, env.ref, n)
        olo, ohi = extent_along(env, env.obj, n)
        return center(env, env.ref) + n * ((hi - lo) / 2 + (ohi - olo) / 2 + self.stop_gap)

    def finish_stroke(self, u):
        env = self.env
        d = gap(env, env.obj, env.ref)
        self.log.append(("gap", round(d * 1000, 1)))
        if d <= 0.0:
            return
        n = -env.face_normal()
        target = self._cmd_pos[:2] + n * (d + self.overlap)
        yield from self.move(np.r_[target, self.push_z], self._cmd_rot, speed=0.006, tol=0.0015, settle=0.4,
                             label="close gap")
        self.log.append(("gap after", round(gap(env, env.obj, env.ref) * 1000, 2)))

    def plan(self):
        env = self.env
        yield from self.gripper(-0.17, 0.4, 0.0)
        yield from self.stroke(self.target_xy(), first=True)
        for k in range(self.corrections):
            if against(env, env.obj) or gap(env, env.obj, env.ref) <= TOUCH:
                break
            # Re-push from behind along the face normal (the object drifted or turned away).
            self.log.append(("repush", k, round(gap(env, env.obj, env.ref) * 1000, 1)))
            if not (yield from self.stroke(self.target_xy(), first=False)):
                break
        yield from self.move(np.r_[self._cmd_pos[:2], self.finish_z], self._cmd_rot, speed=0.08, label="up")
        yield from self.rest()
        yield from self.wait(1.2)


# ----- task 1: red block against the blue block --------------------------------------------------------------

HALF = 0.014
RED, BLUE = (0.8, 0.1, 0.08, 1), (0.1, 0.25, 0.8, 1)


class BlockToBlockEnv(TogetherEnv):
    instruction = "Push the red block against the blue block."
    task_objects = ("red_block",)
    obj, ref = "red_block", "blue_block"

    def scene_objects(self):
        return [Block("red_block", half=(HALF,) * 3, rgba=RED), Block("blue_block", half=(HALF,) * 3, rgba=BLUE)]


# ----- task 2: pudding box up against a standing book --------------------------------------------------------

class BoxToBookEnv(TogetherEnv):
    instruction = "Push the cardboard box up against the standing book."
    task_objects = ("box",)
    obj, ref = "box", "book"
    obj_yaw_along = 0          # long axis along the push: the gripper body stays > 2.5 cm from the book

    def scene_objects(self):
        # A small closed cardboard box (5.6 x 3.4 x 2.2 cm, 25 g). A primitive box: the 0.7x LIBERO pudding
        # mesh pitched 3.6 deg under the 30 Hz pusher pulses (1212 rad/s^2, 1 of 8 calibration seeds).
        return [Block("box", half=(0.028, 0.017, 0.011), rgba=(0.72, 0.55, 0.36, 1), mass=0.025)]

    def scene_fixtures(self):
        return [Fixture("book", "black_book")]

    def layout(self):
        # Only the book's broad faces (its thin local axis): find it from the collision extents once.
        if not hasattr(self, "_broad"):
            self.set_fixture_pose("book", (0.2, 0.0), yaw=0.0)
            ex = [np.subtract(*extent_along(self, "book", a)[::-1]) for a in body_axes(self, "book")]
            k = int(np.argmin(ex))
            self._broad = (k, k + 2)
        self.faces = self._broad
        super().layout()


class BookOracle(TogetherOracle):
    allow_lead = False         # the camera mount must trail: it is lower than the book's top


BlockToBlockEnv.probe_oracle = TogetherOracle
BoxToBookEnv.probe_oracle = BookOracle

TASKS = [
    define_task(name="push_block_against_block", instruction=BlockToBlockEnv.instruction, family=FAMILY,
                env=BlockToBlockEnv, oracle=TogetherOracle, objects=("red_block",), object_kinds=("block",),
                relation="pushed against", goal="block",
                steps=("Push the red block along the table until it touches the blue block, without lifting it.",)),
    define_task(name="push_box_against_book", instruction=BoxToBookEnv.instruction, family=FAMILY,
                env=BoxToBookEnv, oracle=BookOracle, objects=("box",), object_kinds=("cardboard box",),
                relation="pushed against", goal="standing book",
                steps=("Push the cardboard box along the table until it rests against the standing book, "
                       "without lifting it.",)),
]

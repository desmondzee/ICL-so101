"""push_to_target: push an object along the table (closed jaws, never grasped) into a marked target region.

Non-prehensile: the object is never lifted or grasped. The closed fingers push it sideways, a few mm above the
table, approaching from behind along the push line. Every scene has the target region and a confounder region
of a different shape and colour (marker colours are drawn per episode, the instructions name the region by
shape, or by viewer side), plus 2-4 ordinary distractors kept clear of the push corridor.

Strict success, on top of the kit's released/settled/upright(10 deg)/supported-by-the-table goal:

* the object's whole collision footprint lies inside the region (``inside_region``), with ``EDGE_MARGIN``;
* it was never lifted (its height never rose more than ``LIFT_TOL`` above its resting height) and never
  grasped (both tracked every control frame);
* the jaws no longer touch it (the robot has pulled away).

Local kit extensions (no shared file is changed):

* ``FlatMarker``: a printed mat / taped outline flush with the table. It is visual-only: an object pushed
  across a real 1-6 mm mat or tape edge jams against its side face (MuJoCo box-box contact has no bevel), so the
  marker's collision proxy uses contact bit 2, which no other geom in the scene uses (checked at layout), and
  the body floats in place by gravity compensation. The kit has no static, movable, collision-free marker
  primitive (``Fixture`` needs a LIBERO asset); see the report's kit proposal.

Pusher geometry (measured 2026-10-06, closed jaws, vertical TCP): the fingers occupy TCP-frame x 12-32 mm,
|y| <= 5-8 mm at the TCP height (wider higher up: 9.7 mm at +20 mm); their hull reaches 8.2 mm below the TCP. The
side of the closed fingers (TCP -y) is the pushing face: a 2 cm wide flat-ish face, whereas the jaw ends are
only 1 cm wide. The wrist-camera mount sits on the TCP +y side 4.4-7.2 cm above the TCP, so pushing along TCP -y
keeps it trailing, away from the object.
"""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.train.tasks.base import LOW_DISTRACTORS, TRAIN_KEEPOUT, Goal, TrainEnv, TrainOracle, define_task, top_down_mat
from sim.val.scene import Block, Obj

FAMILY = "push_to_target"
EDGE_MARGIN = 0.0          # footprint may reach (not cross) the region's inner edge
LIFT_TOL = 0.004           # max rise of the object above its resting height at any frame (m)
MARK_Z = 0.0001            # marker bottom above the table top (m); the marker is 0.6 mm thick

# Distinct, saturated marker colours (target and confounder never share one).
PALETTE = ((0.85, 0.2, 0.15, 1), (0.15, 0.35, 0.85, 1), (0.2, 0.6, 0.25, 1), (0.9, 0.72, 0.1, 1),
           (0.55, 0.25, 0.7, 1), (0.95, 0.5, 0.1, 1), (0.1, 0.6, 0.65, 1), (0.92, 0.92, 0.9, 1))


# ----- marker primitive --------------------------------------------------------------------------------------

@dataclass
class FlatMarker:
    """A flat region marker flush with the table: a solid mat or (``tape`` = strip width) a taped outline.

    Visual-only for contacts (see the module docstring): the collision proxy (the full footprint, so the kit's
    footprint/extent bookkeeping is correct) uses contact bit 2 only, and the body is gravity-compensated."""

    name: str
    half: tuple = (0.035, 0.035)      # outer half extents (m)
    tape: float | None = None         # strip width for an outline; None: solid mat
    thickness: float = 0.0006
    rgba: tuple = (0.2, 0.3, 0.8, 1.0)

    def build_mjcf(self, mj, asset_el, worldbody):
        body = ET.SubElement(worldbody, "body", name=self.name, pos="0 0 -1", gravcomp="1")
        ET.SubElement(body, "freejoint", name=f"{self.name}_joint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.01", diaginertia="1e-6 1e-6 1e-6")
        hx, hy, h = self.half[0], self.half[1], self.thickness / 2
        ET.SubElement(body, "geom", name=f"{self.name}_proxy", type="box", size=f"{hx:.6g} {hy:.6g} {h:.6g}",
                      contype="2", conaffinity="2", group="3", rgba="0 0 0 0", mass="0")
        color = " ".join(map(str, self.rgba))
        if self.tape is None:
            parts = [((0, 0), (hx, hy))]
        else:
            w = self.tape / 2
            parts = [((hx - w, 0), (w, hy)), ((-(hx - w), 0), (w, hy)),
                     ((0, hy - w), (hx - self.tape, w)), ((0, -(hy - w)), (hx - self.tape, w))]
        for k, ((x, y), (sx, sy)) in enumerate(parts):
            ET.SubElement(body, "geom", name=f"{self.name}_vis_{k}", type="box", pos=f"{x:.6g} {y:.6g} 0",
                          size=f"{sx:.6g} {sy:.6g} {h:.6g}", rgba=color, group="1", contype="0", conaffinity="0",
                          mass="0", material="val_fabric")

    def inner(self):
        """Half extents of the region an object must lie in (inside the tape for an outline)."""
        if self.tape is None:
            return np.asarray(self.half, float)
        return np.asarray(self.half, float) - self.tape


# ----- geometry helpers --------------------------------------------------------------------------------------

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


def in_frame(env, marker, pts):
    """xy of world points in the marker's own frame."""
    yaw = env.yaw(marker)
    c, s = np.cos(yaw), np.sin(yaw)
    return (pts[:, :2] - env.object_pos(marker)[:2]) @ np.array([[c, -s], [s, c]])


def inside_region(marker_spec):
    """Footprint inside the marker's region, never lifted or grasped, and no longer touched by the jaws."""
    inner = marker_spec.inner() - EDGE_MARGIN

    def check(env, obj):
        local = in_frame(env, marker_spec.name, collision_points(env, obj))
        if np.any(np.abs(local) > inner):
            return False
        return obj not in env.lifted and not env.jaws_touching(obj)
    return check


# ----- environment -------------------------------------------------------------------------------------------

class PushEnv(TrainEnv):
    """Common push layout. Subclasses set ``obj``, ``markers`` (target first) and ``scene_objects``.

    ``layout``: target region centre, then the object start at ``push_distance`` from it in a random direction
    (reachable start, reachable pusher start behind it, object entirely outside the region), then the confounder
    region, then distractors kept clear of the push corridor and both regions."""

    obj = ""
    target_marker = "target"
    confounder = "confounder"
    target_region = dict(r=(0.165, 0.25), angle=(-55.0, 55.0))
    start_region = dict(r=(0.145, 0.26), angle=(-62.0, 62.0))
    pusher_region = dict(r=(0.125, 0.27), angle=(-70.0, 70.0))
    push_distance = (0.07, 0.11)
    standoff = 0.03           # pusher face start behind the object's trailing face (m)
    start_gap = 0.01          # the object starts at least this far outside the region (m)
    corridor = 0.055          # half width of the distractor keep-out along the push corridor (m)
    distractor_pool = tuple(n for n in LOW_DISTRACTORS if n != "plate")    # a plate reads as a round region
    slide_friction = 0.4
    # Pushed by the sloped finger side, a light object chattered between finger and table at the default 0.02 s
    # contact time constant (vertical velocity +-12 cm/s within 3 substeps, 1008 rad/s^2); 0.025 damps it (max 780).
    contact_timeconst = 0.025
    # Slow stroke ends stick-slipped: the object stopped, the creeping pusher compressed the contact and it
    # jumped to 10 cm/s (1531 rad/s^2, 0.7x pudding box). An overdamped contact (ratio 2) bleeds that energy.
    contact_damping = 2.0

    def marker_specs(self):
        return {s.name: s for s in self.scene.objects if isinstance(s, FlatMarker)}

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

    def color_markers(self):
        i, j = self.np_random.choice(len(PALETTE), size=2, replace=False)
        for name, rgba in ((self.target_marker, PALETTE[i]), (self.confounder, PALETTE[j])):
            for g in range(self.model.ngeom):
                if self.model.body(int(self.model.geom_bodyid[g])).name == name and self.model.geom_group[g] == 1:
                    self.model.geom_rgba[g] = rgba
        return PALETTE[i], PALETTE[j]

    def _check_marker_bits(self):
        """The markers' contact bit (2) must be unused by every other geom, so they collide with nothing."""
        own = {g for n in self.marker_specs() for g in self._geoms[n]}
        m = self.model
        for g in range(m.ngeom):
            if g not in own and (m.geom_contype[g] & 2 or m.geom_conaffinity[g] & 2):
                raise RuntimeError(f"contact bit 2 used by geom {g}; markers would collide")

    def put_marker(self, name, xy, yaw):
        self.set_object_pose(name, xy, yaw=yaw, z=MARK_Z - 0.001)

    def polar_ok(self, xy, region):
        r, a = np.hypot(*xy), np.degrees(np.arctan2(xy[1], xy[0]))
        return region["r"][0] <= r <= region["r"][1] and region["angle"][0] <= a <= region["angle"][1]

    def clear_of_keepout(self, xy, radius):
        return all(np.linalg.norm(xy - c) >= q + radius for c, q in TRAIN_KEEPOUT)

    def marker_radius(self, name):
        return float(np.hypot(*self.marker_specs()[name].half))

    def object_half_along(self, u):
        lo, hi = extent_along(self, self.obj, u)
        return (hi - lo) / 2

    def sample_target(self):
        """Target region centre and yaw (subclasses may restrict it)."""
        rng = self.np_random
        for _ in range(500):
            rr, aa = rng.uniform(*self.target_region["r"]), np.radians(rng.uniform(*self.target_region["angle"]))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if self.clear_of_keepout(xy, self.footprint(self.obj) + 0.01):
                return xy, float(rng.uniform(-np.pi, np.pi))
        raise RuntimeError("no target region position")

    def sample_start(self, target_xy, target_yaw):
        """Object start: ``push_distance`` from the target centre, fully outside the region, reachable, with a
        reachable pusher start behind it."""
        rng = self.np_random
        spec = self.marker_specs()[self.target_marker]
        for _ in range(800):
            dist = rng.uniform(*self.push_distance)
            phi = rng.uniform(-np.pi, np.pi)
            xy = target_xy + dist * np.array([np.cos(phi), np.sin(phi)])
            if not self.polar_ok(xy, self.start_region) or not self.clear_of_keepout(xy, self.footprint(self.obj)):
                continue
            yaw = float(rng.uniform(-np.pi, np.pi))
            self.set_center(self.obj, xy, yaw)
            local = in_frame(self, self.target_marker, collision_points(self, self.obj))
            out = np.asarray(spec.half) + self.start_gap
            if not any(local[:, k].min() > out[k] or local[:, k].max() < -out[k] for k in (0, 1)):
                continue                                                  # overlaps or nearly touches the region
            u = (target_xy - xy) / np.linalg.norm(target_xy - xy)
            pusher = xy - u * (self.object_half_along(u) + self.standoff)
            side = np.array([u[1], -u[0]]) * 0.022                        # the TCP sits beside the face point
            if not all(self.polar_ok(p, self.pusher_region) for p in (pusher + side, pusher - side)):
                continue
            return xy, yaw, u, pusher
        raise RuntimeError("no object start position")

    def set_center(self, name, xy, yaw):
        """Put free body ``name`` on the table with its bounding-box centre at ``xy``."""
        self.set_object_pose(name, xy, yaw=yaw)
        self.set_object_pose(name, np.asarray(xy) + self.object_pos(name)[:2] - center(self, name), yaw=yaw)

    def corridor_keepout(self, start, end, pusher):
        """Circles covering the push corridor (pusher start -> object start -> target) for distractors."""
        pts = [pusher, start, end]
        circles = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / 0.02)), 1)
            circles += [(a + (b - a) * k / n, self.corridor) for k in range(n + 1)]
        return circles

    def place_confounder(self, placed):
        rng = self.np_random
        radius = self.marker_radius(self.confounder)
        for _ in range(400):
            rr, aa = rng.uniform(0.13, 0.28), np.radians(rng.uniform(-70, 70))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if all(np.linalg.norm(xy - c) >= q + radius + 0.01 for c, q in placed):
                yaw = float(rng.uniform(-np.pi, np.pi))
                self.put_marker(self.confounder, xy, yaw)
                return xy, radius
        raise RuntimeError("no confounder position")

    def set_slide_friction(self):
        """Give the pushed object's geoms contact priority and a sliding friction of ``slide_friction`` so that,
        not the table's 0.95, governs its contacts (plastic/carton on wood is 0.3-0.5). The closed fingers' taper
        puts the push contact near the top of the object (measured: a 2.8 cm cube pushed at 0.95 rode on its edge
        at 12-17 deg); tipping needs friction x contact height > half width, so 0.4 keeps it flat."""
        for g in self._geoms[self.obj]:
            self.model.geom_priority[g] = 1
            self.model.geom_friction[g] = (self.slide_friction, 0.005, 0.0001)
            self.model.geom_solref[g] = (self.contact_timeconst, self.contact_damping)

    def layout(self):
        self.set_slide_friction()
        self.color_markers()
        target_xy, target_yaw = self.sample_target()
        self.put_marker(self.target_marker, target_xy, target_yaw)
        start, yaw, u, pusher = self.sample_start(target_xy, target_yaw)
        corridor = self.corridor_keepout(start, target_xy, pusher)
        placed = list(corridor) + [(target_xy, self.marker_radius(self.target_marker))]
        placed.append(self.place_confounder(placed))
        self.place_distractors(placed)
        self._check_marker_bits()
        mujoco.mj_forward(self.model, self.data)
        self._rest_z = {self.obj: float(self.object_pos(self.obj)[2])}
        self.finish_goal(target_xy)

    def finish_goal(self, target_xy):
        spec = self.marker_specs()[self.target_marker]
        tol = float(np.max(spec.inner()))
        # The goal box bounds the body origin (offset from the bounding-box centre for meshes).
        offset = self.object_pos(self.obj)[:2] - center(self, self.obj)
        tol += float(np.linalg.norm(offset))
        self.set_goals(Goal(self.obj, "table", target=(*target_xy, self._rest_z[self.obj]), tolerance=(tol, tol, 0.004),
                            check=inside_region(spec)))


# ----- oracle ------------------------------------------------------------------------------------------------

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

    def target_xy(self):
        """Aim point for the object's bounding-box centre (the goal is declared for its body origin)."""
        env = self.env
        return env.goal_target(env.goals[0])[:2] + center(env, env.obj) - env.object_pos(env.obj)[:2]

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


# ----- task 1: cube into the taped square --------------------------------------------------------------------

CUBE_HALF = 0.014


class CubeIntoTapeEnv(PushEnv):
    instruction = "Push the cube into the taped square."
    task_objects = ("cube",)
    obj = "cube"

    def scene_objects(self):
        return [Block("cube", half=(CUBE_HALF,) * 3, rgba=(0.85, 0.85, 0.82, 1)),
                FlatMarker("target", half=(0.036, 0.036), tape=0.006),
                FlatMarker("confounder", half=(0.032, 0.032))]

    def color_markers(self):
        target, confounder = super().color_markers()
        rest = [c for c in PALETTE if c not in (target, confounder)]
        self.model.geom_rgba[self.model.geom("cube_geom").id] = rest[int(self.np_random.integers(len(rest)))]
        return target, confounder


# ----- task 2: pudding box onto the mat ----------------------------------------------------------------------

class PuddingOntoMatEnv(PushEnv):
    instruction = "Slide the pudding box onto the mat."
    task_objects = ("pudding",)
    obj = "pudding"
    distractor_pool = tuple(n for n in PushEnv.distractor_pool if n != "chocolate_pudding")

    def scene_objects(self):
        return [Obj("pudding", "chocolate_pudding", scale=0.7, friction=0.7),
                FlatMarker("target", half=(0.045, 0.045)),
                FlatMarker("confounder", half=(0.04, 0.04), tape=0.006)]


TASKS = [
    define_task(name="push_cube_into_tape_square", instruction=CubeIntoTapeEnv.instruction, family=FAMILY,
                env=CubeIntoTapeEnv, oracle=PushOracle, objects=("cube",), object_kinds=("cube",),
                relation="pushed into", goal="taped square",
                steps=("Push the cube along the table into the taped square without lifting it.",)),
    define_task(name="push_pudding_onto_mat", instruction=PuddingOntoMatEnv.instruction, family=FAMILY,
                env=PuddingOntoMatEnv, oracle=PushOracle, objects=("pudding",), object_kinds=("flat food box",),
                relation="pushed onto", goal="mat",
                steps=("Slide the pudding box along the table onto the mat without lifting it.",)),
]

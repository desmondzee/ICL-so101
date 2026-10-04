"""Task: put the frying pan on the stove (pick the pan by its handle and set it flat on the burner).

Scene: LIBERO chefmate_8_frypan at 0.5x (bowl ~10 cm across, 18 cm long with the handle, 0.10 kg), the flat_stove
fixture at 0.5x (9.5 cm burner plate, knob on the far side), one look-alike container (black bowl or plate, chosen per
seed) and 2-3 small pantry distractors. Success: pan centre inside the stove's cook_region, pan flat (< 8 deg) and
resting on the burner plate, not grasped, nearly still.

Oracle: top-down pinch across the handle 8.8 cm from the pan centre, with the handle sitting just above the fingertip
spheres (GRASP_DZ) so the flat jaw pads hold it; the IK runs with a stiff orientation weight so the gripper and the pan
stay level. The carry is planned in pan-centre polar coordinates (radius, azimuth, handle angle) so the hanging pan
never swings into the robot, then the pan position is corrected in closed loop over the burner before set-down.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim.val.env import PARK_X, ValEnv
from sim.val.oracle import CLOSED, IK_ITERS, OPEN, RELEASE, Oracle, min_jerk, top_down_mat
from sim.val.scene import Fixture, Obj, SceneSpec

PAN_SCALE = 0.5
PAN_MASS = 0.10
PAN_FRICTION = 1.5
HANDLE_D = 0.088          # grasp point along the handle, from the pan centre (pan local +x)
HANDLE_WIDTH = 0.0126     # handle width across the closing direction
GRASP_DZ = -0.0035        # TCP height at the grasp relative to the handle centre height at HANDLE_D
BURNER_OFFSET = 0.075     # flat_stove: burner centre is +x of the fixture origin (the knob) at 0.5x
STOVE_Z = 0.01            # the stove base box is centred on its origin; lift it so it rests on the table
COOK_HALF = 0.0375        # cook_region half size at 0.5x
DECOYS = {"bowl": Obj("bowl", "akita_black_bowl", 0.6), "plate": Obj("plate", "plate", 0.65, mass=0.25)}
DISTRACTORS = ("alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "chocolate_pudding", "popcorn", "ramekin")
STOVE_R = 0.067           # stove half-diagonal at 0.5x
PAN_R = 0.052             # pan bowl radius at 0.5x
CARRY_SPEED = 0.06        # m/s of the pan centre during the carry
PAN_CARRY_Z = 0.075       # TCP height while carrying; the pan bottom hangs ~2.7 cm lower
TCP_R = (0.15, 0.255)     # radii where a level top-down TCP is reachable at both set-down and carry height


class PanOnStoveEnv(ValEnv):
    instruction = "Put the frying pan on the stove."

    def make_scene(self):
        objects = [Obj("pan", "frypan", PAN_SCALE, mass=PAN_MASS, friction=PAN_FRICTION), *DECOYS.values()]
        return SceneSpec(arena="kitchen", objects=objects, fixtures=[Fixture("stove", "flat_stove")],
                         distractors=[Obj(n, n) for n in DISTRACTORS])

    # ----- geometry helpers -------------------------------------------------------------------------------------
    def yaw(self, name):
        q = self.object_pose(name)[3:]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def burner_xy(self):
        return self.data.site_xpos[self.model.site("stove_cook_region").id][:2].copy()

    def burner_top(self):
        g = self.model.geom("stove_collision_burner").id
        return float(self.data.geom_xpos[g][2] + self.model.geom_size[g][2])

    def handle_point(self):
        p, a = self.object_pos("pan"), self.yaw("pan")
        return p[:2] + HANDLE_D * np.array([np.cos(a), np.sin(a)])

    # ----- layout -----------------------------------------------------------------------------------------------
    def layout(self):
        rng = self.np_random
        placed = []
        # Stove: burner at r 0.20-0.26, clear of the robot base and of the folded gripper at rest (so the set-down pan
        # is too); knob on the far side of the burner (away from the robot) +- 25 deg.
        burner = self.sample_xy(STOVE_R + 0.005, [], r=(0.20, 0.26), angle=(-55, 55), clearance=0.0)
        ang = np.arctan2(burner[1], burner[0])
        syaw = ang + np.pi + np.radians(rng.uniform(-25, 25))
        knob = burner - BURNER_OFFSET * np.array([np.cos(syaw), np.sin(syaw)])
        self.set_fixture_pose("stove", knob, yaw=syaw, z=STOVE_Z)   # fixture origin = knob; burner at +x
        placed += [(burner, 0.068), (knob, 0.03)]
        # Pan: centre at r 0.18-0.28; handle anywhere but pointing away from the robot; grasp point reachable.
        for _ in range(300):
            xy = self.sample_xy(PAN_R, placed, r=(0.18, 0.28), angle=(-60, 60), clearance=0.015)
            hyaw = np.arctan2(xy[1], xy[0]) + np.pi + np.radians(rng.uniform(-100, 100))
            u = np.array([np.cos(hyaw), np.sin(hyaw)])
            h, tip = xy + HANDLE_D * u, xy + 0.135 * u
            if not TCP_R[0] + 0.015 <= np.linalg.norm(h) <= TCP_R[1] - 0.01 or abs(np.arctan2(h[1], h[0])) > np.radians(70):
                continue
            if any(np.linalg.norm(p - q) < rad + 0.035 for p in (h, tip) for q, rad in [*placed, *self.keepout]):
                continue
            break
        else:
            raise RuntimeError("pan")
        self.set_object_pose("pan", xy, yaw=hyaw)
        placed += [(xy, PAN_R), (h, 0.03), (tip, 0.02)]
        # One look-alike container (black bowl or plate); the other is parked.
        self.decoy = "bowl" if rng.random() < 0.5 else "plate"
        for i, n in enumerate(DECOYS):
            if n == self.decoy:
                dxy = self.sample_xy(self.footprint(n), placed, r=(0.15, 0.33), angle=(-65, 65), clearance=0.025)
                placed.append((dxy, self.footprint(n)))
                self.set_object_pose(n, dxy, yaw=rng.uniform(-np.pi, np.pi))
            else:
                self.set_object_pose(n, (PARK_X + 10 + i, 0.0), z=self._floor_z)
        self.place_distractors(placed)

    def object_poses(self):
        poses = super().object_poses()
        return {k: v for k, v in poses.items() if k not in DECOYS or k == self.decoy}

    # ----- success ----------------------------------------------------------------------------------------------
    def pan_on_stove(self):
        p = self.object_pose("pan")
        b = self.burner_xy()
        q = p[3:]
        up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)          # pan local z . world z
        local = p[:2] - b
        c, s = np.cos(-self.stove_yaw()), np.sin(-self.stove_yaw())
        lx, ly = c * local[0] - s * local[1], s * local[0] + c * local[1]
        inside = abs(lx) < COOK_HALF and abs(ly) < COOK_HALF
        flat = up_z > np.cos(np.radians(8))
        resting = abs(p[2] - self._extent["pan"]["bottom"] - self.burner_top()) < 0.006 and self.touching("pan", "stove")
        v = self._dadr["pan"]
        still = np.linalg.norm(self.data.qvel[v : v + 3]) < 0.05 and np.linalg.norm(self.data.qvel[v + 3 : v + 6]) < 0.5
        return bool(inside and flat and resting and still and not self.is_grasping("pan"))

    def stove_yaw(self):
        return self.yaw("stove")

    def knob_dir(self):
        """Yaw of the direction burner -> knob."""
        return self.stove_yaw() + np.pi

    def success(self):
        return self.pan_on_stove()

    def task_info(self):
        return {"pan_on_stove": self.pan_on_stove()}


class PanOnStoveOracle(Oracle):
    """Top-down pinch across the handle, level carry planned in pan-centre coordinates, closed-loop set-down on the
    burner with the handle pointing anywhere from sideways to back toward the robot (never over the knob)."""

    rot_weight = 0.6

    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng or np.random.default_rng()

    def ik(self, pos, rot, seed=None):
        """`Oracle.ik` with a stiffer orientation weight, so a held pan is not tipped by wrist tilt."""
        env, m, d = self.env, self.model, self.ik_data
        d.qpos[:] = env.data.qpos
        q = (self.q[:5] if seed is None else seed).copy()
        lo, hi = env._target_low[:5], env._target_high[:5]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        dofs = env._arm_qvel_addrs
        w = self.rot_weight
        for _ in range(IK_ITERS):
            d.qpos[env._arm_qpos_addrs] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, env._tcp_site_id)
            cur = d.site_xmat[env._tcp_site_id].reshape(3, 3)
            err_p = pos - d.site_xpos[env._tcp_site_id]
            err_r = 0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))
            J = np.vstack([jacp[:, dofs], w * jacr[:, dofs]])
            e = np.concatenate([err_p, w * err_r])
            free = np.ones(5, dtype=bool)
            for _ in range(3):
                Jf = J * free
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-4 * np.eye(6), e)
                stuck = ((q <= lo + 1e-6) & (dq < 0)) | ((q >= hi - 1e-6) & (dq > 0))
                if not (stuck & free).any():
                    break
                free &= ~stuck
            q = np.clip(q + dq, lo, hi)
        return q

    def plan(self):
        ok = yield from self.pick_handle()
        if not ok:
            return
        yield from self.place_pan()
        yield from self.gripper(OPEN, 0.3, 0.0)
        yield from self.rest()

    def pick_handle(self, attempts=2, approach=0.05):
        env = self.env
        for attempt in range(attempts):
            h, a = env.handle_point(), env.yaw("pan")
            gz = 0.0214 + 0.13 * (HANDLE_D - 0.096) + GRASP_DZ
            rot, grasp, q = self.grasp_plan(np.array([h[0], h[1], gz]), HANDLE_WIDTH, [a + np.pi / 2, a - np.pi / 2])
            yield from self.gripper(OPEN, 0.4, 0.0)
            pre = grasp + [0, 0, approach]
            yield from self.transit(pre, rot, q=self.solve(pre, rot, seeds=[q[4]])[0], label="pregrasp")
            yield from self.move(grasp, rot, speed=0.06, tol=0.003, label="grasp")
            yield from self.gripper(CLOSED, 0.5, 0.3)
            ok = env.is_grasping("pan")
            self.log.append(("pan", "grasp", attempt, ok))
            if ok:
                yield from self.move(grasp + [0, 0, 0.03], rot, speed=0.05, label="lift")
                yield from self.move(np.array([grasp[0], grasp[1], PAN_CARRY_Z]), rot, speed=0.08, label="lift")
                if env.is_grasping("pan"):
                    return True
                self.log.append(("pan", "dropped", attempt))
            yield from self.gripper(OPEN, 0.4, 0.2)
            yield from self.move(self._cmd_pos + [0, 0, approach], rot, label="back off")
        return False

    def pan_state(self):
        """Pan centre (r, azimuth) about the robot base and the handle angle relative to the outward radial."""
        p = self.env.object_pos("pan")
        az = np.arctan2(p[1], p[0])
        return np.linalg.norm(p[:2]), az, np.mod(self.env.yaw("pan") - az, 2 * np.pi)

    def pan_to_tcp(self, r, az, phi, z):
        """TCP pose (pos, rot) that puts the held pan's centre at polar (r, az) with relative handle angle phi."""
        psi = az + phi
        rot = top_down_mat(psi - self.rel)
        pos = np.array([r * np.cos(az), r * np.sin(az), 0.0]) - rot @ self.off_local
        pos[2] = z
        return pos, rot

    def carry_path(self, phi1, n):
        r0, az0, phi0 = self.pan_state()
        rb, azb = np.linalg.norm(self.burner), np.arctan2(self.burner[1], self.burner[0])
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            yield self.pan_to_tcp(r0 + s * (rb - r0), az0 + s * (azb - az0), phi0 + s * (phi1 - phi0), PAN_CARRY_Z)

    def check_path(self, phi1):
        """Max IK error along the carry path for final handle angle phi1 (and the final joint solution)."""
        q, worst = self.q.copy(), 0.0
        for pos, rot in self.carry_path(phi1, 10):
            if not TCP_R[0] - 0.01 <= np.linalg.norm(pos[:2]) <= TCP_R[1] + 0.01:
                return np.inf, q
            for _ in range(2):
                q = self.ik(pos, rot, seed=q)
            worst = max(worst, float(np.linalg.norm(self._fk(q) - pos)))
        return worst, q

    def place_pan(self):
        env = self.env
        R, tcp = self.tcp_rot(), self.tcp()
        self.off_local = R.T @ (env.object_pos("pan") - tcp)                 # pan centre in the TCP frame
        self.rel = env.yaw("pan") - np.arctan2(R[1, 0], R[0, 0])             # handle yaw minus TCP x yaw
        self.burner, top = env.burner_xy(), env.burner_top()
        azb = np.arctan2(self.burner[1], self.burner[0])
        best = None
        for phi1 in np.radians(np.arange(70, 291, 10)):
            if abs(np.angle(np.exp(1j * (azb + phi1 - env.knob_dir())))) < np.radians(75):
                continue                                                     # handle would lie over the knob
            u = np.array([np.cos(azb + phi1), np.sin(azb + phi1)])
            if any(np.linalg.norm(self.burner + d * u - c) < rad + 0.03 for d in (HANDLE_D, 0.135) for c, rad in env.keepout):
                continue                                                     # handle in the way of the rest pose
            err, q = self.check_path(phi1)
            score = 200 * err + 0.3 * abs(q[4] - self.q[4]) + 0.2 * abs(phi1 - np.pi)
            if best is None or score < best[0]:
                best = (score, phi1)
        if best is None or best[0] == np.inf:
            self.log.append(("pan", "no carry path"))
            return
        phi1 = best[1]
        r0, az0, phi0 = self.pan_state()
        length = abs(np.linalg.norm(self.burner) - r0) + 0.25 * abs(azb - az0) + 0.06 * abs(phi1 - phi0)
        n = max(int(np.ceil(max(1.0, length / CARRY_SPEED) / self.dt)), 1)
        for pos, rot in self.carry_path(phi1, n):
            self.q = self.ik(pos, rot)
            yield self._target()
        self._cmd_pos, self._cmd_rot = pos, rot
        yield from self.wait(0.3)
        if not env.is_grasping("pan"):
            self.log.append(("pan", "dropped in carry"))
            return
        # Close the loop on the measured pan position before and during the descent.
        drop = top + env._extent["pan"]["bottom"]
        for hz in (0.0, 0.012):
            for _ in range(2):
                err = self.burner - env.object_pos("pan")[:2]
                if np.linalg.norm(err) > 0.04:
                    self.log.append(("pan", "lost", round(float(np.linalg.norm(err)), 3)))
                    return
                z = PAN_CARRY_Z if hz == 0.0 else self._cmd_pos[2]
                if hz:
                    z = self._cmd_pos[2] - (env.object_pos("pan")[2] - drop - hz)
                yield from self.move(np.array([*(self._cmd_pos[:2] + err), z]), rot, speed=0.04, label="align")
        yield from self.move(self._cmd_pos - [0, 0, 0.0125], rot, speed=0.02, label="set down")
        yield from self.gripper(RELEASE, 0.5, 0.4)
        yield from self.move(self._cmd_pos + [0, 0, 0.05], rot, speed=0.06, label="retreat")

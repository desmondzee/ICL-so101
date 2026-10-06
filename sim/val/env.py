"""`ValEnv`: SO101-Nexus MuJoCo env base for the validation tasks (front + wrist cameras, 30 fps control)."""

from __future__ import annotations

import tempfile
from dataclasses import replace

import mujoco
import numpy as np
from gymnasium import spaces
from so101_nexus import get_so101_mujoco_model_dir
from so101_nexus.config import EnvironmentConfig, RobotConfig
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.observations import JointPositions

from sim.val.scene import SceneSpec, build_scene_xml, camera_xml, table_top_z

FPS = 30
SUBSTEPS = 7
WIDTH, HEIGHT = 640, 480
FRONT_CAM = dict(pos=(0.48, -0.13, 0.38), lookat=(0.15, -0.02, 0.0), fovy=48.0)
WRIST_CAM_POS = (0.0025, 0.06157, -0.01877)
WRIST_CAM_QUAT = (0.98481, -0.17365, 0.0, 0.0)
WRIST_CAM_FOVY = 65.0
REST_DEG = (0.0, -99.0, 95.0, 64.0, 0.0, -9.0)
PARK_X = 20.0
ROBOT_VISUAL_GROUP = 2


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class ValEnv(SO101NexusMuJoCoBaseEnv):
    """Base env. Subclasses implement `make_scene`, `layout` and `success`, and set `instruction`.

    Coordinates are the robot base frame (base at the origin facing +x, table top at z = 0).
    One `step` is one 30 fps frame (7 x 1/210 s physics steps).
    """

    _N_SUBSTEPS = SUBSTEPS
    instruction = ""
    n_distractors = (2, 3)
    distractor_region = dict(r=(0.13, 0.34), angle=(-62, 62))
    keepout = ((np.array([0.0, 0.0]), 0.10), (np.array([0.15, 0.0]), 0.045))
    # Reset/rest joint pose (deg). Validation keeps REST_DEG; training subclasses
    # may override it (sim.train.tasks.base uses a table-clear variant).
    rest_deg = REST_DEG

    def __init__(self, render_images=True, control_mode="pd_joint_pos", robot_init_qpos_noise=0.02,
                 *, visual_config=None):
        config = EnvironmentConfig(
            spawn_center=(0.25, 0.0),
            spawn_max_radius=0.15,
            reset_settle_frames=20,
            terminate_on_success=False,
            robot=RobotConfig(rest_qpos_deg=self.rest_deg),
            observations=[JointPositions()],
        )
        self._init_common(config=config, render_mode=None, control_mode=control_mode, robot_init_qpos_noise=robot_init_qpos_noise)
        self.scene = self.make_scene()
        self.visual_config = visual_config
        if visual_config is not None:
            self.scene = replace(self.scene, arena=visual_config.arena)
        cams = [camera_xml("front", FRONT_CAM["pos"], FRONT_CAM["lookat"], FRONT_CAM["fovy"])]
        xml = build_scene_xml(self.scene, cams, visual_config=visual_config)
        with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=get_so101_mujoco_model_dir()) as f:
            f.write(xml)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.model.opt.timestep = 1.0 / (FPS * SUBSTEPS)
        self.model.actuator_forcerange[self.model.actuator("gripper").id] = [-1.47, 1.47]
        self._recolor_robot(self.scene.robot_rgba)
        self.data = mujoco.MjData(self.model)
        self._floor_z = -table_top_z(self.scene.arena)
        self._index_bodies()
        self._finish_model_setup()
        self.render_images = render_images
        self._renderer_rgb = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self._front_cam_id = self.model.camera("front").id
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "front": spaces.Box(0, 255, (HEIGHT, WIDTH, 3), np.uint8),
            "wrist": spaces.Box(0, 255, (HEIGHT, WIDTH, 3), np.uint8),
        })
        self.active_distractors: list[str] = []

    # ----- subclass hooks -------------------------------------------------------------------------------------
    def make_scene(self) -> SceneSpec:
        raise NotImplementedError

    def layout(self) -> None:
        """Place task objects and fixtures for this reset using `self.np_random`; call `place_distractors()`."""
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    def task_info(self) -> dict:
        return {}

    # ----- model bookkeeping ----------------------------------------------------------------------------------
    def _recolor_robot(self, rgba):
        yellow = np.array([1, 0.82, 0.12, 1])
        for i in range(self.model.nmat):
            if np.allclose(self.model.mat_rgba[i], yellow, atol=1e-3):
                self.model.mat_rgba[i] = rgba

    def _index_bodies(self):
        m = self.model
        self.free_names = [o.name for o in self.scene.free_bodies]
        self.distractor_names = [o.name for o in self.scene.distractors]
        self.fixture_names = [f.name for f in self.scene.fixtures]
        self._qadr = {n: m.jnt_qposadr[m.joint(f"{n}_joint").id] for n in self.free_names}
        self._dadr = {n: m.jnt_dofadr[m.joint(f"{n}_joint").id] for n in self.free_names}
        self._body = {n: m.body(n).id for n in self.free_names + self.fixture_names}
        self._geoms = {}
        for n in self.free_names + self.fixture_names:
            root = self._body[n]
            self._geoms[n] = [g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], root) and (m.geom_contype[g] or m.geom_conaffinity[g])]
        robot_root = m.body("base").id
        robot_vis = {g for g in range(m.ngeom) if self._in_subtree(m.geom_bodyid[g], robot_root) and m.geom_group[g] == ROBOT_VISUAL_GROUP}
        other = {g for g in range(m.ngeom) if m.geom_group[g] == ROBOT_VISUAL_GROUP} - robot_vis
        assert robot_vis and not other, "group 2 must hold exactly the robot visuals"
        self._extent = self._measure_extents()

    def _in_subtree(self, body, root):
        while body > 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return root == 0

    def _measure_extents(self):
        """Per free body: (xy footprint radius, z offset of the body origin above its lowest collision point)."""
        m, d = self.model, mujoco.MjData(self.model)
        out = {}
        for i, n in enumerate(self.free_names):
            a = self._qadr[n]
            d.qpos[a : a + 7] = [100.0 * (i + 1), 0, 1.0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        for n in self.free_names:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for g in self._geoms[n]:
                c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
                half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
                lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
            origin = d.xpos[self._body[n]]
            out[n] = dict(radius=float(np.max(np.linalg.norm(np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]) - origin[:2], axis=1))),
                          bottom=float(origin[2] - lo[2]), height=float(hi[2] - lo[2]))
        return out

    # ----- object helpers -------------------------------------------------------------------------------------
    def footprint(self, name) -> float:
        return self._extent[name]["radius"]

    def height(self, name) -> float:
        return self._extent[name]["height"]

    def object_pose(self, name) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz] of a free body or fixture in the robot base frame."""
        b = self._body[name]
        return np.concatenate([self.data.xpos[b], self.data.xquat[b]])

    def object_pos(self, name) -> np.ndarray:
        return self.data.xpos[self._body[name]].copy()

    def object_poses(self) -> dict:
        names = [n for n in self.free_names if n not in self.distractor_names] + self.active_distractors + self.fixture_names
        return {n: np.round(self.object_pose(n), 5).tolist() for n in names}

    def set_object_pose(self, name, xy, yaw=0.0, z=None, quat=None):
        """Put a free body at `xy` resting on the table (or at height `z` for its lowest point)."""
        a, v = self._qadr[name], self._dadr[name]
        base = self._extent[name]["bottom"] + (0.0 if z is None else z) + 0.001
        self.data.qpos[a : a + 3] = [xy[0], xy[1], base]
        self.data.qpos[a + 3 : a + 7] = _yaw_quat(yaw) if quat is None else quat
        self.data.qvel[v : v + 6] = 0

    def set_fixture_pose(self, name, xy, yaw=0.0, z=0.0):
        b = self._body[name]
        self.model.body_pos[b] = [xy[0], xy[1], z]
        self.model.body_quat[b] = _yaw_quat(yaw)

    def is_grasping(self, name) -> bool:
        self._set_target_geoms(self._geoms[name])
        return bool(self._is_grasping())

    def touching(self, a, b) -> bool:
        ga, gb = set(self._geoms[a]), set(self._geoms[b])
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in ga and c.geom2 in gb) or (c.geom1 in gb and c.geom2 in ga):
                return True
        return False

    def sample_xy(self, radius, placed, r=(0.12, 0.36), angle=(-80, 80), clearance=0.02, tries=500):
        """Rejection-sample a table point in polar coords around the robot base, clear of `placed` [(xy, radius)]
        and of `keepout` (robot base and the folded gripper at rest)."""
        for _ in range(tries):
            rr, aa = self.np_random.uniform(*r), np.radians(self.np_random.uniform(*angle))
            xy = np.array([rr * np.cos(aa), rr * np.sin(aa)])
            if all(np.linalg.norm(xy - p) >= radius + q + clearance for p, q in (*self.keepout, *placed)):
                return xy
        raise RuntimeError("could not place object")

    def place_distractors(self, placed):
        """Place 2-3 distractors (pool order shuffled) clear of `placed` [(xy, radius)]; park the rest off-table."""
        k = int(self.np_random.integers(self.n_distractors[0], self.n_distractors[1] + 1))
        pool = list(self.distractor_names)
        self.active_distractors = []
        for i in self.np_random.permutation(len(pool)):
            n = pool[i]
            if len(self.active_distractors) < k:
                try:
                    xy = self.sample_xy(self.footprint(n), placed, **self.distractor_region, clearance=0.03, tries=200)
                except RuntimeError:
                    xy = None
                if xy is not None:
                    placed.append((xy, self.footprint(n)))
                    self.set_object_pose(n, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
                    self.active_distractors.append(n)
                    continue
            self.set_object_pose(n, (PARK_X + i, 0.0), z=self._floor_z)
        return placed

    # ----- rendering ------------------------------------------------------------------------------------------
    def render_camera(self, camera="front", robot=True) -> np.ndarray:
        """RGB frame from `camera` ("front", "wrist_cam" or a MjvCamera). robot=False hides every robot geom, so
        neither the arm nor its shadow appears."""
        option = mujoco.MjvOption()
        option.geomgroup[ROBOT_VISUAL_GROUP] = int(robot)
        self._renderer_rgb.update_scene(self.data, camera=camera, scene_option=option)
        return self._renderer_rgb.render()

    def render_scene_without_robot(self, camera="front") -> np.ndarray:
        return self.render_camera(camera, robot=False)

    def _randomize_wrist_camera(self):
        self.model.cam_pos[self._wrist_cam_id] = WRIST_CAM_POS
        self.model.cam_quat[self._wrist_cam_id] = WRIST_CAM_QUAT
        self.model.cam_fovy[self._wrist_cam_id] = WRIST_CAM_FOVY

    # ----- gym plumbing ---------------------------------------------------------------------------------------
    def step(self, action, *, substep_observer=None):
        """Optionally observe each solver substep as ``observer(env, index)``.

        The upstream step still owns action conversion, observation, reward and
        termination. Observers must be read-only; exceptions propagate and the
        hook is cleared in all cases. Reset settling is never observed.
        """
        previous = getattr(self, "_substep_observer", None)
        self._substep_observer = substep_observer
        try:
            return super().step(action)
        finally:
            self._substep_observer = previous

    def _advance_physics(self):
        observer = getattr(self, "_substep_observer", None)
        if observer is None:
            return super()._advance_physics()
        for substep in range(self._N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            observer(self, substep)

    def _task_reset(self):
        for _ in range(50):
            self.active_distractors = []
            for i, n in enumerate(self.free_names):
                self.set_object_pose(n, (PARK_X + i, 5.0), z=self._floor_z)
            try:
                self.layout()
                return
            except RuntimeError:
                continue
        raise RuntimeError("layout failed 50 times")

    def _get_obs(self):
        obs = {"state": self._get_current_qpos().astype(np.float32)}
        if self.render_images:
            obs["front"] = self.render_camera("front")
            obs["wrist"] = self.render_camera("wrist_cam")
        return obs

    def _get_info(self):
        return {"success": bool(self.success()), **self.task_info()}

    def _compute_reward(self, info):
        return float(info["success"])

    def lerobot_joints(self, qpos=None) -> np.ndarray:
        """Six joint values in the training-data units: LeRobot degrees, gripper 0-100."""
        q = self._get_current_qpos() if qpos is None else qpos
        return sim_qpos_to_dataset_row(np.asarray(q, dtype=np.float64).copy(), gripper_limits_rad=(self._target_low[-1], self._target_high[-1]))

    def from_lerobot_joints(self, row) -> np.ndarray:
        """Inverse of `lerobot_joints`: a training-format joint row -> `pd_joint_pos` action (radians)."""
        row = np.asarray(row, dtype=np.float64)
        lo, hi = self._target_low[-1], self._target_high[-1]
        return np.concatenate([np.radians(row[:5]), [lo + row[5] / 100.0 * (hi - lo)]])

    def from_lerobot_ee(self, ee, iters=60) -> np.ndarray:
        """A training-format `action.ee` row (curate gripper-site xyz + rotvec, gripper 0-100) -> `pd_joint_pos`
        action, by IK on curate.robot's MJCF seeded from the current joints (orientation weighted 0.1)."""
        from curate.robot import Kinematics
        from scipy.spatial.transform import Rotation

        if not hasattr(self, "_curate_kin"):
            self._curate_kin = Kinematics()
        kin = self._curate_kin
        m, d = kin.model, kin.data
        target_p, target_r = np.asarray(ee[:3], float), Rotation.from_rotvec(ee[3:6]).as_matrix()
        q = self._get_current_qpos()[:5].copy()
        lo, hi = self._target_low[:5], self._target_high[:5]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        dofs = [m.jnt_dofadr[m.joint(j).id] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
        for _ in range(iters):
            d.qpos[:] = 0
            d.qpos[kin.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, kin.site)
            cur = d.site_xmat[kin.site].reshape(3, 3)
            e = np.concatenate([target_p - d.site_xpos[kin.site], 0.1 * 0.5 * sum(np.cross(cur[:, i], target_r[:, i]) for i in range(3))])
            J = np.vstack([jacp[:, dofs], 0.1 * jacr[:, dofs]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lo, hi)
        return np.concatenate([q, self.from_lerobot_joints(np.r_[np.zeros(5), ee[6]])[5:]])

    def close(self):
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = None
        super().close()

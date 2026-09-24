import tempfile

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_dir
from so101_nexus.config import EnvironmentConfig
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.observations import (
    EndEffectorPose,
    JointPositions,
    JointVelocities,
    OverheadCamera,
    WristCamera,
)

from libero_scene import build_scene_xml, table_top_z

SCALE = 0.5
ROBOT_XY = np.array([-0.26, 0.0])
OBJECTS = ("basket", "alphabet_soup", "cream_cheese", "tomato_sauce", "ketchup")
SPAWN = {
    "basket": ((0.20, 0.27), (-50, 50), 0.05),
    "alphabet_soup": ((0.17, 0.29), (-55, 55), 0.025),
    "cream_cheese": ((0.17, 0.29), (-55, 55), 0.025),
    "tomato_sauce": ((0.16, 0.30), (-65, 65), 0.02),
    "ketchup": ((0.16, 0.30), (-65, 65), 0.02),
}
CLEARANCE = 0.03
TARGETS = ("alphabet_soup", "cream_cheese")
TASK = "put both the alphabet soup and the cream cheese box in the basket"
WRIST_CAM_POS = (0.0025, 0.06157, -0.01877)
WRIST_CAM_QUAT = (0.97616, -0.21706, 0.0, 0.0)
WRIST_CAM_FOVY = 48.46


def default_config(width=640, height=480):
    return EnvironmentConfig(
        spawn_center=(0.26, 0.0),
        spawn_max_radius=0.12,
        reset_settle_frames=15,
        terminate_on_success=False,
        observations=[
            JointPositions(),
            JointVelocities(),
            EndEffectorPose(),
            WristCamera(width=width, height=height),
            OverheadCamera(width=width, height=height),
        ],
    )


class LiberoBasketEnv(SO101NexusMuJoCoBaseEnv):
    def __init__(self, config=None, render_mode=None, control_mode="pd_ee_pose", robot_init_qpos_noise=0.02):
        self._init_common(
            config=config or default_config(),
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        xml = build_scene_xml(SCALE, ROBOT_XY, table_top_z(SCALE, -0.1, 0.0))
        with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=get_so101_mujoco_model_dir()) as f:
            f.write(xml)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.model.actuator_forcerange[self.model.actuator("gripper").id] = [-1.47, 1.47]
        self.data = mujoco.MjData(self.model)
        self._objects = {
            name: (
                self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id],
                self.model.body(name).id,
                [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.model.body(f"{name}_upright").id and self.model.geom_contype[g]],
            )
            for name in OBJECTS
        }
        self._contain_site = self.model.site("basket_contain_region").id
        self._set_target_geoms(self._objects[TARGETS[0]][2])
        self._finish_model_setup()

    def _randomize_wrist_camera(self):
        self.model.cam_pos[self._wrist_cam_id] = WRIST_CAM_POS
        self.model.cam_quat[self._wrist_cam_id] = WRIST_CAM_QUAT
        self.model.cam_fovy[self._wrist_cam_id] = WRIST_CAM_FOVY

    @property
    def task_description(self):
        return TASK

    def object_pos(self, name):
        return self.data.xpos[self._objects[name][1]].copy()

    def is_grasping(self, name):
        self._set_target_geoms(self._objects[name][2])
        return bool(self._is_grasping())

    def in_basket(self, name):
        local = self.data.site_xmat[self._contain_site].reshape(3, 3).T @ (self.object_pos(name) - self.data.site_xpos[self._contain_site])
        return bool(np.all(np.abs(local) <= self.model.site_size[self._contain_site]))

    def _task_reset(self):
        placed = []
        for name in OBJECTS:
            (r_lo, r_hi), (a_lo, a_hi), radius = SPAWN[name]
            while True:
                r, a = self.np_random.uniform(r_lo, r_hi), np.radians(self.np_random.uniform(a_lo, a_hi))
                xy = np.array([r * np.cos(a), r * np.sin(a)])
                if all(np.linalg.norm(xy - p) >= radius + q + CLEARANCE for p, q in placed):
                    break
            placed.append((xy, radius))
            adr = self._objects[name][0]
            self.data.qpos[adr : adr + 3] = [*xy, 0.002 if name == "basket" else 0.03]
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

    def object_poses(self):
        return np.concatenate([self.data.qpos[self._objects[n][0] : self._objects[n][0] + 7] for n in (*TARGETS, "basket")])

    def _get_info(self):
        info = {f"{n}_in_basket": self.in_basket(n) for n in TARGETS}
        info["success"] = all(info.values())
        return info

    def _compute_reward(self, info):
        return float(info["success"])

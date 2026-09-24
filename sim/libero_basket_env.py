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
REGIONS = {
    "basket": ((-0.01, 0.25), (0.01, 0.27)),
    "alphabet_soup": ((0.025, -0.125), (0.075, -0.075)),
    "cream_cheese": ((-0.175, 0.035), (-0.125, 0.085)),
    "tomato_sauce": ((0.075, -0.225), (0.125, -0.175)),
    "ketchup": ((-0.225, -0.175), (-0.175, -0.125)),
}
TARGETS = ("alphabet_soup", "cream_cheese")
TASK = "put both the alphabet soup and the cream cheese box in the basket"


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
            WristCamera(
                width=width,
                height=height,
                fov_deg_range=(48.5, 48.5),
                pitch_deg_range=(-32.66, -32.66),
                pos_x_noise=0.0,
                pos_y_center=0.055,
                pos_y_noise=0.0,
                pos_z_center=-0.045,
                pos_z_noise=0.0,
            ),
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
            for name in REGIONS
        }
        self._contain_site = self.model.site("basket_contain_region").id
        self._set_target_geoms(self._objects[TARGETS[0]][2])
        self._finish_model_setup()

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
        for name, (lo, hi) in REGIONS.items():
            adr = self._objects[name][0]
            xy = self.np_random.uniform(lo, hi) * SCALE - ROBOT_XY
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

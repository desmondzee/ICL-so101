"""SO-101 joint conventions, unit conversion to LeRobot degrees, and forward kinematics."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MJCF = Path(__file__).resolve().parents[1] / "sim/so101/so101_new_calib.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
NAMES = [f"{j}.pos" for j in JOINTS] + ["gripper.pos"]
OLD_NAMES = [f"main_{j}" for j in JOINTS] + ["main_gripper"]
REST = {1: -99.0, 2: 96.1}
REST_REACHED = 12.0


class Kinematics:
    def __init__(self, mjcf: Path = MJCF, site: str = "gripperframe"):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(j).id for j in JOINTS]
        self.qadr = [self.model.jnt_qposadr[i] for i in ids]
        self.range_deg = np.rad2deg(np.array([self.model.jnt_range[i] for i in ids]))
        self.site = self.model.site(site).id

    def ee(self, deg: np.ndarray) -> np.ndarray:
        """Joint degrees [N, 5] -> [N, 6] gripper-site xyz (m, base frame) and rotation vector."""
        out = np.empty((len(deg), 6))
        for i, q in enumerate(np.deg2rad(deg)):
            self.data.qpos[:] = 0
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[i, :3] = self.data.site_xpos[self.site]
            out[i, 3:] = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3)).as_rotvec()
        return out

    def to_degrees(self, x: np.ndarray, units: str) -> np.ndarray:
        """[N, 6] arm values in `units` -> LeRobot degrees (zero at mid-range); gripper stays 0-100."""
        x = np.asarray(x, dtype=np.float64).copy()
        if units == "norm":
            lo, hi = self.range_deg[:, 0], self.range_deg[:, 1]
            x[:, :5] = (lo + hi) / 2 + x[:, :5] / 100 * (hi - lo) / 2
        elif units == "old_deg":
            x[:, 1] = 90 - x[:, 1]
            x[:, 2] = x[:, 2] - 90
        elif units != "deg":
            raise ValueError(units)
        return x


def rest_offsets(deg: np.ndarray) -> np.ndarray | None:
    """Per-joint degree offsets that put the folded rest pose on the stops seen in normalized datasets.

    Shoulder lift and elbow rest on mechanical stops; the other joints have none and get 0.
    Returns None when the arm never folds to rest, so no anchor exists.
    """
    lift, elbow = np.percentile(deg[:, 1], 0.5), np.percentile(deg[:, 2], 99.5)
    off = np.zeros(deg.shape[1])
    off[1], off[2] = REST[1] - lift, REST[2] - elbow
    return None if max(abs(off[1]), abs(off[2])) > REST_REACHED else off

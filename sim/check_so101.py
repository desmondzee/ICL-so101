"""Smoke test: load the SO101 + wrist camera scene, drive the joints, render both cameras.

    uv run python sim/check_so101.py   # writes sim/check_so101.png
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(HERE / "so101" / "scene_camera.xml"))
data = mujoco.MjData(model)

print("joints:", [model.joint(i).name for i in range(model.njnt)])
print("actuators:", [model.actuator(i).name for i in range(model.nu)])
print("cameras:", [model.camera(i).name for i in range(model.ncam)])

# Reach down toward the cube in front of the base, gripper open.
target = {"shoulder_pan": 0.0, "shoulder_lift": 0.6, "elbow_flex": -0.2,
          "wrist_flex": 1.2, "wrist_roll": 0.0, "gripper": 1.0}
for name, q in target.items():
    data.ctrl[model.actuator(name).id] = q
for _ in range(2000):
    mujoco.mj_step(model, data)
assert np.all(np.isfinite(data.qpos)), "simulation diverged"

renderer = mujoco.Renderer(model, 480, 640)
frames = []
for cam in ("front", "wrist_cam"):
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())
out = HERE / "check_so101.png"
Image.fromarray(np.concatenate(frames, axis=1)).save(out)
print("wrote", out)

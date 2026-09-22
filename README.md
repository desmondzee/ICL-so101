# ICL-so101

SO-101 arm model for MuJoCo, with the wrist camera attached, set up for RL training later.

## Layout

- `sim/so101/`: MJCF, URDF and meshes vendored from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101` (commit `eecbe3e`).
  - `scene_camera.xml`: **use this one.** It contains the arm, the wrist camera, a floor, a `front` camera and a test cube.
  - `so101_new_calib_camera.xml`: the robot with the wrist camera mount and camera. The only local change is an added `wrist_cam` render camera that sits at the lens and looks where the real camera looks. It is modelled on an OV2710 32x32 module with the stock 3.6 mm lens (1920x1080 sensor; ≈77° × 48° pinhole FOV; lens distortion is not modelled). Render it at 16:9.
  - `so101_new_calib_camera.urdf`: URDF of the same robot, with the camera, if you need it elsewhere.
  - `scene.xml`, `so101_new_calib.xml`, `so101_old_calib.*`: the upstream files without the camera.
- `cad/wrist_camera/`: STEP and STL for the Hex-Nut wrist camera mount (32x32 UVC module), from `Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`. This is the mount the sim model uses.
- `sim/check_so101.py`: smoke test. It loads the scene, drives the joints, and renders both cameras to `sim/check_so101.png`.

## Quick start

```sh
uv sync
uv run python sim/check_so101.py
uv run python -m mujoco.viewer --mjcf sim/so101/scene_camera.xml
```

Actuators are position servos (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), and `ctrl` is the target in radians. "New calib" means joint zero is the middle of each joint's range, which matches LeRobot. The gripper is a hinge in radians: the lower limit (−0.175) is closed and the upper limit (1.745) is fully open. LeRobot uses 0 for closed and 100 for open, so `pct = 100 * (q - q_min) / (q_max - q_min)`.

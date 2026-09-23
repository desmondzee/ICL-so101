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
- `sim/check_grasp.py`: grasp test. At 5 cube positions it uses IK to put the jaws around a 30 mm cube, closes the gripper and lifts; the cube must come up with it.

## Changes from upstream (in `so101_new_calib_camera.xml` / `scene_camera.xml`)

- `wrist_cam`: OV2710 camera at the lens, see above.
- Jaw collisions: MuJoCo collides meshes as their convex hulls, which filled the gap between the jaws. The palm, the fixed finger and the moving finger now use boxes fitted to the meshes (`palm_collision`, `fixed_jaw_*`, `moving_jaw_*`). The finger pads have high friction (condim 4) and stiff contacts.
- Gripper force limit is 1.47 Nm instead of 3.35 Nm, matching the 50% torque cap LeRobot sets on the real gripper.
- The scene uses `cone="elliptic" impratio="10"` so grasped objects don't slip out.

## Workspace notes (for end-effector-space control)

- The arm has 5 DOF, so it cannot reach an arbitrary 6-DoF end-effector pose. Specify position plus approach direction (the IK in `check_grasp.py` constrains position and "fingers pointing down"), and leave the wrist roll free or command it separately.
- A straight-down grasp is reachable only in a limited region: at 0.30 m out it works near the table but not about 10 cm above it, because wrist_flex hits its limit.
- The gripper is asymmetric (one finger is fixed). A ~30 mm object sits about 10 mm to the moving-jaw side of the fixed finger (`GRASP_OFFSET` in `check_grasp.py`). The fixed fingertip is 24 mm below that point.

## Quick start

```sh
uv sync
uv run python sim/check_so101.py
uv run python -m mujoco.viewer --mjcf sim/so101/scene_camera.xml
```

Actuators are position servos (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), and `ctrl` is the target in radians. "New calib" means joint zero is the middle of each joint's range, which matches LeRobot. The gripper is a hinge in radians: the lower limit (−0.175) is closed and the upper limit (1.745) is fully open. LeRobot uses 0 for closed and 100 for open, so `pct = 100 * (q - q_min) / (q_max - q_min)`.

## LIBERO (vanilla)

- `third_party/LIBERO`: git submodule pinned to upstream `8f1084e`. After cloning this repo, run `git submodule update --init`.
- `libero_sim/`: a separate uv environment for it (Python 3.10, robosuite 1.4.0, MuJoCo 2.3.7, NumPy 1.22). It is kept separate because the root environment uses MuJoCo 3 and NumPy 2. The training stack (transformers, wandb, robomimic) is left out; torch is CPU-only.

Setup:

```sh
git submodule update --init
cd libero_sim && uv sync
```

LIBERO reads its paths from `$LIBERO_CONFIG_PATH/config.yaml`. If that file is missing, the first `import libero` prompts interactively and writes `~/.libero/config.yaml`. Keep the config in the repo instead: create `libero_sim/.libero/config.yaml` (gitignored, absolute paths) with the keys `assets`, `bddl_files`, `benchmark_root`, `datasets` and `init_states`, pointing into `third_party/LIBERO/libero/`. Then run with:

```sh
export LIBERO_CONFIG_PATH=$PWD/libero_sim/.libero
```

Note: MuJoCo 2.3.7 has no camera `sensorsize`/`focal` attributes, so an SO-101 robosuite model must use `fovy="48.46"` for the OV2710.

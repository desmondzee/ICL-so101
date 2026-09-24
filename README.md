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

## LIBERO-style task on SO101-Nexus

A LIBERO scene scaled down to the SO-101, running on [SO101-Nexus](https://github.com/johnsutor/so101-nexus) (pinned `0.7.0`, MuJoCo 3, LeRobot 0.5).

- `third_party/LIBERO`: git submodule pinned to upstream `8f1084e`. Only its assets and task definitions are used, not its robosuite code. After cloning, run `git submodule update --init`.
- `sim/libero_scene.py`: builds one MJCF from LIBERO's living-room arena (floor, walls, wall unit, table) and LIBERO objects, all scaled by 0.5, plus the Nexus SO-101. The table top is at robot-base height. LIBERO's hand-made collision boxes are kept (moved to group 3). Objects get density 1000 kg/m³ and stiff, high-priority contacts.
- `sim/libero_basket_env.py`: `LiberoBasketEnv`, a Nexus `SO101NexusMuJoCoBaseEnv` subclass. The task is LIBERO-10 "put both the alphabet soup and the cream cheese box in the basket", with the tomato sauce and ketchup as distractors, and object positions sampled from the scaled LIBERO regions. Details:
  - Control runs at 30 Hz. The default control mode is `pd_ee_pose`, using Nexus's IK with its default orientation weight of 0.01 (orientation leeway).
  - Cameras are `wrist_camera` (the Nexus wrist cam, pinned to its mount pose with no randomisation, `fovy` 48.5) and a top-down `overhead_camera`, both 640x480.
  - `info["success"]` is true when both objects are inside the basket's LIBERO `contain_region`.
  - The gripper is capped at 1.47 Nm, LeRobot's real 50% limit.
- `sim/scripted.py`: `PickPlace` state machine with the gripper pointing down. For each object it approaches, descends, closes, checks the grasp with Nexus's contact-based `_is_grasping` (retrying once if needed), lifts, carries, lowers into the basket, releases and retreats.
- `sim/record_demos.py`: records successful episodes to a LeRobot dataset in `data/so101_libero_basket` (gitignored). Failed episodes are discarded.
  - Videos: `observation.images.wrist` and `observation.images.overhead`.
  - Joint data: `observation.state` and `action` (joint targets), in LeRobot units: degrees, and 0-100 for the gripper.
  - `action.ee`: the commanded TCP pose (xyz, rotation vector, gripper rad).
  - `observation.environment_state`: the TCP pose plus the soup, cheese and basket poses.
  - The task string is stored on every frame, and `demo_summary.json` holds the grasp/lift/in-basket checks for each episode.
- `sim/export_previews.py`: writes H.264 side-by-side (wrist | overhead) previews to `data/so101_libero_basket/previews/`. The dataset videos themselves are AV1.

```sh
git submodule update --init
uv sync
uv run python sim/record_demos.py --episodes 5
uv run python sim/export_previews.py
```

Notes:
- To read the dataset, pass `video_backend="pyav"` to `LeRobotDataset`. LeRobot's default decoder (torchcodec) needs a system FFmpeg (`brew install ffmpeg`).
- The robot is the Nexus (MuJoCo Menagerie) SO-101, whose finger collisions and grasp detection Nexus is tuned for. Its wrist camera mount differs from the hex-nut mount in `sim/so101/`, although the camera intrinsics match the OV2710.

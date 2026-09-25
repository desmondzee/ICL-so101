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
- `sim/libero_scene.py`: builds one MJCF from LIBERO's living-room arena (floor, walls, wall unit, table) and LIBERO objects, all scaled by 0.5, plus the Nexus SO-101. The table top is at robot-base height. LIBERO's hand-made collision boxes are kept (moved to group 3). Physics and rendering come from Nexus rather than ad-hoc values:
  - `MUJOCO_SCENE_OPTION_XML`: timestep 0.005, implicitfast, elliptic cones, impratio 10, noslip 3.
  - `SCENE_VISUAL_XML` and `SCENE_LIGHTS_XML` for rendering.
  - Nexus's graspable-object contacts: condim 4, friction `1 0.05 0.001`, MuJoCo default solref/solimp.
  - Masses use Nexus's GSO method: convex-hull volume of the scaled mesh × an assumed effective density (`EFFECTIVE_DENSITY`).
- `sim/libero_basket_env.py`: `LiberoBasketEnv`, a Nexus `SO101NexusMuJoCoBaseEnv` subclass. The task is LIBERO-10 "put both the alphabet soup and the cream cheese box in the basket", with the tomato sauce and ketchup as distractors. Each reset samples every object (including the basket) over the arm's reachable arc (`SPAWN`: radius and angle range from the base, plus a footprint radius), with at least `CLEARANCE` between footprints, so each rollout starts from a different layout. Details:
  - Control runs at 50 Hz, Nexus's native 4 × 0.005 s. The default control mode is `pd_ee_pose`, using Nexus's IK with its default orientation weight of 0.01 (orientation leeway).
  - Cameras are `wrist_camera` and an `overhead_camera` tilted to 60° elevation and turned −30° in azimuth (`OVERHEAD_ELEVATION`, `OVERHEAD_AZIMUTH`), so the robot sits on the right of the frame and reaches across the table, both 640x480, recorded at 50 fps. The wrist camera is pinned (no randomisation) to the official TheRobotStudio SO-101 camera pose, which is the lens of the hex-nut mount model in `sim/so101/`: pos (2.5, 61.6, −18.8) mm, 25.1° pitch in the gripper body frame, and OV2710 intrinsics (`fovy` 48.46).
  - `info["success"]` is true when both objects are inside the basket's LIBERO `contain_region`.
  - The gripper is capped at 1.47 Nm, LeRobot's real 50% limit.
- `sim/scripted.py`: `PickPlace` state machine with the gripper pointing down. For each object it approaches, descends, closes, checks the grasp with Nexus's contact-based `_is_grasping` (retrying once if needed), lifts, carries, lowers into the basket, releases and retreats.
- `sim/record_demos.py`: records successful rollouts (seeds counting up from `--seed`; about two in three succeed, failures are discarded). Each one goes straight into `data/examples/so101_libero_basket/episode_XXX/` as its own one-episode LeRobot dataset with H.264 videos (tracked in git). Each episode contains:
  - Videos: `observation.images.wrist` and `observation.images.overhead`.
  - Joint data: `observation.state` and `action` (joint targets), in LeRobot units: degrees, and 0-100 for the gripper.
  - `action.ee`: the commanded TCP pose (xyz, rotation vector, gripper rad).
  - `observation.environment_state`: the TCP pose plus the soup, cheese and basket poses.
  - The task string on every frame, and `episode.json` with the task, seed and grasp/lift/in-basket checks.
  - Load one with `LeRobotDataset("x", root="data/examples/so101_libero_basket/episode_000", video_backend="pyav")`.
- `sim/export_frames.py`: for each example, replays its seed and saves `frames/first.png` (scene at reset) and `frames/last.png` (task completed). Both use the overhead camera, with the robot's visual geoms (group 2) hidden, so they show only the environment state before and after the task.

```sh
git submodule update --init
uv sync
uv run python sim/record_demos.py --episodes 5
uv run python sim/export_frames.py
```

Notes:
- To read the dataset, pass `video_backend="pyav"` to `LeRobotDataset`. LeRobot's default decoder (torchcodec) needs a system FFmpeg (`brew install ffmpeg`).
- The robot is the Nexus (MuJoCo Menagerie) SO-101, whose finger collisions and grasp detection Nexus is tuned for. Only the wrist camera pose is changed. The Menagerie camera-mount mesh is still drawn but does not block the camera's view.

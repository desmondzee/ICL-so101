# SO-101 simulated validation set

Scripted-oracle episodes recorded in the same format as the curated real data (`data/so101_export/lerobot/*`),
plus robot-free first and last frames for the image-to-video generator, plus success predicates for closed-loop
policy evaluation. Run everything from the repo root:

```bash
uv run python -m sim.val.record --task sort_blocks --episodes 10 --seed 0   # dataset + frames + meta
uv run python -m sim.val.preview --task sort_blocks --compare                # contact sheets, sim vs real
```

## Files

| file | what |
|---|---|
| `scene.py` | MJCF builder: arena (`kitchen`, `living_room`), LIBERO objects (`Obj`), primitive boxes (`Block`), flat discs (`Disc`), static fixtures (`Fixture`), distractor pool, lights, cameras. `CATALOG` lists the assets with upright rotations and densities. |
| `env.py` | `ValEnv(SO101NexusMuJoCoBaseEnv)`: cameras, 30 fps stepping, object/fixture helpers, distractor placement, robot-free rendering, unit conversion. |
| `oracle.py` | `Oracle`: IK and reusable skills (`move`, `transit`, `pick`, `place`, `gripper`, `wait`, `rest`). |
| `record.py` | Records N successful episodes of a task to LeRobot v3. |
| `preview.py` | Contact sheets, first/last-frame sheets, and the sim-vs-real front-camera comparison. |
| `tasks/__init__.py` | Task registry (`TASKS`). |
| `tasks/sort_blocks.py` | Task 1, a reference implementation to copy. |

## Conventions (keep these identical across tasks)

- **World frame = robot base frame.** The base is at the origin facing +x, +y is to the robot's left, and the table top is z = 0. Object poses, `action.ee` and `observation.state.ee` all use this frame.
- **Timing.** One `env.step` is one 30 fps frame (7 physics steps of 1/210 s). The env runs in `pd_joint_pos`, so the action is 6 joint targets: 5 arm joints in radians, then the gripper in radians over [-0.1745, 1.745].
- **Recorded units** (the training format, written by `record.py`):
  - `observation.state` and `action` are `env.lerobot_joints(q)`: LeRobot degrees, gripper 0-100. The sim joint zeros equal the curated LeRobot zeros, which I checked by comparing FK between the nexus MJCF and `sim/so101/so101_new_calib.xml`.
  - `*.ee` is `curate.robot.Kinematics().ee(deg[:, :5])` plus the gripper, exactly as `curate/convert.py` computes it. It is the curate `gripperframe` site, not nexus's TCP site, which sits 2 cm away in a different frame.
  - `action` is the joint target the oracle commanded, and `observation.state` is the measured joint position before that step.
- **Rest pose.** `REST_DEG = (0, -99, 95, 64, 0, -9 deg)`, which settles to about [0, -99, 93, 64, 0, 0.9%], close to the real datasets' first frame (about [0, -99, 94, 70, 0, 1]). Episodes start and end at rest.
- **Gripper.** The oracle opens to 0.70 rad (about 45%; real data peaks at 40-50%) and closes toward -0.17 rad (0%). On a 2.8 cm block it stalls at about 15-20%.
- **Cameras.** `front` is a fixed scene camera (`FRONT_CAM` in env.py). `wrist_cam` is the nexus wrist mount, overridden by `WRIST_CAM_*`. Both render at 640x480.
- **Geom groups.** Group 2 holds exactly the robot's visual geoms; the env asserts this at build time. Scene visuals are group 1, and LIBERO collision boxes are moved to group 3, which is hidden. `render_scene_without_robot` turns group 2 off, which removes the robot and its shadow, because hidden geoms never enter the shadow pass.

## ValEnv API

```python
env = SortBlocksEnv(render_images=True)        # render_images=False: ~2 s per episode
obs, info = env.reset(seed=s)                  # obs: state (6 rad), front, wrist (HxWx3 uint8)
obs, r, term, trunc, info = env.step(q6)       # info["success"] plus task_info()

env.object_pos(name)    env.object_pose(name)  # [x y z qw qx qy qz], base frame (free bodies and fixtures)
env.object_poses()                             # dict of task objects, active distractors and fixtures
env.set_object_pose(name, xy, yaw=0, z=None)   # rests the object on the table (or with its bottom at height z)
env.set_fixture_pose(name, xy, yaw=0, z=0)     # writes model.body_pos/quat; call it inside layout()
env.footprint(name)   env.height(name)         # xy radius and height of the collision geometry
env.is_grasping(name) env.touching(a, b)       # nexus two-finger pinch test; any contact between two bodies
env.sample_xy(radius, placed, r=(..), angle=(..), clearance=..)  # polar rejection sampler with robot keepouts
env.place_distractors(placed)                  # 2-3 from the scene's distractor pool; the rest are parked off-table
env.render_camera("front" | "wrist_cam", robot=True)
env.render_scene_without_robot("front")
env.lerobot_joints(q)        env.from_lerobot_joints(row)   # sim radians <-> training units
env.from_lerobot_ee(ee_row)  # training-format action.ee -> joint action (IK), for evaluating EE policies
env.instruction              # task string, also written to the dataset
```

Closed-loop policy evaluation: `reset(seed)`, convert each predicted action row with `from_lerobot_joints` (or
`from_lerobot_ee`), call `step`, and read `info["success"]`. Use seeds outside the recorded ones for held-out
layouts. The recorded seeds are in `data/so101_sim_val/frames/<task>/episode_*/meta.json` and
`lerobot/<task>/meta/val_episodes.json`.

## Adding a task

1. Create `tasks/<name>.py` with an env and an oracle:

```python
class MyEnv(ValEnv):
    instruction = "Put the mug on the plate."
    def make_scene(self):
        return SceneSpec(arena="kitchen",                      # or "living_room"
                         objects=[Obj("mug", "porcelain_mug"), Obj("plate", "plate", scale=0.65)],
                         fixtures=[],                            # e.g. Fixture("microwave", "microwave")
                         distractors=[Obj(n, n) for n in ("alphabet_soup", "butter", "ramekin")])
    def layout(self):                                          # use self.np_random only
        placed = []
        for n in ("mug", "plate"):
            xy = self.sample_xy(self.footprint(n), placed, r=(0.15, 0.28), angle=(-60, 60))
            placed.append((xy, self.footprint(n)))
            self.set_object_pose(n, xy, yaw=self.np_random.uniform(-np.pi, np.pi))
        self.place_distractors(placed)
    def success(self): ...                                      # must also hold at the final frame

class MyOracle(Oracle):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = rng
    def plan(self):
        ok = yield from self.pick("mug", width=..., grasp_z=..., yaw=...)
        if not ok:
            return
        yield from self.place(target_xy + self.held_offset("mug"), release_z)
        yield from self.rest()
```

2. Register it in `tasks/__init__.py`: `"my_task": "sim.val.tasks.my_task:MyEnv:MyOracle"`.
3. Check it with a quick loop: `render_images=False`, about 2 s per episode, aiming for at least 95% over 50+ seeds. Look at `preview.py` sheets before you record.

`layout()` may raise `RuntimeError` (for example from `sample_xy`); the env then retries with fresh draws, up to 50
times, and stays deterministic per seed. The success predicate must be strict: object resting, not grasped, and
low velocity, because `record.py` keeps only seeds where it holds at the end of the episode.

## Things task agents must know

- **Reach.** A top-down grasp is kinematically feasible at table height for r = 0.12-0.30 m from the base. At 8 cm above the table it is only feasible to about 0.25 m, because wrist flex saturates at 95 deg. Spawn graspable objects at r of 0.28 m or less. `ROT_WEIGHT = 0.1` in the IK lets the gripper tilt when vertical is infeasible, which is what happens during lift and carry.
- **Wrist roll is limited to ±157 deg.** `pick` picks among the 4 face directions the one that needs the least roll change and moves to the pregrasp in joint space. `place` rotates the grasp yaw with the arm's azimuth so the roll stays put. Keep that pattern for long moves; a Cartesian move with a big yaw change pins the roll at its limit.
- **TCP and grasp geometry.** The oracle controls nexus's `gripperframe` site: x points from the moving jaw toward the fixed finger, z is the approach axis, and the fingertips are about 5 mm past the TCP. `pick` offsets the TCP by `FIXED_FACE - width/2 - 3 mm` along the closing direction so the object ends up between the jaws. Use `grasp_z` of about half the object height and at least 0.012 m (fingertip to table). Leave about 3 cm of finger clearance around graspable objects (see `FINGER_CLEARANCE` in sort_blocks).
- **Carry height.** `CARRY_Z = 0.10` clears every distractor in the sort_blocks pool. Tall LIBERO items (ketchup, milk, orange_juice, moka_pot, books) are 6-8 cm tall at 0.5x. Ketchup tips over easily.
- **Scale.** Objects default to 0.5x, the project's LIBERO convention. Some come out large for the SO-101 at 0.5x: `frypan` has a 0.14 m footprint radius, and `microwave` is about 0.17 m wide. Pass `Obj(..., scale=...)` or `Fixture(..., scale=...)` as needed. Mugs at 0.5x are about 5 cm tall with a 4 cm diameter. The 0.5x `plate` is 9 cm across; sort_blocks uses 0.65x (13 cm).
- **Fixtures** are static bodies placed with `set_fixture_pose`, so they need no free joint. Their internal joints keep a prefixed name: microwave door `microwave_microjoint` (LIBERO open range [-2.094, -1.3] rad), stove knob `stove_button`. The door has no actuator, so set its qpos in `layout()`. Microwave collision boxes are `contype=0 conaffinity=1`. The microwave's default yaw shows its back to the camera, so set `yaw` so the door faces the robot.
- **Upright assets.** Every `CATALOG` entry was checked to rest upright and stable. The books stand on edge.
- **Distractors** must not overlap task objects. Parked distractors sit on the floor at x = 20 m, out of view. `object_poses()` reports only the active ones.
- **Determinism.** A rollout is a pure function of the seed: the oracle's own RNG is `np.random.default_rng(seed)`, and rendering doesn't touch physics. `record.py` relies on this; it screens each seed without rendering, then re-runs only the seeds that succeed with rendering on.
- **Rendering.** It uses the default macOS GL (no `MUJOCO_GL` needed). On Linux, set `MUJOCO_GL=egl`.

# SO-101 simulated validation pairs (sim_val_v1)

50 human-robot pairs over 5 tasks (15 min of robot data). None of these tasks is in the training set (`curated_humangen_v1`). Each pair joins a scripted-oracle SO-101 episode recorded in MuJoCo (`sim/val`, built on so101-nexus with LIBERO assets) with a generated video of a person doing the same task in the same scene. Every scene also has 2-4 distractor objects that the oracle never touches.

Viewer: https://desmondzee.github.io/ICL-so101/#val/sort_blocks (grid) and https://desmondzee.github.io/ICL-so101/viewer.html#val/all/1 (one pair at a time)

| Task | Instruction | Pairs |
|---|---|---|
| `mug_on_plate` | Put the red mug on the blue plate. / Put the red mug on the white plate. | 10 |
| `mugs_in_microwave` | Put both mugs in the microwave. | 10 |
| `pan_on_stove` | Put the frying pan on the stove. | 10 |
| `sort_blocks` | Put the red block on the red plate and the blue block on the blue plate. | 10 |
| `stack_bowls` | Put the white bowl in the black bowl. | 10 |

## How it was made

1. **Episodes:** recorded with a scripted oracle in the training data's format: front and wrist cameras at 640x480 and 30 fps; joint `action`/`observation.state` in LeRobot degrees with the gripper at 0-100; `action.ee`/`observation.state.ee` from the same FK as the real export. Layouts vary per seed. The oracle succeeded on 97-100% of seeds per task; only successful episodes are kept.
2. **Human demos:** the first and last front-camera frames are rendered with the robot hidden (no arm, no shadow), and H3 Max Turbo (fal) generates a 5 s video from them. The prompt is the training data's v5 prompt, with the hand doing the steps in the robot's own order, one object at a time.
3. **Review:** the same two-stage Opus check as the training set. A judge checks task adherence, physics, one acting hand and scene consistency, then an independent verifier tries to refute each accept. Rejected episodes were regenerated with a new seed. Where an episode kept failing (three pan episodes and one block-sorting episode), a newly recorded episode replaced it. First-try pass rates were 10/10 for mug on plate and stack bowls, 6/10 for sort blocks, 5/10 for pan on stove (the generator morphs the pan handle) and 2/10 for the microwave (rising to 8/8 once the prompt named the order). All 50 final pairs passed both stages.
4. **Packaging:** the selected seeds were re-recorded into clean datasets (`sim/val/package.py`). Their robot-free frames are pixel-identical to the frames the human videos were generated from.

## Layout

```
index.json                         tasks and episodes (paths relative to episodes/<task>/<episode>/)
lerobot/<task>/                    LeRobot v3 dataset, episodes 0-9 (AV1 video, training-data columns)
frames/<task>/episode_XXX/         first.png / last.png (robot-free renders) and meta.json (seed, instruction, object poses, oracle log)
episodes/<task>/episode_XXX/
  human.mp4                        generated human demonstration (H.264, no audio, 640x480, 24 fps, ~5 s)
  robot_front.mp4, robot_wrist.mp4 robot views (H.264, 640x480, 30 fps)
  robot_data.parquet               the episode's frames (joint and EE actions and states)
  review.json                      judge and verify verdicts
  source.json                      the recorded episode and generation seed it came from
  thumb.jpg, robot_thumb.jpg
```

For closed-loop evaluation, rebuild a task's scene with `sim.val.tasks.load(task)` and `env.reset(seed=<seed from meta.json>)`, condition the policy on `human.mp4`, and read `info["success"]` from `env.step`.

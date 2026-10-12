# Closed-loop sim_val_v2 evaluation v1 (zero-shot, human-video prompted)

**Status: partial (5 of 50 episodes).** The Modal workspace was disabled partway through the run (spend limit), so only 5 `mug_on_plate` episodes finished. All 5 failed.

## Setup

- **Checkpoint:** `hf://buckets/nikgeo/icl-so101-weights/runs/so101-sft-a100-b32-cosine-v2b-1k-20261011/checkpoints/checkpoint_step_500/transformer`. The VAE, text encoder and tokenizer come from `Robbyant-Research/zero-wam-pretrain@7040c4195df216c900334ef62d5fdcf05c0601aa`.
- **Tasks:** the 50 `sim_val_v2` pairs (`hf://buckets/akoniti/ICL-so101/sim_val_v2`), 5 tasks × 10, run once each. None of these tasks is in training (`task_overlap_with_train: 0`). Training does share skill families and LIBERO assets with them.
- **Zero-shot prompt:** the paired generated human video, as its precomputed ICL latent from `zero_wam_latents_v1/val/human_latents`, which also carries the detailed human-video text. ICL CFG is 5. `video_guidance_scale = -1`, so the short task text is absent; this matches upstream RoboTwin and the training doc's checkpoint-selection protocol. The chunk-causal validation used the short text with CFG 5 instead. No robot demo of the task is given.
- **Sim:** `sim.val.tasks.load(task)` with `env.reset(seed=<index seed>)`. Control is `pd_joint_pos` at 30 Hz. Model outputs (LeRobot degrees, gripper 0-100) go through `env.from_lerobot_joints`. Wrist frames are rendered with the val_v3 near-plane transform (`sim.rebuild_wrist_v2.configure_clean_wrist`, 0.02425 m) so they match training. Both cameras are 640x480.
- **Loop:** follows upstream `evaluation/robotwin/eval_policy_client_openpi.py`. A chunk is 2 latent frames × 8 controls. The first chunk executes only its second frame; later chunks execute all 16 controls. A keyframe is taken every 2 controls and fed back with the executed actions through `compute_kv_cache`. Norm stats come from `zero_wam_loader_v1/val/meta/action_stats.json`.
- **Sampling:** 50 video + 50 action denoising steps. The torch/numpy seed is `20261012 + index position`.
- **Success:** the episode stops at the first control where `info["success"]` is true. Every task predicate requires the object to be released and at rest. Time limit is `ceil(2.0 × oracle robot_frames)` controls, with the last chunk allowed to finish.
- **Hardware:** Modal H100, sim and model in the same container, up to 8 in parallel. About 14-20 s per chunk, about 20 min per episode.
- **Code:** `zero_wam/modal_so101_closed_loop.py` and `zero_wam/so101_closed_loop.py` at ICL-so101 commit `814ab1c`.

## Results (v2b step 500)

| episode | success | controls / limit | final state |
|---|---|---|---|
| mug_on_plate/episode_000 | no | 1128 / 1120 | mug upright, not on a plate |
| mug_on_plate/episode_002 | no | 1016 / 1006 | mug knocked over |
| mug_on_plate/episode_004 | no | 952 / 948 | mug knocked over |
| mug_on_plate/episode_008 | no | 1128 / 1124 | mug knocked over |
| mug_on_plate/episode_009 | no | 1032 / 1022 | mug upright, not on a plate |

In episode 000 the arm reaches the red mug, hovers over it and moves toward the blue plate, but never closes the gripper on the mug; later it drifts toward the other mug. Not run yet: the other 45 episodes (mug_on_plate 001/003/005/006/007, and every episode of mugs_in_microwave, pan_on_stove, sort_blocks and stack_bowls).

## Layout

```
v2b_step500/summary.json                 settings + all episode results
v2b_step500/<task>/<episode>/
  rollout.mp4    front | wrist side by side, 15 fps (one frame every 2 controls = real time)
  actions.npy    executed actions [controls, 6]: LeRobot deg x5, gripper 0-100
  result.json    success, first_success_control, controls, chunk timing, task_info
smoke/mug_on_plate/episode_000/          first smoke run of the same episode/seed (same outcome; video only in the bucket copy)
```

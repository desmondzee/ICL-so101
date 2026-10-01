# Zero-WAM SO-101 smoke test

Date: 2026-10-01. Goal: check that upstream Zero-WAM post-training runs on SO-101 data, that the loss falls and that training is stable, by overfitting one episode.

## Setup

- **Training code:** upstream Zero-WAM (`third_party/Zero-WAM`, commit `08e2c4a`), unmodified. It uses the `robotwin_train` config and the torchrun command from `script/train.sh`, starting from `zero-wam-pretrain` (revision `7040c41`).
- **Hardware:** Modal workspace `jedwardzee`, 8× H100 with 1 sample per GPU (the released code requires exactly that), 480 GB host memory.
- **Run settings changed for the smoke test:** `--num-steps 300` (the fixed 200-step warmup, then 100 steps at lr 1e-4), one checkpoint at the end, `--load-worker 1`.
- **Data source:** `--datasets oxe:1.0`. It copies every `robotwin_train` setting but drops the check that requires 43 RoboTwin tasks.
- **WandB:** turned on by `zero_wam/so101_wandb_train.py`, because the preset hard-codes it off. Project `wamh3/zero-wam-so101`.
- **Data:** `pbvr__so101_test002` ("pick the orange block and put it in the box"), with front and wrist cameras at 224×288 each, placed side by side.
  - Train: export episode 168 (211 frames), paired with human demo `episode_173`.
  - Validation: export episode 84 (257 frames), paired with human demo `episode_085`.
  - The human demos come from `hf://buckets/nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo`, and both are confirmed good in its review.
- **Actions:** end-effector pose relative to the segment's first state goes in channels 0–6, and the absolute gripper in channel 28. Upstream's `LeRobotActionProcessor` does the mapping from `meta/action_transform.yaml`.
- **Latents:**
  - Robot: video stride 4, so 16 actions per latent frame, at 48×14×14×36.
  - Human: 12 fps at 320×448, giving 48×16×20×28, the same as the released HumanGen latents.
- **Code:** `zero_wam/so101_smoke_data.py` builds the data, `zero_wam/modal_so101_smoke.py` runs everything on Modal, and `zero_wam/so101_wandb_train.py` is the WandB launcher.

## Results

**Training** (WandB run `9vbcsvbs`): 300 steps at 3.7 s per step, about 18 minutes. There were no NaNs and no skipped optimizer steps, and the gradient norm stayed at 0.1–0.26, with one spike to 1.84 during warmup that settled straight away.

| Steps | Video | Action | Future-chunk |
|---|---|---|---|
| 0–24 | 0.315 | 0.0164 | 0.414 |
| 100–124 | 0.024 | 0.0043 | 0.185 |
| 275–299 | 0.012 | 0.0052 | 0.041 |

**Validation:** the same training step at learning rate 0, with the human video and the text always given (`drop_icl 0`, `droptext 0`), averaged over 20 steps × 8 GPUs.

| Checkpoint | Episode | Video | Action | Future-chunk | WandB |
|---|---|---|---|---|---|
| pretrain | train 168 | 0.365 | 0.0244 | 0.432 | `5utb89zw` |
| step 300 | train 168 | **0.012** | **0.0049** | **0.042** | `hnf7iehv` |
| pretrain | val 84 | 0.324 | 0.0217 | 0.416 | `5ew6o2jq` |
| step 300 | val 84 | 1.021 | 0.0546 | 1.085 | `ypkfcorm` |

## Takeaways

- The pipeline works end to end, and training is stable. On the training episode, video loss falls about 30×, action loss about 5× and future-chunk loss about 10×.
- On the held-out episode, loss gets about 3× worse than pretrain. That is expected after 300 steps on a single episode, and it shows the run memorised that one episode rather than learning the task. A real run needs many episodes and tasks.
- Action loss settles at about 0.004–0.005 within about 40 steps. It is measured at a random noise level each step, so part of that floor is noise.
- **Pairing gotcha:** the human-demo runs name episodes by `curated_episode_index`, not the export `episode_index`. They differ for 711 accepted episodes across 11 datasets. Pair through `data/so101_export/accepted.json`.
- **Differences from the paper's RoboTwin post-training:**
  - 8 GPUs at 1 sample per GPU, against 64 GPUs packing samples up to 160K tokens each.
  - No mix with general robot data or HumanGen data.
  - SO-101 runs at 30 Hz against RoboTwin's 50 Hz, so each latent frame covers 0.53 s instead of 0.32 s.
  - The model has not seen front and wrist cameras placed side by side before.

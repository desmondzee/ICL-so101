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

## Smoke test 2: short warmup and cosine decay (2026-10-02)

Smoke test 1's action loss stopped improving at about step 50 (≈0.004) and then drifted up about 9% per 100 steps as the learning rate climbed. Upstream's schedule is linear warmup then constant, with no decay, and 200 of the 300 steps were warmup. Smoke test 2 changes the following; everything else is the same: episodes, preset dropout, 8× H100, upstream files untouched.

- **Schedule:** 150 steps, with a 15-step warmup to 1e-4, then cosine decay to 0. `zero_wam/so101_wandb_train.py` does this when `SO101_WARMUP_STEPS` and `SO101_COSINE_TOTAL_STEPS` are set; it overrides `warmup_steps` and swaps in `wan_va.utils.warmup_constant_lambda`.
- **Gripper:** normalised over the dataset's own 1st–99th percentiles (−0.45 to 45.6) instead of the nominal 0–100, which left the gripper in about [−1, −0.1].
- **Launch:** `modal run --detach`, with training and validation chained in the Modal `pipeline` function. The first attempt (WandB `f4kpjcr7`) was cancelled at step 51 when the local client lost its connection to Modal; it was not a training failure.

**Training** (WandB run `gpetw71p`): 150 steps at 3.7 s per step. The gradient norm peaked at 0.35 (mean 0.11), and no steps were skipped.

| Steps | Action (mean ± sd) | Video | Future-chunk | LR at end |
|---|---|---|---|---|
| 0–24 | 0.0123 ± 0.0067 | 0.223 | 0.354 | 9.9e-5 |
| 50–74 | 0.0051 ± 0.0012 | 0.024 | 0.195 | 5.9e-5 |
| 100–124 | 0.0026 ± 0.0010 | 0.011 | 0.109 | 8.2e-6 |
| 125–149 | 0.0019 ± 0.0015 | 0.0095 | 0.097 | 0 |

**Validation:** same method as smoke test 1. "Per channel" is the action loss × 30/8, because upstream averages over all 30 action channels and only 8 are active.

| Checkpoint | Episode | Video | Action | Action per channel | Future-chunk | WandB |
|---|---|---|---|---|---|---|
| pretrain | train 168 | 0.341 | 0.0271 | 0.102 | 0.455 | `zl2p4lbk` |
| step 150 | train 168 | **0.0082** | **0.0015** | **0.0057** | 0.090 | `lhp4b6za` |
| pretrain | val 84 | 0.328 | 0.0232 | 0.087 | 0.416 | `rfm4i6it` |
| step 150 | val 84 | 1.184 | 0.0862 | 0.323 | 0.946 | `kanez7vl` |

**Smoke test 1 against smoke test 2, on the training episode:**

| | Smoke 1 (300 steps, constant LR) | Smoke 2 (150 steps, cosine) |
|---|---|---|
| Action | 0.0049 | **0.0015** (3.3× lower) |
| Video | 0.0118 | **0.0082** |
| Future-chunk | **0.042** | 0.090 |

- Decaying the learning rate removes the action-loss plateau. Action loss keeps falling to the last step, reaching 18× below pretrain against 5× in smoke test 1. The pretrain baseline moves slightly between the two runs because the gripper scaling changed.
- The future-chunk loss ends higher than in smoke test 1, because there were half as many steps and the learning rate was low for the last third of the run.
- The held-out episode gets worse again, and more than in smoke test 1, because the model fits the single training episode more tightly. This is still expected from one-episode overfitting.
- **Next:** a real run needs many episodes and a held-out split by task, and should keep a decaying schedule. Upstream has no decay, so it would come from the launcher override, or from training long enough that the paper's constant rate is fine.

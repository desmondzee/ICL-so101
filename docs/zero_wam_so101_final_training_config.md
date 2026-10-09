# SO-101 Zero-WAM final training configuration

This canonical file supersedes both `docs/zero_wam_so101_updated_config_pending_spatial.md` and the previous contents of this file.

## Objective

Post-train Zero-WAM on 1,109 paired SO-101 simulated robot/human demonstrations. Measure held-out loss and closed-loop task success on 50 episodes from five task-disjoint validation tasks before training and again from the loss-selected trained checkpoint.

This file specifies the intended run; training has **not** started. No Modal training job may resume until all launch gates pass.

## Final decisions

- Effective batch **8 → 64** using `gradient_accumulation_steps: 8` at per-GPU microbatch 1 on 8 H100s. Keep **4,000 optimizer steps**, 200-step warmup, AdamW LR `1e-4`.
- Robot spatial preprocessing is **resolved**: direct bilinear resize of each recorded 640×480 frame to H×W `[224, 288]` with `align_corners: false`; each camera is VAE-encoded separately to latent H×W `[14, 18]`, then the existing camera-latent concatenation applies. Evidence: released RoboTwin and RoboCOIN VAE reconstruction checks (`docs/zero_wam_spatial_preprocessing_tracker.md`).
- Checkpoint-selection validation: **human ICL video + detailed human-video text present, short target-task text absent**. Training retains short target-task text with **40%** dropout and human ICL bundle with **10%** dropout.
- Use **full robot episodes** — confirmed memory-feasible on the longest 64 under the measured FSDP policy (every-microstep reduce-scatter + `expandable_segments`); no temporal segmentation. **No length bucketing**; maintain task-balanced random sampling.
- Use **one host containing 8× H100s** for FSDP (topology measured below).
- Keep **absolute joint targets**, existing IFP weights, action normalization, robot 30 Hz → 15 FPS, and human ICL 12 FPS.
- Retain original 640×480 robot recordings as the immutable source.


## Frozen provenance

```yaml
model:
  repository: Robbyant-Research/zero-wam-pretrain
  revision: 7040c4195df216c900334ef62d5fdcf05c0601aa

code:
  zero_wam_revision: 08e2c4ae41e2b63573a299825cebe6753481407c

train_data:
  source: hf://buckets/akoniti/ICL-so101/sim_train_v1
  wrist_overlay: hf://buckets/akoniti/ICL-so101/train_v2
  episodes: 1109
  tasks: 61
  skill_families: 29

validation_data:
  source: hf://buckets/akoniti/ICL-so101/sim_val_v2
  wrist_overlay: hf://buckets/akoniti/ICL-so101/val_v3
  episodes: 50
  tasks: 5
  episodes_per_task: 10
  task_overlap_with_train: 0
```

## Robot observations

```yaml
robot_observations:
  cameras:
    - observation.images.front
    - observation.images.wrist
  camera_order: [front, wrist]
  recorded_resolution_per_camera: [480, 640]   # source H,W; originals stay unchanged
  resolution_per_camera: [224, 288]            # model input H,W
  latent_resolution_per_camera: [14, 18]       # Wan VAE spatial downsample x16
  spatial_preprocessing:
    method: direct_resize
    interpolation: bilinear
    align_corners: false
  source_fps: 30
  frame_stride: 2
  effective_model_video_fps: 15
  controls_per_latent_frame: 8
  wrist_absolute_near_plane_m: 0.02425
  camera_dropout:
    front: 0.0
    wrist: 0.0
```

The corrected wrist transform retains the camera mount and its shadow while removing direct mount pixels. Source MP4s stay unchanged; train, validation, and online rendering all use the identical direct bilinear resize to `[224, 288]`.

## Text and human-video conditioning

```yaml
target_task_text:
  source: episode instruction
  style: short imperative
  examples:
    - Put the red mug on the blue plate.
    - Put both mugs in the microwave.

human_icl:
  video_resolution: [320, 448]
  latent_resolution: [20, 28]
  spatial_preprocessing:
    method: direct_resize
    interpolation: bilinear
    align_corners: false
  video_fps: 12
  video_source: paired human.mp4
  text_source: source.json["human"]["prompt"]
  text_preprocessing:
    retain: integrated_multimodal_description action content
    remove:
      - reference-picture timestamp boilerplate
      - overall_soundscape
```

Human ICL geometry follows the inspected released human latent artifacts (`[320, 448]` → latent `[20, 28]`), despite the public Robotwin config's stale/contradictory width of 480. This supersedes the prior 320×480 provisional encoding; incompatible human latents must be regenerated.

The released model has two text routes:

```text
robot video/action tokens (sequence 0) -> short target-task text (sequence 0)
human ICL video tokens    (sequence 1) -> detailed human-video text (sequence 1)
robot video tokens                     -> human ICL video tokens via self-attention
```

Do not encode the short task instruction as the human latent's detailed text. Existing partial human latents made with the short instruction are provisional and must be regenerated.

### Training dropout

```yaml
conditioning_dropout:
  human_icl_bundle: 0.10
  target_task_text: 0.40
  robot_front_camera: 0.0
  robot_wrist_camera: 0.0

human_icl_bundle:
  components:
    - human_video_latent
    - detailed_human_video_text_embedding
  dropped_together: true
```

Approximate sample conditions:

| Human ICL bundle | Target task text | Probability |
|---|---|---:|
| Present | Present | 54% |
| Present | Dropped | 36% |
| Dropped | Present | 6% |
| Dropped | Dropped | 4% |

`human_icl_bundle` does not refer to the robot front or wrist cameras.

## Action interface

```yaml
action:
  model_dimension: 30
  active_channels:
    arm_joints: [0, 1, 2, 3, 4]
    gripper: 28
  inactive_channels: masked
  representation: absolute_joint_targets
  arm_units: degrees
  gripper_units: 0_to_100
  actions_per_vae_latent_frame: 8
  inference_latent_chunk_size: 2
  actions_per_inference_prediction: 16
  execution_rate_hz: 30
```

### Normalization

Use fixed physical actuator bounds, not train quantiles and never validation-derived statistics:

```yaml
normalization:
  method: physical_actuator_bounds
  q01:
    [-109.99987556, -100.00000051, -96.82987066,
     -94.99983758, -157.21087394, 0.0]
  q99:
    [109.99987556, 100.00000051, 96.82987066,
     94.99983758, 157.21087394, 100.0]
  processor_clip_range: [-2.0, 2.0]
```

The same statistics apply to train, validation loss, and closed-loop inference. Current conversion verification reports no meaningful clipping. The rejected train-quantile configuration made many validation wrist-roll and open-gripper commands unrepresentable.

## Objective and model components

```yaml
loss:
  total: video_flow_loss + action_flow_loss + weighted_ifp_loss

  video:
    coefficient: 1.0
    timestep_reweighting: true
    snr_shift: 5.0

  action:
    coefficient: 1.0
    timestep_reweighting: false
    snr_shift: 1.0

  ifp:
    enabled: true
    outer_coefficient: 1.0
    future_modules: 4
    future_stride: 2
    snr_shift: 10.0
    source_weights: [0.5, 0.25, 0.15, 0.1]

  noisy_robot_condition_probability: 0.5
```

In the pinned executable source, `0.1` is the **fourth/deepest** MCP/IFP module weight (`source_weights[3]`); the paper reports `0.15` — use the pinned source value and record the discrepancy. The per-module weighted sum is added directly to the loss with `outer_coefficient: 1.0`; `source_weights` scale each module's contribution — they are **not** a total IFP multiplier.

## Optimizer

```yaml
optimizer:
  type: AdamW
  learning_rate: 1.0e-4
  betas: [0.9, 0.95]
  epsilon: 1.0e-8
  weight_decay: 0.01
  fused: true

schedule:
  warmup_optimizer_steps: 200
  after_warmup: constant

precision:
  parameters_and_activations: bfloat16
  fsdp_gradient_reduce_dtype: float32
  loss_reductions: float32

gradient:
  max_norm: 1.0
  skip_optimizer_step_if_nonfinite: true
  skip_optimizer_step_above_norm: 20.0

ema:
  enabled: false
```

## Batch and sampling

```yaml
batch:
  per_gpu: 1
  gpu_count: 8
  gradient_accumulation_steps: 8
  effective_samples_per_optimizer_step: 64
  fsdp_reduce_scatter_each_microstep: true
  suppress_gradient_sync_during_accumulation: false

sampling:
  task_balanced: true
  choose_task: uniform_over_61_tasks
  choose_episode: uniform_within_selected_task
  length_bucketing: false
```

Sampling means: pick a task uniformly, then pick an episode uniformly within that task. The current flattened `DistributedSampler` is **not** compliant with this; implement and verify a deterministic distributed task-balanced sampler before launch.

```yaml
sequence_policy:
  preferred_unit: full_episode
  full_episodes_feasible: true              # measured on longest 64 under final FSDP policy
  allow_unapproved_temporal_segmentation: false
  final_artifact_loader_smoke_pending: true

training_robot_video_chunk_size:
  distribution: uniform_integer
  minimum: 1
  maximum: 4
  inherited_config:
    frame_chunk_size: 0        # marker -> trainer samples chunk_size uniformly 1..4
    max_frame_chunk_size: 4
```

The inherited training config carries `frame_chunk_size: 0` and `max_frame_chunk_size: 4`, so the trainer samples a random chunk size in 1–4 per item. This does **not** crop full episodes — it controls temporal grouping within the loss and the MCP/IFP future-frame shifts. It is distinct from `num_mcp_modules: 4`, `future_chunk_stride: 2`, and the fixed inference latent chunk size 2. The longest-64 benchmark inherited and exercised this random 1–4 behavior; do **not** switch it to a fixed value.

Attention is block-causal: permitted tokens attend bidirectionally within the current chunk, while attention between chunks remains causal; clean/noisy masks prevent access to clean targets from the same chunk. This lets Zero-WAM jointly generate a coherent short video/action segment in parallel while preserving closed-loop causality between predictions. Deployment uses two latent frames per prediction, corresponding to 16 controls or approximately 0.533 seconds at 30 Hz; random training chunks 1–4 include that deployed horizon while providing multi-scale temporal training.

The upstream ICL trainer requires per-GPU batch size 1. The approved effective batch is 64: 8 GPUs concurrently process one example each over 8 sequential gradient-accumulation rounds per optimizer step. At 4,000 optimizer steps this corresponds to 256,000 sampled examples (with replacement), **not** distinct episodes. Reduce-scatter must run after every microstep so accumulated gradients remain sharded; the upstream behavior that suppresses gradient synchronization on microsteps 1–7 OOMs on the second longest-episode microstep. Loss scaling remains `1/8`, and the optimizer still steps only after microstep 8.

Full-episode memory feasibility is **verified**: the longest-64 benchmark ran the full objective at final shapes (robot latent `[48,F,14,36]`, human ICL `[48,T,20,28]`) — 59 real episodes plus 5 exact-shape synthetic, making it a shape/memory test, not a loss-correctness test. Loader smoke on the final regenerated artifacts remains pending. Keep random prediction chunk sizes 1–4 (see below); no length bucketing; no temporal segmentation.

## Data mixture

```yaml
data_mixture:
  so101_simulated_icl: 1.0
```

This is a deliberate deviation from the paper's RoboTwin replay mixture because the complete Task-diverse VA and HumanGen robot-training corpora are not available as verified inputs for this run. Report potential forgetting explicitly.

## Run length, checkpoints, and cost gates

```yaml
run:
  maximum_optimizer_steps: 4000
  checkpoint_interval: 500
  checkpoints: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
  checkpoint_sft_validation:
    execution: inline_on_training_host
    gpu_count: 8
    pause_training: true
    resume_after_validation: true
  live_review_gate_step: 500
  review_gate_mode: monitor_while_run_remains_active
  continue_if_healthy: true
  stop_immediately_if_unhealthy: true
  resumable_state_required: [model, adamw_optimizer, lr_scheduler, optimizer_step,
                             sampler_rng_and_state]

modal:
  gpu_request: "H100:8"
  training_function_max_containers: 1
  world_size: 8
  hard_max_training_containers: 1
  hard_max_h100_gpus: 8
  prohibit_overlapping_apps: true
  durable_volume: zero-wam-so101-sim
  environment:
    PYTORCH_ALLOC_CONF: expandable_segments:True
```

Resumable checkpoint state must include model, AdamW optimizer state, LR scheduler, optimizer step, and sampler RNG/state; upstream optimizer state save is currently disabled — this is **launch-blocking** until implemented and verified.

At every 500-step checkpoint, all ranks synchronize and training pauses on the same `H100:8` host while deterministic SFT validation loss runs across eight GPUs. Validation performs no optimizer or scheduler updates and must preserve/restore training mode plus training RNG and sampler state before resuming. This inline phase does not launch an overlapping GPU app.

Step 500 is a **live review gate, not a mandatory termination**: the agent monitors the run through the step-500 checkpoint and inline SFT validation, then reviews training and validation loss, finite gradients/skips, throughput/cost, and loader/conditioning health. If anything is unhealthy the agent stops the active Modal run immediately; if healthy training resumes toward step 4,000. The remote logging/job must survive a local disconnect, and a stop command must remain available throughout. Deployment is a single `H100:8` host: the measured topology probe confirmed a single hostname with unique CUDA devices 0–7 and 64 MiB all-reduce of ~0.360 ms (FP32) / ~0.342 ms (BF16) — intra-host class bandwidth; `nvidia-smi topo -m` link labels were unavailable in the container. Enforce one training host/container and an eight-H100 resource ceiling at the Modal configuration level, not merely by spawning eight calls.

## Validation loss

Evaluate initialization and every retained checkpoint using one frozen segment/timestep/noise manifest:

```yaml
validation_loss:
  episodes: 50
  tasks: 5
  execution: inline_on_training_host_at_each_checkpoint
  distributed_world_size: 8
  updates_enabled: false
  learning_rate: 0.0
  human_icl_dropout: 0.0
  target_text_dropout: 1.0  # primary checkpoint-selection loss: short target text OFF
  human_icl_video: enabled
  human_icl_detailed_text: enabled
  fixed_segments: true
  fixed_diffusion_timesteps: true
  fixed_noise_seeds: true
  preserve_and_restore_training_rng_state: true
  preserve_and_restore_sampler_state: true
  report:
    - video_loss
    - action_loss
    - ifp_weighted_total
    - per_task_mean
    - macro_task_mean
    - standard_deviation
    - median
    - bootstrap_95ci
```

Select the final checkpoint by the lowest macro validation IFP-weighted total **under deployment-matched conditioning** (human-video latent and detailed human text enabled; separate short target-task text disabled), among finite, stable checkpoints. Ensure `target_text_dropout: 1.0` actually produces the released-faithful no-short-text condition rather than another conditioning shortcut. Do not select using repeated closed-loop validation rollouts. Optionally report both-text-present loss separately as a diagnostic; never mix its values into primary checkpoint selection.

## Closed-loop evaluation

Run the exact same 50 validation episodes before training and after checkpoint selection:

```yaml
closed_loop:
  trials_per_checkpoint: 50
  simulator_seed: from_validation_index
  model_seed: frozen_per_episode
  target_task_text_at_icl_inference: disabled
  human_icl_video: enabled
  human_icl_text: enabled
  video_denoising_steps: 50
  action_denoising_steps: 50
  icl_cfg_scale: 5.0
  action_cfg_scale: 1.0
  inference_chunk_size: 2
  success_source: info["success"]
```

Report:

- Overall success / 50 with Wilson 95% interval
- Macro-average across five tasks
- Per-task success / 10
- Paired fail→success and success→fail transitions
- Exact McNemar test
- Runtime failures as failures in the headline intent-to-evaluate metric

### Conditioning controls

After the primary released-faithful comparison, evaluate when feasible:

| Condition | Human video | Human detailed text | Target task text |
|---|---:|---:|---:|
| Released-faithful ICL | yes | yes | no |
| Strict video-only | yes | empty | no |
| Human-text-only | no | yes | no |
| Target-language-only | no | no | yes |
| Unconditioned | no | no | no |

Strict video-only and human-text-only require explicit runtime support because the released server normally couples the human latent with its stored text embedding.

## WandB

```yaml
wandb:
  project: zero-wam-so101
  group: sim-train-v1
  logging_rank: 0
  primary_step_axis: optimizer_step
  resume: must
  run_id_persisted_in_checkpoint: true
  required_job_types:
    - data-prep
    - init-validation-loss
    - init-rollouts
    - train
    - checkpoint-validation-loss
    - final-rollouts
  metric_namespaces:
    train:
      - train/video_loss
      - train/action_loss
      - train/ifp_module_0_loss
      - train/ifp_module_1_loss
      - train/ifp_module_2_loss
      - train/ifp_module_3_loss
      - train/ifp_weighted_total
      - train/total_loss
      - train/learning_rate
      - train/gradient_norm
      - train/skipped_optimizer_steps
      - train/samples_per_second
      - train/tokens_per_second
      - train/gpu_memory_allocated_gib
      - train/gpu_memory_reserved_gib
      - train/episode_exposure_total_draws
      - train/episode_exposure_unique_seen
      - train/episode_exposure_unseen
      - train/episode_exposure_min
      - train/episode_exposure_max
      - train/episode_exposure_mean
      - train/episode_exposure_p95
    validation:
      - validation/video_loss
      - validation/action_loss
      - validation/ifp_weighted_total
      - validation/macro_task_mean
      - validation/standard_deviation
      - validation/median
      - validation/bootstrap_95ci_low
      - validation/bootstrap_95ci_high
    rollout:
      - rollout/overall_success_rate
      - rollout/macro_task_success_rate
      - rollout/per_task_success_rate
      - rollout/runtime_failure_count
```

Only rank 0 writes WandB metrics. All rank-local values must be reduced or gathered explicitly before logging; never emit eight duplicate series. Training and inline SFT-validation metrics share `optimizer_step` as their x-axis, so initialization validation is step 0 and checkpoint validation is logged at steps 500 through 4,000. A resumed job must reuse the persisted WandB run ID with `resume="must"` and continue the same step axis without overwriting prior history. Closed-loop rollouts use separate initialization/final job types but record the evaluated checkpoint step and hash.

Log configuration and provenance before compute, then at minimum:

- Video, action, individual IFP, weighted IFP and total losses
- Learning rate and gradient norm
- Skipped optimizer steps and NaN/Inf counts
- Samples, tasks and tokens per update
- Exact cumulative episode-level exposure counts as a checkpoint artifact, plus compact
  WandB summaries (total draws, unique/unseen episodes, min/max/mean/median/p95). Keys
  include the dataset source, task and episode ID so future data-mixture changes remain
  comparable; do not create one WandB time series per episode.
- Throughput and GPU memory
- Validation metrics per checkpoint and task
- Checkpoint and manifest hashes
- Modal app/call IDs and actual GPU topology
- Closed-loop success metrics and artifact references

WandB is the canonical experiment dashboard, not the only durable record. Validation manifests, per-episode losses, rollout outcomes, checkpoint metadata, and configuration hashes must also be written to the Modal volume so results remain auditable if WandB logging is interrupted.

## Spatial decision provenance

Resolved per `docs/zero_wam_spatial_preprocessing_tracker.md` with durable reports `/sim/reports/spatial_decode_robotwin_ep42.json` and `/sim/reports/spatial_decode_robocoin_high345.json` on the `zero-wam-so101-sim` volume.

## Mandatory launch gates

- [x] **Robot image preprocessing resolved:** direct bilinear resize 640×480 → `[224, 288]` (`align_corners: false`), latent `[14, 18]` per camera then camera-latent concatenation; approved.
- [ ] Re-encode all robot latents at 224×288 and all human latents at 320×448; inventory durable latents on CPU and encode only missing/invalid outputs.
- [ ] After re-encoding, verify train, validation and online rendering use identical camera preprocessing.
- [ ] Keep source robot videos at 640×480 and verify 30 Hz camera/action timestamps, 15 FPS sampled robot observations, VAE boundaries and 8 actions/latent frame.
- [ ] Enforce one training host/container with eight GPUs; ensure other Modal functions cannot create overlapping H100 allocations.
- [ ] Confirm no other Modal app is active.
- [ ] Implement and verify the deterministic distributed task-balanced sampler (uniform task, uniform episode); current flattened `DistributedSampler` is not compliant.
- [ ] Implement resumable checkpoint state (model + AdamW optimizer + LR scheduler + optimizer step + sampler RNG/state); upstream optimizer save is currently disabled — launch-blocking.
- [x] Verify one-sample-per-GPU, 8-rank FSDP and accumulation of 8 on the 64 longest episodes: successful only with reduce-scatter after every microstep and expandable CUDA segments; leave length bucketing disabled.
- [ ] Pass a short representative task-balanced **unprofiled throughput benchmark**, reporting warm steps/s, samples/s, utilization, and projected H100-hours to steps 500 and 4000. The longest-64 result (0.182 samples/s) is worst-case/cold and cannot alone estimate cost.
- [ ] Implement the verified every-microstep reduce-scatter policy in the production trainer and regression-test gradient equivalence before launch.
- [ ] Verify checkpoint-selection loss disables the short task text but keeps the human-video latent and detailed human text.
- [ ] Replace provisional human latent text embeddings with detailed human-video descriptions.
- [ ] Verify exactly 1,109 train and 50 validation episodes, each with two robot latents and one human latent.
- [ ] Pass deterministic upstream loader smoke with active action mask `[0,1,2,3,4,28]`.
- [ ] Run and freeze initialization validation loss.
- [ ] Run and freeze all 50 initialization closed-loop rollouts.
- [ ] Record WandB run IDs and frozen configuration hash.
- [ ] Obtain explicit approval before launching optimizer training.

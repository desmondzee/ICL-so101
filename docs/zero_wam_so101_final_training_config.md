# SO-101 Zero-WAM final training configuration

## Objective

Post-train Zero-WAM on 1,109 paired SO-101 simulated robot/human demonstrations. Measure held-out loss and closed-loop task success on 50 episodes from five task-disjoint validation tasks before training and again from the loss-selected trained checkpoint.

This file specifies the intended run. It does not claim that training has started. No Modal job may resume until the latent inventory, loader smoke, initialization loss, and initialization rollouts pass.

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
  resolution_per_camera: [256, 256]
  source_fps: 30
  frame_stride: 2
  controls_per_latent_frame: 8
  wrist_absolute_near_plane_m: 0.02425
  camera_dropout:
    front: 0.0
    wrist: 0.0
```

The corrected wrist transform retains the camera mount and its shadow while removing direct mount pixels. Use the same transform for training videos, validation-loss videos, and online validation rendering.

## Text and human-video conditioning

```yaml
target_task_text:
  source: episode instruction
  style: short imperative
  examples:
    - Put the red mug on the blue plate.
    - Put both mugs in the microwave.

human_icl:
  video_resolution: [320, 480]
  video_fps: 12
  video_source: paired human.mp4
  text_source: source.json["human"]["prompt"]
  text_preprocessing:
    retain: integrated_multimodal_description action content
    remove:
      - reference-picture timestamp boilerplate
      - overall_soundscape
```

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
  action_horizon: 8
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

The pinned executable source uses final IFP weight `0.1`; the paper reports `0.15`. Use the pinned source value and record the discrepancy.

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
  gradient_accumulation_steps: 1
  effective_samples_per_optimizer_step: 8

sampling:
  task_balanced: true
  choose_task: uniform_over_61_tasks
  choose_episode: uniform_within_selected_task

training_robot_video_chunk_size:
  distribution: uniform_integer
  minimum: 1
  maximum: 4
```

The upstream ICL trainer requires per-GPU batch size 1. Do not increase gradient accumulation merely to imitate an image-model batch size. First log token counts, gradient variance, memory, throughput, and task composition. Any batch/accumulation change requires a new frozen configuration and a step-budget adjustment.

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
  first_review_gate_step: 500

modal:
  gpu: H100
  gpu_count: 8
  hard_max_gpu_containers: 8
  prohibit_overlapping_apps: true
  durable_volume: zero-wam-so101-sim
```

The run must stop for review at step 500. Continuing requires acceptable training/validation loss, finite gradients, expected throughput/cost, and no data-loader or conditioning failure. The eight-GPU limit must be enforced in Modal configuration, not only by spawning eight calls.

## Validation loss

Evaluate initialization and every retained checkpoint using one frozen segment/timestep/noise manifest:

```yaml
validation_loss:
  episodes: 50
  tasks: 5
  updates_enabled: false
  learning_rate: 0.0
  human_icl_dropout: 0.0
  target_text_dropout: 0.0
  fixed_segments: true
  fixed_diffusion_timesteps: true
  fixed_noise_seeds: true
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

Select the final checkpoint by the lowest macro validation IFP-weighted total among finite, stable checkpoints. Do not select it using repeated closed-loop validation rollouts.

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
  required_job_types:
    - data-prep
    - init-validation-loss
    - init-rollouts
    - train
    - checkpoint-validation-loss
    - final-rollouts
```

Log configuration and provenance before compute, then at minimum:

- Video, action, individual IFP, weighted IFP and total losses
- Learning rate and gradient norm
- Skipped optimizer steps and NaN/Inf counts
- Samples, tasks and tokens per update
- Throughput and GPU memory
- Validation metrics per checkpoint and task
- Checkpoint and manifest hashes
- Modal app/call IDs and actual GPU topology
- Closed-loop success metrics and artifact references

## Mandatory launch gates

- [ ] Enforce Modal `max_containers=8` on every H100 function.
- [ ] Confirm no other Modal app is active.
- [ ] Replace provisional human latent text embeddings with detailed human-video descriptions.
- [ ] Inventory existing durable latents on CPU and encode only missing/invalid outputs.
- [ ] Verify exactly 1,109 train and 50 validation episodes, each with two robot latents and one human latent.
- [ ] Pass deterministic upstream loader smoke with active action mask `[0,1,2,3,4,28]`.
- [ ] Run and freeze initialization validation loss.
- [ ] Run and freeze all 50 initialization closed-loop rollouts.
- [ ] Record WandB run IDs and frozen configuration hash.
- [ ] Obtain explicit approval before launching optimizer training.

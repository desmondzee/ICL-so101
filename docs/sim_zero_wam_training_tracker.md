# Zero-WAM simulated-data training and validation tracker

## Goal

Post-train Zero-WAM on the 1,109 paired SO-101 simulated training episodes, track held-out loss on the 50 paired simulated validation episodes, and measure closed-loop validation-task success **before any optimizer update** and again from the selected trained checkpoint. The comparison must use the same scenes, human-video prompts, action interface, model-noise seeds, horizons, and success predicates.

This tracker is the execution record. A lower validation loss is not task success; closed-loop success rate is the primary outcome. Do not start the training job until the initial-checkpoint rollout baseline and the data/loader gates below are complete.

## Frozen inputs

| Role | Source | Count | Notes |
|---|---|---:|---|
| Training pairs | `hf://buckets/akoniti/ICL-so101/sim_train_v1` | 1,109 episodes, 61 tasks, 29 families | Front video, robot data, paired human video and metadata remain immutable. |
| Corrected training wrist overlay | `hf://buckets/akoniti/ICL-so101/train_v2` | 1,109/1,109 episodes | `absolute-near-plane-v3`; 24.25 mm absolute near plane; mount shadow retained; full local and remote verification passed. Replace only each train episode's `robot_wrist.mp4`. |
| Validation pairs | `hf://buckets/akoniti/ICL-so101/sim_val_v2` | 50 episodes, 5 held-out tasks | Ten episodes each: `mug_on_plate`, `mugs_in_microwave`, `pan_on_stove`, `sort_blocks`, `stack_bowls`. |
| Corrected validation wrist overlay | `hf://buckets/akoniti/ICL-so101/val_v3` | 50/50 episodes | `absolute-near-plane-v3`; `wrist_v3.json` per episode; verified 50/50, frozen, uploaded without deletion, and hash-verified by remote readback. Replace only each val episode's `robot_wrist.mp4`. |
| Initialization checkpoint | `Robbyant-Research/zero-wam-pretrain` revision `7040c4195df216c900334ef62d5fdcf05c0601aa` | one frozen checkpoint | Existing SO-101 smoke training starts from this exact revision. Reconfirm exact-load audit before baseline. |
| Upstream training code | `third_party/Zero-WAM` revision `08e2c4ae41e2b63573a299825cebe6753481407c` | pinned submodule | Initialize before implementation; do not silently update or patch upstream. Keep adapters in this repository. |

All generated manifests must record complete input hashes, repository revision/diff state, model revision, dependency versions, camera keys, action transform/statistics, simulator/model hashes, and random seeds. Never mutate a frozen input release.

## Required experiment contract

### Data and model interface

- Robot observations: front and corrected wrist views, in that order, with geometry/resampling matching the selected Zero-WAM config. Apply the same 24.25 mm absolute-near-plane transform to validation wrist data and online validation rendering; do not train on corrected wrist frames and evaluate loss on the original near-plane artifact.
- Conditioning: the episode's paired `human.mp4`; text-conditioning/dropout behavior must be recorded separately for train and evaluation.
- Robot cadence: source is 30 Hz. The existing smoke path uses stride 4 and 16 controls per latent frame; verify this against the pinned upstream loader before freezing the full manifest.
- Action target: select one executable SO-101 representation before conversion. Prefer direct five arm joints plus gripper if the pinned processor can represent the packaged absolute LeRobot-degree targets without an unverified reference-frame conversion. Otherwise use the audited EE representation. Record active Zero-WAM channels and inverse mapping explicitly.
- Closed-loop execution must feed measured simulator state and actually executed actions back into model history. Predicted-target history is invalid.
- Simulator action commands must use `ValEnv.from_lerobot_joints` or an equivalently tested adapter. Any rate/range clipping must be reported, not silently absorbed.

### Validation-loss protocol

Validation loss uses all 50 `sim_val_v2` paired trajectories and never updates weights. For every evaluated checkpoint:

1. Force `learning_rate=0`, disable weight decay through the zero LR, and keep the model in the same training-step loss path used by the existing smoke evaluation.
2. Force human-video and task-text conditioning on (`drop_icl=0`, `droptext_target=0`) unless the final training contract intentionally excludes one; record the choice.
3. Use a fixed, checkpoint-independent manifest of sampled segments, diffusion timesteps, and noise seeds. Randomly resampling per checkpoint is not an acceptable comparison.
4. Report per-episode and aggregate video loss, action loss, future-chunk/MCP-weighted total, sample count, mean, standard deviation, median, and bootstrap 95% confidence interval.
5. Report losses per task as well as the macro-average across the five tasks so one long task cannot dominate.
6. Evaluate at initialization, every retained training checkpoint, and the selected final checkpoint. Select the final checkpoint by the frozen rule below—not by post-hoc rollout inspection.

### Closed-loop rollout protocol

The primary baseline/final comparison consists of the same 50 packaged validation episodes. For each episode:

- Load its task with `sim.val.tasks.load(task)` and reset using the exact seed in `sim_val_v2/index.json`/`source.json`.
- Condition on that episode's paired `human.mp4` and instruction.
- Use one frozen model-noise seed per episode, derived and stored in the evaluation manifest. Seed Python, NumPy, Torch and model denoising explicitly.
- Use identical initial state, policy seed, cameras, 24.25 mm absolute near plane, action mapping, normalization, denoising steps, control cadence, safety limits and maximum horizon before and after training.
- Read success from `info["success"]`; do not substitute distance, grasp or object-motion proxies.
- Save `trial.json`, action/history JSONL, runtime/checkpoint provenance, front and wrist rollout videos, contact sheet, final state, success step, and failure reason.
- Treat runtime errors/timeouts as failures in the headline intent-to-evaluate rate, while also reporting completed-rollout success separately.

Primary metrics:

- Overall success: successful episodes / 50, with Wilson 95% confidence interval.
- Macro task success: mean of the five per-task success rates.
- Per-task success: successes / 10.
- Paired pre/post transition table: fail→fail, fail→success, success→fail, success→success.
- Paired significance: exact McNemar test on discordant episodes.
- Secondary diagnostics: horizon, clipping fractions, inference/runtime failures, grasp/contact/task-specific telemetry. These cannot replace success.

If stochastic repeatability remains material after explicit seeding, freeze a secondary three-model-seed protocol for every episode and run all 150 trials for both checkpoints; do not repeat only favorable episodes.

### Checkpoint-selection rule

Before training, freeze one rule such as: lowest macro validation MCP-weighted total among checkpoints with finite losses and no training-instability gate failure. Closed-loop validation success is measured at initialization and only after selection; do not use validation rollouts repeatedly to tune checkpoints unless they are explicitly reclassified as development data and a new untouched test set is created.

## Training configuration table

Complete the **Frozen SO-101 run** column and attach a source/config hash before launching the initialization baseline or optimizer. `TBD` is a blocker, not an invitation to inherit an undocumented default. The paper reports some pretraining and RoboTwin post-training settings but does not specify every optimizer/runtime field; unresolved values must be read from the pinned `08e2c4a` source after initializing the submodule.

| Field | Paper / released reference | Existing SO-101 smoke | Frozen SO-101 run | Status / rationale |
|---|---|---|---|---|
| Initialization checkpoint | Zero-WAM checkpoint pretrained on Task-diverse VA + HumanGen | `zero-wam-pretrain` revision `7040c419…` | `Robbyant-Research/zero-wam-pretrain@7040c419…` | Proposed; exact-load audit required. |
| Upstream config | RoboTwin post-training configuration | `robotwin_train`, copied through `oxe:1.0` | `robotwin_train` plus hashed local adapter overrides | Proposed; enumerate every mutation. |
| Trainable modules | Paper trains causal video/action model with IFP; exact frozen-module list not stated | Pinned trainer calls `transformer.requires_grad_(True)` after FSDP setup | Entire ICL video/action Transformer including four MCP/IFP modules | Source-audited; record realized trainable/total parameter counts at launch. |
| Optimizer | AdamW | Inherits upstream | AdamW | Proposed. |
| Peak learning rate | `1e-4` | `1e-4` in smoke | `1e-4` | Frozen to paper and pinned preset. |
| Weight decay | `0.01` | Inherits upstream | `0.01` unless source audit contradicts | Proposed. |
| Adam betas / epsilon | Not reported in paper | Pinned preset/source: `(0.9, 0.95)`, `1e-8` | `(0.9, 0.95)`, `1e-8` | Source-audited. |
| LR warmup | Not reported in paper | Pinned preset uses 200 steps; second smoke used 15/150 | 200 optimizer steps | Frozen to pinned post-training preset. |
| LR schedule / minimum LR | Not reported; released code uses warmup then constant | Warmup-constant in smoke 1; optional cosine in smoke 2 | Linear warmup then constant `1e-4`; no minimum/decay | Frozen source-faithful schedule. |
| Total optimizer steps | RoboTwin: 4,000 | Smoke: 300 and 150 | 4,000 | Frozen to paper post-training length; checkpoint every 500. |
| GPU topology | RoboTwin: 64 GPUs | 8×H100, one sample/GPU smoke path | Modal 8×H100, FSDP, world size 8 | Frozen to tested available path; validate allocation before launch. |
| Per-GPU token cap | Up to 160K packed tokens | Inherits `robotwin_train` | TBD from pinned config | Blocker; record observed peak and packing efficiency. |
| Effective batch / accumulation | Variable packed samples; paper does not report a fixed sample batch | Pinned preset: batch 1/rank, accumulation 1 | 8 samples/optimizer step (1 × 8 ranks), accumulation 1 | Frozen; log realized packed tokens and task IDs. |
| Precision / loss scaling | Not reported in paper | Pinned shared config/FSDP use bfloat16; losses cast to float32 | BF16 params/activations, FP32 loss reductions, fused AdamW | Source-audited; record optimizer-state dtype at launch. |
| Gradient clipping | Not reported | Pinned preset clips global norm to `1.0`, skips optimizer step above `20.0` or non-finite | max norm `1.0`, skip threshold `20×` | Frozen source-faithful behavior. |
| EMA | Not reported | No EMA in pinned trainer | None | Source-audited. |
| Task sampling | Task-balanced construction is a central paper contribution | One task in smoke | Task-balanced across 61 train tasks | Proposed; specify with/without replacement and epoch semantics. |
| Data-family mixture | Pretrain Task-diverse VA:HumanGen = `1:5`; RoboTwin post-train Task-diverse VA:HumanGen:RoboTwin = `2:10:3` | SO-101-only smoke (`oxe:1.0`) | SO-101 simulated ICL only, weight `1.0` | Frozen deliberate deviation: replay corpora are unavailable in this workspace. Interpret forgetting risk explicitly. |
| Non-ICL language dropout | `0.1` | Inherits preset | N/A if every SO-101 sample is ICL; otherwise `0.1` | Freeze with mixture decision. |
| ICL human-video latent dropout | `0.1` | Pinned `drop_icl=0.1` | `0.1` | Frozen; source-audited. |
| ICL language-instruction dropout | `0.4` (raised from Wan default `0.1`) | Pinned `droptext_target=0.4`; dataset replaces target text embedding with empty embedding | `0.4` | Frozen; evaluation overrides both conditioning dropouts to zero. |
| Other conditioning dropout | Not reported | TBD from preset | TBD | Audit robot-history, state/action-history, camera/view and future-target masking separately. |
| Training evaluation dropout | N/A | LR-zero eval sets `drop_icl=0`, `droptext_target=0` | Human and text dropout both `0` | Proposed for deterministic checkpoint comparison. |
| Video flow-matching loss weight | Video loss is a primary term; scalar not separately reported | Pinned trainer sums mean video loss directly | `1.0` implicit | Source-audited. |
| Action loss weight `lambda_a` | Present in Eq. 19–20; numeric value not reported | Pinned trainer sums masked mean action loss directly | `1.0` implicit | Source-audited. |
| IFP loss weight `lambda_ifp` | Present in Eq. 20; numeric value not reported | Pinned trainer sums weighted MCP losses directly | `1.0` outer coefficient | Source-audited; per-depth weights below. |
| IFP future chunks | `K=4`, stride `s=2` | Inherits upstream ICL model | `K=4`, `s=2` reference | Proposed; verify checkpoint/config compatibility. |
| IFP per-future weights | Paper: `(0.5, 0.25, 0.15, 0.15)` | Pinned source: `(0.5, 0.25, 0.15, 0.1)` | `(0.5, 0.25, 0.15, 0.1)` | Freeze to executable pinned source; record paper/source discrepancy. |
| Training robot-video chunk size | Pretraining samples uniformly/randomly from 1–4 | Pinned source samples integer uniformly from 1–4 each batch | Uniform integer 1–4 | Frozen source-faithful. |
| Inference chunk size | Fixed at 2 | Existing worker is config-dependent | `2` unless checkpoint runtime requires otherwise | Freeze identically for pre/post rollouts. |
| Video CFG | Language mode `5`; ICL mode disables language | Existing workers vary by mode | ICL mode; language disabled as in paper | Proposed; verify empty-text behavior and runtime evidence. |
| ICL CFG | `5` | Existing Zero-WAM workers use `5` | `5` | Proposed. |
| Action CFG | `1.0` | Existing config-dependent | `1.0` | Proposed. |
| Video/action denoising steps | Not reported in paper | Demo preset `5/10`; RoboTwin preset `50/50` | 50 video / 50 action for both pre/post rollout | Frozen high-fidelity comparison; record runtime. |
| Robot frame stride / control cadence | Not reported as a raw-frame stride | Demo preset has 8 actions/latent frame | Sample robot video every 2 source frames at 30 Hz; VAE temporal stride 4 gives 8 controls/latent frame | Frozen pending loader gate. |
| Camera order / resolution | Paper uses encoded robot video | SO-101 demo preset: top+wrist, 256×256 each | front then corrected wrist, each resized to 256×256 | Frozen to pretrain demo geometry. |
| Action representation / channels | Shared 30-channel space | Demo preset uses channels 0–4 + gripper 28 with degree-like absolute stats | Absolute LeRobot-degree arm joints in channels 0–4; absolute 0–100 gripper in 28 | Frozen direct executable mapping; gate through replay and processor tests. |
| Normalization | Not reported numerically | Processor supports collection stats | Per-channel train-only q01/q99 over all 1,109 action rows; clip normalized values to `[-2,2]` | Frozen method; compute/hash exact values and audit val saturation. |
| Checkpoint interval / retention | Not reported | Smoke saves final checkpoint | Every 500 steps; retain 500–4000 plus final | Frozen for eight candidate checkpoints. |
| Validation-loss interval | Not reported | Smoke evaluates initialization/final separately | Initialization + every retained checkpoint + selected final | Proposed. Fixed segment/noise manifest required. |
| Validation-loss samples | Not reported | 20 steps × 8 ranks for one episode | TBD fixed segments per all 50 episodes | Choose for adequate per-task uncertainty before launch. |
| Early stopping / instability gates | Not reported | No general full-run rule | TBD before training | Include NaN/Inf, skipped-step, gradient and loss gates. |
| Logging | Not specified | WandB + durable JSONL | WandB + append-only JSONL + immutable run summary | Proposed; record run IDs and artifact hashes. |
| Random seeds | RoboTwin reports 3 evaluation seeds | Smoke/eval paths vary | Data order, diffusion noise, model init, rollout and simulator seeds all frozen | Required for paired comparisons. |

### Paper-derived design implications

- The paper's RoboTwin result is **not** a pure seen-task fine-tune: it retains Task-diverse VA and HumanGen replay at a `2:10:3` ratio with RoboTwin. Training solely on our 1,109 samples is a deliberate deviation and must be labeled as such.
- Human-video conditioning is deliberately regularized: drop the video 10% of ICL samples and language 40% of ICL samples. Evaluation uses neither dropout.
- The action Transformer does not directly attend to the human video in the paper; its action loss remains language-conditioned while the video branch absorbs human-video semantics. The implementation/config audit must verify this behavior rather than assuming action loss directly measures video following.
- IFP is removed at inference but materially improves unseen-task success in the paper; its loss configuration and future-target construction are release-critical training fields.
- The paper evaluates RoboTwin with 100 rollouts per task for each of three seeds and reports mean/standard deviation. Our primary 50 paired episodes are smaller, so confidence intervals and the predefined repeatability escalation remain necessary.

## Acceptance gates

### A. Data conversion and leakage

- [x] Initialize/pin `third_party/Zero-WAM` and record the exact revision and clean/dirty state —
      pinned at `08e2c4ae41e2b63573a299825cebe6753481407c`, clean (2026-10-08).
- [x] Prove all 1,109 train IDs resolve to immutable front/human/data files plus the corrected `train_v2` wrist overlay; verify hashes against both release inventories —
      `zero_wam.sim_data verify` checked all 1,109 episodes: front/human media hashes equal
      `sim_train_v1` bytes, wrist hashes equal `train_v2` overlay bytes; source index sha256
      `618d062b…0e64`, overlay index `e9734148…61c858`, overlay inventory `6eda5dc6…188793`
      recorded in `data/zero_wam_sim/audit.json` (sha256 `4f8b9b0b…774b`).
- [x] Prove all 50 validation IDs resolve to robot data, front/wrist videos, human video, source seed and task metadata —
      same verify pass covered all 50 `sim_val_v2` episodes against `val_v3` overlays; seeds are
      carried per episode in index/`wrist_v3.json` and checked there during the overlay build.
- [x] Build and verify a 50-episode corrected validation-wrist overlay with the same
      `absolute-near-plane-v3` transform; retain mount shadows and keep `sim_val_v2` immutable.
      **Done 2026-10-08**: local output `data/hf_bucket_ICL-so101/val_v3`, 50/50 episodes rebuilt
      (`rebuild-val --workers 3 --encoder nvenc`, 4m20s, 0 failed), `verify-val`
      checked=ok=50 (frames == parquet rows, h264 640x480@30 yuv420p, all hashes match),
      `freeze-val` gated on 50/50 then wrote `index.json` sha256 `6e335778…437fc`,
      `README.md` `60541245…c1746`, `inventory.json` `242b0d4e…ada27` (102 files,
      158,887,416 bytes, self-excluding). Every episode meta records
      `transform_version=absolute-near-plane-v3`, `near_meters=0.02425`, `geom_group=2`,
      encoder `nvenc`. Per-task frame-0 MuJoCo segmentation: mount geom renders **0 direct
      pixels** and stays in group 2 for all five tasks; audit sheets at
      `/tmp/sim_val_wrist_v3_audit/` (temporary, not in release) show no circular aperture,
      textured corners, gray jaws/task objects, and no collision-group pixels (near-pure-red
      identical to packaged: 0.00–0.31%).
- [x] Confirm the five validation task IDs are absent from all 61 simulated training task IDs; save the machine-readable leakage report —
      leakage `{ok: true, overlap: [], train_tasks: 61, val_tasks: 5}` recorded in
      `data/zero_wam_sim/audit.json`; conversion aborts on any overlap.
- [x] Choose and document action channels, units, reference frame, gripper mapping, camera ordering, resize/crop, cadence, segment boundaries and normalization population —
      `meta/action_transform.yaml`: packaged `action[0:5]` absolute joint degrees →
      `action.hand.position` channels 0–4; gripper `action[5]` (0–100) →
      `action.effector.position` channel 28; `absolute_value`+`use_absolute`+`format: joint`;
      the other 24 model channels stay masked/zero (verified mask == [0,1,2,3,4,28] on every
      checked task). Cameras: `observation.images.front` then `observation.images.wrist`, source
      bytes hard-linked (no resize/crop/re-encode). Cadence 30 Hz; one `action_config` segment
      per whole episode. **Normalization = fixed physical actuator bounds** (`source_method:
      physical_actuator_bounds`, method `physical_bounds`, loader-facing `method: abs`) derived
      from `sim/val/env.py::ValEnv.from_lerobot_joints` target limits mapped to packaged units
      (model `038b7983…13b1`): arm ±[110.00, 100.00, 96.83, 95.00, 157.21] deg, gripper 0–100 —
      frozen constants `ACTION_Q01/ACTION_Q99` in `zero_wam/sim_data.py`, re-derivable via
      `python -m zero_wam.sim_data bounds` (live-env diff 4.0e-6 deg). **Train-only q01/q99
      quantiles were rejected**: they made val executable commands unrepresentable — 8,845 val
      wrist-roll rows hit the loader's ±2 clip, gripper q99 ≈32.9 vs physical open=100, and up
      to 42.6% of val rows normalized outside [-1,1]. Physical bounds use the full actuator
      range, so outside [-1,1] and ±2 clips are both **zero** (0.02 deg tolerance covers
      pre-clamp recording noise: strict count is 114,674+497 train and 7,304 val rows sitting
      ≤0.0098° above the bound — reported as `strict_outside_1`, not gated). Utilization
      (fraction of physical range), train: pan 12.7–86.6%, lift 0.5–71.7%, elbow 24.3–96.5%,
      wrist_flex 81.2–100%, wrist_roll 0–99.8%, gripper 0.24–33.6%; val: pan 16.5–86.0%,
      lift 0–73.4%, elbow 22.3–100%, wrist_flex 33.5–100%, wrist_roll 22.0–80.5%,
      gripper 0.24–45.6%. Stats file sha256 `71a5e79e…fb87` (identical bytes both splits);
      canonical sorted-JSON hash `15c70fb9…2f91`.
- [x] Convert train and validation data to the pinned Zero-WAM loader format without copying mutable paths into manifests —
      `python -m zero_wam.sim_data build --bucket-root data/hf_bucket_ICL-so101
      --output-root data/zero_wam_sim --workers 8` (44m48s incl. verify): 61 v2.1 task roots /
      1,109 episodes (train), 5 / 50 (val); per-task `meta/{info,tasks,episodes,
      episodes_stats}` + `data/chunk-000` + `videos/chunk-000`; collection `meta/`,
      `icl_manifest.json` (1,109+50 samples resolving through the pinned loader's
      `_episode_key_without_interval` — verified 100%), `human_data/so101/run_<split>/samples/
      <task>__<episode>/generated_video.mp4`; built-in verify checked=1,159 ok. Output 5.3 GB.
      Latent paths are collection-relative, not host-absolute. Metadata refresh: re-running
      `build` skips `.build_ok` task roots (no ffprobes) and rewrites collection
      meta/stats/manifest/audit; full `verify` (15m42s) then re-validates every episode
      against source — checked=1,159 ok, zero tolerated bound violations.
- [ ] Encode robot and human latents; verify source-to-latent ID coverage, shapes, frame IDs, text embeddings and hashes for 1,109 train + 50 validation episodes.
- [ ] Loader smoke: retrieve deterministic batches spanning multiple tasks, short/long episodes and legacy/new visual configurations; decode IDs back to source records.
- [ ] Overfit a tiny multi-task subset and confirm finite gradients and falling train loss without altering the full-run protocol.

### B. Closed-loop evaluator

- [ ] Implement a `sim.val` Zero-WAM rollout runner for all five held-out task classes; existing `zero_wam/so101_eval.py` covers other SO-101 scenes and is not sufficient by itself.
- [ ] Gate the action adapter by replaying packaged validation actions through each task and confirming the expected success predicate is reachable.
- [ ] Gate observations/history: verify front/wrist pixels, measured state, executed-action cache, control cadence and model action shapes against the frozen interface.
- [ ] Gate deterministic reset and model seeding by repeating at least one trial per task and comparing initial observations and first action blocks.
- [ ] Freeze `validation_rollouts.json`: 50 IDs, simulator seeds, model seeds, paired human-video hashes, instructions, horizons and success predicates.
- [ ] Save oracle upper bound on the exact 50 scenes: 50/50 packaged trajectories must satisfy the current task success predicates or discrepancies must be resolved before model scoring.

### C. Initialization baseline — must precede training

- [ ] Exact-load audit the initialization checkpoint and save loading-info evidence.
- [ ] Compute initialization validation loss using the frozen 50-episode loss manifest.
- [ ] Run all 50 initialization closed-loop rollouts using the frozen rollout manifest.
- [ ] Verify every initialization trial artifact and publish overall/macro/per-task success with confidence intervals and failure counts.
- [ ] Freeze the initialization results and manifest hashes before allowing the training job to launch.

### D. Full training and loss tracking

- [x] Review the Zero-WAM paper and record reported optimizer, data-mixture, conditioning-dropout, IFP, packing, post-training and inference settings in the training configuration table.
- [ ] Audit every unresolved/defaulted field against pinned upstream revision `08e2c4a`; replace all blocking `TBD` values with frozen values and source references.
- [ ] Freeze run configuration: GPU topology, effective batch/token budget, optimizer, LR, warmup/decay, dropout, precision, gradient accumulation/clipping, checkpoint interval, total steps and selection rule.
- [ ] Launch through a detached/persistent job; write local JSONL plus WandB and durable checkpoint/log storage.
- [ ] Track training video/action/future-chunk total losses, LR, gradient norm, skipped steps, throughput, data/task sampling, GPU memory and NaN/Inf counters.
- [ ] Run frozen validation loss at initialization and each retained checkpoint; append results without overwriting previous evaluations.
- [ ] Apply predefined instability/early-stop gates only; record cancellations/restarts and whether optimizer/data-order state was resumed exactly.
- [ ] Select one final checkpoint using the frozen loss-based rule and record its immutable hash.

### E. Post-training comparison

- [ ] Exact-load audit the selected trained checkpoint.
- [ ] Run the same 50 closed-loop trials with no manifest changes except checkpoint identity.
- [ ] Verify all post-training artifacts and compute overall/macro/per-task success with confidence intervals.
- [ ] Produce the paired pre/post transition table and exact McNemar result.
- [ ] Compare initialization/final validation losses globally and per task; report uncertainty and any regressions.
- [ ] Review a task-stratified set of successful and failed rollout contact sheets without changing the headline metrics.
- [ ] Publish a final evidence table linking checkpoint, loss logs, rollout manifest, trial artifacts and summary hashes.

## Current state

- The prior one-episode smoke test established that upstream training and LR-zero loss evaluation can run end to end on 8×H100, but it also overfit and worsened held-out loss. It does not validate the full simulated corpus.
- `sim_train_v1`, corrected `train_v2`, and corrected validation overlay `val_v3` are complete, locally verified, and synced to Hugging Face. The `val_v3` post-upload dry run reports 103/103 files identical; README/index/inventory and beginning/middle/end videos match local SHA-256 on readback.
- `sim_val_v2` provides 50 held-out paired episodes and documents the exact closed-loop reconstruction path through `sim.val`.
- The existing smoke converter is hard-coded to one real-data task and two episodes. A full simulated-data converter/latent manifest is still required.
- The existing pretrain Modal worker rejects ICL video and the existing SO-101 runner targets cube/basket scenes. A checkpoint-agnostic ICL worker plus `sim.val` runner is still required for the requested pre/post success comparison.
- The paper configuration review is complete and captured above. Reported reference values are distinguished from unresolved pinned-source defaults and proposed SO-101 settings; blocking `TBD` fields remain intentionally unfrozen.
- No initialization success-rate baseline, full-corpus validation loss, training run, or post-training success rate has been executed yet.

## Results table

| Stage | Checkpoint | Validation video loss | Validation action loss | Validation MCP total | Overall success | Macro task success | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| Initialization | pretrain `7040c419…` | pending | pending | pending | pending / 50 | pending | pending |
| Selected final | pending | pending | pending | pending | pending / 50 | pending | pending |

## Artifact roots (planned)

```text
data/zero_wam_sim/                         converted manifests and local audit reports
outputs/zero_wam/sim_training/<run_id>/    immutable run config, loss summaries and checkpoint inventory
outputs/zero_wam/sim_val/<eval_id>/        frozen rollout manifest and pre/post trial artifacts
```

Large datasets, latents, checkpoints and rollout media belong in bucket/volume storage, not Git. Commit only code, compact manifests/summaries, and this tracker.

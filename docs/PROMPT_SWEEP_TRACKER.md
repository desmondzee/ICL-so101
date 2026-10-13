# Zero-WAM -> SO-101 Prompt Sweep Tracker

## Investigation

- [x] Inventory the existing Zero-WAM rollout, SO-101 simulation, multitask scene, conditioning experiment, tests, and causal-validation artifacts.
- [x] Trace the closed-loop rollout path from checkpoint loading through prompt encoding, denoising, KV-cache updates, action execution, and result persistence.
- [x] Verify V1 and V2b checkpoint locations, identities, transformer subpaths, and required base-model components.
- [x] Audit text conditioning, human-video conditioning, classifier-free guidance, ICL guidance, null prompts, and modality-disable semantics.
- [x] Audit simulator reset/replay behavior and identify the state that must be held identical across prompt conditions.
- [x] Record relevant prior findings and distinguish what is already evidenced from what this sweep must establish.

## Implementation

- [x] Make the SO-101 checkpoint selection explicit and configurable; reject accidental fallback to the RoboTwin checkpoint.
- [x] Define validated prompt-condition configurations for no conditioning, correct/incorrect text, correct/mismatched/shuffled video, matching text+video, and conflicting text+video.
- [x] Implement true text and video modality removal without changing unrelated inference behavior.
- [x] Preserve the existing closed-loop observation/action-history/KV-cache update and action execution path.
- [x] Guarantee paired conditions reuse the same simulator state, seed, initial observations, robot history, and model-noise schedule.
- [x] Add a two-goal soup-versus-cheese sweep using the existing multitask scene and success predicates.
- [x] Persist videos, actions, trajectories, prompt metadata, checkpoint identity, seeds, object selection, grasp success, task success, and a sweep manifest.
- [x] Add a report generator that consumes saved results without rerunning rollouts.

## Validation

- [x] Add automated tests for condition selection and modality-disable semantics.
- [x] Add automated tests for paired-condition reproducibility inputs and model-noise seeds.
- [x] Add automated tests that prove the selected checkpoint is V1/V2b and not the default RoboTwin checkpoint.
- [x] Add automated tests for manifest/result serialization and report generation.
- [x] Run focused tests and record exact commands and outcomes.
- [x] Run a local CPU/simulator smoke test where feasible and record limitations.
- [x] Run a local H100 model smoke test before any expanded experiment.

## Experiments

- [x] Run a minimal paired soup-versus-cheese smoke sweep with identical initial state across conditions.
- [x] Inspect videos, trajectories, selected objects, grasp outcomes, and task predicates rather than relying only on aggregate success.
- [ ] Expand only after the smoke gate passes, across multiple seeds and both V1 and V2b checkpoints.
- [x] Run all five held-out `sim_val_v2` tasks with five seeds each, as explicitly requested despite the failed paired-goal semantic-switch gate.
- [x] Preserve each validation episode's dataset seed, paired human prompt, model-noise seed, bounded horizon, and native task success predicate.
- [x] Inspect validation rollouts for shaking/oscillation, target-object approach or confusion, grasp attempts, drift, and release behavior.
- [x] Aggregate per-task success, closest-target approach, grasp proxies, action jitter, joint-limit saturation, and trajectory smoothness.
- [x] Compute prompt-switch accuracy, task success, object-selection accuracy, grasp success, and behavioral trajectory differences for the V2b seed-0 pilot.

## Results

- [x] Generate `docs/results/zero_wam_so101_prompt_sweep/report.md` from frozen rollout artifacts.
- [x] Include checkpoint identities, condition tables, seed-level rollout links, failures, exclusions, and unexecuted tests.
- [x] Separate evidence for action prediction quality, prompt sensitivity, and genuine in-context learning.
- [x] State conclusions conservatively and do not infer ICL from text-only sensitivity or open-loop action differences.

## Decisions

- [x] Reuse the existing SO-101 closed-loop evaluator and multitask basket environment rather than creating a second rollout stack.
- [x] Use mutually exclusive alphabet-soup versus cream-cheese goals as the primary causal scene.
- [x] Finalize exact null-text/null-video and CFG behavior after tracing upstream model code.
- [x] Finalize the smoke/expansion seed and checkpoint matrix after validating artifact availability.

## Commands And Findings

- [x] `git status --short --branch` — clean `main` at `b51d810` when work began.
- [x] `find zero_wam docs tests ...` and repository-wide conditioning/checkpoint search — found the existing closed-loop, multitask, chunk-causal, RoboTwin conditioning, and V2b rollout artifacts.
- [x] Existing published SO-101 closed-loop evidence uses V2b step 500 with human ICL latent and ICL CFG 5; it does not implement the requested controlled prompt sweep.
- [x] `zero_wam/modal_app.py` defaults to the published RoboTwin checkpoint, so the SO-101 sweep must not inherit that default.
- [x] Local hardware check: one NVIDIA H100 80 GB is available with approximately 81 GB free; local inference will replace Modal execution.
- [x] Initialized pinned `third_party/Zero-WAM` and `third_party/LIBERO` submodules and built `.venv-zero-wam`.
- [x] Downloaded the pinned base components plus explicit V1 and V2b step-500 transformers; both transformer config SHA256 values are `f30afc3dc6393199678e485580151eeb932cc8d8c7223b0f988c2651fe7a7d52`.
- [x] Conditioning audit: `use_icl=False` omits the ICL cache; `video_guidance_scale=-1` selects the learned empty-text embedding; text-only uses target CFG 5; video conditions use ICL CFG 5. Reset clears transformer and streaming-VAE caches.
- [x] Human-video controls use deterministic halves of the existing generated human demonstration (soup first, cheese second); shuffled controls deterministically permute the selected clip's frames. This is a pragmatic existing-asset reuse and must be disclosed when interpreting results.
- [x] `.venv-zero-wam/bin/python -m pytest -q tests/zero_wam/test_prompt_sweep.py tests/zero_wam/test_chunk_causal_validation.py` — 10 passed.
- [x] Simulator smoke: two independent seed-0 resets produced identical state and observation hashes; front/wrist shapes are `(224, 288, 3)`, state is `(6,)`, and one closed-loop-format step succeeded.
- [x] Local H100 load/reset smoke: V2b loaded exactly with empty missing/unexpected/mismatched keys; no-conditioning and correct-text conditions completed one chunk with identical paired inputs.
- [x] Eight-condition modality smoke: all text/video/conflict/shuffle branches completed at 5-step denoising and 16 controls. This validates execution only, not behavior.
- [x] Raw-video smoke finding/fix: resample to 12 fps, trim to `1+4k` frames, and stream-encode one initial frame followed by four-frame chunks. The original all-frames VAE call fails.
- [x] Full-denoising pilot command: `.venv-zero-wam/bin/python scripts/run_so101_prompt_sweep.py --checkpoints v2b --tasks soup_lift,cheese_lift --seeds 0 --max-controls 32 --video-steps 50 --action-steps 50 --output outputs/zero_wam/prompt_sweep_pilot_v2b_seed0`.
- [x] Pilot finding: all 16 trajectories differ under non-identical prompts, while payload-identical cross-task pairs replay bit-identically. Nevertheless all conditions select/approach soup, with 0/16 grasps, 0/16 successes, and text/video switch accuracy 0/2.
- [x] Expansion decision revised by user: run 5-10 seeds for every held-out validation task and analyze behavioral failure modes even though the paired-goal semantic-switch gate failed.
- [x] Validation assets: downloaded and verified 50 paired human latents, a 50-row manifest, and the 50-episode `sim_val_v2` index with clean one-to-one mappings.
- [x] Validation command: `.venv-zero-wam/bin/python scripts/run_so101_validation_batch.py --checkpoint v2b --per-task 5 --max-controls 64 --video-steps 50 --action-steps 50 --output outputs/zero_wam/validation_batch_v2b_5x5_c64`.
- [x] Validation execution: 25/25 episodes completed in approximately 37 minutes with zero runtime errors; every task has five dataset seeds.
- [x] Horizon finding: the existing chunk executor completed the active chunk, so a nominal 64-control bound produced 72 recorded controls in every episode. Results are early-behavior diagnostics, not full-horizon success estimates.
- [x] Validation outcome: 0/25 task successes and 0/25 target grasps. Closest target distances remained 7.3-18.3 cm depending on task/object.
- [x] Motion finding: normalized second-difference RMS was tightly grouped at `0.00303-0.00387`; severe shaking was not dominant. The largest reversal outlier was microwave episode 002 at 15.0%.
- [x] Behavioral finding: rollouts generally moved toward the correct object region, but failed precise target acquisition and grasp. Two-mug microwave scenes showed the greatest target/distractor ambiguity.
- [x] Wrote `docs/results/zero_wam_validation_batch/report.md` with protocol, aggregates, rollout locations, limitations, and failure-mode analysis.
- [x] Repeat all 25 validation episodes at the established `2x robot_frames` full horizon; all 25 completed with zero runtime errors at 13:16 UTC.
- [x] Full-run workload estimate: 34,760 requested controls / approximately 2,200 chunks. At the measured bounded-batch rate (37 minutes for 125 chunks), ETA is approximately 10.9 hours on the single H100, excluding setup/analysis.
- [x] Interim full-run analysis at 22 completed episodes: one transient `pan_on_stove` success; no other task successes. Long-horizon reversal fractions cluster near 10-13%, while normalized second-difference RMS remains near `0.003`, suggesting repetitive wandering/backtracking more than violent high-frequency shaking.
- [x] Interim target finding: the policy often identifies and approaches the intended object. Mug episode 001 briefly grasps/lifts the red mug; pan episodes 002/004 hold and move the pan; stack-bowls episode 000 holds/lifts the white bowl. Failures are therefore not uniformly object-recognition failures.
- [x] Interim manipulation finding: grasp stability, precise placement, and persistence are major failures. Pan episode 004 reaches success at control 365 but loses final-state success before the generated chunk ends; mug episode 001 lifts then leaves the mug non-upright; stack-bowls episode 000 lifts/moves the white bowl without nesting it.
- [x] Interim ambiguity finding: microwave rollouts frequently alternate between mugs/distractors or displace a mug without grasping it; sort-block rollouts approach within 1.2-3.4 cm in several episodes but record no grasps.
- [x] Interim saturation finding: several failures spend 40-69% of controls at a joint/gripper boundary, consistent with stalled/repetitive motion after an initially meaningful approach.
- [x] Action-frequency analysis over 24 completed full episodes: about 32.8% of predicted action-velocity spectral power lies at or above 5 Hz; microwave is highest at 38.7%. This supports the visually observed jitter.
- [x] Jitter source finding: normalized chunk-boundary delta RMS is slightly lower than within-chunk delta RMS overall (`0.00285` vs `0.00300`), so discontinuities between generated chunks are not the primary cause. High-frequency variation is present inside predicted chunks.
- [x] Dynamics finding: measured/executed second-difference RMS (`~0.0040-0.0042`) exceeds predicted-action RMS (`~0.0030-0.0032`), indicating the PD/physics/contact response amplifies rather than removes target roughness.
- [x] Temporal finding: quarter-episode second-difference RMS is broadly stable; jitter begins early and persists rather than appearing only after cache accumulation. Microwave increases in the final quarter.
- [x] Saved per-episode and per-task action-frequency metrics to `outputs/zero_wam/validation_batch_v2b_5x5_full/action_dynamics_analysis.json`.
- [x] Root-cause audit for miss recovery: the validation loop cached the model's predicted action chunk as history rather than the measured joints actually reached by the simulator. After a miss, the transformer could therefore retain an action history that implied successful execution even though the new images showed otherwise.
- [x] Implemented executed-state KV-cache feedback: each control now records predicted, commanded, and measured states; the cache receives normalized measured joint history, matching the corrected SO-101 evaluator seam.
- [x] Added optional closed-loop action EMA (`--action-ema-alpha`, default `1.0` for compatibility) so smoothness can be tested without silently changing the established protocol.
- [x] Added predicted-versus-executed motion diagnostics and focused smoothing tests; prompt/chunk tests now pass `12 passed`.
- [ ] After the current unmodified batch finishes, run matched full-denoising A/B episodes with corrected feedback at EMA alpha `1.0` versus a conservative smoothed value. Do not claim the smoothness or miss-recovery fix works until rollout evidence confirms it.
- [x] Corrected the cache experiment design after review: commanded targets are the appropriate default action-history tokens in this simulator; measured joints remain observation state. Measured-action caching is retained only as an explicit A/B option, not treated as a confirmed fix.
- [x] Audited upstream Zero-WAM and the paper for deployment smoothing. Upstream executes every predicted control directly; neither code nor paper specifies EMA, rate/acceleration limits, overlapping temporal ensembling, or receding-horizon blending. The paper fixes inference chunk size to two; its four-future-chunk IFP module is training-only and removed at inference.
- [x] Smoothing priority changed to `pan_on_stove`, where the unsmoothed batch has one transient success. The exact successful episode 004 is queued first with the same scene/noise seed and causal EMA alpha `0.35`.
- [ ] Compare smoothed pan episode 004 against its frozen unsmoothed rollout before expanding smoothing to the other four pan episodes.
- [x] Launched the matched pan episode 004 EMA-0.35 smoothing pilot at 13:17 UTC immediately after confirming the full validation batch had exited.
- [ ] Replace the bounded-run conclusions with full-horizon success and failure-mode analysis after all 25 episodes complete.
- [x] Synced a frozen snapshot of completed evaluation artifacts to `hf://buckets/akoniti/ICL-so101/zero_wam_evaluation_results/2026-10-13`.
- [x] Upload verification: 353 files / 282,366,897 bytes; remote manifest SHA256 `fdf9babf39b2a5ba2db2ef3bb7de550ab2f659117191d0a86e43f99adef3ef76` matches locally. The full-horizon subset contains 24 complete episodes and intentionally excludes the then-running 25th.

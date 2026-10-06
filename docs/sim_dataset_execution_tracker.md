# SO-101 simulated training dataset tracker

Updated: 2026-10-05

## Requested outcome

Produce as many high-quality simulated SO-101 robot/human pairs as approximately $120 of fal credits supports, aiming for approximately 100 diverse tasks. Quality and available credits override the original 1,000+ size aspiration. Review each oracle robot video and its physics before submitting to fal; vary lighting, scene and camera within each task. Publish accepted pairs to the existing Hugging Face bucket and expose a separate simulated-training dataset on the GitHub viewer. This tracker is the durable session record.

## Current status

Approved design and implementation underway. Task 1 provides immutable episode manifests, explicit workflow review gates, resumable state history and separate human-attempt records. Task 2 adds frozen deterministic visual configurations and optional scene hooks, with actual rendered visibility/occlusion screening and immutable resample evidence. No paid generation requests, uploads, or website publication have been performed in this session.

## Validation dataset understanding

Source: `docs/sim_val_v2.md`, `data/so101_sim_val_v2/README.md`, `sim/val/README.md`, `humangen/sim_val_demos.py`.

- Documented validation corpus: 50 pairs, ten each for `mug_on_plate`, `mugs_in_microwave`, `pan_on_stove`, `sort_blocks`, and `stack_bowls`.
- Robot episodes use MuJoCo through SO101 Nexus, LIBERO scene assets, deterministic seed layouts, and scripted IK oracles. Front/wrist views are 640×480 at 30 fps.
- Training-compatible records include joint targets/states in LeRobot degrees (gripper 0–100) and end-effector targets/states using the real-data FK convention.
- Human generation uses robot-free first/last front-camera renders and `minimax/h3-max-turbo/image-to-video` on fal, with the v5 prompt and oracle grasp order. Current duration is five seconds.
- Documentation reports two independent human-video review stages (judge then verifier), regeneration of rejected attempts, and replacement of persistently failing robot scenes.
- v2 preserves the v1 human videos and initial scenes while re-recording smoother robot trajectories, with arc carries, joint velocity limits, and wrist unwind.
- The simulator supports seeded resets and online success predicates. Recorded examples can support offline loss evaluation; a complete functioning loss/rollout evaluation integration still needs verification.
- The existing five validation task definitions and scenes must remain held out from the proposed training corpus. Seed-only separation is insufficient for task-level generalization claims.

Independent local audit confirms 50 indexed pairs, all 50 `review.final` values accepted and all 50 frame metadata success flags true; robot duration totals 1,171.49 seconds. All 50 v2 human videos and first robot-free frames match v1 byte-for-byte. However, **none of the 50 v2 last frames match the original generation inputs**. The packaging paragraph's blanket pixel-identity claim therefore overstates endpoint identity for v2. Existing visual review is mainly human-focused and does not establish a fresh comprehensive physics audit of v2 robot trajectories. Exact task/instruction separation exists, but real training already includes related manipulation families; do not describe this as complete family-level separation.

## Work queue

- [x] Audit local validation manifests, provenance and review artifacts against documentation.
- [x] Audit simulation assets, oracle skills, success predicates and physics quality gaps.
- [x] Audit fal generation/retries, per-pair review evidence, HF destination and viewer integration.
- [ ] Define genuine task diversity, split policy, episode quotas and generation cost limits.
- [x] Write the approved design and robot-corpus implementation plan grounded in the audit.
- [ ] Implement task registry, recording and reproducible per-episode quality evidence.
- [ ] Validate a small pilot before large-scale human generation.
- [ ] Generate and review a credit-limited set of accepted pairs; retain rejected attempts and reasons outside the release.
- [ ] Independently verify every proposed human-pair acceptance after generation; generation completion alone cannot count as acceptance.
- [ ] Verify packaged training compatibility and held-out evaluation isolation.
- [ ] Publish the release card, provenance and measured results in HF and repository documentation.
- [ ] Upload accepted data to HF and verify remote inventory/readback.
- [ ] Push scoped code/docs/viewer updates to GitHub and verify Pages deployment.
- [ ] Verify the live simulated-training gallery, pair routes and video playback plus existing real/validation views.

## Required acceptance evidence

Proposed release gates: final settled task success; reproducible reset and rollout; finite and correctly scaled action/state streams; motion limits; grasp/contact and placement evidence; no unexplained object motion, penetrations or scene changes; video decoding and alignment; successful human task execution in the robot's order; independent robot and human-pair visual review with disagreement resolved before acceptance. Visual review alone does not prove physical correctness: simulator telemetry must corroborate it.

## Open decisions

- Confirm whether the existing five validation task families should be excluded entirely or only their exact task definitions. Default recommendation: exclude their exact semantics, with stricter family separation reported where feasible.
- Spending ceiling resolved: at most $120, further limited by verified available fal balance. No automatic top-up; use subagents for review rather than assuming funding for external review APIs.
- Confirm the exact HF release destination from existing publishing configuration.
- Decide task taxonomy after asset/skill audit: object or color substitutions alone do not establish 100 different manipulation tasks.

## Parallel audit assignments

- `validation_audit`: manifest counts, provenance, reviews and split isolation (read-only).
- `simulation_audit`: expansion feasibility and physics QA (read-only).
- `generation_audit`: fal pairing/review workflow and publication targets (read-only).

## Session log

- 2026-10-05: User authorized simulated dataset creation, independent subagent episode QA, HF publication and GitHub presentation; requested persistent Markdown tracking.
- 2026-10-05: Read the existing validation documentation and generation code; launched three independent read-only audits. Generation has not started.
- 2026-10-05: Audits found five implemented tasks, 24 catalog assets and two arenas. Installed runtime dependencies are present. Existing recorder deletes its output directories, so training must use a separate resumable pipeline. Existing task success checks lack sustained success and full-trajectory physics checks; `sort_blocks` does not require target contact or low angular velocity.
- 2026-10-05: Existing HF destination is `akoniti/ICL-so101`, with validation under `sim_val_v2`; viewer lives in `site/`. Proposed training prefix: `sim_train_v1`. Approximately 18 GB projected for 1,000 pairs, before scratch/retries.
- 2026-10-05: Official fal model page lists a current 480p promotional rate of $0.015/s through October 15, then $0.025/s; the same page also contains stale conflicting pricing notes, so recheck account/API pricing before submitting. Five-second clips imply approximately $75–125 per 1,000 attempts before retries/review costs. Source: https://fal.ai/models/minimax/h3-max-turbo/image-to-video . Spending ceiling requested; no paid requests submitted.
- 2026-10-05: Three read-only audits completed. Existing human review history contains 88 reviewed attempts (53 accepts, 35 rejects), of which 50 were packaged. No dedicated Zero-WAM runner wiring this validation corpus into closed-loop evaluation was found. Remote publication completeness has not been verified.
- 2026-10-05: Drafted `docs/sim_dataset_design.md` for concrete design review: proposed 100 qualified task definitions × 12 accepted pairs, frozen episode endpoints, substep physics telemetry, independent robot/human review and verifier, resumable isolated output, bucket publication and viewer integration. Detailed task catalog and implementation plan remain pending design review and spending limit.
- 2026-10-05: User refined the design: size follows approximately $120 fal credits; quality takes precedence; approve oracle robot videos before fal spend; randomize lighting, scene and camera across episodes of a task. Updated design supersedes fixed 1,200-pair quota. Plan conservative charge reservations, a ≤$5 pilot, initial 25% retry reserve, balanced task coverage rounds and frozen per-episode visual configuration.
- 2026-10-05: User approved the design. Wrote `docs/superpowers/plans/2026-10-05-sim-train-robot-corpus.md`, covering immutable workflow state, deterministic procedural variation, substep physics QA, validation-isolated task registry, resumable recording, independent robot review and a frozen fal-ready pilot manifest. The plan intentionally makes no paid request; human generation and release follow as separately reviewed phases after measured robot-pilot evidence.
- 2026-10-05: User requested a full delivery-plan recheck. Found downstream work was only summarized as follow-on plans and the state machine allowed `human_complete → accepted` without explicit review states. Added required Tasks 7–11 for generation review/verifier, packaging audit, HF upload/readback, documentation, GitHub push/Pages deployment and live viewer checks; added explicit human-review and verifier states. Robot-only milestone completion does not complete the overall request. Existing authorization covers downstream delivery; stage boundaries are quality gates.
- 2026-10-05: Implemented Task 1 in the isolated `sim-train-v1` worktree: immutable `manifest.json`, atomic `state.json`, checked ordered review history and immutable keyed human attempts. Generation completion requires subsequent human-review and verifier approvals before acceptance; failed human-attempt records preserve the source's workflow state. Captured RED with missing `sim.train`; focused and full pytest checks pass (28 tests). Existing standalone transform/mask/composite checks also pass. Validation files were not modified. Interrupted or corrupt state fails explicitly rather than resetting an episode.
- 2026-10-05: Implemented Task 2: SHA-256-derived per-task/seed/resample RNG, frozen qualified arena/front-camera/key/fill/background configurations, JSON/hash integrity and layout-inclusive episode identity. Optional `visual_config` hooks preserve validation defaults and the fixed wrist transform; task reset owns all object layouts. Front screening measures actual segmentation pixels, projects object bounds, casts rays over task-defined goal surfaces (including self-occlusion), rejects existing store hashes and records reasons plus deterministic retry indices immutably. All 65 collected tests pass with CoreGraphics access, including 15 variation cases and 50 existing tests. Initial 640×480 calibration blocks occupy 1,322–1,493 pixels against the configurable 100-pixel floor; task pilot qualification must refine visibility thresholds. Used worktree `.venv` commands because uv cache access was sandbox-blocked. Compatibility recorder failed at sandbox CoreGraphics initialization before any dataset write; controller must rerun the smoke with graphics access. Validation dataset directories were not written.
- 2026-10-06: Task 2 review fix enforces explicit persisted identities: `EpisodeManifest.config_hash` remains the complete visual-plus-layout identity; required keyword-only `visual_config_hash` carries `VisualConfig.config_hash` separately. Store creation/load validates and preserves both fields; duplicate screening uses the explicit visual field alone, including when metadata is empty. Missing visual fields fail clearly rather than inferring from a composite digest or optional metadata. No production training manifests exist yet, so schema version remains 1 and all current constructors were updated. Regression-first checks and the complete 72-test suite pass. Task 5 recorder must persist both identities.

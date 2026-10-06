# SO-101 simulated paired training dataset design

Design refined by user feedback, 2026-10-05. Execution state and findings: [tracker](sim_dataset_execution_tracker.md).

## Outcome and scope

Release as many high-quality accepted robot/human pairs as the approximately $120 fal credit balance supports, aiming for approximately 100 distinct task definitions for SO-101 Zero-WAM post-training. The initial 1,000+ goal is aspirational: quality and the credit cap take precedence over count. Every robot episode completes its task in MuJoCo through SO101 Nexus; every generated human clip completes the same ordered manipulation in the corresponding scene. Publish under `hf://buckets/akoniti/ICL-so101/sim_train_v1` and add a separate simulated training view to `site/`.

The existing 50-pair validation corpus stays held out. This work creates training data, simulator task definitions, reproducible QA evidence and publication integration. Actual model post-training is subsequent work; this release must be loadable with the current joint and EE training conventions.

## Approach and alternatives

Recommended: extend the existing Nexus/LIBERO environment and oracle skills with a separate training task registry and resumable pipeline. This preserves embodiment, camera conventions and action units, while allowing stronger success and physics checks.

Alternative: import an external task suite wholesale. This offers task breadth but requires requalifying reach, articulation, grasp geometry, camera framing and success predicates for the SO-101. Use external assets selectively when existing assets cannot supply meaningful diversity.

Alternative: procedurally reskin the five validation tasks. This is fast but does not meet meaningful task diversity and compromises the evaluation split; reject as the primary approach.

## Task diversity and allocation

Aim for 100 explicit task definitions and approximately ten accepted pairs per task when credits permit. Final quotas follow measured price, clip duration and pilot acceptance rate. Allocate in coverage rounds across qualified tasks before adding depth to easy tasks; retain retry credits rather than spending everything on first attempts. Count both distinct task definitions and manipulation families in the dataset card. Color/layout substitutions are episode diversity, not automatically new task definitions.

Within each task, randomize object layouts, lighting intensity/direction/color, qualified scene backgrounds and front-camera angle/position. Keep objects visible, task outcomes unambiguous and camera poses relevant to deployment. The camera stays fixed during an episode. Preserve the physically attached wrist-camera transform; lighting and arena changes still affect its images. Camera, scene and lighting configurations are deterministic per episode, frozen before robot review, and reproduced exactly in robot-free human-generation endpoints. Record sampled values and seeds, and report coverage of variation dimensions. Screen framing, shadows and visibility before accepting a recording.

Candidate families: container packing and unpacking; stacking and unstacking; spatial arrangements and alignment; ordered assembly or rearrangement; relocation and exchange; pushing along constrained routes; handle-based transport; orientation-sensitive placement; articulated fixtures once qualified. Each task must have a reachable randomized layout, physical goal predicate, oracle implementation, action-order description and family identifier.

The exact 100-task catalog must be validated against available assets and reachable geometry before quota generation. Do not claim that 100 feasible tasks already exist: only five are implemented today. Begin with representative tasks from the available skills, then qualify articulation and pushing separately. Failed task families must be repaired or replaced with honestly labeled definitions.

Exclude the five exact validation semantics/instructions, their episode seeds/scenes and their human clips. Record any remaining family-level overlap explicitly. Preserve fixed validation identities rather than silently redefining its benchmark.

## Episode production and provenance

Use `data/so101_sim_train_v1` as an isolated root; never invoke the existing destructive recorder on existing task outputs. Persist per-episode states and atomic artifacts so interrupted runs resume without deleting accepted work or silently replacing videos.

State progression: proposed seed → oracle screen → recorded robot candidate → physics QA → independent robot-video approval → human request → human QA → independent verification → packaged acceptance. Submission to fal is forbidden unless both physics QA and robot visual review accept the exact hashed recording and its frozen endpoints. Rejection records retain reasons, source hashes, attempt seeds and request IDs. Only accepted episodes enter release indexes.

Freeze scene/assets/code revision, simulator configuration, seed, instruction, oracle order, joint/EE arrays, start/end images and video hashes before generation. Human generation uses the exact frozen endpoints from that recorded robot episode. Do not re-record selected episodes after human generation unless the entire correspondence is revalidated; first-frame equality alone is insufficient.

## Physics and completion gates

Require task-specific geometry, support/contact, released grasp, orientation and low linear/angular velocity where applicable. Final success must persist for a settled interval, initially one second, rather than a single-frame flag. Thresholds must be calibrated against qualified pilot trajectories and saved in versioned QA configuration.

Audit every physics substep for finite state, simulator warnings, joint bounds, unexpected penetration/contact, object motion and unintended distractor displacement. Distinguish permitted manipulation contact from arm/table/fixture collisions. Track actuator targets and actual motion separately; oracle target limits do not prove actual joint velocity limits.

Retain object/contact trajectories and events sufficient to investigate rejected episodes. Physical realism claims must disclose simplified assets, collision geometry, densities and fixtures. Visual reviewers cannot establish numerical physical validity by themselves.

## Human generation and independent review

Reuse fal `minimax/h3-max-turbo/image-to-video` with frozen start/end renders and v5-style task descriptions derived from structured action order. Use five-second clips for qualified simple tasks; longer sequences need durations that allow visible completion and a final hold. Retry with bounded attempts and a global spending ledger; cost limits include failed attempts.

Hard fal spending cap: $120 (user confirmed 2026-10-06 that the full $120 balance is available and unspent). No automatic top-up. Start with a pilot allocation of at most $5, initially reserve 25% of the remaining budget for retries, and revise allocation using observed acceptance costs. Verify live/account pricing before submission. Atomically reserve a conservative per-request charge before concurrent jobs are sent; unresolved requests retain their reservation. Track actual billed usage where available and never resubmit an uncertain POST merely because retrieval failed. Remaining credit determines whether another job fits. Review services outside fal do not have an authorized paid budget; use the requested subagents unless separate funding is provided.

For planning only: five-second requests at $0.015/s cost $0.075 each, while $0.025/s costs $0.125. $120 permits at most 1,600 or 960 such attempts respectively, before longer clips and retries. At an illustrative 60–80% acceptance rate, that implies roughly 960–1,280 or 576–768 accepted pairs. These are scenarios, not measured yield forecasts; account pricing and pilot yield determine the actual target.

Every episode receives an independent robot-video review before any paid human request, then a human-pair review followed by an independent verifier that tries to refute acceptance. Subagents review assigned episodes in batches with a per-episode verdict, timestamped evidence and artifact hashes. Inspect front and wrist robot views, corresponding human movement, intermediate task transitions and final states. Increase frame density or inspect contiguous segments when contact, occlusion or motion is ambiguous; sparse contact sheets alone are insufficient.

Review criteria: completed ordered task, stable object identities/counts, physical hand/object contact, no teleportation or morphing, scene consistency, valid release and settled goal. Reviewer disagreement or missing evidence yields pending/reject, never automatic acceptance. The orchestrator resolves failures through regeneration or replacement and verifies coverage for every released episode.

## Packaging and evaluation compatibility

Retain the existing pair layout: per-task LeRobot v3 robot data, front/wrist videos, human video, robot-free endpoints, episode/source metadata, thumbnails and review records. Add task taxonomy, split assignment, hashes, physics QA and attempt provenance. Preserve 30 fps robot streams, joint degrees/gripper 0–100, and current FK-based EE units.

Verify loader compatibility with Zero-WAM's actual latent/pair indexing before release. For online evaluation, retain seeded task resets, action conversions and `info["success"]`; run oracle replay and policy-interface smoke checks. Loss evaluation is a separate held-out consumption path and must not mix training episodes into the existing validation set.

## Pilot and release checks

First qualify representative task families using at least 50 unseen oracle seeds each; aim for ≥95% strict success and inspect failures. The recorder admits tasks at ≥86% (43/50) with no systematic failure region, because every recorded episode is individually physics-gated and reviewed (decision 2026-10-06 after the user asked for whatever method gives the most high-quality completions). Then record and review a small diverse paired pilot before bulk requests. Use measured generation acceptance rate, review workload and rendering throughput to adjust retry budget and scheduling.

Release requires complete per-episode QA coverage, no duplicate/cross-split artifact identities, valid decoded video/array alignment, frozen-endpoint correspondence, training-loader compatibility and HF readback verification. Pair count and achieved task coverage are determined by the credit cap, with approximately 100 qualified task definitions as the diversity objective. Dataset card reports actual counts, family and visual-variation coverage, rejection rates, spend, generation model/settings and realism limits. GitHub viewer must label simulated training separately from real training and simulated validation.

## Next implementation work

User supplied the $120 credit ceiling, robot-first review requirement and within-task lighting/scene/camera diversity. Implement the budget and approval gates before paid production, qualify the task catalog and randomization, then measure the pilot yield. The exact balance and current request pricing remain runtime checks, not additional authorization requests.

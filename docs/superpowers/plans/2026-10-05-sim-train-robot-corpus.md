# Simulated Training Robot Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, deterministic robot-only production pipeline that records diverse procedural SO-101 episodes and admits only physics-checked, independently reviewed recordings to the fal-ready queue.

**Architecture:** A training-specific package under `sim/train` owns task definitions, deterministic visual randomization, trajectory telemetry, immutable episode manifests, and workflow state. It reuses the qualified `sim.val` environment/oracle primitives without writing under the validation roots. Automated QA precedes an artifact-hashed visual-review queue; fal code can later consume only records in `robot_approved` state.

**Tech Stack:** Python 3.12, MuJoCo, so101-nexus 0.7.0, LeRobot 0.5, NumPy, Pillow, ffmpeg/ffprobe, pytest.

**Spec:** `docs/sim_dataset_design.md`

**Delivery scope:** Tasks 1–6 are the robot-corpus milestone, not completion of the user's request. The required generation, post-generation review, HF/GitHub publication, documentation and live viewer verification are tracked below as Tasks 7–11. Execution proceeds through all milestones under the existing authorization; milestone boundaries are quality gates, not additional permission requests.

## Global Constraints

- Write all production data beneath `data/so101_sim_train_v1`; never modify `data/so101_sim_val*`.
- Keep the existing validation task identities, episode seeds/scenes, clips and exact instructions out of training.
- Every episode must have a distinct deterministic procedural scene configuration, including object layout and sampled visual variation.
- Preserve the wrist-camera attachment transform; randomize only qualified front-camera, arena and lighting parameters.
- Freeze and hash the approved robot recording, metadata and robot-free endpoints before any fal submission.
- Require sustained task success, substep physics QA and independent robot-video approval before an episode becomes fal-ready.
- Do not call fal, upload to Hugging Face, or change the public viewer in this plan.
- Update `docs/sim_dataset_execution_tracker.md` as each task completes and record measured results rather than projections.

## Review Focus

- A resumed run must preserve accepted artifacts byte-for-byte and skip completed states.
- Two seeds that collide on sampled visual configuration must be rejected or deterministically resampled so every episode configuration remains unique.
- A terminal goal that holds for only one frame must fail the one-second settle gate.
- Distractor displacement, non-finite dynamics, joint-limit violations and disallowed contact must reject an episode even when `success()` is true.
- A review verdict whose stored artifact hashes do not match the current files must never advance the episode to `robot_approved`.

---

### Task 1: Training package, immutable manifests and workflow state

**Files:**
- Create: `sim/train/__init__.py`
- Create: `sim/train/model.py`
- Create: `sim/train/store.py`
- Create: `tests/sim_train/test_store.py`
- Modify: `pyproject.toml`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: filesystem root and JSON-serializable episode metadata.
- Produces: `EpisodeKey(task: str, seed: int)`, `EpisodeState`, `EpisodeManifest`, `sha256_file(path)`, `EpisodeStore.create_candidate(manifest)`, `EpisodeStore.transition(key, expected, target, evidence)` and atomic JSON writes.

- [x] **Step 1: Add pytest and write failing state-machine tests**

Add `pytest>=8,<9` to the dev dependency group. Test that a candidate may advance through `recorded`, `physics_approved`, `robot_approved`; illegal skips fail; repeated identical transitions are idempotent; changed evidence fails; and atomic writes leave no partial manifest.

```python
def test_cannot_skip_robot_review(tmp_path):
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 7)
    store.create_candidate(EpisodeManifest(key=key, config_hash="a" * 64))
    with pytest.raises(InvalidTransition):
        store.transition(key, EpisodeState.CANDIDATE, EpisodeState.ROBOT_APPROVED, {})
```

- [x] **Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/sim_train/test_store.py -q`
Expected: collection fails because `sim.train.store` does not exist.

- [x] **Step 3: Implement typed records and atomic state transitions**

Use string enums with the ordered path `candidate → recorded → physics_approved → robot_approved → human_submitted → human_complete → human_review_approved → verifier_approved → accepted`, plus terminal `rejected`. `human_complete` means generation/download finished, never quality acceptance. Failed human attempts are recorded separately so a robot-approved source remains eligible for a budgeted retry. Write JSON to a sibling `.tmp`, flush and `os.fsync`, then `os.replace`. Every transition appends timestamp, old/new state and evidence hashes; existing final artifacts are never overwritten.

- [x] **Step 4: Add corruption, resume and hash mismatch tests**

Exercise truncated JSON, duplicate key with a different config hash, idempotent resume, changed evidence, and a transition from `rejected`. Expected behavior is an explicit exception without mutation.

- [x] **Step 5: Run tests and record the result in the tracker**

Run: `uv run pytest tests/sim_train/test_store.py -q`
Expected: PASS.

- [x] **Step 6: Commit Task 1**

```bash
git add pyproject.toml sim/train tests/sim_train/test_store.py docs/sim_dataset_execution_tracker.md
git commit -m "feat: add immutable sim training episode state"
```

### Task 2: Deterministic procedural visual variation

**Files:**
- Create: `sim/train/variation.py`
- Create: `tests/sim_train/test_variation.py`
- Modify: `sim/val/scene.py`
- Modify: `sim/val/env.py`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: task name, episode seed and allowed arena/camera/light ranges.
- Produces: immutable `VisualConfig` with `arena`, `front_camera`, `lights`, `background`, `variation_seed`, `config_hash`; `sample_visual_config(task, seed, policy)`; `ValEnv(..., visual_config=...)`.

- [x] **Step 1: Write deterministic variation tests**

Test identical task/seed equality, different-seed diversity over 100 seeds, numeric bounds, config JSON round-trip, unique hashes, fixed wrist transform, and stable XML generation.

```python
def test_variation_is_deterministic_and_changes_across_seeds():
    a = sample_visual_config("put_can_in_basket", 11, DEFAULT_POLICY)
    assert a == sample_visual_config("put_can_in_basket", 11, DEFAULT_POLICY)
    assert a.config_hash != sample_visual_config("put_can_in_basket", 12, DEFAULT_POLICY).config_hash
```

- [x] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/sim_train/test_variation.py -q`
Expected: collection fails because `sim.train.variation` does not exist.

- [x] **Step 3: Implement bounded sampling and scene injection**

Derive a stable RNG seed from SHA-256 of task plus episode seed. Sample from the two qualified arenas, bounded front-camera offsets/look-at/fovy, and bounded key/fill light position, intensity and neutral-to-warm color. Extend scene XML generation to accept the frozen values. Do not sample the wrist transform.

- [x] **Step 4: Add visibility and uniqueness screening**

Reject configurations when task objects fall outside the front image, foreground task objects occupy fewer than a calibrated pixel-area threshold, the goal region is occluded, or the hash is already present in the store. Store the rejection reason and next deterministic resample index.

- [x] **Step 5: Run variation and existing validation smoke tests**

Run: `uv run pytest tests/sim_train/test_variation.py -q`
Run: `uv run python -m sim.val.record --task mug_on_plate --episodes 1 --seed 900000 --out /tmp/icl_so101_val_compat`
Expected: tests pass and the compatibility recording succeeds without changing validation data.

- [x] **Step 6: Commit Task 2**

```bash
git add sim/train/variation.py sim/val/scene.py sim/val/env.py tests/sim_train/test_variation.py docs/sim_dataset_execution_tracker.md
git commit -m "feat: add deterministic sim visual variation"
```

### Task 3: Substep trajectory telemetry and strict physics QA

**Files:**
- Create: `sim/train/telemetry.py`
- Create: `sim/train/physics.py`
- Create: `tests/sim_train/test_physics.py`
- Modify: `sim/val/env.py`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: per-substep qpos/qvel/actuator/contact/object state, declared task objects, distractors and allowed contact rules.
- Produces: compressed `telemetry.npz`; `PhysicsReport(accepted, checks, maxima, violations, config_hash)`; `audit_trajectory(telemetry, policy)`.

- [x] **Step 1: Write synthetic physics-gate tests**

Construct small telemetry fixtures and assert rejection for NaN, one-frame success, excess linear/angular speed, joint-limit breach, unsupported final object, distractor displacement, deep penetration and disallowed robot/fixture collision. Assert acceptance for a settled supported placement with declared manipulation contacts.

- [x] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/sim_train/test_physics.py -q`
Expected: collection fails because physics modules do not exist.

- [x] **Step 3: Expose all MuJoCo substeps and collect telemetry**

Add an optional substep observer to `ValEnv.step` without changing default behavior. Store timestamps, robot joint state/targets, free-body poses/velocities, contacts with distance/force where available, success flags, grasp state and simulator warnings. Write compressed arrays plus a schema/version manifest.

- [x] **Step 4: Implement strict configurable checks**

Require finite values, joint ranges, speed/acceleration bounds, permitted contacts, penetration tolerance, distractor displacement tolerance and 30 consecutive success frames after release. Task definitions provide support bodies, goal tolerances and allowed task contacts. Reports include exact offending timestamps and measured maxima.

- [x] **Step 5: Run focused tests and a real validation-task audit**

Run: `uv run pytest tests/sim_train/test_physics.py -q`
Run a single seed of `mug_on_plate` through the telemetry collector under `/tmp/icl_so101_physics_smoke` and confirm the report is deterministic on replay.
Expected: tests pass; repeated report JSON hashes match.

- [x] **Step 6: Commit Task 3**

```bash
git add sim/train/telemetry.py sim/train/physics.py sim/val/env.py tests/sim_train/test_physics.py docs/sim_dataset_execution_tracker.md
git commit -m "feat: audit sim trajectories at physics substeps"
```

### Task 4: Training task schema, registry and validation isolation

**Files:**
- Create: `sim/train/tasks/__init__.py`
- Create: `sim/train/tasks/schema.py`
- Create: `sim/train/tasks/catalog.py`
- Create: `tests/sim_train/test_task_catalog.py`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: env/oracle class, canonical instruction, manipulation family, task objects, support/goal/contact policy and action-order renderer.
- Produces: `TaskDefinition`; `TRAIN_TASKS`; `load_train_task(name)`; `validate_catalog(definitions, validation_index)`.

- [ ] **Step 1: Write catalog contract and held-out isolation tests**

Test unique names/instructions, required taxonomy fields, deterministic action descriptions, no exact validation task/instruction, no validation seeds, and rejection of cosmetic aliases with identical normalized semantics.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/sim_train/test_task_catalog.py -q`
Expected: collection fails because the training registry does not exist.

- [ ] **Step 3: Implement the typed task definition and validator**

Normalize instructions and semantic signatures from family, manipulated objects, relation, goal and order. Catalog validation emits a machine-readable overlap report rather than claiming family-level isolation.

- [ ] **Step 4: Register the first representative pilot tasks**

Register at least one task from each feasible pilot family using existing qualified assets: container insertion, container removal, spatial arrangement, ordered relocation and orientation-sensitive placement. Each task supplies strict success/support/contact rules and avoids the five exact validation semantics.

- [ ] **Step 5: Qualify each pilot task over 50 unseen seeds**

Run the non-rendered oracle screen for 50 seeds per task. Record success rate and all failure reasons in `data/so101_sim_train_v1/qualification/<task>.json`. Repair or exclude any task below 95% strict success; never lower thresholds to meet the target.

- [ ] **Step 6: Run catalog tests and commit**

Run: `uv run pytest tests/sim_train/test_task_catalog.py -q`
Expected: PASS.

```bash
git add sim/train/tasks tests/sim_train/test_task_catalog.py data/so101_sim_train_v1/qualification docs/sim_dataset_execution_tracker.md
git commit -m "feat: add isolated sim training task registry"
```

### Task 5: Resumable robot recorder and frozen review artifacts

**Files:**
- Create: `sim/train/record.py`
- Create: `sim/train/review.py`
- Create: `tests/sim_train/test_record.py`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: `TaskDefinition`, episode seed, `VisualConfig`, `EpisodeStore` and physics policy.
- Produces: immutable episode directory with `robot_front.mp4`, `robot_wrist.mp4`, `robot_data.parquet`, `telemetry.npz`, `physics.json`, `first.png`, `last.png`, `manifest.json`, contact sheets and `robot_review_request.json`.

- [ ] **Step 1: Write resume, immutability and approval-gate tests**

Use a fake recorder backend to test atomic staging, crash/resume, byte-identical rerun, no overwrite of completed episodes, endpoint/video hashes, rejection before review, and advancement only when an accept verdict contains matching hashes.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/sim_train/test_record.py -q`
Expected: collection fails because `sim.train.record` does not exist.

- [ ] **Step 3: Implement one-episode atomic recording**

Render into `data/so101_sim_train_v1/staging/<task>/<seed>-<uuid>`; run physics QA; encode videos; verify decoding, frame counts and timestamps; hash all artifacts; then atomically rename into `candidates/<task>/episode_<seed>`. Never remove another episode directory.

- [ ] **Step 4: Build dense robot-review evidence**

Generate front/wrist contact sheets containing start, every skill transition, grasp/release events, any maximum-violation neighborhood and the final settled second. Include task text, variation metadata, physics report and artifact hashes in the request. The verdict schema is `accept|reject|needs_detail`, reasons and inspected evidence.

- [ ] **Step 5: Implement verdict import and fal-ready index**

Validate reviewer identity, required reasons and exact hashes. Only `accept` after `physics_approved` advances to `robot_approved`. Regenerate `fal_ready.json` from manifests so it cannot include any other state.

- [ ] **Step 6: Run tests and record a five-task robot pilot**

Run: `uv run pytest tests/sim_train/test_record.py -q`
Record three diverse candidates for every qualified pilot task without fal access. Dispatch independent subagents to inspect each robot episode, resolve `needs_detail`, and store signed verdicts. Confirm every `fal_ready.json` entry is hash-valid.

- [ ] **Step 7: Commit Task 5**

```bash
git add sim/train/record.py sim/train/review.py tests/sim_train/test_record.py data/so101_sim_train_v1 docs/sim_dataset_execution_tracker.md
git commit -m "feat: produce reviewed sim robot candidates"
```

### Task 6: Robot-corpus verification and handoff to paid-generation plan

**Files:**
- Create: `docs/sim_train_robot_pilot_report.md`
- Modify: `docs/sim_dataset_execution_tracker.md`

**Interfaces:**
- Consumes: qualification records, robot candidate artifacts, physics reports and independent review verdicts.
- Produces: a go/no-go report and a frozen fal-ready pilot manifest for the later human-generation plan.

- [ ] **Step 1: Run the complete robot-only verification suite**

Run: `uv run pytest tests/sim_train -q`
Run: `uv run python -m sim.train.record --verify-all data/so101_sim_train_v1/candidates`
Expected: all tests pass; every accepted artifact decodes, hashes match, physics passes and review evidence is complete.

- [ ] **Step 2: Audit validation isolation and diversity**

Run the catalog validator against `data/so101_sim_val_v2/index.json`. Report exact overlaps, family overlap, unique configuration hashes, arena/camera/light coverage and per-task layout uniqueness.

- [ ] **Step 3: Write the pilot report**

Document task qualification rates, recording/rejection counts, physics failures, reviewer decisions, runtime/storage, visual-variation coverage and remaining risks. State explicitly that no fal calls occurred.

- [ ] **Step 4: Freeze the fal-ready pilot manifest**

Write an immutable manifest containing only robot-approved episode IDs, task/action text, duration request, endpoint image hashes and artifact hashes. This becomes the sole input to the human-generation plan.

- [ ] **Step 5: Commit Task 6**

```bash
git add docs/sim_train_robot_pilot_report.md docs/sim_dataset_execution_tracker.md data/so101_sim_train_v1/fal_ready.json
git commit -m "docs: qualify sim robot corpus pilot"
```

## Required downstream delivery milestones

Expand each milestone into implementation steps using actual pilot interfaces and measurements before touching its code. These milestones remain part of the original authorized deliverable.

### Task 7: Credit-limited generation and complete post-generation review

**Files:** `humangen/sim_train_demos.py`, `humangen/sim_train_review.py`, `sim/train/budget.py`, `tests/sim_train/test_budget.py`, `tests/sim_train/test_pair_review.py`, production attempt/review manifests.

- [ ] Verify current fal price and available credits; enforce a cap of the lesser of $120 and available balance, with conservative atomic request reservations and unresolved-request accounting.
- [ ] Generate a ≤$5 diverse pilot only from hash-valid robot-approved entries; decode every downloaded video and preserve exact request settings, seeds, IDs and endpoint hashes.
- [ ] Have independent subagents review **every generated attempt**, checking completed task/order, continuous hand/object contact, object identities, scene consistency, final goal and agreement with the approved robot episode. Use denser/contiguous evidence whenever sampled frames are insufficient.
- [ ] Have a separate verifier challenge every proposed acceptance; unresolved disagreement or missing evidence cannot become `accepted`.
- [ ] Retain rejected attempts/reasons and regenerate only within the credit ledger. Measure accepted-pair cost before allocating remaining credits in balanced task-coverage rounds.
- [ ] Qualify additional tasks toward approximately 100 definitions; record real task/family coverage and visual diversity. Do not fabricate a fixed count when credits or qualification restrict it.
- [ ] Test that completed generation without both review stages cannot enter the accepted set, modified hashes invalidate verdicts, and concurrent submissions cannot exceed the cap.

### Task 8: Package and audit the final accepted dataset

**Files:** `sim/train/package.py`, `sim/train/verify.py`, `data/so101_sim_train_v1/release/`, `tests/sim_train/test_release.py`.

- [ ] Package only pairs with physics acceptance, robot approval, human review acceptance and independent verifier acceptance. Copy the frozen recordings/endpoints; never silently re-record them.
- [ ] Export compatible LeRobot data, human/front/wrist videos, thumbnails, source metadata, variation configuration, checksums and review records. Keep rejected attempts outside the published release.
- [ ] Audit **every released episode** for artifact hashes, complete review coverage, decoded frame counts/durations, state/action alignment, finite correctly scaled arrays, endpoint correspondence, duplicates and validation isolation.
- [ ] Smoke-test the actual Zero-WAM dataset/latent loader and seeded simulator action interface. Report any unsupported integration explicitly rather than inferring compatibility from file names.
- [ ] Freeze `index.json` and a release inventory with file counts, sizes and hashes; test that a missing/rejected/stale-review pair makes release verification fail.

### Task 9: Documentation and Hugging Face bucket publication

**Files:** `data/so101_sim_train_v1/release/README.md`, `docs/sim_train_v1.md`, `docs/sim_dataset_execution_tracker.md`, root `README.md`.

- [ ] Document actual accepted pairs/tasks/families, procedural uniqueness, generation model/settings, credit spend, attempts/rejections, review method, simulator limitations, provenance/licenses, train/validation split and loading/evaluation instructions.
- [ ] Upload the frozen release to `hf://buckets/akoniti/ICL-so101/sim_train_v1` without deleting other bucket prefixes. Include data, index, card, provenance and review evidence.
- [ ] Verify remote inventory against the release inventory; download/read back the published index/card and representative videos/parquet files across tasks, then compare their hashes and decode them.
- [ ] Record upload completion, remote verification evidence and public bucket links in the tracker. A successful upload command alone does not establish publication completeness.

### Task 10: GitHub repository and Pages viewer update

**Files:** `site/data/sim_train_index.json`, `site/index.html`, `site/app.js`, `site/viewer.html`, `site/viewer.js`, `site/style.css` if needed, root `README.md`, `.github/workflows/pages.yml` only if required.

- [ ] Add a third dataset labeled **Training (simulation)**, separate from real training and simulated validation; copy the verified release index with the correct HF base URL.
- [ ] Add the simulated-training gallery, actual counts, task browsing and single-pair viewer selection. Use `#sim/<task>` gallery routes and `#sim/<task>/<n>` / `#sim/all/<n>` single-pair routes.
- [ ] Verify human/front/wrist playback, thumbnails, task switching, previous/next controls, keyboard navigation and direct-link reload. Preserve existing real-training and `#val/` routes.
- [ ] Handle unavailable/empty dataset indexes without crashes or misleading counts. Test media URLs and route behavior locally before publication.
- [ ] Commit and push the scoped code, documentation and site changes to GitHub using the repository's existing workflow; store large videos/data in HF and link them from GitHub rather than committing the full corpus.
- [ ] Confirm `.github/workflows/pages.yml` deploys the intended commit and completes successfully. Record the deployed commit and workflow evidence.

### Task 11: Live release verification and completion report

- [ ] Open the deployed Pages gallery and single-pair viewer; check `#sim/` links, actual counts, several task pairs, both robot views and HF-backed media playback.
- [ ] Verify live real-training and validation views still work and the published index matches the frozen release.
- [ ] Complete `docs/sim_dataset_execution_tracker.md` with accepted counts, covered tasks/families, spend/remaining credits, rejection totals, quality evidence, HF/GitHub/Pages links and material limitations.
- [ ] Deliver those links and measured results to the user. Mark the overall task complete only after post-generation review, verified HF upload, GitHub publication, documentation and live Pages checks all pass.

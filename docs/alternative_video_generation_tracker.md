# Alternative video generation tracker

Created: 2026-09-30. Status: matched comparison completed: **33/33 clips across 11 tasks**, plus **one successful targeted shape-placement clip on H3 Max Turbo**. Selected model: H3 Max Turbo.

## Objective

Compare Seedance 1.5 Pro, H3 Max Turbo and H3 Max using the same first/last images and identical submitted prompts, at 480p and a requested duration of five seconds. Determine whether higher cost improves task execution and yields more usable human demonstrations.

Existing baseline: [generated human demos first-pass analysis](generated_human_demos_first_pass_analysis.md). FastH3 has a confirmed example in 26/28 reviewed tasks, but many episodes fail; the inspected sample is not a representative full-batch failure rate. Tape sliding and lower-drawer closing remain 0/16. Model quality, prompt limitations and endpoint conflicts are separate hypotheses to track.

## Models and cost

Priority follows the proposed order. Prices below were checked against official fal pages on 2026-09-30; recheck at submission time and record actual billed cost.

| Priority | Model | fal endpoint | 480p price/minute | Estimated five-second cost | Purpose |
|---|---|---|---:|---:|---|
| 1 | Seedance 1.5 Pro, audio disabled | `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` | Approximately $0.69 at 16:9; varies with output dimensions/FPS | Approximately $0.0575 at 16:9 | Low-cost baseline |
| 2 | H3 Max Turbo | `minimax/h3-max-turbo/image-to-video` | $0.75 promotional; $1.50 standard | $0.0625 promotional; $0.125 standard | Affordable alternative to FastH3 |
| 3 | H3 Max | `minimax/h3-max/image-to-video` | $1.50 promotional; $3.00 standard | $0.125 promotional; $0.25 standard | Test whether maximum quality improves usability |

Seedance is token-priced: $1.20 per million video tokens without audio, with tokens = height × width × FPS × duration / 1024. The $0.69/min estimate assumes roughly 854×480 at 24 FPS; actual aspect ratio and dimensions can change it. Its native audio is optional, not absent: set `generate_audio=false`. [Official Seedance pricing](https://fal.ai/models/fal-ai/bytedance/seedance/v1.5/pro/image-to-video).

H3 model pages currently state that the 50% promotion ends **September 30**. Use standard prices for runs after expiry unless the live quote confirms another promotion. [Official Turbo pricing](https://fal.ai/models/minimax/h3-max-turbo/image-to-video), [official Max pricing](https://fal.ai/models/minimax/h3-max/image-to-video).

## Verified API controls

All three official schemas expose `image_url` and `end_image_url`, support 480p, and accept five seconds. First/last inputs must both be supplied; a starting-frame-only request does not satisfy this comparison.

| Model | Resolution value | Duration | Additional controls |
|---|---|---|---|
| Seedance 1.5 Pro | `480p` | `"5"` | `generate_audio=false`, `camera_fixed=true`; preserve source aspect ratio |
| H3 Max Turbo | `480P` | `5` | `prompt_expansion_mode="disabled"` to preserve submitted wording |
| H3 Max | `480P` | `5` | `prompt_expansion_mode="disabled"` to preserve submitted wording |

Sources: [Seedance API](https://fal.ai/models/fal-ai/bytedance/seedance/v1.5/pro/image-to-video/api), [Turbo API](https://fal.ai/models/minimax/h3-max-turbo/image-to-video/api), [Max API](https://fal.ai/models/minimax/h3-max/image-to-video/api). Record any model-generated audio separately; do not evaluate soundtrack quality or assume H3 exposes Seedance's audio-disable parameter.

## Controlled comparison protocol

1. Freeze the episode list, original `result_native.png` files, image SHA-256 hashes and submitted prompt text before generating. Start with the stored `v5_exclusion` episode prompts from baseline `meta.json`, byte-identical across the three models for each episode. Record template version and prompt SHA-256.
2. Request five seconds at 480p using the same source aspect ratio. Record returned dimensions, FPS and actual duration; do not assume nominal 480p implies identical canvases.
3. Disable H3 prompt expansion. Record all provider settings, seeds where supported, request IDs, timestamps, latency, retries and actual charges. A shared numeric seed does not imply equivalent randomness across models.
4. Generate one independent clip per episode/model. Preserve all first attempts, including failures; retries are separate attempts. Do not choose only the best result to calculate failure rate.
5. Save under `outputs/robot_removal/alternative_demos/<experiment>/<model>/<task>/episode_XXX/`, including `video.mp4`, `contact.png`, `meta.json` and exact prompt. Keep existing `demos` as the baseline.
6. Inspect every contact sheet; inspect the full clip when motion, hand count or contact is unclear. Compare native endpoints against supplied images and inspect intermediate action independently of endpoint similarity.
7. If modifying the prompt later, create a new version and rerun the same small varied panel across all models. Keep the initial unchanged-prompt comparison available.

Known prompt caveats: some stored FastH3 prompts reference 5.17 seconds despite a five-second requested duration, and `v5_exclusion` restricts person visibility more than the user's current acceptance rule. Record these as shared conditions rather than silently editing one model's prompt. Endpoint hardware/content mismatches and underspecified drawer mechanisms also remain confounders.

## Harder-task panel (after control validation)

Six varied transitions, three alternatives each: **18 clips**. At the indicative 16:9 Seedance rate, generation alone is approximately $1.47 with H3 promotional rates or $2.60 with standard H3 rates, excluding retries and aspect-ratio differences. These are proposed harder-stage inputs. Three Seedance requests were submitted before the staged-plan clarification; they are retained separately and do not count as control validation.

| Task / episode | Reason | Baseline evidence |
|---|---|---|
| `ReubenLim__so101_tape_in_square/episode_030` | Explicitly requested surface slide; baseline lifts instead | [sheet](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_030/contact.png) |
| `aiden-li__so101-close-lower-drawer/episode_000` | Mechanism error: tray lifted/rotated instead of slid | [sheet](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_000/contact.png) |
| `sattgle__clean2-test/episode_006` | Multi-object completion failure | [sheet](../outputs/robot_removal/demos/sattgle__clean2-test/episode_006/contact.png) |
| `Rorschach4153__so101_30_fold/episode_005` | Single-acting-hand constraint under cloth manipulation | [sheet](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_005/contact.png) |
| `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000` | Object inventory and hand consistency | [sheet](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000/contact.png) |
| `ricky0526__so101_pick_toy_to_plate_v3/episode_000` | Successful simple transfer as a control | [sheet](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) |

Inputs for each pair: first image under `outputs/robot_removal/v4/<task>/<episode>/result_native.png`; last image under `outputs/robot_removal/v4_last/<task>/<episode>/result_native.png`; baseline prompt under `outputs/robot_removal/demos/<task>/<episode>/meta.json`.

## Evaluation rubric

| Dimension | Pass criterion |
|---|---|
| Endpoint accuracy | Beginning and ending preserve supplied arrangement, inventory and appearance; flag input disagreements separately |
| Movement realism | Correct task motion/mechanism, plausible grasp/contact/support, no teleportation or lifted substitute for a required slide |
| Visual consistency | Stable objects and background; no duplication, morphing, entering hardware or extra acting hand |
| Completion | All requested transfers/actions complete; no unrelated or repeated manipulation |
| Usable demo | All semantic criteria pass; visible person and idle hands are acceptable, but only one hand performs, assists or stabilizes the action |

Report semantic failures / completed clips separately from API failures / submitted requests. List counts with denominators, task coverage with at least one success, and cost per usable demo. Six first attempts per model provide an exploratory comparison, not a reliable general failure-rate estimate. Additional matched repeats can measure variability after this first panel.

## Results tracker

Control and harder-task results are recorded below.

| Model | Submitted | Completed | Usable / reviewed | Endpoint failures | Motion failures | Consistency failures | Completion failures | Billed cost | Outputs |
|---|---:|---:|---|---|---|---|---|---|---|
| Seedance 1.5 Pro | 11 | 11 | 6/11 | 0/11 final arrangements | 2/11 (tape, drawer) | 3/11 (tape, folding, pouring) | 2/11 (drawer, wipe continuity) | Unknown | Linked below |
| H3 Max Turbo | 11 | 11 | 9/11 | 0/11 final arrangements | 1/11 (drawer) | 2/11 (drawer hands, pouring geometry) | 1/11 (drawer action) | Unknown | Linked below |
| H3 Max | 11 | 11 | 9/11 | 0/11 final arrangements | 1/11 (drawer) | 2/11 (drawer hands, pouring geometry) | 1/11 (drawer action) | Unknown | Linked below |

Per-attempt records must include task/episode, model, attempt, image/prompt hashes, settings, request ID, output paths, criterion verdicts, observed failure cause and review notes. Failure categories can overlap.

## Experiment log

| Date | Experiment | Status | Notes |
|---|---|---|---|
| 2026-09-30 | Alternative-model comparison setup | Planned | Created tracker; verified official endpoint schemas and current pricing; proposed six-transition panel; no paid requests submitted |

## Staged execution: control validation first

User-confirmed settings: **five seconds, 480p, first and last frame conditioning plus prompt**. Run all three models on these three known-success tasks, then inspect every result before submitting further harder-task requests. Each model receives identical stored `v5_exclusion` prompt text and endpoint image bytes for a given episode. Seedance audio is disabled; H3 prompt expansion is disabled.

| Control task | Episode | Seedance 1.5 Pro | H3 Max Turbo | H3 Max |
|---|---|---|---|---|
| Toy to plate | `ricky0526__so101_pick_toy_to_plate_v3/episode_000` | Pass | Pass | Pass |
| Brick into marked square | `lerobot__svla_so101_pickplace/episode_000` | Pass | Pass | Pass |
| Can into tray | `chirag1701__can/episode_000` | Pass | Pass | Pass |

Control output root: `outputs/robot_removal/alternative_demos/validation_controls_v5/`. Runner: [humangen/alternative_demos.py](../humangen/alternative_demos.py). Exact inputs and prompt hashes are frozen in each experiment's `manifest.json`; request IDs and settings are stored per episode in `meta.json`.

### Requests submitted before staged-plan clarification

The initial runner submitted Seedance tape/030, close-lower-drawer/000 and clean2/006 before the user requested easy controls first. Further submissions from that process were stopped. Existing provider requests are not resubmitted; their results will be retrieved and recorded separately under `outputs/robot_removal/alternative_demos/first_panel_v5/`. No H3 harder-task submissions were made by that process.

## Batching discount check

Checked official fal pricing and model/API pages on 2026-09-30. No published batch-specific discount was found for Seedance 1.5 Pro, H3 Max Turbo or H3 Max. Queueing/concurrent submission is used for throughput; budget each output at the model's published rate. General enterprise/volume arrangements exist for some fal offerings, but eligibility and discount for these three endpoints have not been established. See [fal pricing](https://fal.ai/pricing) and the model pricing links above. No sales outreach was made.

## Control inspection results

All nine native videos decoded successfully. Every control contact sheet was visually inspected and passed: 3/3 per model. This is contact-sheet validation at 2 Hz, not a full-video artifact certification. No quality advantage for Max is established by these simple controls.

**Requested settings versus returned media:** every request explicitly used five seconds and 480p. Seedance returned 752×560, duration 5.041667 seconds; Turbo and Max returned 640×480, duration 5.184 seconds. Seedance's native output does not match literal 480-pixel height, so the result is not an exact same-resolution comparison. Files are retained at native resolution; no silent resizing or replacement occurred.

| Model | Task / episode | Video | Sheet | Verdict |
|---|---|---|---|---|
| `h3_max` | `chirag1701__can/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/chirag1701__can/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/chirag1701__can/episode_000/contact.png) | Pass |
| `h3_max` | `lerobot__svla_so101_pickplace/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/lerobot__svla_so101_pickplace/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/lerobot__svla_so101_pickplace/episode_000/contact.png) | Pass |
| `h3_max` | `ricky0526__so101_pick_toy_to_plate_v3/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/ricky0526__so101_pick_toy_to_plate_v3/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) | Pass |
| `h3_max_turbo` | `chirag1701__can/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/chirag1701__can/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/chirag1701__can/episode_000/contact.png) | Pass |
| `h3_max_turbo` | `lerobot__svla_so101_pickplace/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/lerobot__svla_so101_pickplace/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/lerobot__svla_so101_pickplace/episode_000/contact.png) | Pass |
| `h3_max_turbo` | `ricky0526__so101_pick_toy_to_plate_v3/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/ricky0526__so101_pick_toy_to_plate_v3/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/h3_max_turbo/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) | Pass |
| `seedance_1_5_pro` | `chirag1701__can/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/chirag1701__can/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/chirag1701__can/episode_000/contact.png) | Pass |
| `seedance_1_5_pro` | `lerobot__svla_so101_pickplace/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/lerobot__svla_so101_pickplace/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/lerobot__svla_so101_pickplace/episode_000/contact.png) | Pass |
| `seedance_1_5_pro` | `ricky0526__so101_pick_toy_to_plate_v3/episode_000` | [video](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/ricky0526__so101_pick_toy_to_plate_v3/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/validation_controls_v5/seedance_1_5_pro/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) | Pass |

Control review provenance: [review.json](../outputs/robot_removal/alternative_demos/validation_controls_v5/review.json). Actual billed costs are not returned by these generation responses and remain unknown; do not treat estimated charges as billed amounts.

After reviewing controls, proceeded to the three harder transitions: tape/030, close-lower-drawer/000, clean2/006. Existing Seedance request IDs are reused for retrieval; H3 harder requests are submitted only after control inspection. No prompt changes.

## Harder-task inspection results

All nine harder clips completed and decoded successfully; all contact sheets were inspected. Native final frames were extracted and visually checked for all 18 clips: the target endpoint arrangements are present, including the failed drawer motions. Endpoint correctness therefore does not establish a correct demonstration. Seedance tape's invented strip disappears near the endpoint. No requests were regenerated.

| Task / episode | Seedance 1.5 Pro | H3 Max Turbo | H3 Max |
|---|---|---|---|
| Tape sliding / 030 | Fail: tilted/lifted roll; unrequested tape strip | Pass: surface translation, checked at 6 Hz | Pass: surface translation, checked at 6 Hz |
| Close lower drawer / 000 | Fail: flips/removes front lid | Fail: tray lifted out; extra support hand | Fail: tray lifted out; extra support hand |
| Two-shelf slippers / 006 | Pass: all slippers placed, one acting hand | Pass: all slippers placed, one acting hand | Pass: all slippers placed, one acting hand; visible head allowed |

| Model | Task / episode | Video | Sheet | Verdict / observation |
|---|---|---|---|---|
| `h3_max` | `ReubenLim__so101_tape_in_square/episode_030` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/ReubenLim__so101_tape_in_square/episode_030/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/ReubenLim__so101_tape_in_square/episode_030/contact.png) | Pass: 6 Hz reinspection shows the flat roll translated across the tabletop onto the blue square with one acting hand; no obvious lift/place substitute. |
| `h3_max` | `aiden-li__so101-close-lower-drawer/episode_000` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/aiden-li__so101-close-lower-drawer/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/aiden-li__so101-close-lower-drawer/episode_000/contact.png) | Fail: Wrong mechanism: Seedance manipulates/flips the front lid and then lifts it away; H3 lifts the compartment tray out instead of sliding closed. H3 sheets also show a second supporting hand. Final endpoint nevertheless matches the supplied closed state. |
| `h3_max` | `sattgle__clean2-test/episode_006` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/sattgle__clean2-test/episode_006/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max/sattgle__clean2-test/episode_006/contact.png) | Pass: One acting hand sequentially places both gray slippers below and all three white slippers on top; visible head in Max is acceptable. |
| `h3_max_turbo` | `ReubenLim__so101_tape_in_square/episode_030` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/ReubenLim__so101_tape_in_square/episode_030/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/ReubenLim__so101_tape_in_square/episode_030/contact.png) | Pass: 6 Hz reinspection shows the flat roll translated across the tabletop onto the blue square with one acting hand; no obvious lift/place substitute. |
| `h3_max_turbo` | `aiden-li__so101-close-lower-drawer/episode_000` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/aiden-li__so101-close-lower-drawer/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/aiden-li__so101-close-lower-drawer/episode_000/contact.png) | Fail: Wrong mechanism: Seedance manipulates/flips the front lid and then lifts it away; H3 lifts the compartment tray out instead of sliding closed. H3 sheets also show a second supporting hand. Final endpoint nevertheless matches the supplied closed state. |
| `h3_max_turbo` | `sattgle__clean2-test/episode_006` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/sattgle__clean2-test/episode_006/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/sattgle__clean2-test/episode_006/contact.png) | Pass: One acting hand sequentially places both gray slippers below and all three white slippers on top; visible head in Max is acceptable. |
| `seedance_1_5_pro` | `ReubenLim__so101_tape_in_square/episode_030` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/ReubenLim__so101_tape_in_square/episode_030/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/ReubenLim__so101_tape_in_square/episode_030/contact.png) | Fail: Roll is tilted/lifted and an unrequested long strip of tape unrolls across the scene, then disappears near the final endpoint. |
| `seedance_1_5_pro` | `aiden-li__so101-close-lower-drawer/episode_000` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/aiden-li__so101-close-lower-drawer/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/aiden-li__so101-close-lower-drawer/episode_000/contact.png) | Fail: Wrong mechanism: Seedance manipulates/flips the front lid and then lifts it away; H3 lifts the compartment tray out instead of sliding closed. H3 sheets also show a second supporting hand. Final endpoint nevertheless matches the supplied closed state. |
| `seedance_1_5_pro` | `sattgle__clean2-test/episode_006` | [video](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/sattgle__clean2-test/episode_006/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/seedance_1_5_pro/sattgle__clean2-test/episode_006/contact.png) | Pass: One acting hand sequentially places both gray slippers below and all three white slippers on top; visible head in Max is acceptable. |

Review provenance: [harder-task review.json](../outputs/robot_removal/alternative_demos/first_panel_v5/review.json). Detailed tape sheets are saved alongside each H3 tape video as `detail.png` (6 Hz). Native final frames are saved as `last.png` alongside all 18 videos. Every paired request's input/prompt SHA-256 values match across models. Requested duration/resolution and returned media properties are in `meta.json` and `probe.json`.

### Initial conclusion

- Controls: 3/3 usable for each model. Harder panel: Seedance 1/3, Turbo 2/3, Max 2/3. Overall semantic failures: Seedance 2/6, Turbo 1/6, Max 1/6. API failures: 0/18. Categories overlap.
- H3 Turbo and Max improve these tape/slipper examples over their FastH3 baselines. The drawer remains unresolved across all three; mechanism/prompt ambiguity is still relevant.
- **No observed pass-rate advantage for Max over Turbo in this six-transition panel.** Results do not establish a general ranking or rule out differences on more varied tasks/repeated samples.
- Seedance returned 752×560 for the 480p request; H3 returned 640×480. Treat this as a nominal-resolution test, with a native-output mismatch, rather than a strictly matched 480-pixel-height benchmark. All generation prompts remain unchanged.
- Pricing estimates are not actual billing. A cost-per-usable-demo conclusion awaits account billing data, particularly around H3's promotion expiry. No published batch discount was established.

| Date | Experiment | Status | Notes |
|---|---|---|---|
| 2026-09-30 | Three known-success controls × three models | Complete | 9/9 generated; 9/9 contact-sheet passes |
| 2026-09-30 | Three harder transitions × three models | Complete | 9/9 generated; Seedance 1/3, Turbo 2/3, Max 2/3 usable; three early Seedance submissions retrieved without duplicates |

## Extension: five new tasks, one episode per model

Status: complete: 15/15 clips generated and contact sheets inspected. Same five-second, nominal 480p, first/last-frame and frozen `v5_exclusion` prompt settings as the first comparison. Fifteen new requests; no prompt revisions.

| New task | Episode | Purpose |
|---|---|---|
| Cloth folding | `Rorschach4153__so101_30_fold/episode_005` | Single-hand feasibility and cloth motion |
| Pouring liquid | `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000` | Spout/geometry consistency and liquid contact |
| Unplug cable | `fbeltrao__so101_unplug_cable_4/episode_005` | Small connector identity and contact |
| Stack three blocks | `tenkau__SO101-Stack3Blocks/episode_004` | Support/contact and multistep completion |
| Sponge wiping | `stsqitx__clean/episode_005` | Duplicate-object failure and wiping continuity |

Outputs: `outputs/robot_removal/alternative_demos/extended_five_v5/`. After reviewing all fifteen new clips, select the model using cumulative usable-demo counts, failure types, nominal resolution discrepancies and published cost. A single sample per task measures this panel, not general failure probabilities.

## Five-task extension results and model selection

**Use H3 Max Turbo (`minimax/h3-max-turbo/image-to-video`) as the default video model for the next generation work.** It ties Max on every task's pass/fail outcome in this 11-task panel, costs half as much at both published promotional and standard rates, and produces literal 640×480 output. Seedance has fewer usable examples and returns 752×560 for the same nominal 480p request. This decision selects the default; it does not start a full-batch replacement.

| Additional task / episode | Seedance 1.5 Pro | H3 Max Turbo | H3 Max |
|---|---|---|---|
| Cloth folding / 005 | Fail: two active hands | Pass | Pass |
| Pouring liquid / 000 | Fail: spout appears/grows; inconsistent input geometry | Same failure | Same failure |
| Unplug cable / 005 | Pass | Pass | Pass |
| Stack three blocks / 004 | Pass | Pass | Pass |
| Sponge wiping / 005 | Fail: large smear disappears only at ending | Pass | Pass |

| Model | New five tasks | All 11 tasks | Semantic failures | Decision |
|---|---:|---:|---:|---|
| Seedance 1.5 Pro | 2/5 | 6/11 (54.5%) | 5/11 | Lower usable-task coverage; retain as an alternative, not the default |
| H3 Max Turbo | 4/5 | 9/11 (81.8%) | 2/11 | **Selected** |
| H3 Max | 4/5 | 9/11 (81.8%) | 2/11 | No additional passing task observed to justify higher price |

All 33 videos decoded successfully. All 33 contact sheets and native final frames were visually inspected; selected ambiguous motions received denser sheet inspection. This is not a claim that every full video frame was manually inspected. Images and submitted prompt hashes are identical across models for each task/episode, with five-second nominal 480p requests and no regenerations. Actual duration/resolution remains recorded in per-video metadata.

### Failure interpretation

- **Pouring input conflict:** inspected the supplied first and last images directly. The first kettle has no visible long spout; the last has one. Every model introduces a spout while animating. These are rejected as solid identity-consistent demos, but the evidence points to an input geometry conflict, not a clean measurement of inherent model capability. Repairing endpoint consistency is needed before judging a pouring-model winner.
- **Seedance wiping:** the late-clip detail sheet at 8 Hz shows a large smear persisting after release; the native final frame is clear. The ending therefore hides incomplete/implausible cleaning. H3 progressively clears the spill instead.
- **Seedance folding:** both hands visibly perform the fold, violating the one-acting-hand criterion. H3's briefly visible idle hand is allowed.
- **Persistent drawer failure:** the earlier drawer example still fails across all alternatives. Model selection does not resolve ambiguous mechanism descriptions.

At standard 480p list rates and five requested seconds, H3 Turbo is approximately $0.125/clip versus Max $0.25/clip. Applying this panel's observed 9/11 usable rate gives an illustrative $0.153 versus $0.306 per usable demo, excluding retries and any billing difference for actual returned duration. Promotional equivalents are half those amounts. **These are estimates, not account charges or guaranteed production costs.** Seedance's token pricing and different returned dimensions prevent a simple fixed cost comparison.

The panel is deliberately varied and includes difficult cases; it is not a random sample of all episodes. One attempt per task is insufficient to estimate repeatability or prove a statistical quality tie. Within the evidence collected, Turbo is the practical choice; Max has shown no added usable-task coverage.

### Extension evidence

| Model | Task / episode | Video | Contact sheet | Verdict and observation |
|---|---|---|---|---|
| `h3_max` | `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/contact.png) | Fail I/E: a long spout appears/grows. Source first image has no visible spout but last image has one; all models reconcile inconsistent object geometry. One-hand pour and return otherwise plausible. |
| `h3_max` | `Rorschach4153__so101_30_fold/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/Rorschach4153__so101_30_fold/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/Rorschach4153__so101_30_fold/episode_005/contact.png) | Pass: one acting hand performs both folds; briefly visible other hand does not assist. |
| `h3_max` | `fbeltrao__so101_unplug_cable_4/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/fbeltrao__so101_unplug_cable_4/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | Pass: one hand extracts the connector and sets it down; no extra hardware or assisting hand enters. |
| `h3_max` | `stsqitx__clean/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/stsqitx__clean/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/stsqitx__clean/episode_005/contact.png) | Pass: one hand wipes the spill with one sponge; spill progressively clears and sponge is released. No duplicate sponge observed. |
| `h3_max` | `tenkau__SO101-Stack3Blocks/episode_004` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/tenkau__SO101-Stack3Blocks/episode_004/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | Pass: one hand sequentially moves and stacks all three blocks on the green marker; plausible supported final stack. |
| `h3_max_turbo` | `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/contact.png) | Fail I/E: a long spout appears/grows. Source first image has no visible spout but last image has one; all models reconcile inconsistent object geometry. One-hand pour and return otherwise plausible. |
| `h3_max_turbo` | `Rorschach4153__so101_30_fold/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/Rorschach4153__so101_30_fold/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/Rorschach4153__so101_30_fold/episode_005/contact.png) | Pass: one acting hand performs both folds; briefly visible other hand does not assist. |
| `h3_max_turbo` | `fbeltrao__so101_unplug_cable_4/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/fbeltrao__so101_unplug_cable_4/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | Pass: one hand extracts the connector and sets it down; no extra hardware or assisting hand enters. |
| `h3_max_turbo` | `stsqitx__clean/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/stsqitx__clean/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/stsqitx__clean/episode_005/contact.png) | Pass: one hand wipes the spill with one sponge; spill progressively clears and sponge is released. No duplicate sponge observed. |
| `h3_max_turbo` | `tenkau__SO101-Stack3Blocks/episode_004` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/tenkau__SO101-Stack3Blocks/episode_004/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/h3_max_turbo/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | Pass: one hand sequentially moves and stacks all three blocks on the green marker; plausible supported final stack. |
| `seedance_1_5_pro` | `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/contact.png) | Fail I/E: a long spout appears/grows. Source first image has no visible spout but last image has one; all models reconcile inconsistent object geometry. One-hand pour and return otherwise plausible. |
| `seedance_1_5_pro` | `Rorschach4153__so101_30_fold/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/Rorschach4153__so101_30_fold/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/Rorschach4153__so101_30_fold/episode_005/contact.png) | Fail H: two hands actively grasp, fold and smooth the cloth. |
| `seedance_1_5_pro` | `fbeltrao__so101_unplug_cable_4/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/fbeltrao__so101_unplug_cable_4/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | Pass: one hand extracts the connector and sets it down; no extra hardware or assisting hand enters. |
| `seedance_1_5_pro` | `stsqitx__clean/episode_005` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/stsqitx__clean/episode_005/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/stsqitx__clean/episode_005/contact.png) | Fail C/P: wiping spreads the spill into a large brown smear, which remains after the sponge is released; the smear disappears at the very end without further wiping. Verified with 8 Hz late-clip detail sheet and native final frame. |
| `seedance_1_5_pro` | `tenkau__SO101-Stack3Blocks/episode_004` | [video](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/tenkau__SO101-Stack3Blocks/episode_004/video.mp4) | [sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | Pass: one hand sequentially moves and stacks all three blocks on the green marker; plausible supported final stack. |

Extension review provenance: [review.json](../outputs/robot_removal/alternative_demos/extended_five_v5/review.json). Model decision: [model_selection.json](../outputs/robot_removal/alternative_demos/model_selection.json). Seedance wipe detail: [8 Hz late-clip sheet](../outputs/robot_removal/alternative_demos/extended_five_v5/seedance_1_5_pro/stsqitx__clean/episode_005/detail.png). Native final frames are saved as `last.png` beside every video.

| Date | Experiment | Status | Notes |
|---|---|---|---|
| 2026-09-30 | Five new tasks × three models | Complete | 15/15 generated; Seedance 2/5, Turbo 4/5, Max 4/5 usable |
| 2026-09-30 | Model selection across 11 tasks | Complete | Choose H3 Max Turbo; 9/11 usable, matching Max at half its published price |

## Combined task coverage: original and alternative demos

Combining the original reviewed FastH3 demos with all 33 reviewed alternative-model clips gives **27/28 tasks with at least one confirmed usable human demo**. This counts reviewed evidence, not every unreviewed generated episode.

- **Still no confirmed demo:** `aiden-li__so101-close-lower-drawer` — 0/16 original reviewed episodes plus 0/3 alternative clips (episode 000 on Seedance, Turbo and Max). Wrong drawer mechanism persists: lifting/rotating/removing components rather than sliding closed; extra support hands also appear in H3 alternatives.
- **Newly covered:** `ReubenLim__so101_tape_in_square` — original 0/16; alternative episode 030 passes on both H3 Max Turbo and H3 Max, including 6 Hz motion reinspection.
- Every other task already has a confirmed original demo.

Usable tape example: [H3 Max Turbo video](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/ReubenLim__so101_tape_in_square/episode_030/video.mp4), [contact sheet](../outputs/robot_removal/alternative_demos/first_panel_v5/h3_max_turbo/ReubenLim__so101_tape_in_square/episode_030/contact.png).

## Targeted shape-placement follow-up

Status: complete — **confirmed success**. H3 Max Turbo only; `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000` is the green circular piece with a yellow knob. Original FastH3 introduced extra acting hands/pieces. Freeze the existing `v5_exclusion` prompt and supplied first/last `result_native.png` images. Request five seconds, `480P`, seed 0 and disabled prompt expansion. Inspect acting hand count, object inventory, correct circular recess and physical placement.

Output root: `outputs/robot_removal/alternative_demos/shape_green_v5/`. One attempt; no silent retries or prompt changes.

### Green-piece result

**Pass.** Re-inspected both the 2 Hz contact sheet and a denser 6 Hz sheet: one acting hand lifts the green circular piece, moves it into the matching circular recess on the right side of the board, releases it and withdraws. No extra acting arms/hands, added pieces or bowl were observed. Bystander hands already present at the image edge remain nonparticipating, which is allowed. Returned media is 640×480, 5.184000 seconds (requested five seconds, 480P).

- [Video](../outputs/robot_removal/alternative_demos/shape_green_v5/h3_max_turbo/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000/video.mp4)
- [Contact sheet](../outputs/robot_removal/alternative_demos/shape_green_v5/h3_max_turbo/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000/contact.png)
- [6 Hz inspection sheet](../outputs/robot_removal/alternative_demos/shape_green_v5/h3_max_turbo/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000/detail.png)
- [Review provenance](../outputs/robot_removal/alternative_demos/shape_green_v5/review.json)

This is direct improvement on the previously problematic green-piece episode 000 with unchanged submitted prompt and endpoint images. It establishes one successful example, not reliability across all shapes or episodes. Keep this targeted follow-up separate from the matched 11-task comparison: Turbo now has 10/12 inspected successful clips, while Seedance and Max were not tested on this additional episode. Combined task coverage remains 27/28; lower-drawer closing is the only task without a confirmed reviewed demo.

## Green circular piece: five additional episodes

Status: complete — 5/5 placement and single-hand passes; four have stable piece appearance, 007 follows a pale/speckled supplied ending-frame appearance. User reviewed the results and accepted them. Episodes **001, 003, 005, 007, 009** all move the green circular piece, selected across the available circle episodes to include differing images and stored left/top entry prompts. Episode 000 is excluded because it was already tested. No semicircle or other-color episodes are included.

Each request uses its stored `v5_exclusion` prompt unchanged, both original endpoint `result_native.png` images, five seconds, `480P`, seed 0 and disabled prompt expansion. Outputs: `outputs/robot_removal/alternative_demos/shape_green_more_v5/`. Inspect every result for correct recess, one acting hand and unchanged object inventory, then record outcomes below.

## Full regeneration: H3 Max Turbo

Authorized and launched 2026-09-30. Scope: **1,960 episodes across all 28 tasks**, saved in `outputs/robot_removal/demos_h3_max_turbo/<task>/episode_XXX/`. Existing FastH3 demos remain intact. Each output includes `video.mp4`, `contact.png`, `meta.json`, `prompt.txt`, provider response and media probe.

Settings: five-second requested duration, `480P`, first/last original image conditioning, unchanged stored `v5_exclusion` prompts, seed 0, disabled prompt expansion. Input and prompt hashes were verified for all 1,960 rows before submissions. No revised prompt template.

Runner: [humangen/fal_bulk_demos.py](../humangen/fal_bulk_demos.py), eight independent workers. The process resumes saved queue IDs rather than resubmitting uncertain requests. Transient retrieval failures may retry against the same request. No automatic paid regeneration of semantic failures.

Live artifacts:

- [Status](../outputs/robot_removal/demos_h3_max_turbo/_run/status.json)
- [Generation log](../outputs/robot_removal/demos_h3_max_turbo/_run/generation.log)
- [Frozen manifest](../outputs/robot_removal/demos_h3_max_turbo/_run/manifest.jsonl)

Initial generation estimate: approximately 45–90 minutes, subject to queue/concurrency changes. Full-batch estimated cost: $122.50 at the listed promotional rate or $245 at standard rate, before retries/taxes and actual-duration billing differences. Promotion currently lists September 30 expiry. Published/decoded counts are not semantic pass counts; quality inspection follows generation.

### Full regeneration completion

All **1,960/1,960 episodes** now have complete videos, contact sheets and metadata in `outputs/robot_removal/demos_h3_max_turbo/`. Initial run completed 1,954 in approximately 42 minutes; six saved jobs returned temporary HTTP 500 retrieval errors. On 2026-09-30, recovered those six via their existing request IDs, with no duplicate generation submissions. All output-presence and stored prompt-hash checks passed. Generation and per-clip decode checks are complete; full-batch semantic quality review is not yet complete.

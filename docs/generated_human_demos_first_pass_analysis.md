# Generated human demos: first-pass contact-sheet analysis

Reviewed: 2026-09-30. Model: FastH3. Prompt version: `v5_exclusion`. Requested duration: 5 seconds.

## Scope and success criteria

Inspected **248 distinct episodes across 28 tasks: 16 each for the three initially zero-success tasks, eight each for the other 25 tasks**. This includes the original first/middle/last samples plus **five additional episodes per task (140 new sheets)**, chosen near 10%, 25%, 40%, 65%, and 85% of the sorted available episodes, excluding already reviewed episodes. A further **eight unreviewed episodes each** from tape sliding, lower-drawer closing and two-shelf slipper placement were selected evenly across remaining manifest entries (24 new sheets, no repeats). Counts describe only inspected samples, not all 1,960 generated videos. Contact sheets sample at 2 Hz and can miss brief artifacts.

A confirmed success requires the intended action and target outcome to look plausible, **only one hand doing the task**, and no obvious object-identity, background, or physical-contact failure. **More of the person may be visible, and an idle hand is acceptable.** A second hand manipulating, assisting, stabilizing, or taking over an action is a failure. A correct final frame alone is insufficient. Background changes caused by differing supplied endpoints remain distinguished from model inventions.

Updated following the user's clarification: visibility-only failures are now successes when the task action otherwise appears plausible. Original slipper/023 and 046 are restored to success. The previous assessments are preserved in [_run/task_contact_review_before_visibility_clarification.json](../outputs/robot_removal/demos/_run/task_contact_review_before_visibility_clarification.json). No prompt or video was changed.

## Validated human-demo handoff

**26 of 28 tasks have at least one solid generated human demo among inspected samples.** Every task/episode pair below was visually rechecked from its contact sheet for this handoff. Each passes the stated task, single-acting-hand, object-consistency and plausible-contact criteria at the sampled frames. These are model-generated demos, not recordings of real people. Contact sheets sample at 2 Hz, so this validation does not rule out brief between-frame artifacts.

No confirmed example was found for `ReubenLim__so101_tape_in_square` or `aiden-li__so101-close-lower-drawer` (0/16 each).

One validated example per covered task follows. Links are relative to this document in `docs/`; videos and sheets remain in `outputs/robot_removal/demos/`.

| Task | Episode | Video | Contact sheet | Validation observation |
|---|---|---|---|---|
| `Cornito__so101_tea2` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_000/contact.png) | One hand transfers the tea bag into the mug. |
| `EverNorif__so101-table-cleanup` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_000/contact.png) | One hand collects the pens into the holder. |
| `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place` | `episode_029` | [video.mp4](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_029/video.mp4) | [contact.png](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_029/contact.png) | One hand transfers the pink piece into the board. |
| `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid` | `episode_006` | [video.mp4](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_006/video.mp4) | [contact.png](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_006/contact.png) | One hand pours into the cup and returns the vessel. |
| `Rorschach4153__so101_30_fold` | `episode_042` | [video.mp4](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_042/video.mp4) | [contact.png](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_042/contact.png) | One hand folds the cloth twice. |
| `Tear4Pixelation__lego2` | `episode_002` | [video.mp4](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_002/video.mp4) | [contact.png](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_002/contact.png) | One hand places the green brick onto the yellow brick; background remains stable. |
| `aiden-li__so101-close-upper-drawer` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_000/contact.png) | One hand slides the drawer closed without lifting it clear. |
| `aiden-li__so101-grabtissue` | `episode_014` | [video.mp4](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_014/video.mp4) | [contact.png](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_014/contact.png) | One hand pulls out and holds the tissue. |
| `aiden-li__so101-open-lower-drawer` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_000/contact.png) | One hand pulls the drawer outward along the case. |
| `aiden-li__so101-open-upper-drawer` | `episode_039` | [video.mp4](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_039/video.mp4) | [contact.png](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_039/contact.png) | One hand pulls the upper drawer outward; later visible hand does not assist. |
| `b3rnd__record-50-episodes` | `episode_043` | [video.mp4](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_043/video.mp4) | [contact.png](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_043/contact.png) | One hand stacks the marked cube onto the red cube. |
| `chirag1701__can` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/chirag1701__can/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/chirag1701__can/episode_000/contact.png) | One hand places the can in the tray; other hand stays idle. |
| `chirag1701__sandwich` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/chirag1701__sandwich/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/chirag1701__sandwich/episode_000/contact.png) | One hand places the package into the box. |
| `fbeltrao__so101_unplug_cable_4` | `episode_005` | [video.mp4](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_005/video.mp4) | [contact.png](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | One hand unplugs the connector and sets it down; no entering hardware observed. |
| `k1000dai__standup_petbottle` | `episode_004` | [video.mp4](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_004/video.mp4) | [contact.png](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_004/contact.png) | One hand stands the bottle on the white marker and withdraws. |
| `lerobot__svla_so101_pickplace` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_000/contact.png) | One hand transfers the brick into the marked square. |
| `pbvr__so101_test002` | `episode_001` | [video.mp4](../outputs/robot_removal/demos/pbvr__so101_test002/episode_001/video.mp4) | [contact.png](../outputs/robot_removal/demos/pbvr__so101_test002/episode_001/contact.png) | One hand places the orange block into the box. |
| `pbvr__so101_test005` | `episode_009` | [video.mp4](../outputs/robot_removal/demos/pbvr__so101_test005/episode_009/video.mp4) | [contact.png](../outputs/robot_removal/demos/pbvr__so101_test005/episode_009/contact.png) | One hand places all three blocks into the box. |
| `psg777__combinedtape` | `episode_013` | [video.mp4](../outputs/robot_removal/demos/psg777__combinedtape/episode_013/video.mp4) | [contact.png](../outputs/robot_removal/demos/psg777__combinedtape/episode_013/contact.png) | One hand moves the tape roll out of the blue tray. |
| `ricky0526__so101_pick_toy_to_plate_v3` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) | One hand transfers the toy onto the plate. |
| `sattgle__clean-test` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/sattgle__clean-test/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/sattgle__clean-test/episode_000/contact.png) | One hand places both gray slippers onto the shelf. |
| `sattgle__clean2-test` | `episode_010` | [video.mp4](../outputs/robot_removal/demos/sattgle__clean2-test/episode_010/video.mp4) | [contact.png](../outputs/robot_removal/demos/sattgle__clean2-test/episode_010/contact.png) | One hand places gray slippers below and white slippers on top. |
| `stsqitx__clean` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/stsqitx__clean/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/stsqitx__clean/episode_000/contact.png) | One hand wipes with one sponge and releases it. |
| `tenkau__SO101-Stack3Blocks` | `episode_004` | [video.mp4](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_004/video.mp4) | [contact.png](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | One hand sequentially stacks the three black blocks on the marker. |
| `tenkau__so101_color_block` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_000/contact.png) | One hand stacks the gray and black blocks onto the red block. |
| `un1c0rnio__so101_sock_stowing_3pair2` | `episode_000` | [video.mp4](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_000/video.mp4) | [contact.png](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_000/contact.png) | One hand transfers the red sock into the bottom container. |

During this recheck, bottle/000 was downgraded for unrelated tape adjustment and unplug-cable/051 for entering robot hardware. Bottle/004 and unplug-cable/005 are the validated handoff examples. These corrections change sample success counts but do not reduce task coverage.

Validation provenance and sheet hashes: [_run/handoff_validated_demos.json](../outputs/robot_removal/demos/_run/handoff_validated_demos.json). Earlier assessments: [_run/task_contact_review_before_handoff_recheck.json](../outputs/robot_removal/demos/_run/task_contact_review_before_handoff_recheck.json). No regeneration occurred.

## Results by task

Example links point to contact sheets; `video.mp4` and `meta.json` are alongside each sheet. A dash means no example of that outcome was found among the inspected episodes for that task. “Works” means all inspected episodes passed; “Mixed” means some passed and some failed; zero means no confirmed clean sample.

| Task | Reviewed episodes | Successful / reviewed | Result | Success example | Failure example | Observed failure / cause category |
|---|---|---:|---|---|---|---|
| `Cornito__so101_tea2` | 000, 009, 024, 037, 048, 062, 085, 097 | 6/8 | Mixed | [episode_000](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_000/contact.png) | [episode_024](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_024/contact.png) | E/I/A: robot hardware enters in 024; repeated/duplicate bag transfer in 062. Body visibility is allowed. |
| `EverNorif__so101-table-cleanup` | 000, 008, 020, 032, 040, 052, 068, 081 | 3/8 | Mixed | [episode_000](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_000/contact.png) | [episode_040](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_040/contact.png) | E/H: ending changes holder contents/visibility; extra acting hand in 081. Additional: Two acting hands; filled holder is removed.; Two acting hands collect pens.; Two acting hands and robot hardware; holder contents disappear at ending. |
| `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place` | 000, 007, 018, 029, 037, 047, 063, 074 | 2/8 | Mixed | [episode_074](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_074/contact.png) | [episode_000](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000/contact.png) | H/I: added participating hands and duplicate/new pieces. Additional: Extra hands, bowl and multiple new pieces enter.; Second hand handles a different piece.; Additional red/green pieces and another hand enter.; Second hand handles another piece. |
| `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid` | 000, 006, 014, 023, 029, 038, 049, 058 | 6/8 | Mixed | [episode_029](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_029/contact.png) | [episode_000](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000/contact.png) | I: kettle geometry changes; a long spout grows in 000. Additional: Vessel is carried out of frame after pouring rather than returned; ending vessel absent. |
| `ReubenLim__so101_tape_in_square` | 000, 001, 006, 009, 015, 018, 024, 026, 030, 034, 038, 042, 050, 051, 058, 059 | 0/16 | No confirmed clean sample | — | [episode_000](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_000/contact.png) | M/C: lifted transfer instead of requested surface slide in all eight new samples; 034 removes the ring without placement. |
| `Rorschach4153__so101_30_fold` | 000, 005, 012, 020, 025, 032, 042, 049 | 1/8 | Mixed | [episode_042](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_042/contact.png) | [episode_000](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_000/contact.png) | H: two-hand folding overrides the single-hand constraint. Additional: Two acting hands fold cloth. |
| `Tear4Pixelation__lego2` | 000, 002, 005, 008, 010, 012, 016, 019 | 2/8 | Mixed | [episode_019](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_019/contact.png) | [episode_000](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_000/contact.png) | E: robot/background differs between endpoint images; interpolation adds/removes it. Additional: Background robot disappears during withdrawal.; Background robot appears during ending transition. |
| `aiden-li__so101-close-lower-drawer` | 000, 001, 005, 008, 013, 016, 021, 023, 027, 031, 035, 039, 047, 048, 055, 056 | 0/16 | No confirmed clean sample | — | [episode_000](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_000/contact.png) | M/H: all eight new samples lift/rotate the tray instead of sliding it closed; supporting hand in 001, 008 and 039. |
| `aiden-li__so101-close-upper-drawer` | 000, 015, 035, 054, 067, 087, 112, 132 | 1/8 | Mixed | [episode_000](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_000/contact.png) | [episode_067](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_067/contact.png) | M/I: tray lifted/rotated in 067/132; compartment contents change. Additional: Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |
| `aiden-li__so101-grabtissue` | 000, 014, 035, 056, 071, 092, 120, 141 | 3/8 | Mixed | [episode_014](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_014/contact.png) | [episode_000](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_000/contact.png) | H/P/A: two-hand tissue handling or other-object manipulation in several samples; unsupported tissue in 000/120. Visible body alone is allowed. |
| `aiden-li__so101-open-lower-drawer` | 000, 014, 034, 054, 068, 088, 115, 135 | 1/8 | Mixed | [episode_000](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_000/contact.png) | [episode_068](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_068/contact.png) | M/H: tray lifting and two-hand support in 068/135. Additional: Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |
| `aiden-li__so101-open-upper-drawer` | 001, 016, 039, 060, 075, 096, 127, 148 | 1/8 | Mixed | [episode_039](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_039/contact.png) | [episode_001](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_001/contact.png) | H/M: extra supporting hand; tray lifted/rotated in 148. Additional: Tray is lifted and contents change.; Additional hand/handling appears near the front container.; Tray lifted/rotated; second supporting hand.; Supporting hand appears below the case during opening. |
| `b3rnd__record-50-episodes` | 000, 004, 011, 017, 022, 028, 037, 043 | 1/8 | Mixed | [episode_043](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_043/contact.png) | [episode_000](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_000/contact.png) | E/H: hardware appears in 000; acting-arm handoff in 022. Additional: Robot removed after stacking; additional acting-arm switch.; Two hands/arms hand off the cube.; Robot hardware moves and disappears.; Robot brought into scene; two hands handle cube/hardware.; Acting-arm switch during cube transfer. |
| `chirag1701__can` | 000, 005, 012, 020, 025, 032, 042, 049 | 8/8 | Works in sampled episodes | [episode_000](../outputs/robot_removal/demos/chirag1701__can/episode_000/contact.png) | — | None observed under the clarified criterion. |
| `chirag1701__sandwich` | 000, 005, 012, 020, 025, 032, 042, 050 | 4/8 | Mixed | [episode_000](../outputs/robot_removal/demos/chirag1701__sandwich/episode_000/contact.png) | [episode_025](../outputs/robot_removal/demos/chirag1701__sandwich/episode_025/contact.png) | I/A/E: package duplication and repeated transfer in 025; appearance changes in 050. Additional: Package duplicates and transfer repeats.; Package duplicates/repeated transfer; torso enters. |
| `fbeltrao__so101_unplug_cable_4` | 000, 005, 013, 020, 026, 033, 043, 051 | 1/8 | Mixed | [episode_005](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | [episode_000](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_000/contact.png) | I: connector/cable morphs into a different geometry in 000/026. Additional: Robot hardware appears during unplugging.; Robot hardware appears; connector/cable changes geometry.; Robot enters and is handled by another hand.; Robot hardware enters and moves during ending. Handoff recheck: 051 also has orange robot hardware entering; only 005 remains confirmed. |
| `k1000dai__standup_petbottle` | 000, 004, 009, 014, 018, 023, 031, 036 | 7/8 | Mixed | [episode_004](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_004/contact.png) | [episode_000](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_000/contact.png) | A: episode 000 adjusts unrelated green table tape after standing the bottle. |
| `lerobot__svla_so101_pickplace` | 000, 005, 012, 020, 025, 032, 042, 049 | 8/8 | Works in sampled episodes | [episode_000](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_000/contact.png) | — | None observed under the clarified criterion. |
| `pbvr__so101_test002` | 001, 018, 043, 068, 085, 110, 147, 173 | 8/8 | Works in sampled episodes | [episode_001](../outputs/robot_removal/demos/pbvr__so101_test002/episode_001/contact.png) | — | None observed under the clarified criterion. |
| `pbvr__so101_test005` | 000, 009, 023, 037, 047, 060, 080, 094 | 7/8 | Mixed | [episode_047](../outputs/robot_removal/demos/pbvr__so101_test005/episode_047/contact.png) | [episode_000](../outputs/robot_removal/demos/pbvr__so101_test005/episode_000/contact.png) | H: two acting hands in 000; remaining seven samples pass. |
| `psg777__combinedtape` | 000, 013, 034, 054, 067, 087, 113, 133 | 3/8 | Mixed | [episode_013](../outputs/robot_removal/demos/psg777__combinedtape/episode_013/contact.png) | [episode_000](../outputs/robot_removal/demos/psg777__combinedtape/episode_000/contact.png) | A/H/I: unrelated tool handling, extra hand, and tool entering the scene. Additional: Two acting hands handle roll and box. |
| `ricky0526__so101_pick_toy_to_plate_v3` | 000, 008, 019, 030, 038, 049, 065, 076 | 7/8 | Mixed | [episode_000](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_000/contact.png) | [episode_030](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_030/contact.png) | Additional review: Robot hardware is handled and removed with another hand. |
| `sattgle__clean-test` | 000, 005, 012, 018, 023, 030, 039, 046 | 8/8 | Works in sampled episodes | [episode_000](../outputs/robot_removal/demos/sattgle__clean-test/episode_000/contact.png) | — | None observed under the clarified criterion. |
| `sattgle__clean2-test` | 000, 001, 002, 004, 006, 007, 009, 010, 012, 014, 015, 017, 019, 020, 022, 023 | 3/16 | Mixed | [episode_010](../outputs/robot_removal/demos/sattgle__clean2-test/episode_010/contact.png) | [episode_000](../outputs/robot_removal/demos/sattgle__clean2-test/episode_000/contact.png) | C/H: earlier samples leave white slippers unplaced; five of eight new samples use two acting hands. New 010, 019 and 022 complete the arrangement with one acting hand. |
| `stsqitx__clean` | 000, 002, 005, 008, 010, 012, 016, 019 | 5/8 | Mixed | [episode_000](../outputs/robot_removal/demos/stsqitx__clean/episode_000/contact.png) | [episode_005](../outputs/robot_removal/demos/stsqitx__clean/episode_005/contact.png) | Additional review: A duplicate sponge remains while another is used for wiping. |
| `tenkau__SO101-Stack3Blocks` | 000, 004, 009, 014, 018, 023, 030, 035 | 1/8 | Mixed | [episode_004](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | [episode_000](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_000/contact.png) | P/H: questionable block alignment/contact; two acting hands in 018. Additional: Two acting hands; apparent additional block and unstable arrangement.; Block alignment/contact is questionable.; Two acting hands rearrange and stack blocks.; Additional acting/supporting hand during stacking. |
| `tenkau__so101_color_block` | 000, 003, 007, 012, 015, 019, 025, 029 | 3/8 | Mixed | [episode_000](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_000/contact.png) | [episode_029](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_029/contact.png) | I/E: block appearance changes and robot hardware moves in 029. Additional: Background robot enters after stacking.; Background robot changes/moves during task.; Robot hardware is handled/moves after stacking. |
| `un1c0rnio__so101_sock_stowing_3pair2` | 000, 002, 006, 010, 013, 016, 021, 026 | 7/8 | Mixed | [episode_000](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_000/contact.png) | [episode_026](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_026/contact.png) | H: two-hand handoff in 026. Idle hands and visible body in other samples are allowed. |

**Total after handoff recheck: 107/248 inspected samples confirmed successful (43.1%).** This zero-task follow-up contributed 3/24; the previous five-per-task extension contributed 66/140; original samples contributed 40/84 under the clarified criterion. The handoff recheck corrected bottle/000 and unplug-cable/051 to failures (details below). This is a descriptive sample, not a statistically representative full-batch pass rate.

## Findings from the additional review

- Simple brick transfer, can transfer, single-block box placement and single-shelf slipper placement pass all eight samples. Bottle standing is now 7/8 after the handoff recheck identified unrelated tape adjustment in 000.
- After eight more episodes each, lower-drawer closing and tape sliding remain at **0/16**. The two-shelf slipper task now has **3/16** clean samples (010, 019, 022); five other new samples use two acting hands.
- Folding/042 and Stack3Blocks/004 provide clean examples despite frequent failures in those tasks.
- Sponge duplication in clean/005, 008 and 012 reduces a previously clean task to mixed.
- Toy transfer/030 manipulates/removes robot hardware; the task is now mixed.
- Single-shelf slipper placement passes all eight samples under the clarified criterion; visible heads/bodies are allowed.
- Pouring/049 removes the vessel; its supplied ending frame also omits the vessel, conflicting with the prompt’s return-and-release sequence. This is an endpoint/prompt conflict, rather than evidence that the model invented that disappearance.

## Failure causes: evidence and interpretation

- **M — motion/mechanism:** drawers are lifted or flipped instead of sliding; tape is lifted instead of pushed. Drawer prompts say only “pushes it closed” / “pulls it open.” The template takes only the first clause of role appearance text, which can discard mechanism information. Missing motion constraints are a plausible contributor. For tape, the prompt already explicitly says “slides it across the surface,” so that failure demonstrates model non-adherence, not an absent action instruction.
- **H — acting-hand constraint:** extra acting or supporting hands and handoffs violate the user’s single-acting-hand requirement. Visible heads, torsos and idle hands are acceptable and are not failures. A learned preference for two-hand folding, handling or stabilization is a plausible explanation, not a proven cause.
- **E — endpoint disagreement:** supplied first/last images can differ in robot hardware, background, contents or container visibility. The model must reconcile those differences while the template asks for an unchanged inventory/background. This conflict can explain some additions, disappearances and abrupt endings. It does not justify an incorrect intermediate action.
- **I — identity/geometry:** spouts, cable connectors, packages or blocks change appearance, or objects duplicate. Additional clean/005, 008 and 012 show duplicate sponges, and sandwich/012 and 032 show duplicate packages. The sheets show identity/geometry failures; their exact internal cause cannot be established from images alone.
- **C — incomplete task:** an intended object remains unplaced, notably a white slipper in clean2-test; the five additional samples confirm repeated omissions. Reaching a similar ending composition does not establish that every requested transfer occurred.
- **A — unnecessary action:** repeated transfers or handling unrelated tools/robot hardware appear. These actions are not required by the task.
- **P — physical continuity:** unsupported tissue and questionable stacking contact/alignment. Visual endpoint conditioning does not guarantee plausible contact or support during motion.

## Concrete prompt evidence

- Close-lower-drawer/000: “grasps the black lower drawer … and pushes it closed.” The generated hand lifts and rotates the tray. Its source annotation explicitly describes the drawer sliding in and disappearing under the case lid.
- Tape-in-square/030: “slides it across the surface to the blue square.” The generated hand lifts the roll and sets it down.
- The shared template says “One acting hand and forearm performs the task” and “No other hands or objects enter the scene at any time.” Multiple sampled tasks violate these instructions.

## Status

All 1,960 episodes have been published according to the generation status. That count is separate from semantic success. No regeneration is authorized while this diagnosis is being discussed. The report does not claim every video has been manually reviewed or that the final full-batch technical audit is complete.

Zero-task follow-up: [_run/zero_additional8_assessments.json](../outputs/robot_removal/demos/_run/zero_additional8_assessments.json), combined sheets [_run/review_zero_additional8/index.json](../outputs/robot_removal/demos/_run/review_zero_additional8/index.json). Assessments before this extension: [_run/task_contact_review_before_zero_followup.json](../outputs/robot_removal/demos/_run/task_contact_review_before_zero_followup.json).

Review provenance: [_run/task_contact_review.json](../outputs/robot_removal/demos/_run/task_contact_review.json). Additional episode notes: [_run/additional5_assessments.json](../outputs/robot_removal/demos/_run/additional5_assessments.json). Original combined sheets: [_run/review/index.json](../outputs/robot_removal/demos/_run/review/index.json). Additional combined sheets: [_run/review_additional5/index.json](../outputs/robot_removal/demos/_run/review_additional5/index.json).


## Additional episode judgments

Each newly inspected episode is scored under the clarified criterion.

### Cornito__so101_tea2

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_009](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_009/contact.png) | Yes | One hand transfers/dips bag and withdraws. |
| [episode_024](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_024/contact.png) | No | Torso enters; turquoise robot hardware is brought into scene. |
| [episode_037](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_037/contact.png) | Yes | Single hand transfers bag to mug. |
| [episode_062](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_062/contact.png) | No | A bag is dipped, then another bag is taken from the box and dipped: repeated/duplicate transfer, not merely body visibility. |
| [episode_085](../outputs/robot_removal/demos/Cornito__so101_tea2/episode_085/contact.png) | Yes | Single hand transfers bag into mug. |

### EverNorif__so101-table-cleanup

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_008](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_008/contact.png) | No | Two acting hands; filled holder is removed. |
| [episode_020](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_020/contact.png) | No | Two acting hands collect pens. |
| [episode_032](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_032/contact.png) | Yes | Single hand collects all pens into holder. |
| [episode_052](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_052/contact.png) | No | Two acting hands and robot hardware; holder contents disappear at ending. |
| [episode_068](../outputs/robot_removal/demos/EverNorif__so101-table-cleanup/episode_068/contact.png) | Yes | Single hand transfers pens into holder. |

### LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_007](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_007/contact.png) | No | Extra hands, bowl and multiple new pieces enter. |
| [episode_018](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_018/contact.png) | No | Second hand handles a different piece. |
| [episode_029](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_029/contact.png) | Yes | One acting hand transfers pink piece into board. |
| [episode_047](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_047/contact.png) | No | Additional red/green pieces and another hand enter. |
| [episode_063](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_063/contact.png) | No | Second hand handles another piece. |

### LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_006](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_006/contact.png) | Yes | Single hand pours and returns vessel. |
| [episode_014](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_014/contact.png) | Yes | Single hand pours and returns vessel. |
| [episode_023](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_023/contact.png) | Yes | Single hand pours and returns vessel. |
| [episode_038](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_038/contact.png) | Yes | Single hand pours and returns vessel; blue mark is uncovered beneath vessel. |
| [episode_049](../outputs/robot_removal/demos/LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_049/contact.png) | No | Vessel is carried out of frame after pouring rather than returned; ending vessel absent. |

### ReubenLim__so101_tape_in_square

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_006](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_006/contact.png) | No | Roll is lifted and placed, not slid along the surface. |
| [episode_015](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_015/contact.png) | No | Roll is lifted and placed, not slid along the surface. |
| [episode_024](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_024/contact.png) | No | Roll is lifted and placed, not slid along the surface. |
| [episode_038](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_038/contact.png) | No | Roll is lifted and placed, not slid along the surface. |
| [episode_050](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_050/contact.png) | No | Roll is lifted and placed, not slid along the surface. |

### Rorschach4153__so101_30_fold

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_005/contact.png) | No | Two acting hands fold cloth. |
| [episode_012](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_012/contact.png) | No | Two acting hands fold cloth. |
| [episode_020](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_020/contact.png) | No | Two acting hands fold cloth. |
| [episode_032](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_032/contact.png) | No | Two acting hands fold cloth. |
| [episode_042](../outputs/robot_removal/demos/Rorschach4153__so101_30_fold/episode_042/contact.png) | Yes | One hand makes the folds and withdraws. |

### Tear4Pixelation__lego2

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_002](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_002/contact.png) | Yes | Single hand stacks green brick; background hardware stays in place. |
| [episode_005](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_005/contact.png) | No | Background robot disappears during withdrawal. |
| [episode_008](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_008/contact.png) | No | Background robot appears during ending transition. |
| [episode_012](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_012/contact.png) | No | Background robot disappears during withdrawal. |
| [episode_016](../outputs/robot_removal/demos/Tear4Pixelation__lego2/episode_016/contact.png) | No | Background robot appears during ending transition. |

### aiden-li__so101-close-lower-drawer

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_005/contact.png) | No | Tray is lifted/rotated rather than slid closed; supporting hand appears in several samples. |
| [episode_013](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_013/contact.png) | No | Tray is lifted/rotated rather than slid closed; supporting hand appears in several samples. |
| [episode_021](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_021/contact.png) | No | Tray is lifted/rotated rather than slid closed; supporting hand appears in several samples. |
| [episode_035](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_035/contact.png) | No | Tray is lifted/rotated rather than slid closed; supporting hand appears in several samples. |
| [episode_047](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_047/contact.png) | No | Tray is lifted/rotated rather than slid closed; supporting hand appears in several samples. |

### aiden-li__so101-close-upper-drawer

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_015](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_015/contact.png) | No | Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |
| [episode_035](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_035/contact.png) | No | Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |
| [episode_054](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_054/contact.png) | No | Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |
| [episode_087](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_087/contact.png) | No | Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |
| [episode_112](../outputs/robot_removal/demos/aiden-li__so101-close-upper-drawer/episode_112/contact.png) | No | Tray lifted/rotated instead of slid closed; contents/supporting hands also vary. |

### aiden-li__so101-grabtissue

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_014](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_014/contact.png) | Yes | One acting hand extracts tissue and withdraws. |
| [episode_035](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_035/contact.png) | No | Body and second hand enter; blue container is moved. |
| [episode_056](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_056/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_092](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_092/contact.png) | Yes | One hand extracts tissue and withdraws. |
| [episode_120](../outputs/robot_removal/demos/aiden-li__so101-grabtissue/episode_120/contact.png) | No | Released tissue appears unsupported against wall at ending. |

### aiden-li__so101-open-lower-drawer

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_014](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_014/contact.png) | No | Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |
| [episode_034](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_034/contact.png) | No | Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |
| [episode_054](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_054/contact.png) | No | Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |
| [episode_088](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_088/contact.png) | No | Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |
| [episode_115](../outputs/robot_removal/demos/aiden-li__so101-open-lower-drawer/episode_115/contact.png) | No | Tray/lid is lifted and rotated instead of a constrained drawer slide; several samples use two hands. |

### aiden-li__so101-open-upper-drawer

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_016](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_016/contact.png) | No | Tray is lifted and contents change. |
| [episode_039](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_039/contact.png) | Yes | One acting hand pulls drawer outward in-plane. |
| [episode_060](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_060/contact.png) | No | Additional hand/handling appears near the front container. |
| [episode_096](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_096/contact.png) | No | Tray lifted/rotated; second supporting hand. |
| [episode_127](../outputs/robot_removal/demos/aiden-li__so101-open-upper-drawer/episode_127/contact.png) | No | Supporting hand appears below the case during opening. |

### b3rnd__record-50-episodes

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_004](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_004/contact.png) | No | Robot removed after stacking; additional acting-arm switch. |
| [episode_011](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_011/contact.png) | No | Two hands/arms hand off the cube. |
| [episode_017](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_017/contact.png) | No | Robot hardware moves and disappears. |
| [episode_028](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_028/contact.png) | No | Robot brought into scene; two hands handle cube/hardware. |
| [episode_037](../outputs/robot_removal/demos/b3rnd__record-50-episodes/episode_037/contact.png) | No | Acting-arm switch during cube transfer. |

### chirag1701__can

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/chirag1701__can/episode_005/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_012](../outputs/robot_removal/demos/chirag1701__can/episode_012/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_020](../outputs/robot_removal/demos/chirag1701__can/episode_020/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_032](../outputs/robot_removal/demos/chirag1701__can/episode_032/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_042](../outputs/robot_removal/demos/chirag1701__can/episode_042/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |

### chirag1701__sandwich

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/chirag1701__sandwich/episode_005/contact.png) | Yes | One hand transfers package into tray. |
| [episode_012](../outputs/robot_removal/demos/chirag1701__sandwich/episode_012/contact.png) | No | Package duplicates and transfer repeats. |
| [episode_020](../outputs/robot_removal/demos/chirag1701__sandwich/episode_020/contact.png) | Yes | One hand transfers package into tray. |
| [episode_032](../outputs/robot_removal/demos/chirag1701__sandwich/episode_032/contact.png) | No | Package duplicates/repeated transfer; torso enters. |
| [episode_042](../outputs/robot_removal/demos/chirag1701__sandwich/episode_042/contact.png) | Yes | One hand transfers package into tray. |

### fbeltrao__so101_unplug_cable_4

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_005/contact.png) | Yes | One hand disconnects plug and lays it down. |
| [episode_013](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_013/contact.png) | No | Robot hardware appears during unplugging. |
| [episode_020](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_020/contact.png) | No | Robot hardware appears; connector/cable changes geometry. |
| [episode_033](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_033/contact.png) | No | Robot enters and is handled by another hand. |
| [episode_043](../outputs/robot_removal/demos/fbeltrao__so101_unplug_cable_4/episode_043/contact.png) | No | Robot hardware enters and moves during ending. |

### k1000dai__standup_petbottle

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_004](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_004/contact.png) | Yes | One hand lifts bottle and stands it upright on white marker. |
| [episode_009](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_009/contact.png) | Yes | One hand lifts bottle and stands it upright on white marker. |
| [episode_014](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_014/contact.png) | Yes | One hand lifts bottle and stands it upright on white marker. |
| [episode_023](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_023/contact.png) | Yes | One hand lifts bottle and stands it upright on white marker. |
| [episode_031](../outputs/robot_removal/demos/k1000dai__standup_petbottle/episode_031/contact.png) | Yes | One hand lifts bottle and stands it upright on white marker. |

### lerobot__svla_so101_pickplace

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_005/contact.png) | Yes | One hand transfers brick into transparent box. |
| [episode_012](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_012/contact.png) | Yes | One hand transfers brick into transparent box. |
| [episode_020](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_020/contact.png) | Yes | One hand transfers brick into transparent box. |
| [episode_032](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_032/contact.png) | Yes | One hand transfers brick into transparent box. |
| [episode_042](../outputs/robot_removal/demos/lerobot__svla_so101_pickplace/episode_042/contact.png) | Yes | One hand transfers brick into transparent box. |

### pbvr__so101_test002

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_018](../outputs/robot_removal/demos/pbvr__so101_test002/episode_018/contact.png) | Yes | Single-hand orange block transfer. |
| [episode_043](../outputs/robot_removal/demos/pbvr__so101_test002/episode_043/contact.png) | Yes | Single-hand orange block transfer. |
| [episode_068](../outputs/robot_removal/demos/pbvr__so101_test002/episode_068/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_110](../outputs/robot_removal/demos/pbvr__so101_test002/episode_110/contact.png) | Yes | Single-hand orange block transfer. |
| [episode_147](../outputs/robot_removal/demos/pbvr__so101_test002/episode_147/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |

### pbvr__so101_test005

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_009](../outputs/robot_removal/demos/pbvr__so101_test005/episode_009/contact.png) | Yes | One hand transfers blue, yellow and green blocks sequentially. |
| [episode_023](../outputs/robot_removal/demos/pbvr__so101_test005/episode_023/contact.png) | Yes | One hand transfers blocks sequentially. |
| [episode_037](../outputs/robot_removal/demos/pbvr__so101_test005/episode_037/contact.png) | Yes | One hand transfers blocks sequentially. |
| [episode_060](../outputs/robot_removal/demos/pbvr__so101_test005/episode_060/contact.png) | Yes | One hand transfers blocks sequentially. |
| [episode_080](../outputs/robot_removal/demos/pbvr__so101_test005/episode_080/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |

### psg777__combinedtape

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_013](../outputs/robot_removal/demos/psg777__combinedtape/episode_013/contact.png) | Yes | One hand moves roll from box to table. |
| [episode_034](../outputs/robot_removal/demos/psg777__combinedtape/episode_034/contact.png) | No | Two acting hands handle roll and box. |
| [episode_054](../outputs/robot_removal/demos/psg777__combinedtape/episode_054/contact.png) | No | Two acting hands handle roll and box. |
| [episode_087](../outputs/robot_removal/demos/psg777__combinedtape/episode_087/contact.png) | Yes | One hand transfers roll into box. |
| [episode_113](../outputs/robot_removal/demos/psg777__combinedtape/episode_113/contact.png) | Yes | One hand transfers roll into box. |

### ricky0526__so101_pick_toy_to_plate_v3

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_008](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_008/contact.png) | Yes | Single acting hand transfers toy to dish. |
| [episode_019](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_019/contact.png) | Yes | Single acting hand transfers toy to dish. |
| [episode_030](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_030/contact.png) | No | Robot hardware is handled and removed with another hand. |
| [episode_049](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_049/contact.png) | Yes | Single acting hand transfers toy to dish. |
| [episode_065](../outputs/robot_removal/demos/ricky0526__so101_pick_toy_to_plate_v3/episode_065/contact.png) | Yes | Single acting hand transfers toy to dish. |

### sattgle__clean-test

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_005](../outputs/robot_removal/demos/sattgle__clean-test/episode_005/contact.png) | Yes | One hand transfers both slippers; person remains outside image. |
| [episode_012](../outputs/robot_removal/demos/sattgle__clean-test/episode_012/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_018](../outputs/robot_removal/demos/sattgle__clean-test/episode_018/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_030](../outputs/robot_removal/demos/sattgle__clean-test/episode_030/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_039](../outputs/robot_removal/demos/sattgle__clean-test/episode_039/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |

### sattgle__clean2-test

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_002](../outputs/robot_removal/demos/sattgle__clean2-test/episode_002/contact.png) | No | Two hands and head/body enter during transfers. |
| [episode_006](../outputs/robot_removal/demos/sattgle__clean2-test/episode_006/contact.png) | No | Head enters; white slipper remains unplaced. |
| [episode_009](../outputs/robot_removal/demos/sattgle__clean2-test/episode_009/contact.png) | No | Torso and extra acting hand enter. |
| [episode_015](../outputs/robot_removal/demos/sattgle__clean2-test/episode_015/contact.png) | No | White slipper remains unplaced. |
| [episode_020](../outputs/robot_removal/demos/sattgle__clean2-test/episode_020/contact.png) | No | White slipper remains unplaced; head enters. |

### stsqitx__clean

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_002](../outputs/robot_removal/demos/stsqitx__clean/episode_002/contact.png) | Yes | One hand wipes spill and releases sponge. |
| [episode_005](../outputs/robot_removal/demos/stsqitx__clean/episode_005/contact.png) | No | A duplicate sponge remains while another is used for wiping. |
| [episode_008](../outputs/robot_removal/demos/stsqitx__clean/episode_008/contact.png) | No | A duplicate sponge remains while another is used for wiping. |
| [episode_012](../outputs/robot_removal/demos/stsqitx__clean/episode_012/contact.png) | No | A duplicate sponge remains while another is used for wiping. |
| [episode_016](../outputs/robot_removal/demos/stsqitx__clean/episode_016/contact.png) | Yes | One hand wipes spill and releases sponge. |

### tenkau__SO101-Stack3Blocks

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_004](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_004/contact.png) | Yes | One hand builds a plausible three-block stack. |
| [episode_009](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_009/contact.png) | No | Two acting hands; apparent additional block and unstable arrangement. |
| [episode_014](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_014/contact.png) | No | Block alignment/contact is questionable. |
| [episode_023](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_023/contact.png) | No | Two acting hands rearrange and stack blocks. |
| [episode_030](../outputs/robot_removal/demos/tenkau__SO101-Stack3Blocks/episode_030/contact.png) | No | Additional acting/supporting hand during stacking. |

### tenkau__so101_color_block

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_003](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_003/contact.png) | Yes | One hand stacks gray then black onto red; background hardware stays still. |
| [episode_007](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_007/contact.png) | No | Background robot enters after stacking. |
| [episode_012](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_012/contact.png) | No | Background robot changes/moves during task. |
| [episode_019](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_019/contact.png) | No | Robot hardware is handled/moves after stacking. |
| [episode_025](../outputs/robot_removal/demos/tenkau__so101_color_block/episode_025/contact.png) | No | Background robot enters after stacking. |

### un1c0rnio__so101_sock_stowing_3pair2

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_002](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_002/contact.png) | Yes | Single hand places sock into container. |
| [episode_006](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_006/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_010](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_010/contact.png) | Yes | Single hand places sock into container. |
| [episode_016](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_016/contact.png) | Yes | Task action appears plausible with one acting hand; visible body/head or idle hand is permitted under the clarified criterion. |
| [episode_021](../outputs/robot_removal/demos/un1c0rnio__so101_sock_stowing_3pair2/episode_021/contact.png) | Yes | Single hand places sock into container. |


## Eight-episode follow-up for initially zero-success tasks

No videos or prompts were changed. These are contact-sheet judgments, subject to the 2 Hz sampling limitation.

### ReubenLim__so101_tape_in_square

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_001](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_001/contact.png) | No | Roll is lifted from the left and placed over the marker rather than slid. |
| [episode_009](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_009/contact.png) | No | Roll is lifted and placed rather than slid along the surface. |
| [episode_018](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_018/contact.png) | No | Roll is lifted from above the marker and placed over it rather than slid. |
| [episode_026](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_026/contact.png) | No | Roll is lifted from the right and placed over the marker rather than slid. |
| [episode_034](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_034/contact.png) | No | Large thick ring is brought into view, held above the marker, then removed; no surface slide or completed placement. |
| [episode_042](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_042/contact.png) | No | Roll is lifted from the left and placed over the marker rather than slid. |
| [episode_051](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_051/contact.png) | No | Roll is introduced from below the frame and placed over the marker; no surface slide is shown. |
| [episode_058](../outputs/robot_removal/demos/ReubenLim__so101_tape_in_square/episode_058/contact.png) | No | Roll is introduced from below the frame and placed over the marker; no surface slide is shown. |

### aiden-li__so101-close-lower-drawer

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_001](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_001/contact.png) | No | Tray is lifted clear of the case and rotated toward the front rather than slid closed; second hand supports it. |
| [episode_008](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_008/contact.png) | No | Tray is lifted and rotated rather than slid closed; another hand supports its lower edge. |
| [episode_016](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_016/contact.png) | No | Tray is lifted and rotated rather than translated along the drawer mechanism. |
| [episode_023](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_023/contact.png) | No | Tray is lifted and rotated rather than slid closed. |
| [episode_031](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_031/contact.png) | No | Tray is lifted and rotated rather than slid closed. |
| [episode_039](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_039/contact.png) | No | Tray is lifted and rotated rather than slid closed; another hand supports the lower edge. |
| [episode_048](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_048/contact.png) | No | Tray is lifted and rotated rather than slid closed. |
| [episode_055](../outputs/robot_removal/demos/aiden-li__so101-close-lower-drawer/episode_055/contact.png) | No | Tray is lifted and rotated rather than slid closed. |

### sattgle__clean2-test

| Episode | Confirmed success | Observation / verdict basis |
|---|---|---|
| [episode_001](../outputs/robot_removal/demos/sattgle__clean2-test/episode_001/contact.png) | No | Two acting hands transfer slippers to the shelves. |
| [episode_004](../outputs/robot_removal/demos/sattgle__clean2-test/episode_004/contact.png) | No | Two acting hands transfer slippers; one places a white slipper while the other handles another. |
| [episode_007](../outputs/robot_removal/demos/sattgle__clean2-test/episode_007/contact.png) | No | Two acting hands transfer slippers to the shelves. |
| [episode_010](../outputs/robot_removal/demos/sattgle__clean2-test/episode_010/contact.png) | Yes | One hand sequentially places gray slippers on the lower shelf and all white slippers on top; no assisting hand observed. |
| [episode_014](../outputs/robot_removal/demos/sattgle__clean2-test/episode_014/contact.png) | No | One hand handles gray slippers while another places white slippers; gray slipper remains held until the ending transition. |
| [episode_017](../outputs/robot_removal/demos/sattgle__clean2-test/episode_017/contact.png) | No | Two acting hands take turns placing white slippers. |
| [episode_019](../outputs/robot_removal/demos/sattgle__clean2-test/episode_019/contact.png) | Yes | One hand sequentially places gray slippers below and all white slippers on top; no assisting hand observed. |
| [episode_022](../outputs/robot_removal/demos/sattgle__clean2-test/episode_022/contact.png) | Yes | One hand completes the gray-to-lower and white-to-upper arrangement; visible person is allowed. |

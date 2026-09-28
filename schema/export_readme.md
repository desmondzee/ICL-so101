# ICL-so101

SO-101 robot episodes with a scene, task and step-segment record per episode (the robot side of HumanGen-style human/robot pairs for Zero-WAM post-training). 1960 accepted episodes from 28 community LeRobot datasets, one format, front camera, end-effector pose.

## Layout

```text
accepted.json                         one row per episode (below)
lerobot/<dataset>/                    LeRobot v3.0 dataset holding only the accepted episodes, renumbered 0..n-1
pairs/<dataset>/episode_XXX/pair.json scene/task/segments record; source.episode_index = XXX in lerobot/<dataset>
pairs/<dataset>/tasks/                the dataset's canonical task template (*.json) and task notes (*.notes.txt)
schema/                               JSON schemas (common, scene, task, pair), validate.py and schema README
```

`accepted.json` rows: `{"dataset", "episode_index", "pair", "lerobot", "judge": "pass"|"minor_issues", "audited", "curated_episode_index"}`.

## Robot data

Each `lerobot/<dataset>` has `action` and `observation.state` (6 joints, LeRobot degrees, gripper 0-100), `action.ee` and `observation.state.ee` (gripper-frame position in m and orientation, plus gripper, from MuJoCo forward kinematics on the SO-101 model), `observation.images.front` (640x480 AV1) and, where the source had one, `observation.images.wrist`. Joint conventions of the sources were converted and calibrated to each other; see the curation notes in each dataset's `meta`.

## Pair records

`pair.json` follows `schema/pair.schema.json`: `scene` (entities with names, attributes and first-frame `box_2d` as [ymin, xmin, ymax, xmax] in 0-1000, initial state relations), `task` (instruction, roles, ordered steps with actions from pick_place, lift, slide, stack, open, close, press, pour, fold, wipe, goals), `bindings` (role to entity), `segments` (per step: start/end frame, grasp/release frame), `source` (dataset, episode, outcome and evidence). `human` and the generation prompts are not filled yet. Validate with `python -m schema.validate <pair.json>`.

## How the records were made and checked

1. Drafted per episode by a VLM (Space Bunny via OpenRouter, Gemini Flash fallback) from frame strips, gripper events and a per-dataset task template.
2. Judged by Claude Opus (5 pairs per agent) against a sheet of the first frame with boxes and 12 labelled frames; rejected pairs were corrected by one Opus agent each.
3. Audit: one Opus agent corrected each pair in the hardest datasets (drawers, tea) and every pair a second judge disagreed on; one Sonnet agent then checked each pair independently from the frames; a pair it failed got one more Opus correction and check.

An episode is accepted when its pair validates, its outcome is success and its latest verdict is pass or minor_issues. 1347 accepted pairs were checked by the audit in their final form; the rest were accepted by both the Opus and an earlier Sonnet judge. `minor_issues` pairs may have a loose box or a segment edge off by up to about 1/12 of the episode.

| Dataset | Task (example instruction) | Accepted / episodes |
| --- | --- | ---: |
| `aiden-li__so101-close-lower-drawer` | Close the lower drawer | 54 / 57 |
| `aiden-li__so101-close-upper-drawer` | Close the upper drawer | 124 / 133 |
| `aiden-li__so101-grabtissue` | Grab a piece of tissue. | 142 / 142 |
| `aiden-li__so101-open-lower-drawer` | Open the lower drawer | 136 / 136 |
| `aiden-li__so101-open-upper-drawer` | Open the upper drawer | 139 / 149 |
| `b3rnd__record-50-episodes` | Grab the black cube and stack on the red cube | 44 / 44 |
| `chirag1701__can` | pick the can and put it in the brown box | 50 / 50 |
| `chirag1701__sandwich` | pick the sandwich and put it in the gray box | 51 / 51 |
| `Cornito__so101_tea2` | Make tea | 83 / 98 |
| `EverNorif__so101-table-cleanup` | Grab pens and place into pen holder. | 81 / 82 |
| `fbeltrao__so101_unplug_cable_4` | Unplug the cable. | 52 / 52 |
| `k1000dai__standup_petbottle` | Pick up the plastic bottle and stand it up on the white place | 37 / 37 |
| `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place` | Pick the object with centre yellow pin and put in the respective shape | 74 / 75 |
| `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid` | Pick up a kettle and pour liquid into a cup. | 59 / 59 |
| `lerobot__svla_so101_pickplace` | pink lego brick into the transparent box | 50 / 50 |
| `pbvr__so101_test002` | Grab a orange square and put it in the brown box. | 169 / 174 |
| `pbvr__so101_test005` | Grab the blue, yellow, and green squares one by one and put them into the brown box. | 94 / 95 |
| `psg777__combinedtape` | Pick the red roll from the blue box and place it infront of the so101 arm | 133 / 134 |
| `ReubenLim__so101_tape_in_square` | Push a circle block to the blue square in the center. | 60 / 60 |
| `ricky0526__so101_pick_toy_to_plate_v3` | Pick up the toy and place it on the plate | 77 / 77 |
| `Rorschach4153__so101_30_fold` | Fold the blanket. | 48 / 50 |
| `sattgle__clean-test` | Pick up the grey slippers and put them on the top shelf. | 47 / 47 |
| `sattgle__clean2-test` | Put the grey slippers on the bottom shelf and the white ones on the top shelf. | 24 / 24 |
| `stsqitx__clean` | Grab the sponge and clean the table. | 20 / 20 |
| `Tear4Pixelation__lego2` | Grab the loose lego brick and stack it on top of the fixed one. | 20 / 20 |
| `tenkau__SO101-Stack3Blocks` | Stack 3 Blocks on the chosen place. | 36 / 36 |
| `tenkau__so101_color_block` | Stack the three blocks in an ordered color. | 30 / 30 |
| `un1c0rnio__so101_sock_stowing_3pair2` | Put the sock in the container | 26 / 27 |

## Known limits

- The source datasets carry no success labels; `source.outcome` is judged from the video.
- Hardest cases: identical-looking objects (black blocks, socks, slippers), results hidden inside containers (tea bag in the mug), and drawer motion seen from above.
- Segment boundaries are frame estimates from video, not from contact sensing.

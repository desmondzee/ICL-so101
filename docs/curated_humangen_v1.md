# SO-101 HumanGen pairs (curated_humangen_v1)

216 pairs over 24 tasks, with 74 minutes of robot data. Each pair joins a real SO-101 robot episode from the curated export with a generated video of one person's hand doing the same task in the same scene. The robot is removed from the first and last robot frames, and H3 Max Turbo generates the video from them.

Viewer: https://desmondzee.github.io/ICL-so101/

## Selection

Every accepted export episode has one generated demo (`hf://buckets/nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo`, keyed by `curated_episode_index`). Candidates were drawn per task, spread over the episode range, in rounds until a task had 10 confirmed pairs or stopped yielding them. Each candidate was reviewed in two stages:

1. **Judge.** An Opus agent, five pairs per agent, looked at 12 frames of the human video, 8 robot front-camera frames and the task (instruction, steps, goals), and extracted more frames where unsure. To accept, task adherence and physics both had to score at least 4/5, and these checks had to pass:
   - the same steps on the same objects, in the same order, ending in the robot's end state;
   - objects move only when touched, with nothing appearing, vanishing, duplicating or morphing;
   - one acting hand, with a second hand allowed only to steady an object;
   - the first frame matches the robot scene with the robot removed, and there are no overlays or cuts.
2. **Verify.** A second, independent Opus agent was told to find a reason to reject. It always extracted frames at 6 fps and confirmed only when confident.

A pair is included only if both stages accept it. All 638 reviews, including rejections, are in `reviews_all.json`. Four tasks yielded nothing because the generator fails them the same way every time:
- **Pouring:** the long-spouted kettle becomes a jug.
- **Lower drawers (open and close):** the drawer is lifted out, or the lid morphs into the drawer.
- **Tissue grab:** the tissue floats off.

| Task | Instruction | Pairs | Accepted / reviewed | Demos generated |
|---|---|---|---|---|
| `EverNorif__so101-table-cleanup` | Grab pens and place into pen holder. | 10 | 10 / 68 | 81 |
| `LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place` | Pick the object with centre yellow pin and put in the respective shape | 10 | 12 / 14 | 74 |
| `ReubenLim__so101_tape_in_square` | Push a circle block to the blue square in the center. | 10 | 10 / 14 | 60 |
| `aiden-li__so101-close-upper-drawer` | Close the upper drawer | 10 | 12 / 68 | 124 |
| `aiden-li__so101-open-upper-drawer` | Open the upper drawer | 10 | 12 / 20 | 139 |
| `b3rnd__record-50-episodes` | Grab the black cube and stack on the red cube | 10 | 11 / 20 | 44 |
| `chirag1701__can` | pick the can and put it in the brown box | 10 | 14 / 14 | 50 |
| `chirag1701__sandwich` | pick the sandwich and put it in the gray box | 10 | 14 / 14 | 51 |
| `k1000dai__standup_petbottle` | Pick up the plastic bottle and stand it up on the white place | 10 | 11 / 14 | 37 |
| `lerobot__svla_so101_pickplace` | pink lego brick into the transparent box | 10 | 14 / 14 | 50 |
| `pbvr__so101_test002` | Grab a orange square and put it in the brown box. | 10 | 11 / 14 | 169 |
| `pbvr__so101_test005` | Grab the blue, yellow, and green squares one by one and put them into the brown box. | 10 | 14 / 14 | 94 |
| `psg777__combinedtape` | Pick the red roll from the blue box and place it infront of the so101 arm | 10 | 13 / 27 | 133 |
| `ricky0526__so101_pick_toy_to_plate_v3` | Pick up the toy and place it on the plate | 10 | 13 / 14 | 77 |
| `sattgle__clean-test` | Pick up the grey slippers and put them on the top shelf. | 10 | 13 / 14 | 47 |
| `sattgle__clean2-test` | Put the grey slippers on the bottom shelf and the white ones on the top shelf. | 10 | 14 / 14 | 24 |
| `stsqitx__clean` | Grab the sponge and clean the table. | 10 | 10 / 14 | 20 |
| `tenkau__SO101-Stack3Blocks` | Stack 3 Blocks on the chosen place. | 10 | 11 / 27 | 36 |
| `un1c0rnio__so101_sock_stowing_3pair2` | Put the sock in the container | 10 | 11 / 14 | 26 |
| `Cornito__so101_tea2` | Make tea | 9 | 9 / 44 | 83 |
| `tenkau__so101_color_block` | Stack the three blocks in an ordered color. | 7 | 7 / 30 | 30 |
| `Tear4Pixelation__lego2` | Grab the loose lego brick and stack it on top of the fixed one. | 5 | 5 / 20 | 20 |
| `Rorschach4153__so101_30_fold` | Fold the blanket. | 3 | 3 / 38 | 48 |
| `fbeltrao__so101_unplug_cable_4` | Unplug the cable. | 2 | 2 / 38 | 52 |
| `LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid` | Pick up a kettle and pour liquid into a cup. | 0 | 0 / 14 | 59 |
| `aiden-li__so101-close-lower-drawer` | Close the lower drawer | 0 | 0 / 14 | 54 |
| `aiden-li__so101-grabtissue` | Grab a piece of tissue. | 0 | 0 / 14 | 142 |
| `aiden-li__so101-open-lower-drawer` | Open the lower drawer | 0 | 0 / 14 | 136 |

## Layout

```
index.json                         tasks and episodes (paths are relative to episodes/<task>/<episode>/)
reviews_all.json                   every judge and verify verdict, accepted or not
episodes/<task>/episode_<NNN>/     NNN = curated_episode_index (the human demo's key)
  human.mp4                        generated human demonstration (H.264 + AAC, 640x480, 24 fps, ~5 s)
  robot_front.mp4, robot_wrist.mp4 robot camera views (H.264, 640x480, 30 fps; wrist where the source has one)
  robot_data.parquet               the robot episode's frames: action / observation.state (joints, gripper 0-100),
                                   action.ee / observation.state.ee (gripper-site xyz, rotation vector, gripper)
  pair.json                        scene, task, steps, goals and segments (schema 0.2)
  review.json                      this pair's judge and verify verdicts
  thumb.jpg, robot_thumb.jpg       thumbnails
```

`export_episode_index` in `index.json` is the episode's index in `data/so101_export/lerobot/<task>` (this bucket's root export). It differs from `curated_episode_index` in 11 datasets.

# curate

Screen community SO-101 LeRobot datasets and convert the kept ones into one format.

## Format

LeRobot v3.0, one dataset per source under `data/so101_curated/<contributor>__<name>/`:

| Feature | Shape | Units |
| --- | --- | --- |
| `action`, `observation.state` | 6 | LeRobot degrees (zero at mid-range), gripper 0-100 |
| `action.ee`, `observation.state.ee` | 7 | gripper-site xyz (m, base frame), rotation vector, gripper 0-100; FK of the joints with `sim/so101/so101_new_calib.xml` |
| `observation.images.front` | video | third-person camera |
| `observation.images.wrist` | video | only when the source has one |

`meta/curation.json` records source, revision, units, offsets, camera map and dropped episodes.

## Calibration

Sources store joints three ways: LeRobot `-100..100` of each robot's calibrated range (most 2025 datasets), LeRobot degrees, or old `main_*` degrees with a hand-set zero. Normalized values map to degrees through the URDF joint ranges. Old-format values are shifted so the folded rest pose sits on the shoulder-lift and elbow stops (`rest_offsets`); datasets that never fold to rest are rejected.

Check: FK gripper height at grasp onset. Normalized sets land at 0 ± 1 cm above the table (svla −0.6 cm, table-cleanup −0.3 cm over 465 grasps); anchored old-format sets at −1.4 to +1.4 cm, versus +3 to +6 cm before anchoring. Pan, wrist flex and wrist roll have no mechanical reference.

## Steps

```sh
uv run --extra schema python -m curate.screen <dataset roots> --out screen.json
uv run --extra schema python -m curate.qc <dataset roots>            # contact sheets in data/qc/
uv run --extra schema python -m curate.convert <src> data/so101_curated/<name> --front <cam> --wrist <cam>
```

`screen` rejects: not 6-DoF, unknown units, no rest pose (old format), fps ≠ 30, < 20 episodes, NaNs, ≥ 1% frames outside joint limits, per-frame jumps ≥ 30°, leader-follower offset ≥ 5°, FK below −3 cm, no third-person camera, ≥ 10% still episodes. `convert` drops episodes shorter than 2 s, with under 5° of motion, or with a jump over 30°.

Camera names in sources are unreliable (wrist cameras named `front`, keys swapped between episodes); pass the camera chosen from the contact sheet.

## Curated set

`selection.json` lists the 28 kept datasets (19 passed dataset review outright; 9 added for task variety with per-episode filtering). Each episode was then checked on a contact sheet by a Sonnet reviewer, and drops and camera swaps were confirmed by a second reviewer (`episodes.py`, review records in `data/so101_curated/_episodes/review.json`). `build.py` converts everything; `verify.py` loads every dataset and decodes the first and last frame of every episode.

| Family | Datasets | Episodes | Hours |
| --- | ---: | ---: | ---: |
| pick_place_container | 7 | 568 | 2.66 |
| open_close | 4 | 475 | 1.29 |
| other | 2 | 150 | 0.94 |
| grasp_only | 1 | 142 | 0.16 |
| stack | 4 | 130 | 0.96 |
| pick_place_surface | 2 | 124 | 0.47 |
| sort_multi_object | 2 | 119 | 0.68 |
| insert | 1 | 75 | 0.49 |
| push_slide | 1 | 60 | 0.47 |
| pour | 1 | 59 | 0.28 |
| fold | 1 | 50 | 0.57 |
| reorient_upright | 1 | 37 | 0.34 |
| wipe | 1 | 20 | 0.06 |
| **total** | 28 | 2009 | 9.37 |

Dropped: episodes with a person or leader arm in frame, idle or failed episodes, unreadable video, video and trajectory lengths that disagree, and camera assignments the motion check and the reviewers disagree on. `data/so101_curated/_merged_front` aggregates the front camera of all 28 for training; the lower-drawer sets reach below the table plane by design.

```sh
uv run --extra schema python -m curate.fetch
uv run --extra schema python -m curate.episodes
uv run --extra schema python -m curate.build data/so101_curated/_episodes/review.json
uv run --extra schema python -m curate.verify
```

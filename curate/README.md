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

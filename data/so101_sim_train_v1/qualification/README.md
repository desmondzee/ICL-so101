# Training-task qualification

Each task is admitted only after a complete strict screen of its 50 prescribed seeds, with at least 48/50 strict physics passes. The recorder enforces this through `sim.train.tasks.qualify.require_qualified_task(name, root)`, which checks all of the following:
- every prescribed seed is present;
- the source hash matches (shared kit files plus the task's family module);
- the retained per-seed rows are consistent;
- the hashes of all accepted evidence files match.

Original task success is not physics acceptance.

```sh
# calibration (seeds 9000+, separate output, never qualifies)
.venv/bin/python -m sim.train.tasks.qualify --family <family> --seed-start 9000 --count 10 --output /tmp/<family>_cal --jobs 6
# qualification (validation index found automatically; --validation-index to override)
.venv/bin/python -m sim.train.tasks.qualify --family <family> --jobs 6
```

`<task>.json` is the machine-readable summary. `<task>/seed_<n>/` keeps every rollout:
- `result.json`
- `physics.json` (with every violation)
- `policy.json`
- `episode.json`
- `telemetry.npz` (about 39 MB)

Never relax a gate to reach the target. Any code change produces a new source hash; stale evidence is refused, so rerun into a fresh directory. See `sim/train/README.md`.

## Status 2026-10-06 (pilot family `pilot_blocks`, kit frozen at the qualifying source hashes)

| task | family | strict passes | failures |
|---|---|---|---|
| block_in_bowl | container_insertion | 49/50 | 1 release angular-acceleration spike (1030 vs 1000 rad/s^2) |
| block_out_of_bowl | container_removal | 50/50 | none |
| block_beside_bowl | spatial_arrangement | 48/50 | 1 angular spike (1062); 1 camera-mount/shoulder self-contact (close place target r=0.152 m, -45 deg) |
| blocks_onto_mats_in_order | ordered_relocation | 50/50 | none |
| bar_crosswise_on_mat | orientation_sensitive_placement | 48/50 | 2 release angular spikes (1042, 1113) |

All five are admitted. Across the 250 episodes:
- maximum joint acceleration was 140 rad/s² (limit 150);
- the grasp-contact allowance was used 0 times;
- no oracle speed-guard frames were inserted.

Evidence totals 9.7 GB and is git-ignored.

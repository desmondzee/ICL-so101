# Training-task qualification

The five registry entries are candidates. No completed 50-seed qualification report exists yet, so none is admitted. Test passes and original task success are not strict physics acceptance.

Run calibration first, with seeds disjoint from both validation and the fixed qualification set:

```sh
.venv/bin/python -m sim.train.tasks.qualify --validation-index /Users/desmondzee/Hardware/ICL-so101/data/so101_sim_val_v2/index.json --output /tmp/icl_task4_calibration --seed-start 9000 --count 3
```

Inspect each seed's `physics.json`, `policy.json`, `telemetry.npz`, `episode.json` and `result.json`. Repair unsafe trajectories; add contact exceptions only with measured, explained calibration evidence. Never relax the physics gate to obtain a target pass rate. A code change invalidates all previous evidence by source hash; preserve old attempts and select a new output directory.

After calibration and repair, run all 50 unseen seeds per task:

```sh
.venv/bin/python -m sim.train.tasks.qualify --validation-index /Users/desmondzee/Hardware/ICL-so101/data/so101_sim_val_v2/index.json --output data/so101_sim_train_v1/qualification
```

The CLI uses seeded layout and visual variation but does not render frames. The existing base environment still initializes a graphics context, so macOS may require CoreGraphics access. The command returns nonzero if any requested task lacks complete >=95% strict success. It resumes identical retained results; stale/conflicting results require a new output directory. All reported failures count toward the rate. `--tasks` permits screening a subset.

`<task>.json` is the machine-readable summary. Task/seed subdirectories retain every attempted rollout and all violations. Only candidate discovery should use `TRAIN_TASKS` or `load_train_task`; recording admission must call `require_qualified_task(name, qualification_root)` from `sim.train.tasks.qualify`. Admission verifies all 50 prescribed seeds, source identity, retained result rows and the hashes and consistency of accepted evidence. No task below the threshold may enter production.

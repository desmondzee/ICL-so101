# SO-101 simulated training tasks: authoring guide

Training tasks are separate from the held-out validation set (`sim/val`). Each skill family is a single
module, `sim/train/tasks/families/<family>.py`, that exports `TASKS = [define_task(...), ...]`. The
catalog (`sim/train/tasks/catalog.py`) discovers these modules automatically, so there is no shared
registry to edit. A module that fails to import is skipped and reported in `catalog.DISCOVERY_ERRORS`,
and `tests/sim_train` then fails. The worked example is `families/pilot_blocks.py`, which holds five
pilot tasks, all qualified.

## Workflow for a family author

```bash
# 1. write sim/train/tasks/families/<family>.py (copy pilot_blocks.py)
.venv/bin/python -m pytest tests/sim_train -q                                     # catalog + kit contracts
.venv/bin/python -m sim.train.tasks.preview --family <family> --count 3           # /tmp/so101_task_previews/*.png
.venv/bin/python -m sim.train.tasks.qualify --family <family> --seed-start 9000 --count 10 \
    --output /tmp/<family>_cal --jobs 6                                           # calibration, iterate here
.venv/bin/python -m sim.train.tasks.qualify --family <family> --jobs 6            # final: 50 prescribed seeds
```

- **Qualification bar:** aim for at least 48/50 strict passes on the task's own prescribed seeds; the recorder admits tasks at 43/50 (`record.ADMISSION_RATE`) when the failures are scattered rather than a whole region of the layout space, since every recorded episode is gated and reviewed individually.
- **Source hash:** evidence is keyed by a hash over the shared kit files plus your family module. Editing your module invalidates only your family's evidence; editing the kit invalidates everyone's.
- **Rerun location:** write a new run to a fresh `--output` directory, or delete your own task's directories first.
- **What qualification never does:** it never renders anything and never writes under `data/so101_sim_val*`.
- **Seeds:** calibration seeds start at 9000. Qualification seeds are derived from the task name (a block of 50 in [1e6, 5.1e7)). On a collision, pass `qualification_seeds=qualification_seeds_for(name, salt=1)`.
- **Speed:** about 1.5 s per episode per core.
- **Disk:** the evidence is large, about 39 MB of telemetry per episode (the 250 pilot episodes take 9.7 GB). `data/` is git-ignored.

## Writing a task

```python
from sim.train.tasks.base import (GRASP_REGION, PLACE_REGION, VIEW, Goal, TrainEnv, TrainOracle, Obj, Fixture,
                                  Block, Scanned, mat, define_task, step_text, then)

class MyEnv(TrainEnv):
    instruction = "Put the soup can on the plate."
    task_objects = ("can",)                 # bodies the robot manipulates (goal per object)
    # order = ("a", "b")                    # required temporal grasp/release order (OrderedCompletion)
    # surfaces = ("bowl",)                  # extra surfaces the object starts on / jaws may brush
    # extra_contacts = (("a", "b"),)        # object-object contacts the task needs (stacking)
    # distractor_pool = LOW_DISTRACTORS     # 2-4 drawn per episode, never touching the task
    def scene_objects(self):
        return [Obj("can", "alphabet_soup")]
    def scene_fixtures(self):
        return [Fixture("plate", "plate", 0.65)]
    def layout(self):                        # use self.np_random only
        placed = []
        plate_xy, _ = self.place_fixture("plate", placed, region=PLACE_REGION)
        self.place("can", placed)            # GRASP_REGION, random yaw, collision-free, keep-outs
        self.place_distractors(placed)
        top = self.surface_z(plate_xy, exclude=self.task_objects)
        self.set_goals(Goal("can", "plate", target=(*plate_xy, top + self.height("can") / 2),
                            tolerance=(0.02, 0.02, 0.008)))

class MyOracle(TrainOracle):
    def plan(self):
        env = self.env
        ok = yield from self.pick_and_place("can", env.goal_target(env.goals[0]), width=0.031)
        if ok:
            yield from self.rest()
            yield from self.wait(1.2)

TASKS = [define_task(name="can_on_plate", instruction=MyEnv.instruction, family="placement_on_support",
                     env=MyEnv, oracle=MyOracle, objects=("can",), object_kinds=("can",), relation="on",
                     goal="plate", steps=(step_text("put", "soup can", "on", "plate"),))]
```

### What the declared goals give you

`set_goals` must declare exactly one `Goal` per task object. The kit derives the following from those goals, so they cannot disagree:

- **`success()`.** Every goal must hold: within `tolerance` of `target`, upright within `upright_cos` (cos 10° by default), the optional `check(env, obj)` predicate passes, the object is released and settled (≤ 3 cm/s and ≤ 0.5 rad/s), and it is supported by `support` (positive-force contact with an upward normal). If `order` is set, the temporal order must also hold.
- **Goal options:**
  - `support="table"` means the arena table.
  - `reference=<body>` makes the target follow a free support, such as a mat.
  - `target=None` means anywhere on the support.
- **Success helpers:** `supported_by`, `settled`, `released`, `upright`, `yaw`, `axis_alignment`, `in_region`, `within_radius` and `stacked`. Use them in `check=` predicates.
- **Physics policy (`physics_policy()`).** The strict gate (`sim/train/physics.py`):
  - finite state, joint bounds, joint speed ≤ 3 rad/s and acceleration ≤ 150 rad/s², object speed and acceleration limits;
  - contact penetration ≤ 3 mm;
  - distractors and every non-task free body may move ≤ 5 mm and rotate ≤ 0.1 rad;
  - the goal must hold at every substep for the final 30 frames, after release.
  - Allowed contacts are limited to: task object with jaws; any free body with the table; task object with its goal support and declared `surfaces`; `extra_contacts`; parked distractors with the floor. Any other robot contact fails.
- **Bounded grasp-contact allowance.** The jaws (`gripper`, `moving_jaw_so101_v1`) may touch the table, goal supports or `surfaces` only within these limits. Arm links, wrist, camera mount and self-contact always fail.

  | Limit | Value | Basis (measured) |
  |---|---|---|
  | Normal force | ≤ 6 N | Jaw just touching the table: 2.6-4.6 N. Commanded 0.3/0.5/1.5/2.5 mm into it: 5.6-11 / 11-20 / 15-27 / 27-51 N. |
  | Penetration | ≤ 0.5 mm | Grazing ≤ 0.08 mm; pressing 0.2-0.66 mm (force is the real discriminator). |
  | Continuous contact | ≤ 1.5 s | The oracle's close plus hold time. Measured grasp-phase runs: 0.97-1.08 s. |
  | Contact point to nearest task object (xy) | ≤ 5 cm | Measured ≤ 1.83 cm for a 2.8 cm block; 5 cm covers the full jaw opening. |

  These numbers come from block grasps with the TCP at 8.5/8.2/7.9/7.7/7.5/6.5/5.5 mm above the table, 12 rollouts per height across both arenas. The jaw collision hull reaches about 8.2 mm below the TCP. With the oracle defaults, no pilot episode used the allowance (0 samples in 250 qualification episodes). It exists for thin objects, which need grasps lower than about 9 mm.
- **`goal_regions()`.** Visibility screening regions for the recorder, placed on the goal surface. For containers, put `target` at the floor; the region is placed at the support's surface under the target.

### Action text and semantics

- **`steps`** holds one description per object, in `order`. These feed the human-video prompts. Use `step_text(verb, obj, relation, target)` and `then(...)` for later steps.
- **`instruction`** is the episode instruction; it must equal `Env.instruction`.
- **`object_kinds`, `relation` and `goal`** form the semantic signature that catalog validation compares. Colour words and colour-prefixed names are ignored. Two tasks with the same family, kinds, relation, goal and order are duplicates, whatever their names or instructions.
- **What counts as a distinct task.** A task must differ in what is done: the skill or relation (inside / out of / on / beside / stacked / ordered / oriented), the object kind, the receptacle kind, or the required order. Colour swaps, layout changes, arena or lighting changes, distractor changes and paraphrased instructions are episode diversity, not new tasks.
- **Validation semantics stay held out.** The five validation tasks are sort_blocks, stack_bowls, mug_on_plate, mugs_in_microwave and pan_on_stove; `validate_catalog` rejects their exact semantics.
- **Spatial words follow the front (viewer) camera**, never the robot's own frame. `VIEW["right"]` is world +y, `left` is -y, `front` (toward the viewer) is +x and `back` is -x. Left-right in the image corresponds to world y.

### Arena, layout and visual variation

- **Arenas.** `define_task(arenas=...)` restricts the arena choice; the default is both qualified arenas (`living_room`, `kitchen`). `TaskDefinition.variation_policy()` and `sample_visual_config(seed)` give the per-episode frozen camera, lights and background, which the recorder and qualification both use.
- **Reach.** Top-down TCP tilt, measured in degrees at azimuths -60/0/60:

  | TCP height | r = 0.12 | 0.18 | 0.22 | 0.25 | 0.27 | 0.29 | 0.31 m |
  |---|---|---|---|---|---|---|---|
  | 1.5 cm | 0 | 0 | 0 | 0 | 0 | 4 | 14 |
  | 6 cm | 0 | 0 | 0 | 1 | 5 | 9 | 21 |
  | 10 cm | 5 | 3 | 5 | 9 | 11 | 14 | 18 |

  Keep graspable objects and place targets within r = 0.27 m: `GRASP_REGION` covers r 0.14-0.27 and ±65°; `PLACE_REGION` covers r 0.13-0.27 and ±70°.
- **Keep-outs.** The robot base (r < 0.10 + object radius) and the hovering folded gripper at rest (0.16, 0), radius 0.06. These are built into `sample_xy`.
- **Wrist roll** is limited to ±157°. For equivalent end orientations, use `feasible_rotation(...)`. `carry_to` switches to a joint-space transit when the yaw turn exceeds 60°, because a geodesic Cartesian turn pins the roll at its limit.
- **Carry height** is `carry_z = 0.08` (validation uses 0.10). A held 2.8 cm block clears the default `LOW_DISTRACTORS` (all ≤ 4 cm tall). Raise `carry_z` for tall objects or tall distractors; the arm tilts more at higher carries.
- **Arc carries.** Moves about the shoulder-pan axis (x = 0.0388) arc automatically when a straight line would pass near the axis (`tcp_path`), as in validation.

### Objects

The table below summarises the main options. `sim/train/tasks/asset_inventory.json` has measured dimensions and masses for every asset; regenerate it with `python -m sim.train.tasks.assets`. "Graspable" means the narrowest horizontal side is ≤ 5 cm; the fingertip gap at OPEN is 68.6 mm.

| Source | Spec | Graspable at default scale |
|---|---|---|
| LIBERO, 0.5× (`Obj(name, catalog_name, scale)`) | `CATALOG` | white/red bowl 4.1 cm, ramekin 4.5, porcelain/red mugs 4.7-4.8 (by the body), alphabet soup / tomato sauce cans 3.1 × 3.8 h, cream cheese / butter / pudding / popcorn (0.9-1.4 cm thin: grasp low), ketchup / milk / OJ / bbq (tall, tip easily), books (1.2-1.4 thick), moka pot 4.1. Not graspable (use as receptacles): plate 6.9, akita bowl 5.4, basket 8.5 × 7.9 × 7.1 (deep), frypan, white_yellow_mug. Fixtures: `microwave`, `flat_stove`. |
| Primitives | `Block(name, half, rgba, mass)`, `mat(...)` (thin square box) | Any size up to about 5 cm wide. Use `mat` rather than `Disc`: a block can sink 4-45 mm into the 6 mm cylinder `Disc` (MuJoCo box-on-thin-cylinder contact). |
| GSO via nexus, real scale (`Scanned(name, "gso", id, scale)`) | 12 ids in `assets.GSO_IDS` | At 0.5×: honey dipper 1.5 × 5.6, gaffer-tape roll on edge 2.5 wide, sponge pack 3.7 × 6.4 × 1.0, CoQ10 bottle 4.2 × 3.0 × 3.0, sprinkles canister 2.5 × 6.8, lotion bottle 4.3 × 11.2, candy box 8.2 × 4.4 × 1.1. Receptacle-sized: GSO ramekin 5.8, vinyl tape roll 7.2, can opener, syrup bottle. The C-clamp tips at 0.5×. |
| YCB via nexus (`Scanned(name, "ycb", id, scale)`) | 10 ids in `assets.YCB_IDS` | At 0.5×: gelatin box 4.6 × 5.2 × 1.6, large marker 1.5 × 6.1, golf ball 2.7. Masses are a flat nexus default (10 g at 1×), so pass `mass=`. The knife tips over at both scales and the screwdriver at 1×. |

Notes on scanned objects:
- Meshes are downloaded on first use from the public Hugging Face mirrors (`johnsutor/gso-so101-nexus`, `ai-habitat/ycb`); no paid API is involved. They are cached under `~/.cache/so101_nexus`.
- `coacd` is not installed, so collision is a single convex hull per object. Holes and concavities (tape-roll centre, sponge pack) are filled.
- GSO is licensed CC-BY-4.0. Record attribution for any scanned asset in the dataset card.

## Oracle skills (`TrainOracle`) and why they differ from validation

| Skill | Purpose |
|---|---|
| `pick(name, width, grasp_z, yaw, symmetric=4|2)` | Top-down grasp. The jaw opens to `width + 12 mm` and closes over 1.2 s. The fixed finger stops 1.5 mm from the face, so the closing jaw barely slides the object. |
| `place_object(name, target, rot)` | Carries the object (arc or joint-space), re-measures the held offset in the actual TCP frame, aligns, lowers with a tilt-aware release height, releases (opening to `width + 8 mm` over 1.0 s, then a 4 mm back-off of the fixed finger) and retreats. |
| `pick_and_place(name, target, width)` | Both of the above in one call. |
| `move`, `follow` | Cartesian moves. The IK path is sampled at 48 points, smoothed in joint space (Gaussian σ = 3 samples), splined and time-scaled with min-jerk at `vmax`. |
| `transit`, `joint_move`, `rest`, `wait` | As in validation, but at `vmax` and with the training rest pose. |
| `jaw_gap`, `open_for`, `release_for` | Jaw angle for a given fingertip gap. |
| `feasible_rotation`, `carry_to`, `held` | End-orientation choice, carry mode selection and held-offset measurement. |

Root causes found while qualifying the pilots (2026-10-06), each fixed in the kit rather than whitelisted:

1. **Gripper resting on the table.** Validation `REST_DEG` puts the fixed jaw 8.3 mm into the table, so the PD presses it there (17-53 N, 285 table/gripper samples in the val replay). Training uses `TRAIN_REST_DEG = (0, -99, 90, 60, 0, -9)`, which keeps ≥ 9.2 mm table clearance and ≥ 4.9 mm wrist/shoulder clearance over 200 noisy resets.
2. **Joint-acceleration spikes from wrist-flex saturation.** While lifting to carry height, the wrist flex hits its ±95° limit mid-path. The commanded velocity then stops within one frame and the elbow reverses, so the frame-level target acceleration reached 29.9 rad/s² (measured joint acceleration 150-231 rad/s²). Joint-space smoothing brings this to 10-12 rad/s². In addition, at 30 fps the zero-order-hold staircase alone costs about 110 rad/s² per rad/s of joint speed, so `vmax` is 60/60/60/75/75 deg/s (validation uses 70/70/70/90/90).
3. **Object angular-acceleration spikes** (1400-2200 rad/s² against a 1000 limit):
   - The closing jaw shoved the block about 3 mm and it pitched; fixed by the 1.5 mm grasp clearance and the 1.2 s close.
   - The fixed finger dragged the block on retreat; fixed by the back-off.
   - Releasing up to 4 mm above the support (the 4 mm settle tolerance) made the block slap flat; fixed by the 1.5 mm release tolerance.
4. **Penetration.** Objects pivot 2-7° in the jaws because the arm tilts at carry height, so a release at the flat height pressed a block edge 3 mm into a mat. The release height adds half the tilt depth, and carry height dropped to 0.08 (worst pivot 6.9° → 3.9°).
5. **Misplacement.** The held offset was measured in the commanded (vertical) frame while the real TCP was tilted, and objects slip up to 8 mm in the jaws during a carry. The offset is now measured in the actual frame and re-measured over the target.

## Pitfalls (from `sim/val/README.md` and this pilot)

- **Rest sweep.** `rest()` sweeps the folded gripper low near (0.15, 0) and through x 0.12-0.17, y 0.04-0.08. Keep tall objects and placement targets out of that region; the keep-out handles the rest pose itself.
- **Close targets at large azimuth can fold the camera mount into the shoulder.** The one self-contact failure in 250 pilot episodes was a place target at r = 0.152 m, -45°. Prefer r ≥ 0.16 for place targets, or check `preview` sheets.
- **Stale kinematics in `layout()`.** Raw `ValEnv.set_object_pose` leaves `object_pos` stale until a forward pass. `TrainEnv` refreshes it, so use the kit setters before freezing goals.
- **Containers.** Grasps inside a container need `open_margin` ≈ 8 mm and the object near the centre; jaws pressing a bowl wall reached 29 N. Release with the small `release_for` gap so the opening jaw does not catch the rim (measured 20 N drag). The LIBERO basket (7 cm deep at 0.5×) was replaced by a white bowl at 1.0-1.2×.
- **Scene geometry.** `footprint()` is the bounding-box corner radius; for round objects the true radius is about `footprint/√2`. The stove origin is its knob, and the microwave must face sideways (see `sim/val/README.md`).
- **Determinism.** A rollout is a pure function of the seed. Use only `self.np_random` in layouts and the oracle's `rng`.
- **Validation code.** Never change `sim/val/*` behaviour. Only class attributes and hooks are used (`ValEnv.rest_deg`, the `build_mjcf` hook in `scene.py`); the validation defaults are unchanged.

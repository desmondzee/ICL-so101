# Original-command replay and block-task prefix diagnosis

## Oracle replay: completed

All six original command trajectories succeed through the current deployed
simulator path, and all six end in success. No learned model was involved.
Recorded commands were executed exactly once, in order, at 30 Hz, without
smoothing or early stopping.

| Episode | Controls | Final success | Arm pre-state MAE, degrees | Final task-object position difference, mm |
|---|---:|---|---:|---:|
| Ball 000 | 550 | Yes | 0.0000399 | 0.00644 |
| Ball 001 | 673 | Yes | 0.0004533 | 0.05503 |
| Block 000 | 576 | Yes | 0.0000520 | 0.00511 |
| Block 001 | 641 | Yes | 0.0002412 | 0.00375 |
| Push 000 | 507 | Yes | 0.0001691 | 0.39660 |
| Push 001 | 831 | Yes | 0.0002996 | 0.00680 |

The recording loop stores each observation **before** its corresponding action.
Replay joint states are therefore compared before action i with parquet state i,
not with the state after that action. Mean absolute error averages over all five
arm joints and all original controls. The largest single arm-state deviation
across the six runs is approximately 0.1084 degrees. Final position differences
compare live task-object xyz with the recorded end metadata, which has finite
decimal precision.

This strongly validates complete scene reset, recorded command interpretation,
deployment's degree-to-radian/float32 conversion and the simulator's execution
of these trajectories. It does not establish that every learned command is
feasible, or that model observation processing/cache semantics are correct.
The learned failures cannot be explained by the simulator simply being unable
to execute the original successful demonstrations.

### Visual evidence

Every episode has these artifacts under
`outputs/zero_wam/oracle_command_replay_scenev2/<task>/episode_<NNN>/`:

- `rollout.mp4`: current simulator front/wrist views, 30 fps, N+1 frames.
- `comparison.mp4`: original recording above replay, matching pre-control
  indices, 30 fps, N frames.
- `comparison_sheet.jpg`: four representative comparison frames.
- `telemetry.jsonl`: pre/post joint states, task-object positions, goal errors,
  grasp flags and success at every control.
- `pre_state_errors.npy` and `result.json`: aligned numerical comparisons.

The six comparison sheets were inspected. They reproduce the demonstrated
approach/manipulation/retreat sequence; the push tasks visibly move their cubes
into the tape squares, unlike the learned rollouts. Pixel identity is not an
acceptance requirement: video compression and rendering differences remain.
The two block sheets additionally show the exact selected handover phases.

Frozen numerical evidence: [oracle_summary.json](oracle_summary.json).
Quick visual comparisons: [block 000](assets/oracle_block000.jpg) and
[block 001](assets/oracle_block001.jpg).

## Prefix experiment: design frozen after oracle validation

Restrict the experiment to `block_in_bowl`, with its two recorded scenes.
Use four conditions per scene, packed in one batch of four:

| Scene | Zero prefix | Before closure | Held after lift | Held above bowl |
|---|---:|---:|---:|---:|
| Block 000 | 0 | 96 controls (3.20 s) | 176 (5.87 s) | 288 (9.60 s) |
| Block 001 | 0 | 152 controls (5.07 s) | 232 (7.73 s) | 368 (12.27 s) |

At the post-lift handovers, the simulator confirms a grasp and vertical
displacement of 6.46 cm and 6.33 cm, respectively. At the above-bowl handovers,
it still confirms a grasp; the block is approximately 5.74 cm above its original
height and nearly aligned horizontally with the goal. Before closure, there
is no grasp or meaningful lift.

The conditions test distinct questions: can the policy finish a grasp after a
correct approach; retain and carry an already grasped object; or lower/release
an object already above its destination? A zero-prefix rollout provides the
same-run baseline. Prefix-assisted success must not be presented as solving the
entire task autonomously from reset.

All conditions keep V2b step 500, full human-video/text conditioning, target
CFG 5.0, 50/50 denoising, F2 raw forecasts, eight-control execution, no EMA and
the same canonical noise seed within each source episode. During warmup the
model still predicts at every eight-control boundary, but its forecasts are
discarded and only the recorded prefix commands execute. This preserves noise
draw cadence and commits only actual observations and commanded history.
No future recorded robot observation or action is supplied to the policy.

The simulator is never teleported to a later recorded state. At each handover,
live joint states, task-object xyz, grasp and success must reproduce the
validated oracle replay. Total horizon remains twice the original oracle
length, including the prefix; it is not two fresh oracle-length horizons after
handover. All prefix boundaries precede task success.

Frozen conditions and expected handover states:
[prefix_manifest.json](prefix_manifest.json).
A separate 32-control smoke manifest checks early prefixes and cache feedback
before the full eight-condition experiment.

## Prefix results: all eight rollouts completed

| Handover | Block 000 success | Block 001 success | What the learned policy achieved |
|---|---|---|---|
| Zero prefix | No | No | Scene 000 places in the bowl but fails the upright criterion; scene 001 never grasps |
| Before closure | No | No | Grasps and lifts in both scenes, but releases away from the bowl |
| Held after lift | No | No | Maintains the grasp for about 4.7 s, then releases short of destination |
| Held above bowl | Yes | Yes | Lowers/releases successfully after about 1.4 s |

Both successful runs remain successful at the final eight-control boundary.
Block 000 succeeds at global control 329, 41 policy controls after prefix 288;
block 001 at 410, 42 after prefix 368. Each executes 48 policy controls total
because termination occurs at the end of the eight-control block. The failed
runs use their full 1,152/1,288-control horizons.

These are **prefix-assisted placement successes**, not end-to-end autonomous
successes from reset. Neither successful condition required the policy to find
the object, establish the grasp or carry it across the scene.

### Pickup and transport behavior

| Scene / handover | Learned controls with pinch | Maximum lift above original height, cm | Closest object-to-goal distance during policy, cm |
|---|---:|---:|---:|
| 000 / zero | 333 | 11.70 | 1.18 |
| 000 / pre-closure | 167 | 8.11 | 20.09 |
| 000 / post-lift | 139 | 9.14 | 16.44 |
| 000 / above bowl | 40 | 5.90 | 0.082 |
| 001 / zero | 0 | ≈0 | 40.10 |
| 001 / pre-closure | 197 | 8.07 | 31.03 |
| 001 / post-lift | 141 | 7.70 | 12.13 |
| 001 / above bowl | 41 | 5.68 | 0.337 |

Only controls **after handover** contribute to the pinch counts. Post-lift and
above-bowl conditions inherit height from the oracle prefix, so their maximum
height must not be credited as entirely policy-generated lift. Goal distance is
the Euclidean norm of object-origin xyz minus the declared target; it does not
include orientation, support, release or stability conditions.

The pre-closure conditions demonstrate genuine learned grasp and lift after a
correct oracle approach. Block 000 pinches continuously at controls 108–274;
block 001 at 167–363. Their failures are not missed pickups: they carry only part
of the way and release on the table away from the destination.

In post-lift conditions the first loss of opposing-finger contact is at
316/374, respectively 4.67/4.73 seconds after handover. The object is then
approximately 16.8/12.5 cm from the goal in the horizontal plane and already
near the table. The commanded gripper opens: over the preceding eight controls,
the command rises from about 6.45 to 18.75/17.77 on the 0–100 scale, followed
by settling on the table. This supports **lowering/opening before reaching the
destination**, rather than an immediate inability to keep the inherited grasp.
It does not identify why the predicted transport trajectory is inaccurate.

Above-bowl handovers instead retain contact until the release that coincides
with success. They show that the policy can execute the final local
lower-and-release behavior when the object is properly positioned.

### Short-term prediction versus long-term execution

The first eight learned arm targets after nonzero prefixes have mean absolute
errors of approximately 0.119–0.511 degrees against the corresponding next
oracle commands. All four early handovers nevertheless fail to complete the
task. Small immediate continuation error therefore remains compatible with
large later spatial error.

This comparison uses future oracle commands **only in the offline scorer**.
They were not supplied to inference. Successful adaptive behavior need not
match the oracle timing exactly, so these errors are diagnostics rather than
task-quality scores.

### A changed zero-prefix baseline

The new block-000 zero-prefix run grasps and nearly places, whereas the earlier
F2/raw sweep's block-000 run never registered a grasp. Both use canonical seed
20261103, but the new run occupies slot 0 in a batch containing four copies of
the block scene; the earlier one occupied slot 2 in a mixed-task batch.
The first eight predictions differ by only 0.434% relative L2 (37/48 scalar
targets exactly equal; arm-target MAE 0.110 degrees).

This is further evidence that single autonomous trajectories are sensitive to
small implementation/numerical changes. Batch composition and slot placement
are plausible contributors, not a proven isolated cause here. Use the new
same-experiment zero-prefix controls when interpreting these handovers; do not
attribute the new baseline's pickup to a prefix it never received.

The zero-prefix scene-000 block ends within the coordinate tolerances, released,
settled and supported by the bowl. **Only the upright criterion fails.** Replaying
the saved commands without inference reproduces every final joint state exactly
for all eight conditions and confirms this diagnosis:
[terminal_predicate_audit.json](terminal_predicate_audit.json).

The evaluator requires the block's local upright axis to be within 10 degrees
of world up; the final local-axis tilt is approximately 100.7 degrees using the
rounded recorded terminal quaternion. It has landed on a different face/tilt,
despite being in and supported by the bowl. This is a more restrictive criterion
than the text instruction simply saying to put a cube inside a bowl. Keep the
original success score unchanged, but discuss whether that orientation
requirement is intended for a symmetric cube before using the score alone to
judge semantic task completion.

The other five failures are outside the goal position and unsupported by the
bowl; this is not an orientation-only explanation for those failures. Both
above-bowl successes satisfy every terminal predicate.

### Visual and numerical evidence

- [Scene 000: all four handovers](assets/block_in_bowl__episode_000_prefixes.jpg)
- [Scene 001: all four handovers](assets/block_in_bowl__episode_001_prefixes.jpg)
- [Frozen policy-only measurements](prefix_measurements.json)

Each sheet shows the handover state, one second later, the end of pinch contact
(or a later frame when no pinch occurs), and final state. Times are aligned to
each handover rather than pretending all conditions start autonomous control
at the same global time. Videos and complete per-control telemetry are under
`outputs/zero_wam/block_prefix_handover_scenev2_b4/block_in_bowl/`,
in `episode_<NNN>__prefix_<BBBB>` directories.

The bounded smoke test passed 16 focused tests plus four live 32-control
rollouts. The full experiment reproduced all expected handover joint states,
object xyz, grasp flags and success flags before handing control to the policy.
Offline checks verify exact oracle commands before each boundary and raw policy
commands afterward. No oracle prefix finishes the task itself.

## Interpretation and next discussion

The simulator executes the original demonstrations successfully. Correct
approach prefixes enable learned grasp and lift in both scenes, while correct
above-goal prefixes enable learned placement in both scenes. The policy has
useful local manipulation capabilities, but linking them into reliable
spatially correct transport remains fragile.

This narrows the next discussion: examine inaccurate transport and premature
release as well as approach precision. A blanket conclusion that the checkpoint
cannot grasp or cannot place is inconsistent with these handovers. Conversely,
two favorable-state successes do not prove reliable task understanding,
recovery or in-context learning. There are only two scenes, one seed per scene
and no repeated matched-slot trials; rankings should remain descriptive.

## Deferred discussion

Choose the task and design for a focused training/memorization experiment only
after reviewing these diagnostics with the user. No new training has started.

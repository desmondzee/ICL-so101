# schema

Records for HumanGen-style human-robot pairs. A VLM fills the scene and task; code fills the rest.

| File | Unit | Filled by |
| --- | --- | --- |
| `scene.schema.json` | first frame of one episode, one camera | VLM, per episode |
| `task.schema.json` | the task one episode performs, over role names | VLM, per episode, reusing the dataset's first task as a naming hint |
| `pair.schema.json` | episode + camera: source, scene, task, role bindings, step segments, generation settings, human reference/video, checks | code; `bindings` by the scene call |
| `common.schema.json` | relations, conditions, evidence, checks | |

- Relations: `supported_by`, `inside`, `held_by`, `touching`, `at`, `left_of`, `right_of`, `in_front_of`, `behind`; unary `open`, `switched_on`, `folded`, `upright`.
- Actions: `pick_place`, `lift`, `slide`, `stack`, `open`, `close`, `press`, `pour`, `fold`.
- Task roles carry a short `name`; bound scene entities must reuse it verbatim, so prompts name objects the same way in every episode.
- Segments come from the gripper: a hold is where the gripper state stays above the command while the command is closed (`source.py`).
- `accepted` follows HumanGen's filter: semantic score 5, physics score ≥ 3, other checks pass.
- `validate.py` checks the JSON Schema plus references, relation arity, binding kinds, order cycles, goals not already true, segment ranges and the acceptance rule.
- `bundle(name)` gives one self-contained schema for Gemini structured output.

```sh
uv sync --extra schema
uv run --extra schema python -m schema.annotate lerobot/svla_so101_pickplace --episodes 0 1 2
uv run --extra schema python -m schema.validate schema/examples/*.json
```

`annotate` downloads the dataset to `data/<name>/` (ignored) and writes `data/<name>/pairs/{tasks/<task>.json, episode_XXX/{pair.json, first,grasp,release,last.jpg}}`. Both calls default to `gemini-3.8-flash` with thinking off (`--thinking 0`): on 10 episodes thinking-on added ~2k thinking tokens and 3x latency per call with no gain, and flash-lite guessed attributes and misstated support. `GEMINI_API_KEY` is read from `.env`.

Examples: `examples/svla_pickplace.json` (real, Gemini output for episode 0), `examples/basket.json` (sim, two steps, with the H3 prompt that was sent).

## Pipeline over the curated set

Per episode (`annotate.py`, Space Bunny on OpenRouter, free models only; Gemini Flash answers a call only if it fails):
1. Task from a 12-frame strip, gripper events and the dataset's task template and notes (`data/so101_pairs/<dataset>/tasks/*.json|*.notes.txt`, written by reviewer agents).
2. Scene and role bindings from the first, middle and last frames (two drafts, box IoU vote).
3. Step segments from gripper holds when every step is a grasp and counts match (objects assigned to hold windows), otherwise from a 16-frame strip.
4. Bindings checked against the objects actually grasped; outcome checked on the last seconds.

Then `judge.py sheets` and `judge_workflow.js` (Opus, low effort, five pairs per agent) judge every pair against its frames; `judge.py accept` keeps, per episode, the best judged version that validates and succeeded. `verify_workflow.js` is a per-episode Sonnet second opinion.

`fix_workflow.js` gives each rejected episode one Opus agent that corrects the pair against its frames (or confirms a real failure); fixed pairs are re-judged. `audit_workflow.js` then gives each pair in scope an Opus correction (medium effort) and an independent Sonnet check (low effort) from the frames, with one more correction and check for a failed pair; `data/so101_pairs/_logs/audit_merge.py` collects results across runs. `export.py` builds the final `accepted.json` (latest verdict wins: the audit's, else the judge's) and, with `--out`, the upload folder: LeRobot datasets filtered to the accepted episodes, renumbered pairs, task templates and these schemas (README from `export_readme.md`).

Result: 1960 of 2009 episodes accepted (1454 pass, 506 minor issues), 1347 of them checked by the audit in their final form. The audit covered all drawer and tea episodes (their task notes were corrected), every pair a second judge disagreed on and every pair not yet checked by Sonnet; 31 pairs still contested after two correction rounds are excluded. Lowest yields: tea (83/98) and the upper-drawer datasets (124/133, 139/149).

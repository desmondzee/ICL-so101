# schema

Records for HumanGen-style human-robot pairs. A VLM fills the scene and task; code fills the rest.

| File | Unit | Filled by |
| --- | --- | --- |
| `scene.schema.json` | first frame of one episode, one camera | VLM, per episode |
| `task.schema.json` | one dataset task, over role names | VLM, once per task |
| `pair.schema.json` | episode + camera: source, scene, task, role bindings, step segments, generation settings, human reference/video, checks | code; `bindings` by the scene call |
| `common.schema.json` | relations, conditions, evidence, checks | |

- Relations: `supported_by`, `inside`, `held_by`, `touching`, `at`, `left_of`, `right_of`, `in_front_of`, `behind`; unary `open`, `switched_on`, `folded`, `upright`.
- Actions: `pick_place`, `lift`, `slide`, `stack`, `open`, `close`, `press`, `pour`, `fold`.
- Segments come from the gripper: a hold is where the gripper state stays above the command while the command is closed (`source.py`).
- `accepted` follows HumanGen's filter: semantic score 5, physics score ≥ 3, other checks pass.
- `validate.py` checks the JSON Schema plus references, relation arity, binding kinds, order cycles, goals not already true, segment ranges and the acceptance rule.
- `bundle(name)` gives one self-contained schema for Gemini structured output.

```sh
uv sync --extra schema
uv run --extra schema python -m schema.annotate lerobot/svla_so101_pickplace --episodes 0 1 2
uv run --extra schema python -m schema.validate schema/examples/*.json
```

`annotate` downloads the dataset to `data/<name>/` (ignored) and writes `data/<name>/pairs/{tasks/<task>.json, episode_XXX/{pair.json, first,grasp,release,last.jpg}}`. The task model defaults to `gemini-3.8-flash`, the scene model to `gemini-3.5-flash-lite`. `GEMINI_API_KEY` is read from `.env`.

Examples: `examples/svla_pickplace.json` (real, Gemini output for episode 0), `examples/basket.json` (sim, two steps, with the H3 prompt that was sent).

export const meta = {
  name: 'so101-pair-fix',
  description: 'One Opus (medium effort) agent per rejected SO-101 pair: fix annotation errors against the frames, or confirm a genuine task failure',
  phases: [{ title: 'Fix', detail: 'one Opus agent per rejected episode edits pair.json and validates it' }],
}

const CTX = args.ctx
const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const OUT = {
  type: 'object',
  properties: {
    pair: { type: 'string' },
    changed: { type: 'boolean' },
    fixed_parts: { type: 'array', items: { type: 'string' } },
    genuine_failure: { type: 'boolean' },
    validated: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['pair', 'changed', 'fixed_parts', 'genuine_failure', 'validated', 'notes'],
}

const PROMPT = (i) => `You correct one automatically drafted annotation of a robot episode (single SO-101 arm) so it is true to the video. Work in ${REPO}.
Read ${CTX} and take element [${i}]: pair (the pair JSON to fix), dir (its episode folder: first.jpg, middle.jpg, last.jpg, task_strip.jpg, frames/*.jpg named by frame number, and segment/assign/outcome strips if present), sheet (judge sheet: first frame with boxes, 12 labelled frames with step bars), notes (the dataset's task notes, follow them), why (why it was rejected), judge_issues / judge_notes (what an independent judge found wrong).
Also read ${REPO}/schema/common.schema.json, scene.schema.json, task.schema.json, pair.schema.json for the vocabulary and rules.
Look at the frames yourself. When you need a specific frame, extract it:
cd ${REPO} && uv run --extra schema python -c "from pathlib import Path; from schema.source import load_episode, local_source, frame; r=Path('data/so101_curated/<name>'); ep=load_episode(r,*local_source(r),<ep>); print(frame(ep,'observation.images.front',<frame>,Path('<dir>/frames/<frame>.jpg')))"

Then:
1. Copy pair.json to pair.orig.json in the same folder (only if pair.orig.json does not exist yet).
2. Fix every real error in pair.json: task steps/goals/roles, segments (start/end/grasp/release in frames; segments in step order, non-overlapping, grasp/release inside the segment or null), scene entities/names/attributes/box_2d ([ymin,xmin,ymax,xmax] 0-1000 of the first frame), initial_state (first frame only), bindings (the entity actually manipulated; a bound entity's name and kind must equal its role's), and source.outcome (success only if the task's goals hold at the end of the episode). Keep ids stable where possible. Do not change source fields other than outcome, and keep the evidence list (you may append one evidence item describing your check).
3. If the robot genuinely fails the task, keep outcome "failure" and set genuine_failure=true; do not force a success.
4. Validate: uv run --extra schema python -m schema.validate <pair>  (must print ok; fix until it does).
Return what you changed.`

const n = args.count
const results = await parallel(Array.from({ length: n }, (_, i) => () =>
  agent(PROMPT(i), { label: `fix:${i}`, phase: 'Fix', schema: OUT, model: 'opus', effort: 'medium' })))
const ok = results.filter(Boolean)
log(`${ok.length}/${n} done: ${ok.filter(r => r.changed).length} changed, ${ok.filter(r => r.genuine_failure).length} genuine failures, ${ok.filter(r => r.validated).length} validated`)
return ok

export const meta = {
  name: 'so101-pair-verify',
  description: 'Second-opinion check of judged SO-101 pairs: one Sonnet (high effort) agent per episode with full frame, video and schema context',
  phases: [{ title: 'Verify', detail: 'one Sonnet agent per episode re-checks the pair and the first judge verdict' }],
}

const T = args.root
const VERDICTS = args.verdicts
const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const PART = { type: 'object', properties: { verdict: { enum: ['correct', 'partial', 'wrong'] }, issues: { type: 'array', items: { type: 'string' } } }, required: ['verdict', 'issues'] }
const SCHEMA = {
  type: 'object',
  properties: {
    pair: { type: 'string' },
    task_steps: PART, task_goals: PART, roles: PART, segments: PART, scene_entities: PART, initial_state: PART, boxes: PART, bindings: PART, outcome: PART,
    overall: { enum: ['pass', 'minor_issues', 'fail'] },
    agrees_with_first_judge: { type: 'boolean' },
    disagreements: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['pair', 'task_steps', 'task_goals', 'roles', 'segments', 'scene_entities', 'initial_state', 'boxes', 'bindings', 'outcome', 'overall', 'agrees_with_first_judge', 'disagreements', 'notes'],
}

const PROMPT = (it) => `You are the second, independent verifier of an automatically drafted annotation of one robot episode (single SO-101 arm). A first judge already reviewed it; check the annotation yourself, then compare with that judge.

Context (read with the Read tool):
- Pair JSON: ${it.pair} (task.instruction/roles/steps/goals, scene.entities/initial_state, bindings, segments, source.outcome/frames/fps/camera_key)
- Judge sheet: ${it.sheet} (top-left first frame with entity boxes; top-right task/outcome/segments; bottom 12 labelled frames with coloured bars for step segments; last tile = final frame)
- Episode images in ${it.dir}: first.jpg, middle.jpg, last.jpg, task_strip.jpg, segment_strip.jpg / assign_windows.jpg / outcome_tail.jpg if present, frames/*.jpg (named by frame number)
- Schema definitions: ${REPO}/schema/task.schema.json, ${REPO}/schema/scene.schema.json, ${REPO}/schema/common.schema.json
- Dataset task notes: ${T}/${it.name}/tasks/*.notes.txt (reviewer notes on what this dataset's task really is)
- First judge verdict: in ${VERDICTS}, the entry whose "pair" equals ${it.pair}
- If you need more frames, extract them from the source video with Bash: cd ${REPO} && uv run --extra schema python -c "from pathlib import Path; from schema.source import load_episode, local_source, frame; r=Path('data/so101_curated/${it.name}'); ep=load_episode(r,*local_source(r),${it.ep}); print(frame(ep,'<camera_key>',<frame_number>,Path('${it.dir}/frames/x_<frame_number>.jpg')))"

Judge each part (correct / partial / wrong, concrete issues):
- task_steps: one step per object manipulation, right order/action/object/destination; a tool use is one step.
- task_goals: the intended end state (not whether this episode achieved it).
- roles: every participant, sensible kind, short colour+type name.
- segments: each step's [start, end) covers when it actually happens; order right; about one tile of slop allowed.
- scene_entities: task objects, distractors, surface, robot; each separately movable object separate; nothing hallucinated.
- initial_state: facts about the first frame only.
- boxes: boxes enclose their objects in the first frame (hidden objects may lack boxes).
- bindings: each role maps to the entity actually manipulated.
- outcome: source.outcome matches whether the goals hold at the end of the episode.
overall: pass (all correct), minor_issues (only partials that would not mislead a video-generation prompt), fail (anything wrong or misleading).
Then set agrees_with_first_judge (same overall verdict and no material disagreement on any part) and list concrete disagreements. Be strict and do not invent problems.`

const items = Object.entries(args.datasets).flatMap(([name, eps]) => eps.map(e => {
  const ep = String(e).padStart(3, '0')
  return { name, ep: e, sheet: `${T}/_judge/${name}__ep${ep}.jpg`, pair: `${T}/${name}/episode_${ep}/pair.json`, dir: `${T}/${name}/episode_${ep}` }
}))
const results = await parallel(items.map(it => () => agent(PROMPT(it), {
  label: `verify:${it.name.slice(0, 24)}:${it.ep}`, phase: 'Verify', schema: SCHEMA, model: 'sonnet', effort: 'high',
})))
const ok = results.filter(Boolean)
log(`${ok.length}/${items.length} verified: ${ok.filter(r => r.overall === 'pass').length} pass, ${ok.filter(r => r.overall === 'minor_issues').length} minor, ${ok.filter(r => r.overall === 'fail').length} fail; agree with first judge ${ok.filter(r => r.agrees_with_first_judge).length}`)
return ok

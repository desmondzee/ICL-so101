export const meta = {
  name: 'so101-schema-judge',
  description: 'Judge drafted SO-101 scene/task/segment/outcome records against their frames, one Opus low-effort agent per pair',
  phases: [{ title: 'Judge', detail: 'one Opus (low effort) judge per drafted pair' }],
}

const PART = { type: 'object', properties: { verdict: { enum: ['correct', 'partial', 'wrong'] }, issues: { type: 'array', items: { type: 'string' } } }, required: ['verdict', 'issues'] }
const SCHEMA = {
  type: 'object',
  properties: {
    pair: { type: 'string' },
    task_steps: PART, task_goals: PART, roles: PART, segments: PART, scene_entities: PART, initial_state: PART, boxes: PART, bindings: PART, outcome: PART,
    overall: { enum: ['pass', 'minor_issues', 'fail'] },
    notes: { type: 'string' },
  },
  required: ['pair', 'task_steps', 'task_goals', 'roles', 'segments', 'scene_entities', 'initial_state', 'boxes', 'bindings', 'outcome', 'overall', 'notes'],
}

const RUBRIC = `You judge one automatically drafted annotation of a robot episode (single SO-101 arm) against its frames.
Files: the judge sheet image (top-left: first frame with the scene's entity boxes and names drawn; top-right: task, outcome and step segments as text; bottom: 12 evenly spaced frames labelled with frame numbers, the last tile is the episode's final frame, each with a coloured bar naming the step whose segment contains that frame, no bar = no step), the pair JSON, and first.jpg / last.jpg in the pair's directory.
Read them with the Read tool. In the JSON look at: task.instruction, task.roles, task.steps, task.goals, scene.entities, scene.initial_state, bindings, segments, source.outcome.
Judge each part (correct / partial / wrong, with concrete issues):
- task_steps: one step per object manipulation in the right order with the right action/object/destination; a tool use is one step. The task describes the dataset's intended task, so judge it against what the robot attempts.
- task_goals: goals describe the intended end state of the task (inside/supported_by/open/folded/clean...), including released objects; nothing false or trivial. Do NOT mark goals wrong just because this episode failed; that is what outcome is for.
- roles: every task participant has a role with a sensible kind and short colour+type name.
- segments: each step's [start, end) covers when that step actually happens (reach through release/retreat); order right. Allow about one tile of slop.
- scene_entities: task objects, visible distractors, supporting surface and the robot are listed, each separately movable object separately; nothing hallucinated.
- initial_state: facts about the FIRST frame only are right; occluded things may be null.
- boxes: boxes enclose the named objects in the first frame (a missing box for an object hidden by the robot is fine).
- bindings: each role maps to the entity actually manipulated.
- outcome: source.outcome (success/failure/unknown) matches whether the goals hold in the last frame.
overall: pass if every part is correct; minor_issues if only partial verdicts that would not mislead a video-generation prompt; fail if anything is wrong or misleading.
Be strict and concrete; do not invent problems.`

const T = args.root
const items = args.items.map(b => { const [name, ep] = b.split('__ep'); return { sheet: `${T}/_judge/${b}.jpg`, pair: `${T}/${name}/episode_${ep}/pair.json` } })
const results = await parallel(items.map(it => () => agent(
  `${RUBRIC}\n\nJudge sheet: ${it.sheet}\nPair JSON: ${it.pair}\nSet "pair" to the pair JSON path.`,
  { label: `judge:${it.sheet.split('/').pop()}`, phase: 'Judge', schema: SCHEMA, model: 'opus', effort: 'low' },
)))
const ok = results.filter(Boolean)
log(`${ok.length} judged: ${ok.filter(r => r.overall === 'pass').length} pass, ${ok.filter(r => r.overall === 'minor_issues').length} minor, ${ok.filter(r => r.overall === 'fail').length} fail`)
return ok
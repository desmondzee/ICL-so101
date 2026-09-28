export const meta = {
  name: 'humangen-clip-judge',
  description: 'One judge agent (Sonnet by default, args.model to change) per generated human clip: HumanGen checks (reference preservation, first frame, task consistency, semantic and physics scores)',
  phases: [{ title: 'Judge', detail: 'one Sonnet agent per clip' }],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const STATUS = { type: 'object', properties: { status: { enum: ['pass', 'fail', 'unknown'] }, detail: { type: 'string' } }, required: ['status', 'detail'] }
const SCORE = { type: 'object', properties: { score: { type: 'integer', minimum: 1, maximum: 5 }, detail: { type: 'string' } }, required: ['score', 'detail'] }
const SCHEMA = {
  type: 'object',
  properties: {
    dir: { type: 'string' },
    reference_preservation: STATUS,
    first_frame_visible: STATUS,
    task_consistency: STATUS,
    semantic: { type: 'object', properties: { score: { type: 'integer', minimum: 1, maximum: 5 }, goals: { type: 'array', items: { enum: ['pass', 'fail', 'unknown'] } }, detail: { type: 'string' } }, required: ['score', 'goals', 'detail'] },
    physics: SCORE,
    notes: { type: 'string' },
  },
  required: ['dir', 'reference_preservation', 'first_frame_visible', 'task_consistency', 'semantic', 'physics', 'notes'],
}

const PROMPT = (dir) => `You judge one generated human video that should show a person doing the same task a robot did. Work in ${REPO}. Do not edit any file; you may write extracted frames under ${dir}/judge/.
Files in ${dir}: robot_first.jpg and robot_last.jpg if present (robot episode first and last frame), frame.jpg (edited start: the person's hand(s) in the same scene), end_frame.jpg if present (edited end), kf1.jpg..kfN.jpg if present (edited state after each step; the video is the step clips joined, it should pass through each), video.mp4 (the generated clip, 24 fps), contact.png (the clip sampled at 2 Hz), pair.json (task.instruction, task.steps, task.goals, human.entity_map, generation.video_prompt, generation.idle_hand). Hands allowed: one acting hand, plus at most a second hand of the same person that only steadies or holds a task object already in the scene. A second hand that brings anything in, does the task instead of the acting hand, or any third hand fails.
You must look at the images yourself: open contact.png, and extract more frames where needed, e.g.
  mkdir -p ${dir}/judge && ffmpeg -nostdin -loglevel error -y -i ${dir}/video.mp4 -vf fps=4 ${dir}/judge/f_%03d.jpg
Judge:
- reference_preservation: the clip keeps the scene of frame.jpg (same objects, layout, camera, no robot; a hand entering is expected when frame.jpg has none); pass/fail.
- first_frame_visible: in the clip's first frame every task object is visible, and the acting hand too if frame.jpg shows a hand (if frame.jpg has no hand, the hand is meant to enter during the clip); pass/fail.
- task_consistency: the clip does the same task as the robot episode (same steps on the same objects, same order, same destinations) and its last frame shows the same end state as robot_last.jpg (object places and states); pass/fail.
- semantic: 1-5, how completely and correctly the task in pair.json is carried out (5 = every step and the end state exactly right); goals = one of pass/fail/unknown per entry of task.goals, in order, read at the end of the clip (treat the robot in a goal as the person's hand).
- physics: 1-5, physical plausibility (5 = natural hand motion and contact, objects move only when touched, no passing through, no morphing, appearing or vanishing objects, no extra hands, no burned-in text or graphics). A hand that enters already holding a copy of an object is an appearing object. Count the visible hands in at least 8 frames spread over the clip; a hand not allowed by the rule above caps physics at 2 and fails task_consistency. Any new object, copy of an object, text, caption or graphic overlay caps physics at 2.
Be strict and do not invent problems. In each detail, cite the frames you looked at.`

const dirs = args.dirs.map(d => (args.prefix || '') + d)
const results = await parallel(dirs.map((d, i) => () =>
  agent(PROMPT(d), { label: `judge:${i}`, phase: 'Judge', schema: SCHEMA, model: args.model || 'sonnet', effort: args.effort || 'medium' })))
const ok = results.filter(Boolean)
const accepted = ok.filter(r => r.semantic.score === 5 && r.physics.score >= 3 && ['reference_preservation', 'first_frame_visible', 'task_consistency'].every(k => r[k].status === 'pass'))
log(`${ok.length}/${dirs.length} judged, ${accepted.length} pass the HumanGen filter`)
return ok

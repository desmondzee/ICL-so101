// Runs a review workflow script with mocked agent/pipeline/parallel/log (no model calls).
// usage: node workflow_harness.mjs <workflow.js> <args.json> [rejectKey,...] ; prints the return value as JSON.
import { readFileSync } from 'node:fs'

const [, , script, argsPath, rejects = ''] = process.argv
const reject = new Set(rejects.split(',').filter(Boolean))
const args = JSON.parse(readFileSync(argsPath, 'utf8'))
const body = readFileSync(script, 'utf8').replace(/^export const meta/m, 'const meta')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const calls = []

const agent = async (prompt, opts) => {
  calls.push(opts.label)
  const keys = [...prompt.matchAll(/Pair (\S+), in /g)].map(m => m[1])
  const isVerify = !!opts.schema.properties.pairs.items.properties.confirmed
  const pairs = keys.map(key => isVerify
    ? { key, confirmed: true, reason: 'checked fps=6 frames', reasons: ['checked fps=6 frames 0-5s'], inspected_evidence: [`${key}/human.mp4 @6fps`] }
    : { key, verdict: reject.has(key) ? 'reject' : 'accept', task_adherence: reject.has(key) ? 2 : 5, physics: 5,
        order_ok: true, end_state_match: !reject.has(key), hands_ok: true, scene_ok: true, issues: [],
        reasons: [reject.has(key) ? 'block never reaches the bowl (4.8s)' : 'completes the task in order (1.2-4.0s)'],
        inspected_evidence: ['human_frames.jpg', 'robot_frames.jpg'], summary: 'hand moves the block' })
  return { pairs }
}
const pipeline = async (items, s1, s2) => Promise.all(items.map(async (it, i) => s2(await s1(it, it, i), it, i)))
const parallel = async (fns) => Promise.all(fns.map(f => f()))
const log = () => {}

const result = await new AsyncFunction('args', 'agent', 'pipeline', 'parallel', 'log', body)(args, agent, pipeline, parallel, log)
console.log(JSON.stringify({ result, calls }))

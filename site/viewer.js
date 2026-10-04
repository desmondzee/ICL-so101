// One pair at a time: pick a category, click (or use ← →) through its pairs. The position is kept in the URL hash:
// #<task>/<n> or #all/<n> for training, #val/<task>/<n> or #val/all/<n> for validation. Videos stream from the HF bucket.

const SETS = {}
const $ = (id) => document.getElementById(id)
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const cap = (s) => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1)
const tasksOf = (d) => d.tasks.filter((t) => t.episodes > 0).sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
const pairsOf = (d, task) => task === 'all'
  ? tasksOf(d).flatMap((t) => d.episodes.filter((e) => e.task === t.task))
  : d.episodes.filter((e) => e.task === task)

function state() {
  let h = decodeURIComponent(location.hash.slice(1))
  const set = h.startsWith('val/') && SETS.val ? 'val' : 'train'
  if (h.startsWith('val/')) h = h.slice(4)
  const [task, n] = h.split('/')
  const d = SETS[set]
  const valid = task === 'all' || tasksOf(d).some((t) => t.task === task)
  return { set, task: valid ? task : 'all', n: Math.max(0, parseInt(n || '1', 10) - 1) }
}

function go(set, task, n) {
  const len = pairsOf(SETS[set], task).length
  location.hash = `${set === 'val' ? 'val/' : ''}${task}/${((n % len) + len) % len + 1}`
}

function render() {
  const { set, task, n } = state()
  const d = SETS[set]
  const eps = pairsOf(d, task)
  const i = Math.min(n, eps.length - 1)
  const ep = eps[i]
  const url = (file) => `${d.base_url}episodes/${ep.id}/${file}`
  $('task').value = `${set}:${task}`
  $('count').textContent = `${i + 1} / ${eps.length}`
  $('title').textContent = cap(ep.instruction)
  const views = Object.keys(ep.views)
  const switcher = views.length > 1
    ? `<div class="views">${views.map((v, k) => `<button aria-pressed="${k === 0}" data-file="${esc(ep.views[v])}">${esc(v)}</button>`).join('')}</div>` : ''
  $('videos').innerHTML = `
    <div class="clip"><span class="tag">Human demonstration</span>
      <video controls playsinline preload="metadata" poster="${url(ep.thumb)}" src="${url(ep.human)}"></video></div>
    <div class="clip"><span class="tag">${set === 'val' ? 'Robot video (simulation)' : 'Robot video'}</span>${switcher}
      <video controls muted playsinline preload="metadata" poster="${url(ep.robot_thumb)}" src="${url(ep.views[views[0]])}"></video></div>`
  const robot = $('videos').querySelectorAll('video')[1]
  $('videos').querySelectorAll('.views button').forEach((b, _, all) => b.addEventListener('click', () => {
    all.forEach((x) => x.setAttribute('aria-pressed', x === b))
    robot.src = url(b.dataset.file)
    robot.play().catch(() => {})
  }))
  $('caption').textContent = `${set === 'val' ? 'Validation · ' : ''}Episode ${ep.curated_episode_index ?? ep.episode} · human ${ep.human_duration_s.toFixed(1)} s · robot ${ep.robot_duration_s.toFixed(1)} s`
}

const load = (path) => fetch(path).then((r) => (r.ok ? r.json() : null)).catch(() => null)

Promise.all([load('data/index.json'), load('data/val_index.json')]).then(([train, val]) => {
  if (!train) { $('title').textContent = 'Could not load the dataset index.'; return }
  SETS.train = train
  if (val) SETS.val = val
  const options = (set, d, label) => `<optgroup label="${label}"><option value="${set}:all">All ${set === 'val' ? 'validation' : 'training'} pairs (${d.episodes_total})</option>` +
    tasksOf(d).map((t) => `<option value="${set}:${esc(t.task)}">${esc(cap(t.instruction))} (${t.episodes})</option>`).join('') + '</optgroup>'
  $('task').innerHTML = options('train', train, 'Training (real)') + (val ? options('val', val, 'Validation (simulation)') : '')
  $('task').addEventListener('change', (e) => { const [set, task] = e.target.value.split(':'); go(set, task, 0) })
  $('prev').addEventListener('click', () => { const s = state(); go(s.set, s.task, s.n - 1) })
  $('next').addEventListener('click', () => { const s = state(); go(s.set, s.task, s.n + 1) })
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return
    if (e.key === 'ArrowLeft') { const s = state(); go(s.set, s.task, s.n - 1) }
    if (e.key === 'ArrowRight') { const s = state(); go(s.set, s.task, s.n + 1) }
  })
  window.addEventListener('hashchange', render)
  render()
})

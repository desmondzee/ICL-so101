// One pair at a time: pick a category, click (or use ← →) through its pairs. The position is kept in the URL hash
// (#<task>/<n>, or #all/<n>). Videos stream from the public HF bucket.

let DATA = null
const $ = (id) => document.getElementById(id)
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const cap = (s) => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1)
const url = (ep, file) => `${DATA.base_url}episodes/${ep.id}/${file}`
const tasks = () => DATA.tasks.filter((t) => t.episodes > 0).sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
const pairsOf = (task) => task === 'all'
  ? tasks().flatMap((t) => DATA.episodes.filter((e) => e.task === t.task))
  : DATA.episodes.filter((e) => e.task === task)

function state() {
  const [task, n] = decodeURIComponent(location.hash.slice(1)).split('/')
  const valid = task === 'all' || tasks().some((t) => t.task === task)
  return { task: valid ? task : 'all', n: Math.max(0, parseInt(n || '1', 10) - 1) }
}

function go(task, n) {
  const len = pairsOf(task).length
  location.hash = `${task}/${((n % len) + len) % len + 1}`
}

function render() {
  const { task, n } = state()
  const eps = pairsOf(task)
  const i = Math.min(n, eps.length - 1)
  const ep = eps[i]
  $('task').value = task
  $('count').textContent = `${i + 1} / ${eps.length}`
  $('title').textContent = cap(ep.instruction)
  const views = Object.keys(ep.views)
  const switcher = views.length > 1
    ? `<div class="views">${views.map((v, k) => `<button aria-pressed="${k === 0}" data-file="${esc(ep.views[v])}">${esc(v)}</button>`).join('')}</div>` : ''
  $('videos').innerHTML = `
    <div class="clip"><span class="tag">Human demonstration</span>
      <video controls playsinline preload="metadata" poster="${url(ep, ep.thumb)}" src="${url(ep, ep.human)}"></video></div>
    <div class="clip"><span class="tag">Robot video</span>${switcher}
      <video controls muted playsinline preload="metadata" poster="${url(ep, ep.robot_thumb)}" src="${url(ep, ep.views[views[0]])}"></video></div>`
  const robot = $('videos').querySelectorAll('video')[1]
  $('videos').querySelectorAll('.views button').forEach((b, _, all) => b.addEventListener('click', () => {
    all.forEach((x) => x.setAttribute('aria-pressed', x === b))
    robot.src = url(ep, b.dataset.file)
    robot.play().catch(() => {})
  }))
  $('caption').textContent = `Episode ${ep.curated_episode_index} · human ${ep.human_duration_s.toFixed(1)} s · robot ${ep.robot_duration_s.toFixed(1)} s`
}

fetch('data/index.json').then((r) => r.json()).then((d) => {
  DATA = d
  $('task').innerHTML = `<option value="all">All tasks (${d.episodes_total})</option>` +
    tasks().map((t) => `<option value="${esc(t.task)}">${esc(cap(t.instruction))} (${t.episodes})</option>`).join('')
  $('task').addEventListener('change', (e) => go(e.target.value, 0))
  $('prev').addEventListener('click', () => { const s = state(); go(s.task, s.n - 1) })
  $('next').addEventListener('click', () => { const s = state(); go(s.task, s.n + 1) })
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return
    if (e.key === 'ArrowLeft') { const s = state(); go(s.task, s.n - 1) }
    if (e.key === 'ArrowRight') { const s = state(); go(s.task, s.n + 1) }
  })
  window.addEventListener('hashchange', render)
  render()
}).catch(() => { $('title').textContent = 'Could not load the dataset index.' })

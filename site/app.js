// Static viewer for the curated SO-101 HumanGen pairs. The index ships with the site; videos and images stream from the
// public HF bucket (index.base_url). Routes: #/ (overview), #/task/<task>, #/ep/<task>/<episode>.

const app = document.getElementById('app')
let DATA = null

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const pretty = (task) => task.replace(/^[^_]+__/, '').replace(/[_-]+/g, ' ')
const owner = (task) => task.split('__')[0]
const url = (ep, file) => `${DATA.base_url}episodes/${ep.id}/${file}`
const byTask = (task) => DATA.episodes.filter((e) => e.task === task)
const cap = (s) => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1)

function card(href, thumbs, title, meta) {
  return `<a class="card" href="${href}">
    <div class="thumbs">${thumbs.map((t) => `<img loading="lazy" src="${t}" alt="">`).join('')}</div>
    <div class="body"><div class="title">${title}</div><div class="meta">${meta}</div></div></a>`
}

function overview() {
  const tasks = DATA.tasks.filter((t) => t.episodes > 0)
  const views = new Set(DATA.episodes.flatMap((e) => Object.keys(e.views)))
  const hours = DATA.episodes.reduce((s, e) => s + e.robot_duration_s, 0) / 3600
  app.innerHTML = `
    <h1>SO-101 HumanGen pairs</h1>
    <p class="lede">Real SO-101 robot episodes, each paired with a generated video of a person doing the same task in the same scene.
      Every pair was checked by two independent reviewers for task adherence and physical plausibility; only pairs both accepted are here.</p>
    <div class="stats">
      <div class="stat"><b>${DATA.episodes_total}</b><span>pairs</span></div>
      <div class="stat"><b>${tasks.length}</b><span>tasks</span></div>
      <div class="stat"><b>${views.size + 1}</b><span>views per pair (human + ${[...views].join(', ')})</span></div>
      <div class="stat"><b>${hours >= 1 ? hours.toFixed(1) + ' h' : Math.round(hours * 60) + ' min'}</b><span>robot data</span></div>
    </div>
    <div class="filter"><input id="q" placeholder="Filter tasks…" aria-label="Filter tasks"></div>
    <div class="grid" id="tasks"></div>`
  const draw = (q) => {
    document.getElementById('tasks').innerHTML = tasks
      .filter((t) => !q || (t.task + ' ' + t.instruction).toLowerCase().includes(q))
      .map((t) => {
        const ep = byTask(t.task)[0]
        const under = t.episodes < 5 ? '<span class="badge warn">under target</span>' : ''
        return card(`#/task/${t.task}`, [url(ep, ep.thumb), url(ep, ep.robot_thumb)], esc(cap(t.instruction)),
          `<span>${t.episodes} pairs</span><span>${esc(pretty(t.task))}</span>${under}`)
      }).join('')
  }
  document.getElementById('q').addEventListener('input', (e) => draw(e.target.value.toLowerCase()))
  draw('')
}

function taskView(task) {
  const t = DATA.tasks.find((x) => x.task === task)
  const eps = byTask(task)
  if (!t || !eps.length) return notFound()
  app.innerHTML = `
    <div class="crumbs"><a href="#/">All tasks</a> / ${esc(pretty(task))}</div>
    <h1>${esc(cap(t.instruction))}</h1>
    <p class="lede"><span class="mono">${esc(task)}</span> · ${eps.length} pairs · robot views: ${esc(t.views.join(', '))} ·
      ${t.accepted} accepted of ${t.reviewed} reviewed (${t.available_demos} generated)</p>
    <div class="grid">${eps.map((e) => card(`#/ep/${e.id}`, [url(e, e.thumb), url(e, e.robot_thumb)],
      `Episode ${e.curated_episode_index}`,
      `<span>robot ${e.robot_duration_s.toFixed(1)} s</span><span>human ${e.human_duration_s.toFixed(1)} s</span>
       <span class="badge good">task ${e.review.task_adherence}/5 · physics ${e.review.physics}/5</span>`)).join('')}</div>`
}

function episodeView(id) {
  const ep = DATA.episodes.find((e) => e.id === id)
  if (!ep) return notFound()
  const eps = byTask(ep.task)
  const i = eps.indexOf(ep)
  const prev = eps[(i - 1 + eps.length) % eps.length], next = eps[(i + 1) % eps.length]
  app.innerHTML = `
    <div class="crumbs"><a href="#/">All tasks</a> / <a href="#/task/${ep.task}">${esc(pretty(ep.task))}</a> / episode ${ep.curated_episode_index}</div>
    <h1>${esc(cap(ep.instruction))}</h1>
    <div class="player">
      <div class="view human"><video data-role="human" src="${url(ep, ep.human)}" muted playsinline preload="auto"></video>
        <div class="label"><b>Human demo (generated)</b><span>${ep.human_duration_s.toFixed(1)} s</span></div></div>
      ${Object.entries(ep.views).map(([name, file]) => `
      <div class="view"><video data-role="robot" src="${url(ep, file)}" muted playsinline preload="auto"></video>
        <div class="label"><span>Robot · ${esc(name)}</span><span>${ep.robot_duration_s.toFixed(1)} s</span></div></div>`).join('')}
    </div>
    <div class="controls">
      <button id="play">Play</button>
      <input id="seek" type="range" min="0" max="1000" value="0" aria-label="Progress through the episode">
      <label class="toggle"><input id="align" type="checkbox" checked> Align durations</label>
      <label class="toggle"><input id="sound" type="checkbox"> Sound</label>
      <a class="btn" href="#/ep/${prev.id}">‹ Prev</a><a class="btn" href="#/ep/${next.id}">Next ›</a>
    </div>
    <div class="panels">
      <div class="panel"><h3>Review</h3>
        <div class="scores"><span>Task <b>${ep.review.task_adherence}</b>/5</span><span>Physics <b>${ep.review.physics}</b>/5</span></div>
        <p>${esc(ep.review.summary)}</p>
        ${ep.review.issues.length ? `<ul>${ep.review.issues.map((x) => `<li class="muted">${esc(x)}</li>`).join('')}</ul>` : ''}
        <p class="muted"><b>Second reviewer:</b> ${esc(ep.review.verify)}</p></div>
      <div class="panel"><h3>Episode</h3>
        <dl class="kv">
          <dt>Task</dt><dd class="mono">${esc(ep.task)}</dd>
          <dt>Episode</dt><dd>${ep.curated_episode_index} (export index ${ep.export_episode_index})</dd>
          <dt>Robot</dt><dd>${ep.robot_frames} frames at ${ep.fps} fps</dd>
          <dt>Steps</dt><dd>${ep.steps.length ? `<ol>${ep.steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>` : '—'}</dd>
          <dt>Files</dt><dd class="files"><a href="${url(ep, 'pair.json')}" target="_blank">pair.json</a> ·
            <a href="${url(ep, 'robot_data.parquet')}">robot_data.parquet</a> · <a href="${url(ep, 'review.json')}" target="_blank">review.json</a> ·
            <a href="${url(ep, ep.human)}" target="_blank">human.mp4</a></dd>
        </dl></div>
    </div>`
  syncPlayer()
}

// Play every view together. With "align durations" the robot views are sped up or slowed down so all views start and
// finish together (the human clip is ~5 s, the robot episode 7-40 s); the scrubber works in fractions of each clip.
function syncPlayer() {
  const vids = [...app.querySelectorAll('video')]
  const human = vids[0]
  const play = document.getElementById('play'), seek = document.getElementById('seek')
  const align = document.getElementById('align'), sound = document.getElementById('sound')
  let playing = false
  const rates = () => vids.forEach((v) => {
    if (v === human || !align.checked || !human.duration || !v.duration) v.playbackRate = 1
    else v.playbackRate = Math.min(16, Math.max(0.0625, v.duration / human.duration))
  })
  const setAll = (frac) => vids.forEach((v) => { if (v.duration) v.currentTime = Math.min(v.duration - 0.01, frac * v.duration) })
  vids.forEach((v) => v.addEventListener('loadedmetadata', rates))
  align.addEventListener('change', rates)
  sound.addEventListener('change', () => { human.muted = !sound.checked })
  play.addEventListener('click', () => {
    playing = !playing
    play.textContent = playing ? 'Pause' : 'Play'
    if (playing) {
      if (human.ended || human.currentTime >= human.duration - 0.05) setAll(0)
      rates(); vids.forEach((v) => v.play().catch(() => {}))
    } else vids.forEach((v) => v.pause())
  })
  seek.addEventListener('input', () => setAll(seek.value / 1000))
  human.addEventListener('timeupdate', () => { if (human.duration) seek.value = Math.round(1000 * human.currentTime / human.duration) })
  human.addEventListener('ended', () => { playing = false; play.textContent = 'Play'; vids.forEach((v) => v.pause()) })
  vids.slice(1).forEach((v) => v.addEventListener('ended', () => v.pause()))
}

function notFound() { app.innerHTML = '<p class="muted">Not found. <a href="#/">Back to all tasks</a></p>' }

function route() {
  window.scrollTo(0, 0)
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, ''))
  if (h.startsWith('task/')) return taskView(h.slice(5))
  if (h.startsWith('ep/')) return episodeView(h.slice(3))
  return overview()
}

fetch('data/index.json').then((r) => r.json()).then((d) => { DATA = d; route() })
  .catch(() => { app.innerHTML = '<p class="muted">Could not load the dataset index.</p>' })
window.addEventListener('hashchange', () => DATA && route())

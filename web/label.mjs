import {categories, names, emptyLabels, fromServer, toServer, addInterval, removeInterval, toggleAmbiguous, setEdge, setStatus, finalize, hitTest, addWatched, watchedFraction} from './labelmodel.mjs';
import {createSaveQueue, playbackSpan} from './labelsession.mjs';

const $ = id => document.getElementById(id);
const video = $('clip'), lanes = $('lanes'), ctx = lanes.getContext('2d');
const colors = {nudity:'#e58fa5', sexual_activity:'#c98be0', violence:'#ef8a6a', gore:'#d65c5c', kissing:'#f0c46e'};
const GUTTER = 128, RULER = 18, LANE = 34;
let token = '', project = null, current = null, labels = emptyLabels(), category = categories[0], selection = null;
let inPoint = null, watched = [], lastFrame = null, drafts = null, showDrafts = false, revealArmed = false, saveTimer = null;
let busy = false, recovery = null, watchDirty = false;

function notice(text) {
  for (const id of ['notice', 'action-error']) {$(id).textContent = text; $(id).hidden = !text;}
}
async function api(path, body, method = body === undefined ? 'GET' : 'PUT', retry = true) {
  const response = await fetch('/api/' + path, {method, cache:'no-store', signal:AbortSignal.timeout(15000), headers:{'x-censor-token':token, ...(body === undefined ? {} : {'Content-Type':'application/json'})}, body:body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  // The server may have restarted, or another tab may have renewed the cookie.
  if (response.status === 403 && path !== 'bootstrap' && retry && ['Missing application token', 'Open the app on this server before using its API'].includes(data.detail)) {
    token = (await api('bootstrap')).token;
    return api(path, body, method, false);
  }
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
const safely = fn => async (...args) => {try {await fn(...args);} catch (error) {notice(error.message);}};
const fps = () => current?.fps || 24;
const target = () => current.target;
const frameStart = t => Math.floor(t * fps() + 1e-6) / fps();
const frameEnd = t => (Math.floor(t * fps() + 1e-6) + 1) / fps();
const stamp = seconds => {const s = Math.max(0, seconds); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${(s % 60).toFixed(2).padStart(5, '0')}`;};
const storageKey = () => `label-watched:${project.name}:${current.id}`;
const draftKey = (name = project.name, id = current.id) => `label-unsaved:${name}:${id}`;
function store(key, value) {try {localStorage.setItem(key, JSON.stringify(value));} catch {}}
function persistWatch() {if (current && watchDirty) {store(storageKey(), watched); watchDirty = false;}}
function recordWatched(a, b) {watched = addWatched(watched, a, b); watchDirty = true;}
function readWatched(value) {
  if (!Array.isArray(value)) return [];
  return value.reduce((result, span) => Array.isArray(span) && span.length === 2 && span.every(Number.isFinite) ? addWatched(result, ...span) : result, []);
}

// ---- persistence -----------------------------------------------------------
const saves = createSaveQueue(entry => api(entry.path, entry.body), (state, entry, result) => {
  if (result && state !== 'failed' && project?.name === entry.project) {
    const window = project.windows.find(w => w.id === entry.id);
    if (window) Object.assign(window, {started:true, complete:result.complete});
    renderList();
  }
  $('save-state').textContent = state === 'saved' ? `Saved ${new Date().toLocaleTimeString()}` : state === 'saving' ? 'Saving…' : state === 'failed' ? 'Save failed · retry below' : 'Unsaved…';
  $('save-now').textContent = state === 'failed' ? 'Retry save' : 'Save now';
  $('save-now').disabled = !saves.dirty || busy || !!recovery;
  if (state === 'saved') {try {localStorage.removeItem(draftKey(entry.project, entry.id));} catch {}}
  if (state === 'failed') notice(`Labels have not been saved. ${result.message} Use Retry save; this window will stay open.`);
});
function scheduleSave() {
  if (!current) return;
  const entry = {project:project.name, id:current.id, path:`labeling/${project.name}/labels/${current.id}`, body:toServer(labels)};
  store(draftKey(), {body:entry.body, watched});
  saves.mark(entry); persistWatch();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {saveTimer = null; safely(flush)();}, 500);
}
async function flush() {
  clearTimeout(saveTimer); saveTimer = null; persistWatch();
  await saves.flush();
}
async function transition(fn) {
  if (busy) return;
  busy = true; video.pause(); lastFrame = null;
  document.querySelector('.label-stage').inert = true;
  $('project').disabled = true;
  try {
    if (recovery) throw Error('Choose Restore unsaved labels or Keep saved labels before leaving this window.');
    await fn();
  } finally {
    busy = false; document.querySelector('.label-stage').inert = false; $('project').disabled = false;
    if (project) $('project').value = project.name;
    draw();
  }
}

// ---- loading ---------------------------------------------------------------
async function loadProject(name) {
  return transition(async () => {
    await flush();
    const nextProject = await api(`labeling/${name}`);
    const [, wanted] = decodeURIComponent(location.hash.slice(1)).split('/');
    const first = nextProject.windows.find(w => w.id === wanted) || nextProject.windows.find(w => !w.complete) || nextProject.windows[0];
    if (first) await loadWindow(first.id, nextProject);
    else {project = nextProject; current = null; video.removeAttribute('src'); notice('This project has no windows yet.');}
    $('definitions-revision').textContent = project.definitions_revision;
    $('definitions').replaceChildren(...categories.flatMap(c => {const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = names[c]; dd.textContent = project.definitions[c]; return [dt, dd];}));
    renderList();
  });
}

async function openWindow(id) {
  return transition(() => loadWindow(id));
}
async function loadWindow(id, nextProject = project) {
  await flush();
  const window = nextProject.windows.find(w => w.id === id);
  if (!window) throw Error('Window not found.');
  const data = await api(`labeling/${nextProject.name}/labels/${id}`);
  project = nextProject; current = window; labels = fromServer(data);
  // Validation and test labels must stay blind until deliberately revealed.
  selection = null; inPoint = null; drafts = null; showDrafts = current.has_drafts && !['val', 'test'].includes(current.split); revealArmed = false; lastFrame = null;
  try {watched = readWatched(JSON.parse(localStorage.getItem(storageKey()) || '[]'));} catch {watched = [];}
  watchDirty = false; recovery = null;
  try {
    const backup = JSON.parse(localStorage.getItem(draftKey()) || 'null');
    if (backup?.body?.categories && JSON.stringify(fromServer(backup.body)) !== JSON.stringify(labels)) recovery = backup;
    else localStorage.removeItem(draftKey());
  } catch {}
  $('recovery').hidden = !recovery; $('note').readOnly = !!recovery;
  history.replaceState(null, '', `#${project.name}/${id}`);
  video.src = `/api/labeling/${project.name}/clip/${id}`;
  video.playbackRate = +$('speed').value;
  video.onloadedmetadata = () => {lastFrame = null; video.currentTime = window.target[0]; draw();};
  $('window-title').textContent = `${current.title} · ${current.episode.toUpperCase()}`;
  const collection = ['val', 'test'].includes(current.split) ? 'REVIEW WINDOW' : current.stratum?.toUpperCase() || '';
  $('window-meta').textContent = `${current.id.toUpperCase()} · ${collection} · SOURCE ${stamp(current.clip_offset + target()[0])}`;
  $('split-badge').textContent = current.split.toUpperCase(); $('split-badge').className = current.split;
  $('note').value = labels.note;
  $('draft-note').textContent = current.has_drafts ? (['val', 'test'].includes(current.split) ? 'Held-out window: label first, then compare.' : '') : 'No model drafts for this window yet.';
  if (showDrafts) {
    try {await loadDrafts(); labels.drafts_shown = true;}
    catch (error) {showDrafts = false; $('draft-note').textContent = `Model drafts unavailable: ${error.message}`;}
  }
  $('save-state').textContent = recovery ? 'Unsaved labels available · restore below' : data.updated ? 'Saved labels loaded' : 'No labels saved yet';
  notice(''); renderList(); render();
}

async function loadDrafts() {
  if (!drafts) drafts = await api(`labeling/${project.name}/drafts/${current.id}`);
}

// ---- rendering -------------------------------------------------------------
function renderList() {
  if (!project) return;
  const split = $('split-filter').value, open = $('only-open').checked;
  const complete = project.windows.filter(w => w.complete).length;
  $('progress').textContent = `${complete} / ${project.windows.length} labeled`;
  $('windows').replaceChildren(...project.windows.filter(w => (!split || w.split === split) && (!open || !w.complete || w === current)).map(w => {
    const li = document.createElement('li'), dot = document.createElement('span'), text = document.createElement('span'), small = document.createElement('small');
    dot.className = 'dot' + (w.complete ? ' complete' : w.started ? ' started' : '');
    text.textContent = `${w.title} ${w.episode.toUpperCase()}`; small.textContent = `${w.split} · ${Math.round(w.clip_offset + w.target[0])}s`;
    li.append(dot, text, small); li.className = w === current ? 'current' : ''; li.onclick = safely(() => openWindow(w.id));
    return li;
  }));
}

function render() {
  $('category-rows').replaceChildren(...categories.map((c, i) => {
    const rule = labels.categories[c], row = document.createElement('div');
    row.className = 'category-row' + (c === category ? ' selected' : '');
    const key = document.createElement('kbd'); key.textContent = i + 1;
    const name = document.createElement('strong'); const swatch = document.createElement('span'); swatch.className = 'swatch'; swatch.style.background = colors[c];
    name.append(swatch, names[c]);
    const summary = document.createElement('span'); summary.className = 'summary';
    const seconds = rule.intervals.reduce((s, [a, b]) => s + b - a, 0);
    summary.textContent = rule.intervals.length ? `${rule.intervals.length} span${rule.intervals.length > 1 ? 's' : ''} · ${seconds.toFixed(2)}s` + (rule.ambiguous.length ? ` · ${rule.ambiguous.length} ambiguous` : '') : rule.ambiguous.length ? `${rule.ambiguous.length} ambiguous only` : 'none marked';
    const reviewed = document.createElement('button'), unsure = document.createElement('button');
    reviewed.textContent = 'Reviewed'; reviewed.className = rule.status === 'reviewed' ? 'on' : '';
    unsure.textContent = 'Unsure'; unsure.className = 'unsure' + (rule.status === 'unsure' ? ' on' : '');
    reviewed.disabled = unsure.disabled = !!recovery;
    reviewed.onclick = event => {event.stopPropagation(); change(() => setStatus(labels, c, 'reviewed'));};
    unsure.onclick = event => {event.stopPropagation(); change(() => setStatus(labels, c, 'unsure'));};
    row.append(key, name, summary, reviewed, unsure); row.onclick = () => {category = c; render();};
    return row;
  }));
  const span = selection && labels.categories[selection.category][selection.list][selection.index];
  $('selection-text').textContent = span ? `${names[selection.category]} · ${selection.list === 'ambiguous' ? 'ambiguous' : 'definite'} · ${(span[0] - target()[0]).toFixed(2)}s → ${(span[1] - target()[0]).toFixed(2)}s` : 'No span selected. Use Mark start, then Mark end.';
  $('toggle-ambiguous').disabled = $('delete-span').disabled = !span;
  $('in-point').textContent = inPoint === null ? '' : `Start marked at ${(inPoint.time - target()[0]).toFixed(2)}s for ${names[inPoint.category]} — mark the end`;
  $('drafts-toggle').textContent = showDrafts ? 'Hide model drafts (D)' : 'Show model drafts (D)';
  draw();
}

function x(t) {return GUTTER + (t / Math.max(.01, current.clip_duration)) * (lanes.width - GUTTER - 8);}
function draw() {
  ctx.fillStyle = '#141a23'; ctx.fillRect(0, 0, lanes.width, lanes.height);
  if (!current) return;
  const [low, high] = target(), t = video.currentTime;
  ctx.fillStyle = '#0b0f15'; ctx.fillRect(GUTTER, 0, x(low) - GUTTER, lanes.height); ctx.fillRect(x(high), 0, lanes.width - x(high), lanes.height);
  ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = '#7d8796';
  for (let s = Math.ceil(-low); s <= current.clip_duration - low; s++) {const px = x(low + s); ctx.fillRect(px, 0, 1, 5); if (s % 2 === 0) ctx.fillText(`${s}s`, px + 2, 13);}
  for (const [a, b] of watched) {ctx.fillStyle = '#3d5a3a'; ctx.fillRect(x(Math.max(a, low)), RULER - 3, Math.max(0, x(Math.min(b, high)) - x(Math.max(a, low))), 2);}
  categories.forEach((c, i) => {
    const top = RULER + i * LANE, rule = labels.categories[c];
    ctx.fillStyle = c === category ? '#1d2735' : '#171e28'; ctx.fillRect(GUTTER, top + 2, lanes.width - GUTTER, LANE - 4);
    ctx.fillStyle = c === category ? '#f1f3ef' : '#a4adba'; ctx.font = '12px system-ui, sans-serif'; ctx.fillText(`${i + 1}  ${names[c]}`, 8, top + 21);
    if (rule.status) {ctx.fillStyle = rule.status === 'unsure' ? '#eab977' : '#b7ecac'; ctx.fillText(rule.status === 'unsure' ? '?' : '✓', GUTTER - 16, top + 21);}
    if (showDrafts && drafts) {
      for (const candidate of drafts.candidates.filter(d => d.category === c)) {ctx.fillStyle = '#ffffff40'; ctx.fillRect(x(candidate.start), top + LANE - 9, Math.max(2, x(candidate.end) - x(candidate.start)), 4);}
      for (const event of drafts.events.filter(d => d.category === c)) {
        ctx.setLineDash(event.review_required ? [4, 3] : []); ctx.strokeStyle = '#ffffffb0'; ctx.lineWidth = 1.5;
        ctx.strokeRect(x(event.start), top + 5, Math.max(2, x(event.end) - x(event.start)), LANE - 10);
      }
      ctx.setLineDash([]);
    }
    for (const list of ['intervals', 'ambiguous']) rule[list].forEach(([a, b], index) => {
      const left = x(a), width = Math.max(2, x(b) - left);
      ctx.fillStyle = colors[c] + (list === 'ambiguous' ? '55' : 'dd'); ctx.fillRect(left, top + 7, width, LANE - 14);
      if (list === 'ambiguous') {ctx.strokeStyle = colors[c]; ctx.setLineDash([3, 3]); ctx.strokeRect(left, top + 7, width, LANE - 14); ctx.setLineDash([]);}
      if (selection && selection.category === c && selection.list === list && selection.index === index) {ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.strokeRect(left - 1, top + 6, width + 2, LANE - 12);}
    });
  });
  if (inPoint !== null) {ctx.strokeStyle = '#eab977'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x(inPoint.time), 0); ctx.lineTo(x(inPoint.time), lanes.height); ctx.stroke(); ctx.setLineDash([]);}
  ctx.fillStyle = '#f7fbf2'; ctx.fillRect(x(t) - 1, 0, 2, lanes.height);
  const outside = t < low - 1e-3 || t >= high - 1e-3;
  $('context-badge').hidden = !outside; video.parentElement.classList.toggle('context', outside);
  $('clock').textContent = `${(t - low).toFixed(2)}s · frame ${Math.floor(t * fps())} · source ${stamp(current.clip_offset + t)}`;
  const fraction = watchedFraction(watched, target());
  $('watched').textContent = `Watched ${Math.round(fraction * 100)}%`; $('watched').classList.toggle('ok', fraction >= .95);
  $('play').textContent = video.paused ? 'Play' : 'Pause';
  const allDone = project.windows.every(w => w.complete);
  $('finish-hint').textContent = recovery ? 'Unsaved labels were recovered. Choose which version to use below.' : busy ? 'Saving / loading…' : allDone && !saves.dirty ? 'All windows in this project are saved and reviewed.' : inPoint ? 'Mark the end of your open span before finishing.' : fraction < .95 ? `Watch ${Math.max(0, (.95 - fraction) * (high - low)).toFixed(1)}s more to finish. Labels save as you edit. If already reviewed, Shift+click Done or Shift+Enter.` : 'Ready: Done reviews all five categories, saves, then opens the next unfinished window.';
  $('done').disabled = busy || !!recovery;
  $('save-now').disabled = busy || !!recovery || !saves.dirty;
  $('mark-start').disabled = busy || !!recovery;
  $('mark-end').disabled = busy || !!recovery || inPoint === null;
}

// ---- editing ---------------------------------------------------------------
function change(fn) {
  if (busy || recovery) return;
  try {fn();} catch (error) {notice(error.message); return;}
  notice(''); scheduleSave(); render();
}
function seek(t) {video.pause(); lastFrame = null; video.currentTime = Math.max(0, Math.min(current.clip_duration - 1 / fps() / 2, t));}
function step(frames) {
  const before = video.currentTime, n = Math.floor(before * fps() + 1e-6) + frames;
  seek((n + .5) / fps());
  if (frames > 0) recordWatched(before, video.currentTime + 1 / fps() / 2);
  draw();
}
function markIn() {if (busy || recovery) return; inPoint = {time:frameStart(video.currentTime), category}; selection = null; notice(''); render();}
function markOut() {
  if (busy || recovery) return;
  if (inPoint === null) {notice('Press I at the start of the span first.'); return;}
  const end = frameEnd(video.currentTime), start = inPoint.time, markedCategory = inPoint.category;
  inPoint = null;
  change(() => {selection = addInterval(labels, markedCategory, Math.min(start, end), Math.max(start, end), target());});
}
async function done(force) {
  if (busy) return;
  if (inPoint) {notice('Your span has a start but no end. Use Mark end (O) first, or Esc to cancel it.'); return;}
  const fraction = watchedFraction(watched, target());
  if (fraction < .95 && !force) {notice(`Not finished: ${Math.round(fraction * 100)}% watched. Play the target through, then click Done. If you already reviewed it, Shift+click Done or Shift+Enter overrides this check.`); return;}
  return transition(async () => {
    const reviewed = finalize(labels);
    scheduleSave(); await flush(); render();
    notice('');
    if (reviewed.length) $('save-state').textContent += ` · ${reviewed.length} categories marked reviewed`;
    const order = project.windows, index = order.indexOf(current);
    const next = order.slice(index + 1).find(w => !w.complete) || order.find(w => !w.complete);
    if (next) await loadWindow(next.id); else notice('Every window in this project is labeled. Thank you!');
  });
}
async function toggleDrafts() {
  if (!current.has_drafts) {notice('No model drafts exist for this window yet.'); return;}
  if (!showDrafts && ['val', 'test'].includes(current.split) && !labels.drafts_shown && !revealArmed) {
    revealArmed = true; notice('This is a held-out window. Seeing drafts before labeling biases the ground truth, and it is recorded. Press D again to reveal.'); return;
  }
  showDrafts = !showDrafts; revealArmed = false;
  if (showDrafts) {await loadDrafts(); if (!labels.drafts_shown) {labels.drafts_shown = true; scheduleSave();}}
  notice(''); render();
}

// ---- events ----------------------------------------------------------------
lanes.addEventListener('click', safely(async event => {
  if (!current || busy || recovery) return;
  const box = lanes.getBoundingClientRect(), px = (event.clientX - box.left) * lanes.width / box.width, py = (event.clientY - box.top) * lanes.height / box.height;
  const lane = Math.floor((py - RULER) / LANE), t = (px - GUTTER) / (lanes.width - GUTTER - 8) * current.clip_duration;
  if (lane < 0 || lane >= categories.length || px < GUTTER) return;
  category = categories[lane];
  const tolerance = 3 / (lanes.width - GUTTER - 8) * current.clip_duration;
  if (event.shiftKey && showDrafts && drafts) {
    const hit = drafts.events.find(d => d.category === category && d.start - tolerance <= t && t < d.end + tolerance);
    if (hit) {change(() => {selection = addInterval(labels, category, hit.start, hit.end, target());}); return;}
  }
  selection = hitTest(labels, category, t, tolerance);
  if (selection) seek(labels.categories[category][selection.list][selection.index][0]); else seek(t);
  render();
}));
$('project').onchange = safely(() => loadProject($('project').value));
$('split-filter').onchange = renderList; $('only-open').onchange = renderList;
const playPause = safely(async () => {if (!current || busy) return; if (video.paused) await video.play(); else video.pause();});
$('play').onclick = playPause;
$('frame-back').onclick = () => step(-1); $('frame-forward').onclick = () => step(1);
$('replay').onclick = safely(async () => {seek(target()[0]); await video.play();});
$('speed').onchange = () => {video.playbackRate = +$('speed').value;};
$('mark-start').onclick = markIn; $('mark-end').onclick = markOut;
$('toggle-ambiguous').onclick = () => change(() => {selection = toggleAmbiguous(labels, selection);});
$('delete-span').onclick = () => change(() => {removeInterval(labels, selection); selection = null;});
$('done').onclick = event => safely(done)(event.shiftKey);
$('drafts-toggle').onclick = safely(() => transition(toggleDrafts));
$('save-now').onclick = safely(async () => {await flush(); notice(''); draw();});
$('restore-draft').onclick = () => {
  if (!recovery) return;
  labels = fromServer(recovery.body); watched = readWatched([...watched, ...(recovery.watched || [])]);
  recovery = null; watchDirty = true; $('recovery').hidden = true; $('note').readOnly = false; $('note').value = labels.note;
  scheduleSave(); notice('Recovered your unsaved labels. Saving them now.'); render();
};
$('keep-saved').onclick = () => {
  recovery = null; $('recovery').hidden = true; $('note').readOnly = false;
  try {localStorage.removeItem(draftKey());} catch {}
  $('save-state').textContent = current.started ? 'Saved labels loaded' : 'No labels saved yet'; notice(''); render();
};
$('note').oninput = () => {labels.note = $('note').value; scheduleSave();};
function recordPlayback(time) {
  if (!current || video.seeking || document.hidden) {lastFrame = null; return;}
  const now = performance.now(), span = playbackSpan(lastFrame, time, {now, speed:video.playbackRate});
  if (span) recordWatched(...span);
  lastFrame = {time, now};
}
video.addEventListener('play', () => {lastFrame = null; recordPlayback(video.currentTime); draw();});
video.addEventListener('pause', () => {recordPlayback(video.currentTime); lastFrame = null; persistWatch(); draw();});
video.addEventListener('seeking', () => {lastFrame = null;});
video.addEventListener('seeked', () => {if (!video.paused) recordPlayback(video.currentTime); draw();});
video.addEventListener('ratechange', () => {lastFrame = video.paused ? null : {time:video.currentTime, now:performance.now()};});
video.addEventListener('ended', () => {recordPlayback(video.currentTime); lastFrame = null; persistWatch(); draw();});
video.addEventListener('error', () => notice('This clip could not play. Refresh to renew the server session, or choose another window.'));
if (typeof video.requestVideoFrameCallback === 'function') {
  function onFrame(_now, metadata) {
    if (!video.paused) recordPlayback(metadata.mediaTime);
    draw(); video.requestVideoFrameCallback(onFrame);
  }
  video.requestVideoFrameCallback(onFrame);
} else video.addEventListener('timeupdate', () => {if (!video.paused) recordPlayback(video.currentTime); draw();});

document.addEventListener('keydown', event => {
  if (!current || busy || recovery || event.ctrlKey || event.metaKey || event.altKey) return;
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  if (typing && event.key !== 'Escape') return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (event.repeat && !['ArrowLeft', 'ArrowRight'].includes(key)) {event.preventDefault(); return;}
  if (key === ' ') playPause();
  else if (key === 'ArrowLeft') event.shiftKey ? seek(video.currentTime - 1) : step(-1);
  else if (key === 'ArrowRight') event.shiftKey ? seek(video.currentTime + 1) : step(1);
  else if (/^[1-5]$/.test(key)) {category = categories[+key - 1]; render();}
  else if (key === 'i' && event.shiftKey && selection) change(() => {selection = setEdge(labels, selection, 'start', frameStart(video.currentTime), target());});
  else if (key === 'o' && event.shiftKey && selection) change(() => {selection = setEdge(labels, selection, 'end', frameEnd(video.currentTime), target());});
  else if (key === 'i') markIn();
  else if (key === 'o') markOut();
  else if (key === 'a' && selection) change(() => {selection = toggleAmbiguous(labels, selection);});
  else if ((key === 'Delete' || key === 'Backspace') && selection) change(() => {removeInterval(labels, selection); selection = null;});
  else if (key === 'r') change(() => setStatus(labels, category, 'reviewed'));
  else if (key === 'u') change(() => setStatus(labels, category, 'unsure'));
  else if (key === 'p') $('replay').onclick();
  else if (key === 'd') $('drafts-toggle').onclick();
  else if (key === 'Enter') safely(done)(event.shiftKey);
  else if (key === '[' || key === ']') {const order = project.windows, i = order.indexOf(current) + (key === ']' ? 1 : -1); if (order[i]) safely(openWindow)(order[i].id);}
  else if (key === 'Escape') {inPoint = null; selection = null; if (typing) document.activeElement.blur(); render();}
  else return;
  event.preventDefault();
});
document.addEventListener('visibilitychange', () => {if (document.hidden) {video.pause(); persistWatch(); safely(flush)();}});
setInterval(() => {if (current && !document.hidden && !busy && !recovery) {labels.seconds_spent += 1; persistWatch();}}, 1000);
window.addEventListener('beforeunload', event => {persistWatch(); if (saves.dirty) {event.preventDefault(); event.returnValue = '';}});
window.addEventListener('pagehide', persistWatch);

try {
  token = (await api('bootstrap')).token;
  const projects = await api('labeling');
  if (!projects.length) {notice('No labeling projects yet. Create one with select_windows.py and labeling.py create.');}
  for (const p of projects) {const option = document.createElement('option'); option.value = p.name; option.textContent = `${p.name} · ${p.complete}/${p.windows}`; $('project').append(option);}
  const [wanted] = decodeURIComponent(location.hash.slice(1)).split('/');
  if (projects.some(p => p.name === wanted)) $('project').value = wanted;
  if (projects.length) await loadProject($('project').value);
} catch (error) {notice(`Cannot reach the local engine: ${error.message}`);}

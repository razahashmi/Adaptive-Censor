// Exercise the real page event handlers with deterministic DOM/media/network doubles.
// No browser, live labeling project, or human annotation is modified.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {emptyLabels, toServer, isComplete} from './labelmodel.mjs';

let instance = 0;
const htmlIds = new Set([...readFileSync(new URL('./label.html', import.meta.url), 'utf8').matchAll(/id="([^"]+)"/g)].map(match => match[1]));
const clone = value => structuredClone(value);
const defer = () => {let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve};};

async function page(options = {}) {
  const originals = new Map(), timers = new Map(), storage = new Map(Object.entries(options.storage || {}));
  let now = 0, timerId = 0, writes = 0, bootstraps = 0;
  class Element {
    constructor(tagName = 'DIV') {
      Object.assign(this, {tagName, children:[], listeners:{}, style:{}, dataset:{}, value:'', disabled:false, hidden:false, checked:false, textContent:''});
      this.classList = {toggle() {}};
    }
    append(...nodes) {this.children.push(...nodes);}
    replaceChildren(...nodes) {this.children = nodes;}
    addEventListener(name, handler) {(this.listeners[name] ||= []).push(handler);}
    emit(name, event = {}) {for (const handler of this.listeners[name] || []) handler(event);}
    removeAttribute() {}
    blur() {document.activeElement = null;}
  }
  const elements = new Map(), el = id => {
    assert.ok(htmlIds.has(id), `Page references missing HTML element ${id}`);
    if (!elements.has(id)) elements.set(id, new Element(id === 'note' ? 'INPUT' : 'DIV'));
    return elements.get(id);
  };
  el('project').value = 'p'; el('speed').value = '1'; el('only-open').checked = true;
  const video = el('clip');
  Object.assign(video, {currentTime:0, paused:true, seeking:false, playbackRate:1, parentElement:new Element()});
  video.pause = () => {if (!video.paused) {video.paused = true; video.emit('pause');}};
  video.play = async () => {video.paused = false; video.emit('play');};
  if (!options.fallback) video.requestVideoFrameCallback = handler => {video.frame = handler;};
  Object.assign(el('lanes'), {width:1200, height:196, getContext:() => new Proxy({}, {get:(_object, name) => () => {}})});
  const stage = new Element(), doc = new Element(), win = new Element();
  Object.assign(doc, {getElementById:el, createElement:tag => new Element(tag.toUpperCase()), querySelector:() => stage, activeElement:null, hidden:false});
  const location = {hash:'#p/a'};
  const windows = ['a', 'b', 'c'].map(id => ({id, title:'Show', episode:'s01e01', target:[2, 14], clip_duration:16, clip_offset:0, fps:24, split:options.split || 'train', stratum:options.stratum || '', complete:false, started:false, has_drafts:!!options.drafts}));
  const saved = new Map(Object.entries(options.labels || {}));
  const requests = [];
  const response = (data, status = 200) => ({ok:status < 400, status, json:async () => clone(data)});
  const setGlobal = (key, value) => {originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key)); Object.defineProperty(globalThis, key, {configurable:true, writable:true, value});};
  for (const [key, value] of Object.entries({
    document:doc, window:win, location,
    history:{replaceState(_state, _unused, hash) {location.hash = hash;}},
    localStorage:{getItem:key => storage.get(key) ?? null, setItem:(key, value) => storage.set(key, value), removeItem:key => storage.delete(key)},
    performance:{now:() => now}, setTimeout:fn => {timers.set(++timerId, fn); return timerId;}, clearTimeout:id => timers.delete(id), setInterval:() => 1,
    fetch:async (url, init) => {
      requests.push({url, ...init});
      if (url === '/api/bootstrap') {bootstraps++; return response({token:'token-' + bootstraps});}
      if (url === '/api/labeling') return response([{name:'p', windows:3, complete:0}]);
      if (url === '/api/labeling/p') return response({name:'p', definitions_revision:1, definitions:{}, windows:windows.map(w => ({...w, complete:isComplete(saved.get(w.id) || emptyLabels()), started:saved.has(w.id)}))});
      if (url.startsWith('/api/labeling/p/drafts/')) return response({events:[], candidates:[]});
      const id = url.split('/').at(-1);
      if (init.method === 'GET') {
        if (options.onRead) {const failure = await options.onRead(id); if (failure) return response({detail:failure}, 500);}
        return response(saved.get(id) || {});
      }
      const body = JSON.parse(init.body); writes++;
      if (options.onWrite) {const failure = await options.onWrite({id, body, writes}); if (failure) return response({detail:failure.detail || failure, ...failure}, failure.status || 500);}
      const result = {...body, updated:'save-' + writes, complete:isComplete(body)}; saved.set(id, result); return response(result);
    }
  })) setGlobal(key, value);
  const cleanup = () => {for (const [key, descriptor] of originals) {if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];}};
  try {await import(`./label.mjs?test=${++instance}`); video.onloadedmetadata?.();}
  catch (error) {cleanup(); throw error;}
  const click = async (id, event = {}) => {const element = typeof id === 'string' ? el(id) : id; if (!element.disabled) await element.onclick?.({shiftKey:false, stopPropagation() {}, ...event});};
  const frame = time => {
    now += 1000; video.currentTime = time;
    if (video.frame) video.frame(now, {mediaTime:time}); else video.emit('timeupdate');
  };
  const watch = async () => {await click('replay'); for (let time = 3; time <= 14; time++) frame(time); video.pause();};
  const mark = async () => {video.currentTime = 3; await click('mark-start'); video.currentTime = 4; await click('mark-end');};
  return {el, doc, win, video, stage, storage, saved, location, requests, click, watch, mark, frame, cleanup,
    autosave:async () => {for (const [id, fn] of [...timers]) {timers.delete(id); fn();} for (let i = 0; i < 20; i++) await Promise.resolve();}};
}

test('Done explains the watch check beside the button, saves every category, and advances', async () => {
  const p = await page();
  try {
    await p.click('done');
    assert.equal(p.location.hash, '#p/a'); assert.match(p.el('action-error').textContent, /0% watched/); assert.equal(p.saved.size, 0);
    await p.watch(); assert.equal(p.el('watched').textContent, 'Watched 100%');
    await p.mark(); await p.click('done');
    assert.equal(p.location.hash, '#p/b'); assert.equal(p.saved.get('a').complete, true);
    assert.equal(p.saved.get('a').categories.nudity.intervals.length, 1);
    assert.equal(p.el('save-state').textContent, 'No labels saved yet');
  } finally {p.cleanup();}
});

for (const split of ['val', 'test']) {
  test(`${split} drafts stay hidden through labeling and require a second explicit reveal`, async () => {
    const p = await page({split, drafts:true, stratum:'mined:sexual_activity'});
    try {
      const draftReads = () => p.requests.filter(r => r.url.includes('/drafts/'));
      assert.equal(draftReads().length, 0);
      assert.match(p.el('draft-note').textContent, /label first/);
      assert.doesNotMatch(p.el('window-meta').textContent, /SEXUAL_ACTIVITY|MINED/);
      await p.click('done', {shiftKey:true});
      assert.equal(p.saved.get('a').drafts_shown, false);
      assert.equal(draftReads().length, 0);
      await p.click('drafts-toggle');
      assert.equal(draftReads().length, 0);
      assert.match(p.el('action-error').textContent, /Press D again/);
      await p.click('drafts-toggle'); await p.click('save-now');
      assert.equal(draftReads().length, 1);
      assert.equal(p.saved.get('b').drafts_shown, true);
    } finally {p.cleanup();}
  });
}

test('training drafts still load automatically and record their exposure', async () => {
  const p = await page({split:'train', drafts:true});
  try {
    assert.equal(p.requests.filter(r => r.url.includes('/drafts/')).length, 1);
    await p.click('save-now');
    // The exposure flag is persisted on the next actual label save.
    await p.click('done', {shiftKey:true});
    assert.equal(p.saved.get('a').drafts_shown, true);
  } finally {p.cleanup();}
});

test('failed Done keeps the clip and labels; retry saves before advancing', async () => {
  let fail = true;
  const p = await page({onWrite:() => fail ? 'disk unavailable' : null});
  try {
    await p.watch(); await p.mark(); await p.click('done');
    assert.equal(p.location.hash, '#p/a'); assert.equal(p.el('save-state').textContent, 'Save failed · retry below');
    assert.match(p.el('action-error').textContent, /disk unavailable/);
    assert.equal(p.el('save-now').textContent, 'Retry save');
    assert.ok(p.storage.has('label-unsaved:p:a')); assert.equal(p.saved.size, 0);
    await p.click(p.el('windows').children[1]); assert.equal(p.location.hash, '#p/a');
    fail = false; await p.click('save-now');
    assert.equal(p.saved.get('a').complete, true); assert.equal(p.storage.has('label-unsaved:p:a'), false);
    await p.click('done'); assert.equal(p.location.hash, '#p/b');
  } finally {p.cleanup();}
});

test('a restarted server renews authentication and retries the same label write once', async () => {
  const p = await page({onWrite:({writes}) => writes === 1 ? {status:403, detail:'Missing application token'} : null});
  try {
    await p.watch(); await p.click('done');
    assert.equal(p.location.hash, '#p/b'); assert.equal(p.saved.get('a').complete, true);
    const puts = p.requests.filter(r => r.method === 'PUT');
    assert.equal(puts.length, 2); assert.deepEqual(JSON.parse(puts[0].body), JSON.parse(puts[1].body));
    assert.equal(puts[1].headers['x-censor-token'], 'token-2');
  } finally {p.cleanup();}
});

test('watched coverage survives a reload even when no labels were edited', async () => {
  let storage;
  const first = await page();
  try {await first.watch(); storage = Object.fromEntries(first.storage); assert.deepEqual(JSON.parse(storage['label-watched:p:a']), [[2, 14]]);}
  finally {first.cleanup();}
  const second = await page({storage});
  try {assert.equal(second.el('watched').textContent, 'Watched 100%'); await second.click('done'); assert.equal(second.location.hash, '#p/b');}
  finally {second.cleanup();}
});

test('unsaved edits can be restored after reload without replacing saved labels automatically', async () => {
  const labels = emptyLabels(); labels.note = 'recover this'; labels.categories.gore.intervals = [[3, 4]];
  const p = await page({storage:{'label-unsaved:p:a':JSON.stringify({body:toServer(labels), watched:[[2, 14]]})}});
  try {
    assert.equal(p.el('recovery').hidden, false); assert.equal(p.el('done').disabled, true); assert.equal(p.saved.size, 0);
    await p.click('restore-draft'); await p.click('done');
    assert.equal(p.saved.get('a').note, 'recover this'); assert.deepEqual(p.saved.get('a').categories.gore.intervals, [[3, 4]]);
    assert.equal(p.location.hash, '#p/b');
  } finally {p.cleanup();}
});

test('Done cannot discard a start marker and marking stays with its original category', async () => {
  const p = await page();
  try {
    await p.watch(); p.video.currentTime = 3; await p.click('mark-start');
    await p.click('done'); assert.match(p.el('action-error').textContent, /start but no end/); assert.equal(p.saved.size, 0);
    await p.click(p.el('category-rows').children[3]); p.video.currentTime = 4; await p.click('mark-end'); await p.click('save-now');
    assert.equal(p.saved.get('a').categories.nudity.intervals.length, 1); assert.equal(p.saved.get('a').categories.gore.intervals.length, 0);
  } finally {p.cleanup();}
});

test('double Done and rapid navigation cannot finish or load unintended windows', async () => {
  const gate = defer();
  const p = await page({onWrite:async () => {await gate.promise;}});
  try {
    await p.watch(); const first = p.click('done'); await p.click('done'); await p.click(p.el('windows').children[2]);
    assert.equal(p.location.hash, '#p/a'); gate.resolve(); await first;
    assert.equal(p.location.hash, '#p/b'); assert.equal(p.saved.size, 1); assert.equal(p.stage.inert, false);
  } finally {p.cleanup();}
});

test('a failed window load leaves the previous label context intact', async () => {
  const p = await page({onRead:id => id === 'b' ? 'cannot load labels' : null});
  try {
    await p.click(p.el('windows').children[1]); assert.equal(p.location.hash, '#p/a');
    assert.match(p.el('action-error').textContent, /cannot load labels/);
    await p.mark(); await p.click('save-now'); assert.equal(p.saved.has('a'), true); assert.equal(p.saved.has('b'), false);
  } finally {p.cleanup();}
});

test('watch coverage works without video frame callbacks and seeking does not count skipped footage', async () => {
  const p = await page({fallback:true});
  try {
    await p.video.play(); p.video.emit('seeking'); p.video.currentTime = 12; p.video.emit('seeked'); p.frame(13); p.video.pause();
    assert.match(p.el('watched').textContent, /8%/); await p.click('replay');
    for (let time = 3; time <= 14; time++) p.frame(time); p.video.pause();
    assert.equal(p.el('watched').textContent, 'Watched 100%'); await p.click('done'); assert.equal(p.location.hash, '#p/b');
  } finally {p.cleanup();}
});

test('autosave sends partial labels and Done clearly reports the last completed window', async () => {
  const reviewed = emptyLabels(); for (const c of Object.values(reviewed.categories)) c.status = 'reviewed';
  const p = await page({labels:{b:toServer(reviewed), c:toServer(reviewed)}});
  try {
    await p.mark(); await p.autosave();
    assert.equal(p.saved.get('a').complete, false); assert.match(p.el('save-state').textContent, /^Saved /);
    await p.watch(); await p.click('done');
    assert.equal(p.location.hash, '#p/a'); assert.equal(p.saved.get('a').complete, true);
    assert.match(p.el('finish-hint').textContent, /All windows.*saved and reviewed/);
    assert.match(p.el('action-error').textContent, /Every window/);
  } finally {p.cleanup();}
});

test('keeping the server version discards only the unsaved browser backup', async () => {
  const labels = emptyLabels(); labels.note = 'server version';
  const backup = emptyLabels(); backup.note = 'old unsaved edit';
  const p = await page({labels:{a:toServer(labels)}, storage:{'label-unsaved:p:a':JSON.stringify({body:toServer(backup)})}});
  try {
    await p.click('keep-saved');
    assert.equal(p.el('note').value, 'server version'); assert.equal(p.saved.get('a').note, 'server version');
    assert.equal(p.storage.has('label-unsaved:p:a'), false); assert.equal(p.el('done').disabled, false);
    assert.equal(p.requests.some(r => r.method === 'PUT'), false);
  } finally {p.cleanup();}
});

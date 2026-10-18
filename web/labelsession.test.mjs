import test from 'node:test';
import assert from 'node:assert/strict';
import {createSaveQueue, playbackSpan} from './labelsession.mjs';

const deferred = () => {let resolve, reject; const promise = new Promise((a, b) => {resolve = a; reject = b;}); return {promise, resolve, reject};};

test('edits during a slow save are sent after it, with immutable snapshots', async () => {
  const first = deferred(), calls = [], states = [];
  const queue = createSaveQueue(async entry => {calls.push(entry); if (calls.length === 1) await first.promise; return {complete:true};}, state => states.push(state));
  const snapshot = {path:'p/a', body:{note:'first'}};
  queue.mark(snapshot); snapshot.body.note = 'mutated outside queue';
  const saved = queue.flush();
  queue.mark({path:'p/a', body:{note:'second'}});
  queue.mark({path:'p/a', body:{note:'newest'}});
  assert.equal(calls.length, 1);
  assert.equal(calls[0].body.note, 'first');
  first.resolve(); await saved;
  assert.deepEqual(calls.map(e => e.body.note), ['first', 'newest']);
  assert.equal(queue.dirty, false);
  assert.equal(states.at(-1), 'saved');
});

test('failed saves reject navigation, retain edits, and can be retried', async () => {
  let fail = true, advances = 0, saved;
  const queue = createSaveQueue(async entry => {if (fail) throw Error('offline'); saved = entry;});
  queue.mark({path:'p/a', body:{note:'keep me'}});
  await assert.rejects(async () => {await queue.flush(); advances++;}, /offline/);
  assert.equal(advances, 0); assert.equal(queue.dirty, true);
  fail = false; await queue.flush(); advances++;
  assert.equal(saved.body.note, 'keep me'); assert.equal(advances, 1); assert.equal(queue.dirty, false);
});

test('multiple flush callers share all writes, including a pending newer edit', async () => {
  const gate = deferred(), calls = [];
  const queue = createSaveQueue(async entry => {calls.push(entry.path); if (calls.length === 1) await gate.promise;});
  queue.mark({path:'p/a'}); const first = queue.flush();
  queue.mark({path:'p/b'}); const second = queue.flush();
  assert.equal(first, second);
  gate.resolve(); await second;
  assert.deepEqual(calls, ['p/a', 'p/b']); assert.equal(queue.dirty, false);
});

test('a failure while editing retries the latest snapshot', async () => {
  const gate = deferred(), calls = [];
  const queue = createSaveQueue(async entry => {calls.push(entry.body); if (calls.length === 1) await gate.promise;});
  queue.mark({body:'old'}); const first = queue.flush();
  queue.mark({body:'latest'}); gate.reject(Error('timeout'));
  await assert.rejects(first, /timeout/);
  await queue.flush(); assert.deepEqual(calls, ['old', 'latest']);
});

test('watch tracking counts stalled callbacks and changed playback speed but not skipped video', () => {
  assert.deepEqual(playbackSpan({time:2, now:0}, 3, {now:1000}), [2, 3]);
  assert.deepEqual(playbackSpan({time:2, now:0}, 3, {now:2000, speed:.5}), [2, 3]);
  assert.equal(playbackSpan({time:2, now:0}, 10, {now:100}), null);
  assert.equal(playbackSpan(null, 3, {now:1000}), null);
  assert.equal(playbackSpan({time:3, now:0}, 2, {now:1000}), null);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyLabels, addInterval, removeInterval, toggleAmbiguous, setEdge, setStatus, finalize, isComplete, hitTest, addWatched, watchedFraction, fromServer, toServer} from './labelmodel.mjs';

const target = [2, 14];

test('intervals are clipped to the target, merged, and selectable', () => {
  const labels = emptyLabels();
  assert.equal(addInterval(labels, 'nudity', 1, 1.5, target), null);
  addInterval(labels, 'nudity', 1, 3, target);
  const selection = addInterval(labels, 'nudity', 4, 2.5, target);
  assert.deepEqual(labels.categories.nudity.intervals, [[2, 4]]);
  assert.deepEqual(selection, {category:'nudity', list:'intervals', index:0});
  addInterval(labels, 'nudity', 13, 20, target);
  assert.deepEqual(labels.categories.nudity.intervals, [[2, 4], [13, 14]]);
  assert.deepEqual(hitTest(labels, 'nudity', 13.5), {category:'nudity', list:'intervals', index:1});
  assert.equal(hitTest(labels, 'nudity', 8), null);
  removeInterval(labels, {category:'nudity', list:'intervals', index:1});
  assert.deepEqual(labels.categories.nudity.intervals, [[2, 4]]);
});

test('ambiguous spans move between lists and edges can be adjusted', () => {
  const labels = emptyLabels();
  let selection = addInterval(labels, 'violence', 5, 6, target);
  selection = toggleAmbiguous(labels, selection);
  assert.deepEqual(labels.categories.violence, {status:null, intervals:[], ambiguous:[[5, 6]]});
  selection = setEdge(labels, selection, 'end', 7.5, target);
  assert.deepEqual(labels.categories.violence.ambiguous, [[5, 7.5]]);
  assert.equal(selection.list, 'ambiguous');
});

test('unsure cannot hide definite spans; done reviews every unset category', () => {
  const labels = emptyLabels();
  addInterval(labels, 'gore', 3, 4, target);
  assert.throws(() => setStatus(labels, 'gore', 'unsure'));
  setStatus(labels, 'kissing', 'unsure');
  assert.equal(isComplete(labels), false);
  assert.deepEqual(finalize(labels), ['nudity', 'sexual_activity', 'violence', 'gore']);
  assert.equal(labels.categories.kissing.status, 'unsure');
  assert.equal(isComplete(labels), true);
  setStatus(labels, 'kissing', 'unsure');
  assert.equal(labels.categories.kissing.status, null);
});

test('watched coverage only counts the target', () => {
  let watched = addWatched([], 0, 5);
  watched = addWatched(watched, 4, 8);
  assert.deepEqual(watched, [[0, 8]]);
  assert.equal(watchedFraction(watched, target), .5);
  assert.equal(watchedFraction(addWatched(watched, 8, 20), target), 1);
});

test('server round trip keeps statuses and spans', () => {
  const labels = emptyLabels();
  addInterval(labels, 'kissing', 6, 7, target, true);
  labels.categories.nudity.status = 'reviewed';
  const restored = fromServer(JSON.parse(JSON.stringify(toServer(labels))));
  assert.deepEqual(restored.categories, labels.categories);
});

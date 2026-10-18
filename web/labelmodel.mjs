// Pure label-editing logic for the labeling page. Times are clip seconds, half-open.
export const categories = ['nudity', 'sexual_activity', 'violence', 'gore', 'kissing'];
export const names = {nudity:'Nudity', sexual_activity:'Sexual activity', violence:'Violence', gore:'Gore', kissing:'Kissing'};

export function emptyLabels() {
  return {categories:Object.fromEntries(categories.map(c => [c, {status:null, intervals:[], ambiguous:[]}])), drafts_shown:false, seconds_spent:0, note:''};
}

export function fromServer(data) {
  const labels = emptyLabels();
  for (const c of categories) {
    const value = data?.categories?.[c];
    if (value) labels.categories[c] = {status:value.status ?? null, intervals:value.intervals.map(i => [...i]), ambiguous:value.ambiguous.map(i => [...i])};
  }
  labels.drafts_shown = !!data?.drafts_shown; labels.seconds_spent = data?.seconds_spent || 0; labels.note = data?.note || '';
  return labels;
}

export function merge(intervals) {
  const result = [];
  for (const [a, b] of [...intervals].filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0] || x[1] - y[1])) {
    const last = result.at(-1);
    if (last && a <= last[1]) last[1] = Math.max(last[1], b); else result.push([a, b]);
  }
  return result;
}

export function clip([a, b], [low, high]) {
  return [Math.max(a, low), Math.min(b, high)];
}

// Adds a span, clipped to the target. Returns the index of the span that now contains it.
export function addInterval(labels, category, a, b, target, ambiguous = false) {
  const [start, end] = clip([Math.min(a, b), Math.max(a, b)], target);
  if (end - start < 1e-3) return null;
  const rule = labels.categories[category], list = ambiguous ? 'ambiguous' : 'intervals';
  if (rule.status === 'unsure' && !ambiguous) rule.status = null;
  rule[list] = merge([...rule[list], [start, end]]);
  return {category, list, index:rule[list].findIndex(([x, y]) => x <= start + 1e-9 && end <= y + 1e-9)};
}

export function removeInterval(labels, selection) {
  labels.categories[selection.category][selection.list].splice(selection.index, 1);
}

export function toggleAmbiguous(labels, selection) {
  const rule = labels.categories[selection.category], span = rule[selection.list][selection.index];
  removeInterval(labels, selection);
  const list = selection.list === 'intervals' ? 'ambiguous' : 'intervals';
  rule[list] = merge([...rule[list], span]);
  return {category:selection.category, list, index:rule[list].findIndex(([x, y]) => x <= span[0] && span[1] <= y)};
}

export function setEdge(labels, selection, edge, time, target) {
  const rule = labels.categories[selection.category], span = [...rule[selection.list][selection.index]];
  span[edge === 'start' ? 0 : 1] = time;
  removeInterval(labels, selection);
  return addInterval(labels, selection.category, span[0], span[1], target, selection.list === 'ambiguous');
}

export function setStatus(labels, category, status) {
  const rule = labels.categories[category];
  if (status === 'unsure' && rule.intervals.length) throw Error('Mark definite spans ambiguous before calling the whole category unsure.');
  rule.status = rule.status === status ? null : status;
}

// Done: every category without a status is reviewed as marked, including "none present".
export function finalize(labels) {
  const changed = categories.filter(c => !labels.categories[c].status);
  for (const c of changed) labels.categories[c].status = 'reviewed';
  return changed;
}

export function isComplete(labels) {
  return categories.every(c => ['reviewed', 'unsure'].includes(labels.categories[c].status));
}

export function hitTest(labels, category, time, tolerance = 0) {
  const rule = labels.categories[category];
  for (const list of ['intervals', 'ambiguous']) {
    const index = rule[list].findIndex(([a, b]) => a - tolerance <= time && time < b + tolerance);
    if (index >= 0) return {category, list, index};
  }
  return null;
}

// Viewed coverage guards against marking a window clean without watching it.
export function addWatched(watched, a, b) {
  return b > a ? merge([...watched, [a, b]]) : watched;
}

export function watchedFraction(watched, [low, high]) {
  const seen = watched.reduce((sum, [a, b]) => sum + Math.max(0, Math.min(b, high) - Math.max(a, low)), 0);
  return high > low ? seen / (high - low) : 1;
}

export function toServer(labels) {
  // Saving from this page means a person reviewed the window, even if it began as an AI label.
  return {categories:labels.categories, drafts_shown:labels.drafts_shown, labeler:'human', seconds_spent:Math.round(labels.seconds_spent), note:labels.note};
}

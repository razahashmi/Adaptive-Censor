// Serialize label writes so a slow older save cannot overwrite a newer edit.
// A rejected write remains pending; callers must successfully flush before leaving.
export function createSaveQueue(write, onState = () => {}) {
  let pending = null, running = null, revision = 0;
  function mark(snapshot) {
    pending = {...structuredClone(snapshot), revision:++revision};
    onState('dirty', pending);
  }
  async function drain() {
    while (pending) {
      const entry = pending;
      onState('saving', entry);
      let result;
      try {result = await write(entry);}
      catch (error) {onState('failed', entry, error); throw error;}
      if (pending.revision === entry.revision) pending = null;
      onState(pending ? 'dirty' : 'saved', entry, result);
    }
  }
  function flush() {
    if (!running && pending) running = drain().finally(() => {running = null;});
    return running || Promise.resolve();
  }
  return {mark, flush, get dirty() {return pending !== null;}};
}

// Only count continuous playback. Seeks reset the previous sample in the UI.
export function playbackSpan(previous, time, {now, speed = 1}) {
  if (!previous || time <= previous.time) return null;
  const elapsed = Math.max(0, now - previous.now) / 1000;
  return time - previous.time <= elapsed * speed + .15 ? [previous.time, time] : null;
}

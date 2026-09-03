// Local storage for SIGNOO's words and recorded samples.
//
// Everything lives in IndexedDB on this device — nothing is uploaded. Recorded
// samples are landmark vectors, not video: the camera footage is never kept.

const DB_NAME = 'signoo';
const STORE   = 'state';
const KEY     = 'main';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const store = db.transaction(STORE, mode).objectStore(STORE);
    const req = fn(store);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

export async function loadState() {
  try {
    const db = await openDb();
    return (await tx(db, 'readonly', s => s.get(KEY))) ?? null;
  } catch {
    return null;   // private browsing, storage disabled — app still runs in memory
  }
}

export async function saveState(state) {
  try {
    const db = await openDb();
    await tx(db, 'readwrite', s => s.put(state, KEY));
    return true;
  } catch {
    return false;
  }
}

/** Serialises the whole dataset to a JSON file the user can back up or move. */
export function exportFile(state) {
  const payload = {
    format: 'signoo-dataset',
    version: 1,
    exportedAt: new Date().toISOString(),
    words: state.words,
    samples: state.samples.map(s => ({
      label: s.label,
      createdAt: s.createdAt,
      vec: Array.from(s.vec),        // Float32Array is not valid JSON
    })),
  };

  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload)], { type: 'application/json' })
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = `signoo-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Reads a previously exported file back into app state. Throws on bad input. */
export async function importFile(file, expectedDim) {
  const data = JSON.parse(await file.text());

  if (data.format !== 'signoo-dataset') throw new Error('هذا ليس ملف بيانات SIGNOO.');
  if (!Array.isArray(data.words) || !Array.isArray(data.samples)) {
    throw new Error('الملف تالف أو غير مكتمل.');
  }

  const samples = data.samples
    .filter(s => Array.isArray(s.vec) && s.vec.length === expectedDim)
    .map(s => ({
      label: s.label,
      createdAt: s.createdAt ?? Date.now(),
      vec: Float32Array.from(s.vec),
    }));

  if (data.samples.length && !samples.length) {
    throw new Error('العيّنات في هذا الملف غير متوافقة مع هذه النسخة من البرنامج.');
  }
  return { words: data.words, samples };
}

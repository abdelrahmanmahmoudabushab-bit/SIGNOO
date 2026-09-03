// SIGNOO recording studio — collects a sign language dataset in the browser.
//
// Produces exactly the layout ml/extract_landmarks.py expects:
//
//     S07/marhaba/S07_marhaba_r03.webm
//     S07/marhaba/S07_marhaba_r03.json
//
// so a recording session drops straight into the training pipeline with no manual
// filing. Clips are held in memory and written out as one ZIP at the end; nothing is
// uploaded anywhere.

import { STARTER_WORDS } from './vocab.js';
import { createZip } from './zip.js';

const $ = sel => document.querySelector(sel);
const arNum = n => n.toLocaleString('ar-EG');

const el = {
  video: $('#video'), camEmpty: $('#camEmpty'), camBadge: $('#camBadge'),
  countdown: $('#countdown'), chkMirror: $('#chkMirror'),
  btnRecord: $('#btnRecord'), btnRetake: $('#btnRetake'), clipCount: $('#clipCount'),
  progressLabel: $('#progressLabel'), progressPct: $('#progressPct'),
  progressFill: $('#progressFill'),
  setup: $('#setup'), session: $('#session'), setupHint: $('#setupHint'),
  curAr: $('#curAr'), curEn: $('#curEn'), curRep: $('#curRep'),
  logTable: $('#logTable'), sizeNote: $('#sizeNote'), toast: $('#toast'),
};

const state = {
  meta: null,          // session metadata, fixed once recording starts
  words: STARTER_WORDS,
  index: 0,            // which sign we are on
  clips: [],           // { sign_id, rep, blob, meta }
  stream: null,
  recording: false,
};

let toastTimer;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2600);
}

const currentWord = () => state.words[state.index];
const repsDone = signId => state.clips.filter(c => c.sign_id === signId).length;

// ── session start ───────────────────────────────────────────────────────────
async function startSession() {
  const signerId = $('#signerId').value.trim();

  if (!$('#chkConsent').checked) {
    el.setupHint.textContent = 'يجب تأكيد الحصول على موافقة المُشارك قبل التسجيل.';
    return;
  }
  if (!/^[A-Za-z]\d{2,}$/.test(signerId)) {
    el.setupHint.textContent = 'استخدم معرّفًا مثل S01 — لا تستخدم الاسم الحقيقي.';
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,          // sign language needs no audio, and it is extra personal data
    });
  } catch (err) {
    el.setupHint.textContent = err.name === 'NotAllowedError'
      ? 'تم رفض إذن الكاميرا. اسمح بالوصول من إعدادات المتصفح.'
      : `تعذّر تشغيل الكاميرا: ${err.message}`;
    return;
  }

  state.meta = {
    signer_id: signerId,
    handedness: $('#handedness').value,
    lighting: $('#lighting').value,
    background: $('#background').value,
    distance_m: Number($('#distance').value),
    device: $('#device').value.trim() || 'unspecified',
    reps: Number($('#reps').value),
    clip_ms: Number($('#clipMs').value),
  };

  el.video.srcObject = state.stream;
  await el.video.play();

  el.camEmpty.classList.add('hidden');
  el.camBadge.textContent = 'الكاميرا تعمل';
  el.camBadge.className = 'cam-badge live';
  el.setup.hidden = true;
  el.session.hidden = false;
  el.btnRecord.disabled = false;

  render();
}

function endSession() {
  if (state.clips.length && !confirm('هل نزّلت ملف ZIP؟ سيتم فقدان التسجيلات غير المحفوظة.')) {
    return;
  }
  state.stream?.getTracks().forEach(t => t.stop());
  state.stream = null;
  el.video.srcObject = null;
  el.session.hidden = true;
  el.setup.hidden = false;
  el.btnRecord.disabled = true;
  el.camEmpty.classList.remove('hidden');
  el.camBadge.textContent = 'الكاميرا متوقفة';
  el.camBadge.className = 'cam-badge';
}

// ── recording ───────────────────────────────────────────────────────────────
function pickMimeType() {
  // Browsers disagree on codecs; take the first the engine admits to supporting.
  const candidates = [
    'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4',
  ];
  return candidates.find(t => MediaRecorder.isTypeSupported(t)) ?? '';
}

async function recordClip() {
  if (state.recording || !state.stream) return;
  state.recording = true;
  el.btnRecord.disabled = true;

  const word = currentWord();
  const rep = repsDone(word.id);

  // Countdown, so the signer can reach a resting position before capture starts.
  el.countdown.hidden = false;
  for (let n = 3; n > 0; n--) {
    el.countdown.textContent = arNum(n);
    await new Promise(r => setTimeout(r, 700));
  }
  el.countdown.hidden = true;

  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(state.stream, mimeType ? { mimeType } : undefined);
  const parts = [];
  recorder.ondataavailable = e => { if (e.data.size) parts.push(e.data); };

  const finished = new Promise(resolve => { recorder.onstop = resolve; });

  el.camBadge.textContent = '● جارٍ التسجيل';
  el.camBadge.className = 'cam-badge rec';
  recorder.start();

  await new Promise(r => setTimeout(r, state.meta.clip_ms));
  recorder.stop();
  await finished;

  el.camBadge.textContent = 'الكاميرا تعمل';
  el.camBadge.className = 'cam-badge live';

  const blob = new Blob(parts, { type: mimeType || 'video/webm' });
  const extension = blob.type.includes('mp4') ? 'mp4' : 'webm';

  state.clips.push({
    sign_id: word.id,
    rep,
    blob,
    extension,
    meta: {
      ...state.meta,
      sign_id: word.id,
      label_ar: word.ar,
      label_en: word.en,
      repetition: rep,
      recorded_at: new Date().toISOString(),
      resolution: [el.video.videoWidth, el.video.videoHeight],
      duration_ms: state.meta.clip_ms,
      validated_by: null,     // filled in by a Deaf reviewer, later
      notes: '',
    },
  });

  state.recording = false;
  el.btnRecord.disabled = false;
  el.btnRetake.disabled = false;

  // Move on automatically once this sign has enough repetitions.
  if (repsDone(word.id) >= state.meta.reps && state.index < state.words.length - 1) {
    state.index++;
    toast(`اكتملت «${word.ar}» — التالية: ${currentWord().ar}`);
  }
  render();
}

function retake() {
  const last = state.clips.pop();
  if (!last) return;
  // Step back to the sign the discarded clip belonged to.
  const idx = state.words.findIndex(w => w.id === last.sign_id);
  if (idx >= 0) state.index = idx;
  el.btnRetake.disabled = state.clips.length === 0;
  toast('تم حذف آخر تسجيل.');
  render();
}

// ── export ──────────────────────────────────────────────────────────────────
async function download() {
  if (!state.clips.length) { toast('لا توجد تسجيلات بعد.'); return; }

  toast('جارٍ تجهيز الملف…');
  const files = [];

  for (const clip of state.clips) {
    const base = `${state.meta.signer_id}/${clip.sign_id}/`
               + `${state.meta.signer_id}_${clip.sign_id}_r${String(clip.rep).padStart(2, '0')}`;
    files.push({
      name: `${base}.${clip.extension}`,
      data: new Uint8Array(await clip.blob.arrayBuffer()),
    });
    files.push({
      name: `${base}.json`,
      data: new TextEncoder().encode(JSON.stringify(clip.meta, null, 2)),
    });
  }

  // A session summary, so the recording conditions survive alongside the clips.
  files.push({
    name: `${state.meta.signer_id}/session.json`,
    data: new TextEncoder().encode(JSON.stringify({
      ...state.meta,
      clips: state.clips.length,
      signs: [...new Set(state.clips.map(c => c.sign_id))],
      exported_at: new Date().toISOString(),
    }, null, 2)),
  });

  const zip = createZip(files);
  const url = URL.createObjectURL(zip);
  const a = document.createElement('a');
  a.href = url;
  a.download = `signoo-${state.meta.signer_id}-${new Date().toISOString().slice(0, 10)}.zip`;
  a.click();
  URL.revokeObjectURL(url);

  toast(`تم تنزيل ${arNum(state.clips.length)} مقطع.`);
}

// ── rendering ───────────────────────────────────────────────────────────────
function render() {
  const word = currentWord();
  const done = repsDone(word.id);
  const target = state.meta?.reps ?? 5;

  el.curAr.textContent = word.ar;
  el.curEn.textContent = word.en;
  el.curRep.textContent = done >= target
    ? `اكتملت ✓ (${arNum(done)} تكرارات)`
    : `التكرار ${arNum(done + 1)} من ${arNum(target)}`;

  const totalTarget = state.words.length * target;
  const pct = Math.min(100, Math.round((state.clips.length / totalTarget) * 100));
  const signsDone = state.words.filter(w => repsDone(w.id) >= target).length;

  el.progressLabel.textContent =
    `${arNum(signsDone)} من ${arNum(state.words.length)} إشارة مكتملة`;
  el.progressPct.textContent = `${arNum(pct)}٪`;
  el.progressFill.style.width = `${pct}%`;
  el.clipCount.textContent = `${arNum(state.clips.length)} مقطع`;

  const bytes = state.clips.reduce((sum, c) => sum + c.blob.size, 0);
  el.sizeNote.textContent = bytes
    ? `${arNum(state.clips.length)} مقطع · ${(bytes / 1e6).toFixed(1)} م.ب`
    : '';

  const tbody = el.logTable.querySelector('tbody');
  tbody.innerHTML = '';
  for (const w of state.words) {
    const n = repsDone(w.id);
    if (!n) continue;
    const tr = document.createElement('tr');
    const a = document.createElement('td'); a.textContent = w.ar;
    const b = document.createElement('td');
    b.textContent = n >= target ? `${arNum(n)} ✓` : arNum(n);
    tr.append(a, b);
    tbody.append(tr);
  }
}

// ── events ──────────────────────────────────────────────────────────────────
$('#btnStart').onclick = startSession;
$('#btnEnd').onclick = endSession;
el.btnRecord.onclick = recordClip;
el.btnRetake.onclick = retake;
$('#btnDownload').onclick = download;

$('#btnNext').onclick = () => {
  state.index = Math.min(state.index + 1, state.words.length - 1);
  render();
};
$('#btnPrev').onclick = () => {
  state.index = Math.max(state.index - 1, 0);
  render();
};
$('#btnSkip').onclick = $('#btnNext').onclick;

el.chkMirror.onchange = () => el.video.classList.toggle('mirrored', el.chkMirror.checked);
el.video.classList.add('mirrored');

// Space to record: the signer is standing away from the keyboard and an operator
// runs the session, so one large, memorable key beats aiming at a button.
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !el.btnRecord.disabled && el.session.hidden === false) {
    e.preventDefault();
    recordClip();
  }
});

// Losing a session's recordings to a stray tab close would waste a participant's time.
window.addEventListener('beforeunload', e => {
  if (state.clips.length) { e.preventDefault(); e.returnValue = ''; }
});

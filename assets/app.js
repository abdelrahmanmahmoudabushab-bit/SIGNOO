// SIGNOO — camera-based Jordanian Sign Language translator.
//
// Flow: camera → MediaPipe hand landmarks → feature vector → nearest-neighbour
// match against signs the user recorded → Arabic + English text.

import { HandTracker, drawHands, clearCanvas } from './vision.js';
import { Recognizer, frameFeature, sequenceFeature, hasHands, FEATURE_DIM } from './recognizer.js';
import { loadState, saveState, exportFile, importFile } from './store.js';
import { STARTER_WORDS } from './vocab.js';

// ── tuning constants ────────────────────────────────────────────────────────
const BUFFER_MS        = 2600;  // rolling landmark history kept for matching
const CLASSIFY_EVERY   = 140;   // how often live matching runs
const MATCH_WINDOW_MS  = 1500;  // slice of history compared against samples
const MIN_SAMPLES_WORD = 3;     // recordings needed before a word is usable
const REC_COUNTDOWN    = 3;

const $  = sel => document.querySelector(sel);
const el = {
  video: $('#video'), overlay: $('#overlay'), camEmpty: $('#camEmpty'),
  camBadge: $('#camBadge'), countdown: $('#countdown'), handsReadout: $('#handsReadout'),
  btnCamera: $('#btnCamera'), chkMirror: $('#chkMirror'), chkSkeleton: $('#chkSkeleton'),
  btnTranslate: $('#btnTranslate'), liveLabel: $('#liveLabel'), liveConf: $('#liveConf'),
  liveFill: $('#liveFill'), outAr: $('#outAr'), outEn: $('#outEn'),
  chkAutoSpeak: $('#chkAutoSpeak'), translateHint: $('#translateHint'),
  selWord: $('#selWord'), selDuration: $('#selDuration'), btnRecord: $('#btnRecord'),
  sampleCount: $('#sampleCount'), newAr: $('#newAr'), newEn: $('#newEn'),
  stats: $('#stats'), wordTable: $('#wordTable'), fileImport: $('#fileImport'),
  toast: $('#toast'),
};

// ── application state ───────────────────────────────────────────────────────
const state = { words: [], samples: [] };

const tracker    = new HandTracker();
const recognizer = new Recognizer();

let stream = null;
let running = false;
let translating = false;
let recording = null;                       // { frames, until } while capturing

const buffer = [];                          // { t, f } rolling landmark history
const sentence = { ar: [], en: [] };

let lastClassifyAt = 0;
let streakLabel = null, streakCount = 0;
let lastEmit = { label: null, at: 0 };

const settings = { confidence: 0.70, stability: 3, cooldown: 1000 };

// ── helpers ─────────────────────────────────────────────────────────────────
const arNum = n => n.toLocaleString('ar-EG');
const wordById = id => state.words.find(w => w.id === id);
const samplesFor = id => state.samples.filter(s => s.label === id).length;

let toastTimer;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2600);
}

function speak(text, lang) {
  if (!text || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang === 'ar' ? 'ar-SA' : 'en-US';
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

/** Words with enough recordings to be recognised. */
function trainableWords() {
  return state.words.filter(w => samplesFor(w.id) >= MIN_SAMPLES_WORD);
}

function refit() {
  const usable = new Set(trainableWords().map(w => w.id));
  recognizer.fit(state.samples.filter(s => usable.has(s.label)));

  const ready = usable.size >= 2;
  el.btnTranslate.disabled = !ready || !running;

  el.translateHint.textContent = ready
    ? `جاهز للترجمة · ${arNum(usable.size)} كلمة مُدرَّبة.`
    : `تحتاج إلى كلمتين على الأقل، لكل واحدة ${arNum(MIN_SAMPLES_WORD)} تسجيلات، `
      + `من تبويب «تعليم الإشارات». (${arNum(usable.size)} جاهزة الآن)`;

  if (!ready && translating) stopTranslating();
}

const persist = () => saveState({ words: state.words, samples: state.samples });

// ── camera ──────────────────────────────────────────────────────────────────
async function startCamera() {
  el.btnCamera.disabled = true;
  el.btnCamera.textContent = 'جارٍ التحميل…';
  try {
    await tracker.load();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
    });
    el.video.srcObject = stream;
    await el.video.play();

    running = true;
    el.camEmpty.classList.add('hidden');
    el.camBadge.textContent = 'الكاميرا تعمل';
    el.camBadge.className = 'cam-badge live';
    el.btnCamera.textContent = 'إيقاف الكاميرا';
    el.btnRecord.disabled = false;
    refit();
    requestAnimationFrame(loop);
  } catch (err) {
    // Distinguish "no internet for the tracker" from "camera refused" — they look
    // the same to the user but need completely different fixes.
    const messages = {
      LIB_UNAVAILABLE:   'تعذّر تحميل نظام تتبّع اليد. تحقّق من الاتصال بالإنترنت، '
                       + 'أو شغّل scripts/fetch-vendor.sh للعمل بدون إنترنت.',
      MODEL_UNAVAILABLE: 'تعذّر تحميل ملف النموذج. تحقّق من الاتصال بالإنترنت، '
                       + 'أو شغّل scripts/fetch-vendor.sh للعمل بدون إنترنت.',
      NotAllowedError:   'تم رفض إذن الكاميرا. اسمح بالوصول من إعدادات المتصفح.',
      NotFoundError:     'لم يتم العثور على كاميرا متصلة بالجهاز.',
      NotReadableError:  'الكاميرا مستخدمة من قبل تطبيق آخر. أغلقه وحاول مجددًا.',
    };
    toast(messages[err.message] ?? messages[err.name] ?? `تعذّر التشغيل: ${err.message}`);
    el.btnCamera.textContent = 'تشغيل الكاميرا';
  } finally {
    el.btnCamera.disabled = false;
  }
}

function stopCamera() {
  running = false;
  stopTranslating();
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  el.video.srcObject = null;
  clearCanvas(el.overlay);
  buffer.length = 0;

  el.camEmpty.classList.remove('hidden');
  el.camBadge.textContent = 'الكاميرا متوقفة';
  el.camBadge.className = 'cam-badge';
  el.btnCamera.textContent = 'تشغيل الكاميرا';
  el.btnRecord.disabled = true;
  el.btnTranslate.disabled = true;
  el.handsReadout.textContent = 'لا توجد يد';
}

// ── frame loop ──────────────────────────────────────────────────────────────
function loop() {
  if (!running) return;

  const now = performance.now();
  const hands = tracker.detect(el.video, now);

  if (el.chkSkeleton.checked) drawHands(el.overlay, el.video, hands);
  else clearCanvas(el.overlay);

  el.handsReadout.textContent =
    hands.length === 0 ? 'لا توجد يد'
    : hands.length === 1 ? 'يد واحدة' : 'يدان';

  const feature = frameFeature(hands);

  buffer.push({ t: now, f: feature });
  while (buffer.length && now - buffer[0].t > BUFFER_MS) buffer.shift();

  if (recording) {
    recording.frames.push(feature);
    if (now >= recording.until) finishRecording();
  } else if (translating && now - lastClassifyAt >= CLASSIFY_EVERY) {
    lastClassifyAt = now;
    classifyLive(now);
  }

  requestAnimationFrame(loop);
}

// ── live translation ────────────────────────────────────────────────────────
function classifyLive(now) {
  const frames = buffer.filter(b => now - b.t <= MATCH_WINDOW_MS).map(b => b.f);

  if (!hasHands(frames)) {           // hands away — reset and stay quiet
    streakLabel = null; streakCount = 0;
    showLive(null, 0);
    return;
  }

  const vec = sequenceFeature(frames);
  const match = vec && recognizer.classify(vec);

  if (!match || match.confidence < settings.confidence) {
    streakLabel = null; streakCount = 0;
    showLive(match ? wordById(match.label) : null, match?.confidence ?? 0);
    return;
  }

  streakCount = match.label === streakLabel ? streakCount + 1 : 1;
  streakLabel = match.label;
  showLive(wordById(match.label), match.confidence);

  // Emit once the same word has held steady, and it is not an immediate repeat.
  const isRepeat = match.label === lastEmit.label && now - lastEmit.at < settings.cooldown;
  if (streakCount >= settings.stability && !isRepeat) {
    emit(match.label);
    lastEmit = { label: match.label, at: now };
    streakCount = 0;
  }
}

function showLive(word, confidence) {
  el.liveLabel.textContent = word ? word.ar : '—';
  el.liveConf.textContent  = `${Math.round(confidence * 100)}%`;
  el.liveFill.style.width  = `${Math.round(confidence * 100)}%`;
}

function emit(label) {
  const word = wordById(label);
  if (!word) return;

  sentence.ar.push(word.ar);
  sentence.en.push(word.en);
  renderSentence();

  if (el.chkAutoSpeak.checked) speak(word.ar, 'ar');
}

function renderSentence() {
  el.outAr.textContent = sentence.ar.join(' ') || '…';
  el.outEn.textContent = sentence.en.join(' ') || '…';
}

function startTranslating() {
  translating = true;
  streakLabel = null; streakCount = 0;
  lastEmit = { label: null, at: 0 };
  el.btnTranslate.textContent = 'إيقاف الترجمة';
  el.btnTranslate.classList.replace('btn-primary', 'btn-ghost');
}

function stopTranslating() {
  translating = false;
  showLive(null, 0);
  el.btnTranslate.textContent = 'بدء الترجمة';
  el.btnTranslate.classList.replace('btn-ghost', 'btn-primary');
}

// ── recording samples ───────────────────────────────────────────────────────
async function recordSample() {
  const wordId = el.selWord.value;
  if (!wordId || recording) return;

  const wasTranslating = translating;
  stopTranslating();
  el.btnRecord.disabled = true;

  // Countdown gives the signer time to get into position.
  el.countdown.hidden = false;
  for (let n = REC_COUNTDOWN; n > 0; n--) {
    el.countdown.textContent = arNum(n);
    await new Promise(r => setTimeout(r, 700));
  }
  el.countdown.hidden = true;

  el.camBadge.textContent = 'جارٍ التسجيل…';
  el.camBadge.className = 'cam-badge rec';

  recording = {
    wordId,
    frames: [],
    until: performance.now() + Number(el.selDuration.value),
    resumeTranslating: wasTranslating,
  };
}

async function finishRecording() {
  const { frames, wordId, resumeTranslating } = recording;
  recording = null;

  el.camBadge.textContent = 'الكاميرا تعمل';
  el.camBadge.className = 'cam-badge live';
  el.btnRecord.disabled = !running;

  if (!hasHands(frames)) {
    toast('لم تُرصد أي يد أثناء التسجيل — حاول مرة أخرى أمام الكاميرا.');
    return;
  }

  const vec = sequenceFeature(frames);
  if (!vec) { toast('فشل التسجيل، حاول مرة أخرى.'); return; }

  state.samples.push({ label: wordId, vec, createdAt: Date.now() });
  await persist();

  const word = wordById(wordId);
  toast(`تم حفظ عيّنة لكلمة «${word.ar}» (${arNum(samplesFor(wordId))})`);
  renderTrain();
  renderData();
  refit();

  if (resumeTranslating && !el.btnTranslate.disabled) startTranslating();
}

// ── rendering ───────────────────────────────────────────────────────────────
function renderTrain() {
  const selected = el.selWord.value;
  el.selWord.innerHTML = '';

  for (const w of state.words) {
    const n = samplesFor(w.id);
    const opt = document.createElement('option');
    opt.value = w.id;
    opt.textContent = `${w.ar} — ${w.en}${n ? `  (${arNum(n)})` : ''}`;
    el.selWord.append(opt);
  }
  if (selected && wordById(selected)) el.selWord.value = selected;

  const n = samplesFor(el.selWord.value);
  el.sampleCount.textContent = n >= MIN_SAMPLES_WORD
    ? `${arNum(n)} عيّنات — جاهزة ✓`
    : `${arNum(n)} من ${arNum(MIN_SAMPLES_WORD)} عيّنات مطلوبة`;
}

function renderData() {
  const trained = trainableWords().length;
  el.stats.innerHTML = `
    <div><b>${arNum(state.words.length)}</b>كلمة</div>
    <div><b>${arNum(trained)}</b>كلمة مُدرَّبة</div>
    <div><b>${arNum(state.samples.length)}</b>عيّنة مسجّلة</div>`;

  const tbody = el.wordTable.querySelector('tbody');
  tbody.innerHTML = '';

  for (const w of [...state.words].sort((a, b) => samplesFor(b.id) - samplesFor(a.id))) {
    const n = samplesFor(w.id);
    const tr = document.createElement('tr');

    const ar = document.createElement('td'); ar.textContent = w.ar;
    const en = document.createElement('td'); en.dir = 'ltr'; en.textContent = w.en;
    const ct = document.createElement('td');
    ct.textContent = n >= MIN_SAMPLES_WORD ? `${arNum(n)} ✓` : arNum(n);

    const actions = document.createElement('td');
    if (n) {
      const clear = document.createElement('button');
      clear.className = 'row-del';
      clear.textContent = 'مسح العيّنات';
      clear.onclick = async () => {
        state.samples = state.samples.filter(s => s.label !== w.id);
        await persist(); renderTrain(); renderData(); refit();
      };
      actions.append(clear);
    }

    tr.append(ar, en, ct, actions);
    tbody.append(tr);
  }
}

// ── events ──────────────────────────────────────────────────────────────────
function wireEvents() {
  el.btnCamera.onclick    = () => (running ? stopCamera() : startCamera());
  el.btnTranslate.onclick = () => (translating ? stopTranslating() : startTranslating());
  el.btnRecord.onclick    = recordSample;
  el.selWord.onchange     = renderTrain;

  const applyMirror = () => {
    el.video.classList.toggle('mirrored', el.chkMirror.checked);
    el.overlay.classList.toggle('mirrored', el.chkMirror.checked);
  };
  el.chkMirror.onchange = applyMirror;
  applyMirror();

  // tabs
  for (const tab of document.querySelectorAll('.tab')) {
    tab.onclick = () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('is-active', t === tab));
      for (const view of ['translate', 'train', 'data']) {
        $(`#view-${view}`).hidden = view !== tab.dataset.view;
      }
    };
  }

  // sentence actions
  $('#btnClear').onclick     = () => { sentence.ar = []; sentence.en = []; renderSentence(); };
  $('#btnBackspace').onclick = () => { sentence.ar.pop(); sentence.en.pop(); renderSentence(); };
  $('#btnCopy').onclick = async () => {
    await navigator.clipboard.writeText(`${sentence.ar.join(' ')}\n${sentence.en.join(' ')}`);
    toast('تم النسخ.');
  };
  for (const btn of document.querySelectorAll('[data-speak]')) {
    const lang = btn.dataset.speak;
    btn.onclick = () => speak(sentence[lang].join(' '), lang);
  }

  // sensitivity sliders
  const bind = (id, apply) => {
    const input = $(id);
    input.oninput = () => apply(Number(input.value));
    apply(Number(input.value));
  };
  bind('#rngConf', v => { settings.confidence = v / 100; $('#valConf').textContent = `${v}%`; });
  bind('#rngStable', v => { settings.stability = v; $('#valStable').textContent = arNum(v); });
  bind('#rngCooldown', v => {
    settings.cooldown = v * 100;
    $('#valCooldown').textContent = (v / 10).toFixed(1);
  });

  // new word
  $('#btnAddWord').onclick = async () => {
    const ar = el.newAr.value.trim(), en = el.newEn.value.trim();
    if (!ar || !en) { toast('اكتب الكلمة بالعربية والإنجليزية.'); return; }

    const id = `w_${Date.now().toString(36)}`;
    state.words.push({ id, ar, en });
    await persist();
    el.newAr.value = el.newEn.value = '';
    renderTrain(); renderData();
    el.selWord.value = id;
    renderTrain();
    toast(`تمت إضافة «${ar}».`);
  };

  // data management
  $('#btnExport').onclick = () => exportFile(state);
  $('#btnImport').onclick = () => el.fileImport.click();
  el.fileImport.onchange = async () => {
    const file = el.fileImport.files[0];
    if (!file) return;
    try {
      const data = await importFile(file, FEATURE_DIM);
      state.words = data.words;
      state.samples = data.samples;
      await persist();
      renderTrain(); renderData(); refit();
      toast(`تم الاستيراد: ${arNum(data.words.length)} كلمة، ${arNum(data.samples.length)} عيّنة.`);
    } catch (err) {
      toast(err.message);
    }
    el.fileImport.value = '';
  };
  $('#btnReset').onclick = async () => {
    if (!confirm('سيتم حذف كل الكلمات والعيّنات المسجّلة. هل أنت متأكد؟')) return;
    state.words = [...STARTER_WORDS];
    state.samples = [];
    await persist();
    renderTrain(); renderData(); refit();
    toast('تمت إعادة التعيين.');
  };
}

// ── boot ────────────────────────────────────────────────────────────────────
async function init() {
  const saved = await loadState();
  state.words   = saved?.words?.length ? saved.words : [...STARTER_WORDS];
  state.samples = (saved?.samples ?? []).map(s => ({ ...s, vec: Float32Array.from(s.vec) }));

  wireEvents();
  renderTrain();
  renderData();
  renderSentence();
  refit();
}

init();

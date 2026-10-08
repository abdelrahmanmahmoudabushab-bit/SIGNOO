// SIGNOO kiosk frontend — the visitor flow (docs/REQUIREMENTS.md FR-K1-K14).
//
// Camera capture and rendering stay in the browser; every processed frame is sent
// to the kiosk/ FastAPI backend over a WebSocket as a base64 JPEG, and perception
// (MediaPipe Holistic), segmentation, recognition, assembly, dialogue and sign
// playback all happen server-side — see kiosk/main.py's module docstring for the
// full wire-protocol reference. This file only owns screens, the camera, drawing,
// and turning WebSocket messages into what the visitor sees.

import { drawLandmarks, clearCanvas } from './vision.js';

const $ = sel => document.querySelector(sel);
const views = {
  hello: $('#view-hello'), language: $('#view-language'), consent: $('#view-consent'),
  capture: $('#view-capture'), answer: $('#view-answer'), thankyou: $('#view-thankyou'),
};

const el = {
  video: $('#video'), overlay: $('#overlay'), camEmpty: $('#camEmpty'),
  camState: $('#camState'), recIndicator: $('#recIndicator'),
  framingBanner: $('#framingBanner'),
  saidText: $('#saidText'),
  candidatesBox: $('#candidatesBox'), candidateGrid: $('#candidateGrid'),
  retryBanner: $('#retryBanner'),
  outAr: $('#outAr'), outEn: $('#outEn'),
  barLabelPrimary: $('#barLabelPrimary'), answerLabelPrimary: $('#answerLabelPrimary'),
  answerAr: $('#answerAr'), answerEn: $('#answerEn'), answerSource: $('#answerSource'),
  escalationNote: $('#escalationNote'), playbackRow: $('#playbackRow'),
  idleOverlay: $('#idleOverlay'), idleCountdown: $('#idleCountdown'),
  toast: $('#toast'), presenceVideo: $('#presenceVideo'),
};

// FR-K13: idle countdown, inactivity, and a "one touch" full reset all funnel
// through showView('hello') + resetSessionUI(), so the next visitor never sees
// anything left over (NFR-S3).
function showView(name) {
  for (const [key, node] of Object.entries(views)) node.hidden = key !== name;
}

// Arabic-Indic digits wherever a number sits in Arabic text (deaf-kiosk-ux:
// "use Arabic-Indic digits where the surrounding Arabic text uses them").
const arNum = n => Number(n).toLocaleString('ar-EG');

// NFR-X1: the system's state is always visible, and "watching" must be
// distinguishable from "thinking" at a glance — if it isn't, people re-sign
// mid-sign and corrupt the very clip being captured.
//
// `thinking` is currently only set around an explicit ask (question -> answer).
// A true per-utterance "recognising" state needs the backend to emit a signal at
// the B3 segmentation boundary before B4 returns; that is a backend-phase item,
// so it is left honestly absent rather than faked with a timer here.
function setCamState(state) {
  const text = {
    off: 'الكاميرا متوقفة · <span dir="ltr">camera off</span>',
    watching: 'أراك · <span dir="ltr">I can see you</span>',
    thinking: 'أفكّر… · <span dir="ltr">thinking…</span>',
  }[state];
  el.camState.className = `cam-state${state === 'off' ? '' : ` is-${state}`}`;
  el.camState.innerHTML = text;
}

// A translation bar shows either real text or a resting placeholder — never an
// empty box, which reads as "broken" rather than "nothing signed yet".
function setBar(node, text) {
  node.textContent = text || '…';
  node.classList.toggle('is-empty', !text);
}

let toastTimer;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2600);
}

// ── vocabulary (single source of truth: core/vocab.json, per docs/API_CONTRACTS.md §1) ──
let vocab = new Map();
async function loadVocab() {
  try {
    const res = await fetch('/core/vocab.json');
    const entries = await res.json();
    vocab = new Map(entries.map(e => [e.id, e]));
  } catch {
    vocab = new Map();   // candidate buttons fall back to showing the raw gloss id
  }
}

// ── language metadata & localisation ──────────────────────────────────────────
// The chosen language's own name/script drives the consent screen and the
// "your language" caption bar. Only `liu` has real translated consent copy
// today (the rest are disabled cards per docs/ROADMAP.md's "one language
// end-to-end" rule — signoo-language-scaling skill); every other code falls
// back to the existing bilingual AR/EN copy rather than showing nothing,
// flagged here so it isn't mistaken for a finished translation.
const CONSENT_TEXT = {
  liu: {
    whatNative: '<b>ماذا نسجّل؟</b> نقاط تتبّع اليدين والجسم فقط — لا يُحفظ الفيديو ولا صورة وجهك، ولا يغادر شيء هذا الجهاز.',
    processingNative: 'معالجة إشارتي لعرض الترجمة والإجابة',
    retentionNative: 'حفظ نقاط هذه الجلسة لتحسين النظام',
  },
  // TODO(signoo-language-scaling): real per-language consent translations,
  // once that language's card is live. Falls back to the liu/AR copy until then.
};

function nativeLabelFor(langCode) {
  const card = document.querySelector(`.lang-card[data-lang="${langCode}"] .lang-name`);
  return card ? card.textContent : 'العربية';
}

// FR-K2 extended: once a second language ships, its consent and caption-bar
// labels should read in its own language, not always Arabic (deaf-kiosk-ux:
// "written Arabic is not automatically accessible" generalises to every
// language here, not just English speakers).
function applyLanguageLocale(langCode) {
  const native = nativeLabelFor(langCode);
  el.barLabelPrimary.textContent = native;
  el.answerLabelPrimary.textContent = native;

  const copy = CONSENT_TEXT[langCode] || CONSENT_TEXT.liu;
  $('#consentWhatNative').innerHTML = copy.whatNative;
  $('#consentProcessingNative').textContent = copy.processingNative;
  $('#consentRetentionNative').textContent = copy.retentionNative;
}

// ── WebSocket ────────────────────────────────────────────────────────────────
let ws = null;
let wsReady = false;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/session`);
  ws.onopen = () => { wsReady = true; };
  ws.onclose = () => { wsReady = false; toast('انقطع الاتصال بالخادم — أعد تحميل الصفحة. / Connection lost — reload the page.'); };
  ws.onerror = () => { wsReady = false; };
  ws.onmessage = ev => handleMessage(JSON.parse(ev.data));
}

function send(message) {
  if (wsReady) ws.send(JSON.stringify(message));
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'state': break;   // server-side session state; UI already drives the flow
    case 'landmarks':
      if (el.overlay.dataset.skeletonOn === '1') drawLandmarks(el.overlay, el.video, msg);
      break;
    case 'framing':
      el.framingBanner.hidden = msg.ok !== false;
      if (!el.framingBanner.hidden) {
        el.framingBanner.textContent = msg.reason === 'no_body'
          ? 'الجسم غير مرئي — تراجع قليلًا · Body not detected — step back a little'
          : 'لا تظهر اليدان بوضوح · Hands not clearly visible';
      }
      break;
    case 'hypothesis': onHypothesis(msg); break;
    case 'rendered':
      setBar(el.outAr, msg.ar);
      setBar(el.outEn, msg.en);
      break;
    case 'tts_done': break;   // text was already shown before this ever arrives
    case 'answer': onAnswer(msg); break;
    case 'playback': onPlayback(msg.words); break;
    case 'idle_warning': onIdleWarning(msg.seconds_left); break;
    case 'session_reset': onSessionReset(); break;
    case 'error': toast(msg.message || msg.code); break;
    default: break;
  }
}

function onHypothesis(msg) {
  el.candidatesBox.hidden = true;
  el.retryBanner.hidden = true;

  if (msg.status === 'committed') {
    // Committed words land in the two translation bars via the 'rendered'
    // message; nothing extra to show here.
  } else if (msg.status === 'candidates') {
    // FR-K8: below-threshold candidates are large touch targets, never a silent guess.
    el.candidatesBox.hidden = false;
    el.candidateGrid.innerHTML = '';
    for (const [glossId] of msg.alternatives) {
      const entry = vocab.get(glossId);
      const btn = document.createElement('button');
      btn.className = 'candidate-btn';
      btn.textContent = entry ? `${entry.ar} / ${entry.en}` : glossId;
      btn.onclick = () => {
        send({ type: 'select_candidate', gloss_id: glossId });
        el.candidatesBox.hidden = true;
      };
      el.candidateGrid.append(btn);
    }
  } else if (msg.status === 'rejected') {
    el.retryBanner.hidden = false;
  }
}

function onAnswer(msg) {
  setCamState(stream ? 'watching' : 'off');   // question answered, back to watching
  // Show the visitor what the machine thought they signed, in sign-language
  // order — they need to see whether it understood them, not just its reply.
  el.saidText.textContent = el.outAr.textContent || '…';
  showView('answer');
  el.answerAr.textContent = msg.ar;
  el.answerEn.textContent = msg.en;
  el.escalationNote.hidden = !msg.refused;
  // NFR-P3b: the cloud/local-fallback switch must be seamless to the visitor —
  // no retry button, no error — but Answer.source is still shown, quietly, per
  // core/contracts.py's own comment on why the flag exists.
  el.answerSource.className = `answer-source${msg.source === 'local_fallback' ? ' is-local' : ''}`;
  el.answerSource.textContent = msg.source === 'local_fallback'
    ? 'إجابة محلية · Local answer'
    : 'إجابة سحابية · Cloud answer';
}

let playbackTimer = null;

function onPlayback(words) {
  // Without this, asking a second question before the first answer's
  // playback finished cycling left the old interval running indefinitely on
  // now-detached nodes — harmless to look at (nothing visible references
  // them), but a real leak, and it would eventually run on two independent
  // timers if this happened more than once.
  clearInterval(playbackTimer);
  el.playbackRow.innerHTML = '';
  const nodes = words.map(w => {
    const box = document.createElement('div');
    box.className = 'playback-word';
    box.textContent = w.ar;
    box.title = w.has_clip ? w.en : `${w.en} — لم تُسجَّل بعد`;
    el.playbackRow.append(box);
    return box;
  });
  // B8: "highlights each word as its clip plays" — with no real clips yet, the
  // highlight still sequences at a fixed cadence so the mechanism is visible and
  // testable, per docs/SYSTEM_DESIGN.md B8 / models/README.md's stated gap.
  let i = 0;
  playbackTimer = setInterval(() => {
    nodes.forEach(n => n.classList.remove('is-current'));
    if (i >= nodes.length) { clearInterval(playbackTimer); return; }
    nodes[i].classList.add('is-current');
    i += 1;
  }, 700);
}

function onIdleWarning(secondsLeft) {
  el.idleOverlay.hidden = false;
  el.idleCountdown.textContent = arNum(secondsLeft);
}

// NFR-S3: someone who walks away has consented to nothing — the next visitor
// must never see any trace of the previous conversation.
function onSessionReset() {
  el.idleOverlay.hidden = true;
  stopCamera();
  setBar(el.outAr, '');
  setBar(el.outEn, '');
  el.saidText.textContent = '…';
  el.candidatesBox.hidden = true;
  el.candidateGrid.innerHTML = '';
  el.retryBanner.hidden = true;
  el.framingBanner.hidden = true;
  el.recIndicator.hidden = true;
  el.answerAr.textContent = el.answerEn.textContent = '…';
  clearInterval(playbackTimer);
  el.playbackRow.innerHTML = '';
  el.escalationNote.hidden = true;
  $('#chkConsentProcessing').checked = false;
  $('#chkConsentRetention').checked = false;
  $('#btnConsentContinue').disabled = true;
  selectedLang = null;

  // Say goodbye before resetting to idle — covers both the explicit "Finish"
  // button and an idle-timeout-triggered reset; either way the next visitor
  // must land on a clean hello screen, never mid-thank-you (NFR-S3).
  showView('thankyou');
  clearTimeout(thankyouTimer);
  thankyouTimer = setTimeout(enterHello, THANKYOU_DISPLAY_MS);
}

// ── camera (FR-K4) ───────────────────────────────────────────────────────────
let stream = null;
let frameTimer = null;
const FRAME_INTERVAL_MS = 130;   // ~7.5 fps over the wire — plenty for pause-based
                                  // segmentation (SYSTEM_DESIGN.md B3), far below
                                  // the 60fps capture budget that matters on-device.
const sendCanvas = document.createElement('canvas');

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
    });
    el.video.srcObject = stream;
    await el.video.play();
    el.overlay.dataset.skeletonOn = '1';
    el.camEmpty.classList.add('hidden');
    setCamState('watching');

    frameTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
  } catch (err) {
    const messages = {
      NotAllowedError: 'تم رفض إذن الكاميرا. اسمح بالوصول من إعدادات المتصفح. / Camera permission denied.',
      NotFoundError: 'لم يتم العثور على كاميرا. / No camera found.',
      NotReadableError: 'الكاميرا مستخدمة من تطبيق آخر. / Camera is in use by another app.',
    };
    el.camEmpty.querySelector('p').textContent =
      messages[err.name] ?? `تعذّر تشغيل الكاميرا: ${err.message}`;
    toast('تعذّر تشغيل الكاميرا — يمكنك المتابعة بدون كاميرا والمحاولة لاحقًا.');
  }
}

// Perception only needs enough resolution to find hands/pose, not the full
// camera frame — encoding at the camera's native 960x720 on every tick was
// real, measured CPU cost on modest hardware (jetson-realtime: "profile
// before optimising", but canvas-encode cost scales directly with pixel
// count, so this is the obvious first lever). Downscaling to a 480px side
// cuts the pixels to encode ~4x, which cuts JS-thread encode time, wire
// payload size, and the server's decode+MediaPipe cost together.
const SEND_MAX_SIDE = 480;

function sendFrame() {
  if (!stream || !wsReady || el.video.readyState < 2) return;
  const vw = el.video.videoWidth, vh = el.video.videoHeight;
  if (!vw || !vh) return;
  const scale = Math.min(1, SEND_MAX_SIDE / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  sendCanvas.width = w; sendCanvas.height = h;
  sendCanvas.getContext('2d').drawImage(el.video, 0, 0, w, h);
  const dataUrl = sendCanvas.toDataURL('image/jpeg', 0.6);
  send({ type: 'frame', ts_ms: Math.round(performance.now()), data: dataUrl });
}

function stopCamera() {
  clearInterval(frameTimer);
  frameTimer = null;
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  el.video.srcObject = null;
  clearCanvas(el.overlay);
  el.camEmpty.classList.remove('hidden');
  el.camEmpty.querySelector('p').textContent = 'جارٍ تشغيل الكاميرا…';
  el.recIndicator.hidden = true;
  setCamState('off');
}

// ── presence auto-detect (hello screen) ─────────────────────────────────────
// Boring choice first (jetson-realtime: "prefer the boring choice"): a simple
// frame-difference motion heuristic on a low-res sample, not a face detector —
// works offline, no extra model, no flaky experimental browser API. Good
// enough to tell "someone is standing here" from "no one is"; swap for a
// dedicated presence sensor later per docs/SYSTEM_DESIGN.md's open B1a
// question. Tapping the screen remains a working fallback at all times
// (deaf-kiosk-ux: never make an automatic feature the only way in) — wired in
// wireEvents() below.
let presenceStream = null;
let presenceTimer = null;
let presenceHits = 0;
let presencePrevFrame = null;
let presenceWatchActive = false;
const PRESENCE_SAMPLE_MS = 400;
const PRESENCE_HITS_NEEDED = 3;        // ~1.2s of sustained motion, not a passer-by
const PRESENCE_MOTION_THRESHOLD = 18;  // mean abs pixel delta, 0-255 scale
const presenceCanvas = document.createElement('canvas');

// A tap can land while getUserMedia()/play() is still pending (both are async).
// Without this flag, stopPresenceWatch() runs on a stream that doesn't exist
// yet, and the in-flight promise resolves afterwards and starts the camera
// anyway — a live camera the visitor has already walked away from, on a
// screen that no longer wants it. presenceWatchActive is checked after each
// await so a stop during either one aborts cleanly instead of racing it.
async function startPresenceWatch(onDetected) {
  presenceWatchActive = true;
  let localStream;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 160 }, height: { ideal: 120 }, facingMode: 'user' },
    });
  } catch {
    return;   // no camera / permission denied — tap-to-start is still available
  }
  if (!presenceWatchActive) { localStream.getTracks().forEach(t => t.stop()); return; }

  presenceStream = localStream;
  el.presenceVideo.srcObject = presenceStream;
  await el.presenceVideo.play();
  if (!presenceWatchActive) { stopPresenceWatch(); return; }

  presenceHits = 0;
  presencePrevFrame = null;
  presenceTimer = setInterval(() => samplePresence(onDetected), PRESENCE_SAMPLE_MS);
}

function samplePresence(onDetected) {
  const v = el.presenceVideo;
  if (v.readyState < 2) return;
  presenceCanvas.width = 32; presenceCanvas.height = 24;
  const ctx = presenceCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(v, 0, 0, 32, 24);
  const frame = ctx.getImageData(0, 0, 32, 24).data;

  if (presencePrevFrame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i += 4) sum += Math.abs(frame[i] - presencePrevFrame[i]);
    const mean = sum / (frame.length / 4);
    presenceHits = mean > PRESENCE_MOTION_THRESHOLD ? presenceHits + 1 : 0;
    if (presenceHits >= PRESENCE_HITS_NEEDED) {
      stopPresenceWatch();
      onDetected();
      return;
    }
  }
  presencePrevFrame = frame;
}

function stopPresenceWatch() {
  presenceWatchActive = false;
  clearInterval(presenceTimer);
  presenceTimer = null;
  presenceStream?.getTracks().forEach(t => t.stop());
  presenceStream = null;
  el.presenceVideo.srcObject = null;
  presencePrevFrame = null;
}

function enterHello() {
  showView('hello');
  startPresenceWatch(enterLanguage);
}

function enterLanguage() {
  stopPresenceWatch();
  showView('language');
}

// ── flow wiring ──────────────────────────────────────────────────────────────
let selectedLang = null;
let thankyouTimer = null;
const THANKYOU_DISPLAY_MS = 5000;

function wireEvents() {
  views.hello.onclick = enterLanguage;   // explicit tap, always works

  for (const card of document.querySelectorAll('.lang-card:not([disabled])')) {
    card.onclick = () => {
      selectedLang = card.dataset.lang;
      applyLanguageLocale(selectedLang);
      send({ type: 'language', lang: selectedLang });
      showView('consent');
    };
  }

  const updateContinueEnabled = () => {
    $('#btnConsentContinue').disabled = !$('#chkConsentProcessing').checked;
  };
  $('#chkConsentProcessing').onchange = updateContinueEnabled;
  updateContinueEnabled();

  $('#btnConsentContinue').onclick = () => {
    const retention = $('#chkConsentRetention').checked;
    send({
      type: 'consent',
      processing: $('#chkConsentProcessing').checked,
      retention,
    });
    // "Show an unmistakable indicator whenever recording is happening"
    // (deaf-kiosk-ux) — and only then. Declining retention leaves it absent.
    el.recIndicator.hidden = !retention;
    showView('capture');
    startCamera();
  };

  $('#btnClear').onclick = () => send({ type: 'control', action: 'clear' });
  $('#btnBackspace').onclick = () => send({ type: 'control', action: 'backspace' });
  $('#btnAsk').onclick = () => {
    setCamState('thinking');
    send({ type: 'ask' });
  };

  $('#btnSpeakAr').onclick = () => send({ type: 'speak', lang: 'ar' });
  $('#btnSpeakEn').onclick = () => send({ type: 'speak', lang: 'en' });

  $('#btnBackToCapture').onclick = () => showView('capture');

  const endSession = () => send({ type: 'control', action: 'end_session' });
  $('#btnEndSession').onclick = endSession;
  $('#btnEndSessionFromAnswer').onclick = endSession;

  $('#btnIdleExtend').onclick = () => {
    el.idleOverlay.hidden = true;
    send({ type: 'control', action: 'extend' });
  };
}

async function init() {
  await loadVocab();
  wireEvents();
  connect();
  enterHello();

  // Keep-alive so a mid-flow WebSocket idle timeout on the server side never
  // fires just because the visitor is reading, not signing.
  setInterval(() => send({ type: 'ping' }), 15000);
}

init();

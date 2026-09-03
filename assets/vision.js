// Hand tracking for SIGNOO.
//
// Wraps MediaPipe Hand Landmarker: reads frames from a <video>, returns 21 3-D
// landmarks per hand (up to two hands), and draws the skeleton on a <canvas>.
// Everything runs locally in the browser — frames never leave the device.

// The tracker is loaded from a local copy when one exists, and from the CDN
// otherwise. Run `scripts/fetch-vendor.sh` to create the local copy and make the
// whole app work offline.
const CDN_LIB   = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const CDN_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const LOCAL_LIB   = new URL('../vendor/tasks-vision/', import.meta.url).href;
const LOCAL_MODEL = new URL('../vendor/models/hand_landmarker.task', import.meta.url).href;

async function exists(url) {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
  }
}

// Pairs of landmark indices that form the drawn hand skeleton.
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],            // thumb
  [0,5],[5,6],[6,7],[7,8],            // index
  [5,9],[9,10],[10,11],[11,12],       // middle
  [9,13],[13,14],[14,15],[15,16],     // ring
  [13,17],[17,18],[18,19],[19,20],    // pinky
  [0,17],                             // palm base
];

export class HandTracker {
  constructor() {
    this.landmarker = null;
    this.lastTs = -1;
  }

  /**
   * Loads the tracker. Safe to call more than once.
   * Throws with a readable message if the library or model cannot be fetched,
   * so the caller can tell that apart from the camera itself being refused.
   */
  async load() {
    if (this.landmarker) return;

    const useLocalLib   = await exists(`${LOCAL_LIB}vision_bundle.mjs`);
    const useLocalModel = await exists(LOCAL_MODEL);
    const libBase = useLocalLib ? LOCAL_LIB : `${CDN_LIB}/`;
    const model   = useLocalModel ? LOCAL_MODEL : CDN_MODEL;

    let FilesetResolver, HandLandmarker, fileset;
    try {
      ({ FilesetResolver, HandLandmarker } = await import(`${libBase}vision_bundle.mjs`));
      fileset = await FilesetResolver.forVisionTasks(`${libBase}wasm`);
    } catch (cause) {
      throw new Error('LIB_UNAVAILABLE', { cause });
    }

    try {
      this.landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: model, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch (cause) {
      throw new Error('MODEL_UNAVAILABLE', { cause });
    }
  }

  /**
   * Detects hands in the current video frame.
   * @returns {Array<{handedness:string, landmarks:Array<{x,y,z}>}>}
   */
  detect(video, timestampMs) {
    if (!this.landmarker || video.readyState < 2) return [];

    // MediaPipe requires strictly increasing timestamps in VIDEO mode.
    const ts = timestampMs <= this.lastTs ? this.lastTs + 1 : timestampMs;
    this.lastTs = ts;

    const res = this.landmarker.detectForVideo(video, ts);
    const out = [];
    for (let i = 0; i < (res.landmarks?.length ?? 0); i++) {
      out.push({
        handedness: res.handedness?.[i]?.[0]?.categoryName ?? 'Right',
        landmarks: res.landmarks[i],
      });
    }
    return out;
  }
}

/** Draws the hand skeleton onto a canvas sized to match the video. */
export function drawHands(canvas, video, hands) {
  const ctx = canvas.getContext('2d');
  const w = video.videoWidth  || canvas.width;
  const h = video.videoHeight || canvas.height;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

  ctx.clearRect(0, 0, w, h);

  for (const hand of hands) {
    const pts = hand.landmarks;
    // One hue per hand so left and right stay visually distinct.
    const hue = hand.handedness === 'Left' ? 155 : 200;

    ctx.lineWidth = Math.max(2, w / 300);
    ctx.strokeStyle = `hsla(${hue},80%,60%,.9)`;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
      ctx.stroke();
    }

    ctx.fillStyle = `hsl(${hue},85%,72%)`;
    const r = Math.max(3, w / 220);
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// Hand-marker drawing for SIGNOO's kiosk mode.
//
// Landmark tracking itself runs server-side (kiosk/perception.py, MediaPipe Holistic in
// Python — see docs/SYSTEM_DESIGN.md B2). This module only draws what the backend sends
// back over the WebSocket: plain [x,y,z] arrays for pose, left hand, and right hand, each
// in 0..1 normalised video coordinates. The browser's job is limited to "does the visitor
// see that the camera sees them" (.claude/skills/deaf-kiosk-ux).

// Pairs of landmark indices that form the drawn hand skeleton (MediaPipe hand
// topology: 21 points per hand).
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],            // thumb
  [0,5],[5,6],[6,7],[7,8],            // index
  [5,9],[9,10],[10,11],[11,12],       // middle
  [9,13],[13,14],[14,15],[15,16],     // ring
  [13,17],[17,18],[18,19],[19,20],    // pinky
  [0,17],                             // palm base
];

// A few upper-body pose connections (shoulders/arms), enough to show "the system
// sees your body", without trying to reproduce the full 33-point topology.
const POSE_CONNECTIONS = [
  [11,12],           // shoulder line
  [11,13],[13,15],   // left arm
  [12,14],[14,16],   // right arm
];

// The dots-and-lines overlay is off: on the kiosk it covered the visitor's own hands and
// face. Tracking is unaffected (it all happens server-side); this only stops it being
// drawn. Set a flag to true to bring that part of the overlay back.
const SHOW_HANDS = false;
const SHOW_BODY = false;
// What is drawn instead: four rounded viewfinder corners around each hand. It shows the
// camera has the hand without a pile of dots over the fingers.
const SHOW_HAND_MARKERS = true;
// The same corner marks around the face, as in the reference. Pose points 0..10 are the face;
// they cover eyes to mouth, so the box is padded wider than a hand's to take in the head.
const SHOW_FACE_MARKER = true;
const FACE_POINTS = 11;
const FACE_PAD_SCALE = 1.7;

// ── smooth motion ───────────────────────────────────────────────────────────────
// The backend can deliver about ten hand positions a second at best (MediaPipe on the
// Jetson's CPU, measured). Drawing each one the instant it arrives makes a marker hop from
// place to place. So each marker eases toward the newest position on every animation
// frame, and one that loses its hand for a moment is held, then faded, instead of blinking.
const EASE_TAU_MS = 90;     // time constant of the glide: covers ~63% of a jump in 90 ms
const HOLD_MS = 250;        // a hand lost for less than this keeps its marker (tracking flickers)
const FADE_IN_MS = 100;
const FADE_OUT_MS = 160;
const DRAW_EVERY_MS = 33;   // ~30 fps is plenty for corner marks, and light on the board
const MAX_STEP_MS = 100;    // a stalled page must not jump a marker across the picture

/** Normalised bounding box of a hand's landmarks, or null if it has none. */
export function boxFromPoints(points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of points || []) {
    if (!p) continue;
    x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]);
    y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
  }
  return x0 === Infinity ? null : { x0, y0, x1, y1 };
}

/** Moves `box` toward `target` by the share of the gap that `dtMs` covers. Frame-rate
 *  independent: two 16 ms steps land where one 32 ms step does. */
export function easeToward(box, target, dtMs, tauMs = EASE_TAU_MS) {
  const k = 1 - Math.exp(-dtMs / tauMs);
  return {
    x0: box.x0 + (target.x0 - box.x0) * k, y0: box.y0 + (target.y0 - box.y0) * k,
    x1: box.x1 + (target.x1 - box.x1) * k, y1: box.y1 + (target.y1 - box.y1) * k,
  };
}

const tracks = new Map();        // 'left' | 'right' -> { box, target, seenAt, opacity }
let surface = null;              // { canvas, video } the markers are drawn onto
let latest = null;               // newest landmarks, only used by the optional skeleton
let loopId = null;
let lastDraw = 0;

/** Advances every marker by `dtMs` and drops those that have faded out. Pure state, no
 *  drawing, so it can be run and checked without a canvas. */
export function stepMarkers(dtMs, nowMs) {
  for (const [side, track] of tracks) {
    track.box = easeToward(track.box, track.target, dtMs);
    const held = nowMs - track.seenAt <= HOLD_MS;
    track.opacity = held
      ? Math.min(1, track.opacity + dtMs / FADE_IN_MS)
      : Math.max(0, track.opacity - dtMs / FADE_OUT_MS);
    if (!held && track.opacity <= 0) tracks.delete(side);
  }
}

function resizeToVideo(canvas, video) {
  const w = video.videoWidth || canvas.width || 640;
  const h = video.videoHeight || canvas.height || 480;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { w, h };
}

function drawSkeleton(ctx, points, connections, w, h, hue) {
  ctx.lineWidth = Math.max(2, w / 300);
  ctx.strokeStyle = `hsla(${hue},80%,60%,.9)`;
  for (const [a, b] of connections) {
    if (!points[a] || !points[b]) continue;
    ctx.beginPath();
    ctx.moveTo(points[a][0] * w, points[a][1] * h);
    ctx.lineTo(points[b][0] * w, points[b][1] * h);
    ctx.stroke();
  }
  ctx.fillStyle = `hsl(${hue},85%,72%)`;
  const r = Math.max(3, w / 220);
  for (const p of points) {
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p[0] * w, p[1] * h, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHandMarker(ctx, box, w, h, opacity, padScale = 1) {
  const pad = Math.max(24, w / 22) * padScale;
  const x0 = box.x0 * w - pad, x1 = box.x1 * w + pad;
  const y0 = box.y0 * h - pad, y1 = box.y1 * h + pad;
  // Four rounded corner marks, like a camera viewfinder, floating round the hand. Only the
  // corners are drawn, so nothing crosses the fingers.
  const side = Math.min(x1 - x0, y1 - y0);
  const arm = Math.max(18, side * 0.28);         // how far each mark runs along the box
  const r = Math.min(arm * 0.7, side / 3);       // how round the corner is
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.globalAlpha = opacity;
  const corners = () => {
    ctx.beginPath();
    ctx.moveTo(x0, y0 + arm);  ctx.lineTo(x0, y0 + r);  ctx.arcTo(x0, y0, x0 + r, y0, r);  ctx.lineTo(x0 + arm, y0);   // top-left
    ctx.moveTo(x1 - arm, y0);  ctx.lineTo(x1 - r, y0);  ctx.arcTo(x1, y0, x1, y0 + r, r);  ctx.lineTo(x1, y0 + arm);   // top-right
    ctx.moveTo(x1, y1 - arm);  ctx.lineTo(x1, y1 - r);  ctx.arcTo(x1, y1, x1 - r, y1, r);  ctx.lineTo(x1 - arm, y1);   // bottom-right
    ctx.moveTo(x0 + arm, y1);  ctx.lineTo(x0 + r, y1);  ctx.arcTo(x0, y1, x0, y1 - r, r);  ctx.lineTo(x0, y1 - arm);   // bottom-left
    ctx.stroke();
  };
  // White on a slightly wider dark line: white alone disappears against a bright wall. Two
  // strokes rather than a blurred shadow, which is far dearer to redraw 30 times a second.
  const line = Math.max(3, w / 200);
  ctx.lineWidth = line * 2.2;
  ctx.strokeStyle = 'rgba(0,0,0,.35)';
  corners();
  ctx.lineWidth = line;
  ctx.strokeStyle = 'rgba(255,255,255,.96)';
  corners();
  ctx.globalAlpha = 1;
}

function render() {
  const ctx = surface.canvas.getContext('2d');
  const { w, h } = resizeToVideo(surface.canvas, surface.video);
  ctx.clearRect(0, 0, w, h);
  if (SHOW_BODY && latest?.pose) drawSkeleton(ctx, latest.pose, POSE_CONNECTIONS, w, h, 45);
  if (SHOW_HANDS && latest?.left) drawSkeleton(ctx, latest.left, HAND_CONNECTIONS, w, h, 155);
  if (SHOW_HANDS && latest?.right) drawSkeleton(ctx, latest.right, HAND_CONNECTIONS, w, h, 200);
  if (SHOW_HAND_MARKERS) {
    for (const track of tracks.values()) {
      if (track.opacity > 0) drawHandMarker(ctx, track.box, w, h, track.opacity, track.padScale);
    }
  }
}

function loop(nowMs) {
  loopId = requestAnimationFrame(loop);
  const dt = nowMs - lastDraw;
  if (dt < DRAW_EVERY_MS) return;
  lastDraw = nowMs;
  stepMarkers(Math.min(dt, MAX_STEP_MS), nowMs);
  render();
  if (!tracks.size && !SHOW_HANDS && !SHOW_BODY) stopLoop();   // nothing left to animate
}

function stopLoop() {
  if (loopId !== null) cancelAnimationFrame(loopId);
  loopId = null;
}

/**
 * Hands over the newest landmarks. Drawing happens on the animation loop, easing toward
 * these positions, not here: this is called only as fast as the backend answers.
 * @param {{pose:number[][]|null, left:number[][]|null, right:number[][]|null}} landmarks
 */
export function drawLandmarks(canvas, video, landmarks) {
  surface = { canvas, video };
  latest = landmarks;
  const now = performance.now();
  // The pose's first eleven points are the face (nose, eyes, ears, mouth), so the face needs
  // no extra data from the backend. Signing uses the face (grammar, mouth shapes), and
  // showing it is held keeps the visitor looking at the camera.
  const sources = [['left', landmarks?.left, 1], ['right', landmarks?.right, 1]];
  if (SHOW_FACE_MARKER) sources.push(['face', landmarks?.pose?.slice(0, FACE_POINTS), FACE_PAD_SCALE]);
  for (const [name, points, padScale] of sources) {
    const box = boxFromPoints(points);
    if (!box) continue;                       // an absent hand/face is held, then faded, by stepMarkers
    const track = tracks.get(name);
    if (track) { track.target = box; track.seenAt = now; }
    else tracks.set(name, { box, target: box, seenAt: now, opacity: 0, padScale });   // appears where it is, fading in
  }
  if (loopId === null) {
    lastDraw = now;
    loopId = requestAnimationFrame(loop);
  }
}

export function clearCanvas(canvas) {
  stopLoop();
  tracks.clear();
  latest = null;
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

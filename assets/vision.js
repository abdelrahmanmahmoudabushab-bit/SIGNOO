// Skeleton overlay drawing for SIGNOO's kiosk mode.
//
// Landmark tracking itself now runs server-side (kiosk/perception.py, MediaPipe
// Holistic in Python — see docs/SYSTEM_DESIGN.md B2). This module only draws what
// the backend sends back over the WebSocket: plain [x,y,z] arrays for pose, left
// hand, and right hand, each in 0..1 normalised video coordinates. The browser's
// job is limited to "does the visitor see that the camera sees them"
// (.claude/skills/deaf-kiosk-ux) — the single highest-value element on screen.

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

/**
 * Draws pose + both hands onto a canvas sized to match the video.
 * @param {{pose:number[][]|null, left:number[][]|null, right:number[][]|null}} landmarks
 */
export function drawLandmarks(canvas, video, landmarks) {
  const ctx = canvas.getContext('2d');
  const { w, h } = resizeToVideo(canvas, video);
  ctx.clearRect(0, 0, w, h);

  if (landmarks?.pose) drawSkeleton(ctx, landmarks.pose, POSE_CONNECTIONS, w, h, 45);
  if (landmarks?.left) drawSkeleton(ctx, landmarks.left, HAND_CONNECTIONS, w, h, 155);
  if (landmarks?.right) drawSkeleton(ctx, landmarks.right, HAND_CONNECTIONS, w, h, 200);
}

export function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

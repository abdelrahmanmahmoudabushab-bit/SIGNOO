// Sign recognition for SIGNOO.
//
// Two stages:
//   1. Feature extraction — turns a short clip of hand landmarks into one fixed-size
//      vector, invariant to where the signer stands and how big they appear.
//   2. Classification — nearest-neighbour matching against the samples the user
//      recorded, with a rejection gate so unknown movements stay silent.

export const SEQ_LEN     = 12;   // frames every clip is resampled to
export const HAND_DIM    = 67;   // values describing one hand in one frame
export const FRAME_DIM   = HAND_DIM * 2;
export const FEATURE_DIM = SEQ_LEN * FRAME_DIM;

// How much where-the-hands-travel counts against what-shape-the-hands-make.
// Below ~0.2 signs that differ only by movement blur together; above ~1 the
// handshape stops mattering enough.
const TRAJECTORY_WEIGHT = 0.5;

/**
 * Describes one frame as 134 numbers: two hand slots, each holding presence,
 * where the hand is in the frame, how big it is, and its 21 landmarks normalised
 * relative to the wrist.
 */
export function frameFeature(hands) {
  const out = new Float32Array(FRAME_DIM);

  // Slot 0 = left hand, slot 1 = right. A second hand with duplicate handedness
  // falls into whichever slot is still free.
  const slots = [null, null];
  for (const hand of hands) {
    const preferred = hand.handedness === 'Left' ? 0 : 1;
    if (!slots[preferred]) slots[preferred] = hand;
    else if (!slots[1 - preferred]) slots[1 - preferred] = hand;
  }

  slots.forEach((hand, slot) => {
    if (!hand) return;                       // absent hand stays all-zero
    const base = slot * HAND_DIM;
    const pts  = hand.landmarks;
    const wrist = pts[0];

    // Hand size = furthest landmark from the wrist. Dividing by it makes the
    // shape the same whether the signer is near or far from the camera.
    let scale = 0;
    for (const p of pts) {
      const d = Math.hypot(p.x - wrist.x, p.y - wrist.y, p.z - wrist.z);
      if (d > scale) scale = d;
    }
    if (scale < 1e-6) scale = 1e-6;

    out[base]     = 1;            // present
    out[base + 1] = wrist.x;      // raw for now; sequenceFeature turns these into
    out[base + 2] = wrist.y;      // an offset from the clip's own centre
    out[base + 3] = scale;

    for (let i = 0; i < 21; i++) {
      const o = base + 4 + i * 3;
      out[o]     = (pts[i].x - wrist.x) / scale;
      out[o + 1] = (pts[i].y - wrist.y) / scale;
      out[o + 2] = (pts[i].z - wrist.z) / scale;
    }
  });

  return out;
}

/**
 * Resamples a clip to SEQ_LEN frames, normalises it, and flattens it to one vector.
 *
 * Normalisation is what lets the same sign match from a different seat: the whole
 * clip is centred on the signer's own hands and measured in units of their hand
 * size, so standing closer, further away or off to one side produces the same
 * vector. Movement *within* the clip survives, because both hands are centred on
 * one shared origin — so a travelling hand still reads as travelling, and the gap
 * between two hands is still the gap between two hands.
 */
export function sequenceFeature(frames) {
  if (!frames.length) return null;

  // Resample to a fixed number of frames, so a sign made quickly and the same
  // sign made slowly line up.
  const seq = [];
  for (let t = 0; t < SEQ_LEN; t++) {
    const pos  = frames.length === 1 ? 0 : (t / (SEQ_LEN - 1)) * (frames.length - 1);
    const i0   = Math.floor(pos);
    const i1   = Math.min(i0 + 1, frames.length - 1);
    const frac = pos - i0;
    const a = frames[i0], b = frames[i1];

    const f = new Float32Array(FRAME_DIM);
    for (let k = 0; k < FRAME_DIM; k++) f[k] = a[k] + (b[k] - a[k]) * frac;
    seq.push(f);
  }

  // One origin and one unit of length for the whole clip, averaged over every
  // frame in which a hand was visible.
  let sumX = 0, sumY = 0, sumScale = 0, seen = 0;
  for (const f of seq) {
    for (let slot = 0; slot < 2; slot++) {
      const base = slot * HAND_DIM;
      if (f[base] !== 1) continue;
      sumX += f[base + 1]; sumY += f[base + 2]; sumScale += f[base + 3]; seen++;
    }
  }
  if (!seen) return null;                       // no hands anywhere in the clip

  const originX = sumX / seen;
  const originY = sumY / seen;
  const unit    = Math.max(sumScale / seen, 1e-6);

  // Rewrite each hand's position as an offset from that shared origin, in hand-size
  // units. The raw hand size itself is dropped: it only measures camera distance.
  const out = new Float32Array(FEATURE_DIM);
  for (let t = 0; t < SEQ_LEN; t++) {
    const f = seq[t];
    for (let slot = 0; slot < 2; slot++) {
      const base = slot * HAND_DIM;
      if (f[base] === 1) {
        f[base + 1] = ((f[base + 1] - originX) / unit) * TRAJECTORY_WEIGHT;
        f[base + 2] = ((f[base + 2] - originY) / unit) * TRAJECTORY_WEIGHT;
      } else {
        f[base + 1] = 0; f[base + 2] = 0;
      }
      f[base + 3] = 0;
    }
    out.set(f, t * FRAME_DIM);
  }
  return out;
}

/** True when at least one hand was visible in a reasonable share of the clip. */
export function hasHands(frames, minRatio = 0.5) {
  if (!frames.length) return false;
  let seen = 0;
  for (const f of frames) if (f[0] === 1 || f[HAND_DIM] === 1) seen++;
  return seen / frames.length >= minRatio;
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
  return Math.sqrt(sum);
}

/**
 * Nearest-neighbour classifier over recorded samples.
 *
 * Confidence compares the best match against the best match from any *other*
 * word: a sign that looks much more like one word than any other scores high,
 * while a movement sitting between two words scores low. Matches further away
 * than the rejection threshold are reported as unknown, which is what keeps the
 * translator quiet during ordinary hand movement.
 */
export class Recognizer {
  constructor() {
    this.samples   = [];      // { label, vec }
    this.threshold = Infinity;
  }

  fit(samples) {
    this.samples = samples;
    this.threshold = this.#rejectionThreshold();
  }

  get labels() {
    return [...new Set(this.samples.map(s => s.label))];
  }

  /** @returns {{label:string, confidence:number, distance:number}|null} */
  classify(vec) {
    if (this.samples.length < 2 || this.labels.length < 2) return null;

    // Closest sample per word.
    const best = new Map();
    for (const s of this.samples) {
      const d = distance(vec, s.vec);
      if (d < (best.get(s.label) ?? Infinity)) best.set(s.label, d);
    }

    const ranked = [...best.entries()].sort((a, b) => a[1] - b[1]);
    const [label, d1] = ranked[0];
    const d2 = ranked[1][1];

    if (d1 > this.threshold) return null;                 // nothing we recognise

    const confidence = (d2 - d1) / (d1 + d2 + 1e-9);      // 0 = ambiguous, 1 = clear
    return { label, confidence, distance: d1 };
  }

  /**
   * How far a match may sit and still count. Derived from how tightly the user's
   * own repetitions of the same word cluster: mean + 2 standard deviations of
   * each sample's distance to its nearest same-word neighbour.
   */
  #rejectionThreshold() {
    const within = [];
    for (const a of this.samples) {
      let nearest = Infinity;
      for (const b of this.samples) {
        if (a === b || a.label !== b.label) continue;
        nearest = Math.min(nearest, distance(a.vec, b.vec));
      }
      if (Number.isFinite(nearest)) within.push(nearest);
    }
    if (!within.length) return Infinity;   // one sample per word — cannot judge yet

    const mean = within.reduce((s, v) => s + v, 0) / within.length;
    const std  = Math.sqrt(within.reduce((s, v) => s + (v - mean) ** 2, 0) / within.length);
    return mean + 2 * std;
  }
}

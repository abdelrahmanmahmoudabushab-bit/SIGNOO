"""Landmark features for Jordanian Sign Language recognition.

Turns per-frame body/hand landmarks into the tensors the model trains on. Used
identically by `ml/` at training time and by `kiosk/` (B2 Perception → B4 Recognition)
at inference time — see `core/__init__.py` for why that matters.

Landmarks rather than raw video: a sign is defined by the pose of the hands and
body, and everything else in the pixels — the room, the clothing, the signer's
appearance — is a chance for the model to learn the wrong thing. With a dataset of
realistic size (thousands, not millions, of clips) landmark models generalise to new
signers far better than video models, and train in minutes on a laptop.
"""
from __future__ import annotations

import numpy as np

# MediaPipe Holistic landmark counts.
N_POSE = 33
N_HAND = 21

# Upper-body pose landmarks only. Legs tell us nothing about signing and add noise
# when the signer is seated or cropped.
POSE_UPPER = list(range(25))

SEQ_LEN = 64          # frames every clip is resampled to (~2 s at 30 fps)
POSE_DIM = len(POSE_UPPER) * 3
# MediaPipe pose joints that come in left/right pairs within POSE_UPPER: eyes (inner,
# centre, outer), ears, mouth corners, shoulders, elbows, wrists, pinkies, index
# fingers, thumbs, hips.
POSE_MIRROR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                     (15, 16), (17, 18), (19, 20), (21, 22), (23, 24))
HAND_DIM = N_HAND * 3
FRAME_DIM = POSE_DIM + 2 * HAND_DIM + 2      # + presence flag per hand
FEATURE_SHAPE = (SEQ_LEN, FRAME_DIM)


def normalise_frame(pose, left, right):
    """Places one frame in a body-centred coordinate system.

    The origin sits between the shoulders and one unit is the shoulder width, so the
    same sign gives the same numbers whether the signer is near or far, centred or
    off to one side, tall or short. Shoulder width is used rather than hand size
    because it is stable across the clip and does not change as hands move toward or
    away from the camera.

    Args:
        pose: (N_POSE, 3) array, or None when no body was detected.
        left, right: (N_HAND, 3) arrays, or None when that hand is not visible.

    Returns:
        (FRAME_DIM,) float32 array. All-zero when the body is missing.
    """
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    if pose is None:
        return out

    pose = np.asarray(pose, dtype=np.float32)
    l_sh, r_sh = pose[11], pose[12]          # MediaPipe shoulder indices

    origin = (l_sh + r_sh) / 2.0
    scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    if scale < 1e-6:                          # shoulders collapsed — unusable frame
        return out

    def place(points):
        return ((np.asarray(points, dtype=np.float32) - origin) / scale).ravel()

    out[:POSE_DIM] = place(pose[POSE_UPPER])

    off = POSE_DIM
    for hand in (left, right):
        if hand is not None:
            out[off] = 1.0
            out[off + 1: off + 1 + HAND_DIM] = place(hand)
        off += 1 + HAND_DIM

    return out


def resample(frames, length=SEQ_LEN):
    """Resamples a clip to a fixed number of frames by linear interpolation.

    Signers vary in speed by a factor of two or more, and the same person is not
    consistent. Resampling puts fast and slow productions of a sign on the same
    time axis; the model then only has to learn the shape of the movement.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return np.zeros((length, FRAME_DIM), dtype=np.float32)
    if len(frames) == 1:
        return np.repeat(frames, length, axis=0)

    src = np.linspace(0, len(frames) - 1, num=length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    frac = (src - lo)[:, None]
    return (frames[lo] * (1 - frac) + frames[hi] * frac).astype(np.float32)


def add_velocity(clip):
    """Appends frame-to-frame differences to the features.

    Position says where the hands are; velocity says where they are going. Signs
    that share a handshape and differ only in direction of movement — a very common
    minimal pair in sign languages — are hard to separate from position alone, and
    easy once the model can see motion directly.
    """
    clip = np.asarray(clip, dtype=np.float32)
    velocity = np.diff(clip, axis=0, prepend=clip[:1])
    return np.concatenate([clip, velocity], axis=1)


# ── Hand-shape features (feature version 2) ──────────────────────────────────────────────
#
# Joint positions say *where* the hands are; they say little about *what shape* the fingers
# make, and many signs differ only in handshape. These are derived from the 21 landmarks per
# hand already in every frame, so no clip has to be re-tracked, and they are computed from
# the frame the model will actually see (after any augmentation), so a mirrored or jittered
# clip gets handshape that matches it. Angles and distances are scale-free: lengths are
# divided by the hand's own size (wrist to middle-finger knuckle).

HAND_SHAPE_PER_HAND = 31
HANDSHAPE_DIM = 2 * HAND_SHAPE_PER_HAND
# Fingers as landmark chains [base, ..., tip]; the thumb and each finger have four bones.
_FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
_TIPS = (4, 8, 12, 16, 20)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def _hand_shape(points, planar=False):
    """(T, 21, 3) hand landmarks -> (T, 31) shape descriptor; zeros where the hand is degenerate.

    `planar` ignores depth: MediaPipe's z is noisy, and frames filled in by RTMW (2-D) have none.
    """
    if planar:
        points = np.array(points, dtype=np.float32, copy=True)
        points[..., 2] = 0.0
    t = points.shape[0]
    out = np.zeros((t, HAND_SHAPE_PER_HAND), dtype=np.float32)
    size = np.linalg.norm(points[:, 9] - points[:, 0], axis=-1)
    ok = size > 1e-4
    if not ok.any():
        return out
    p = points[ok]
    s = size[ok][:, None]

    cols = []
    for chain in _FINGER_CHAINS:                          # 5 fingers x 3 flexion cosines
        bones = [_unit(p[:, b] - p[:, a]) for a, b in zip(chain[:-1], chain[1:])]
        cols.extend((bones[i] * bones[i + 1]).sum(-1, keepdims=True) for i in range(3))
    thumb_tip = p[:, 4]
    for tip in _TIPS[1:]:                                 # thumb tip to the other four tips
        cols.append(np.linalg.norm(p[:, tip] - thumb_tip, axis=-1, keepdims=True) / s)
    for tip in _TIPS:                                     # each tip's reach from the wrist
        cols.append(np.linalg.norm(p[:, tip] - p[:, 0], axis=-1, keepdims=True) / s)
    dirs = [_unit(p[:, tip] - p[:, base]) for base, tip in ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20))]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):         # spread between neighbouring fingers
        cols.append((dirs[a] * dirs[b]).sum(-1, keepdims=True))
    cols.append(_unit(np.cross(p[:, 5] - p[:, 0], p[:, 17] - p[:, 0])))   # palm normal
    out[ok] = np.concatenate(cols, axis=1)
    return out


def handshape_features(clip, planar=False):
    """(T, FRAME_DIM) clip -> (T, HANDSHAPE_DIM): left-hand then right-hand shape; zero if absent."""
    clip = np.asarray(clip, dtype=np.float32)
    parts = []
    for flag in (POSE_DIM, POSE_DIM + 1 + HAND_DIM):
        present = clip[:, flag] > 0.5
        pts = clip[:, flag + 1: flag + 1 + HAND_DIM].reshape(len(clip), N_HAND, 3)
        shape = _hand_shape(pts, planar)
        shape[~present] = 0.0
        parts.append(shape)
    return np.concatenate(parts, axis=1)


# ── Face features (feature version 3) ────────────────────────────────────────────────────
#
# Eyebrows, eyes, mouth and head tilt carry grammar in sign languages (a raised brow marks a
# question, a puffed cheek or pursed lip changes a verb). They are summarised as 12 numbers
# per frame from MediaPipe's 468-point face mesh, each a ratio of the face's own width, so
# they do not depend on distance from the camera.

FACE_DIM = 12
_FACE_MESH_POINTS = 468


def face_features(face):
    """(>=468, 3) face-mesh landmarks -> (FACE_DIM,) float32; all zeros when no face was found.

    Order: brow raise L/R, eye openness L/R, mouth opening, mouth width, lip-corner lift,
    mouth opening/width ratio, head roll, head yaw, head pitch, brow furrow.
    """
    out = np.zeros(FACE_DIM, dtype=np.float32)
    if face is None:
        return out
    f = np.asarray(face, dtype=np.float32)
    if f.ndim != 2 or len(f) < _FACE_MESH_POINTS:
        return out
    width = float(np.linalg.norm(f[454, :2] - f[234, :2]))        # cheek to cheek
    if width < 1e-6:
        return out

    def d(a, b):
        return float(np.linalg.norm(f[a, :2] - f[b, :2])) / width

    mouth_open = d(13, 14)
    mouth_width = d(61, 291)
    eye_mid = (f[33, :2] + f[263, :2]) / 2
    cheek_mid = (f[234, :2] + f[454, :2]) / 2
    lip_mid_y = (f[13, 1] + f[14, 1]) / 2
    corner_y = (f[61, 1] + f[291, 1]) / 2
    roll_vec = f[263, :2] - f[33, :2]
    out[:] = (
        d(105, 159), d(334, 386),                                  # brow raise: brow to upper lid
        d(159, 145), d(386, 374),                                  # eye openness
        mouth_open, mouth_width,
        (lip_mid_y - corner_y) / width,                            # corners above the lips = smile
        mouth_open / max(mouth_width, 0.01),                       # bounded: 0 closed, ~1 wide open
        float(np.arctan2(roll_vec[1], roll_vec[0])),               # head roll, radians
        float(f[1, 0] - cheek_mid[0]) / width,                     # nose offset: head yaw proxy
        float(f[1, 1] - eye_mid[1]) / width,                       # nose drop: head pitch proxy
        d(107, 336),                                               # inner-brow gap (furrow)
    )
    return out


def mirror_face(face):
    """Face features of the mirrored clip: left and right brows and eyes swap; roll and yaw flip."""
    face = np.array(face, dtype=np.float32, copy=True)
    face[:, [0, 1]] = face[:, [1, 0]]
    face[:, [2, 3]] = face[:, [3, 2]]
    face[:, 8] *= -1
    face[:, 9] *= -1
    return face


# ── Time handling (feature versions 4 and 5) ─────────────────────────────────────────────
#
# Every clip is resampled to SEQ_LEN frames, which discards how long the sign took and shows the
# model a hand that the tracker lost for a few frames as a flicker of "absent". These two fix that:
# brief dropouts are bridged by interpolation, and the clip's real duration travels with it.

MAX_FILLED_GAP = 3            # frames (of SEQ_LEN): about 150 ms of a 2 s sign, i.e. tracker noise, not signing
DURATION_REF_S = 1.5


def fill_hand_gaps(clip, max_gap: int = MAX_FILLED_GAP):
    """Bridges short hand dropouts: a run of at most `max_gap` frames where a hand is absent, with that
    hand present on both sides, is replaced by a straight-line interpolation and marked present.

    Longer gaps, and hands that appear or leave at the clip's ends, are left alone: those are the
    signer's hands leaving the frame, not the tracker blinking. Applied identically in training and on
    the kiosk (inside `model_input`), so a model never sees one treatment and meets the other.
    """
    clip = np.array(clip, dtype=np.float32, copy=True)
    frames = len(clip)
    for flag in (POSE_DIM, POSE_DIM + 1 + HAND_DIM):
        coords = slice(flag + 1, flag + 1 + HAND_DIM)
        present = clip[:, flag] > 0.5
        i = 0
        while i < frames:
            if present[i]:
                i += 1
                continue
            j = i
            while j < frames and not present[j]:
                j += 1
            if 0 < i and j < frames and j - i <= max_gap:
                a, b = i - 1, j
                for k in range(i, j):
                    w = (k - a) / (b - a)
                    clip[k, coords] = (1 - w) * clip[a, coords] + w * clip[b, coords]
                    clip[k, flag] = 1.0
            i = j
    return clip


def duration_column(seconds: float, frames: int):
    """(frames, 1): log of the sign's real duration relative to 1.5 s, the same value on every frame."""
    return np.full((frames, 1), np.log(max(float(seconds), 0.05) / DURATION_REF_S), dtype=np.float32)


FEATURE_VERSIONS = (1, 2, 3, 4, 5)
INPUT_DIMS = {1: FRAME_DIM * 2, 2: (FRAME_DIM + HANDSHAPE_DIM) * 2,
              3: (FRAME_DIM + HANDSHAPE_DIM + FACE_DIM) * 2,
              4: (FRAME_DIM + HANDSHAPE_DIM + 1) * 2, 5: (FRAME_DIM + HANDSHAPE_DIM + 1) * 2}


def model_input(clip, version: int = 1, face=None, duration=None):
    """The exact tensor a model of this feature version is trained and run on.

    Training, evaluation, calibration, shadow scoring and the kiosk all call this, never
    `add_velocity` directly, and a checkpoint records its `feature_version`, so a model
    cannot be fed features it was not trained on. Version 1 is position + velocity (what
    every model before 2026-10-12 used); version 2 adds hand shape; version 3 adds the
    (T, FACE_DIM) `face` features as well, and refuses to run without them.
    """
    clip = np.asarray(clip, dtype=np.float32)
    if version == 1:
        return add_velocity(clip)
    if version == 2:
        return add_velocity(np.concatenate([clip, handshape_features(clip)], axis=1))
    if version == 3:
        if face is None or np.asarray(face).shape != (len(clip), FACE_DIM):
            raise ValueError(f"feature version 3 needs face features of shape ({len(clip)}, {FACE_DIM})")
        return add_velocity(np.concatenate([clip, handshape_features(clip), np.asarray(face, np.float32)], axis=1))
    if version in (4, 5):
        # 4: hand shape in 3-D; 5: planar (xy only). Both bridge short hand dropouts and carry the
        # sign's real duration in seconds, which `duration` must supply.
        if duration is None:
            raise ValueError(f"feature version {version} needs the clip's duration in seconds")
        clip = fill_hand_gaps(clip)
        columns = [clip, handshape_features(clip, planar=version == 5), duration_column(duration, len(clip))]
        return add_velocity(np.concatenate(columns, axis=1))
    raise ValueError(f"unknown feature version {version!r}; known: {FEATURE_VERSIONS}")


def mirror(clip):
    """Left-right flip, for augmentation and for relating left- and right-handed signing.

    Mirroring negates the x axis and swaps the two hand blocks. Handedness in sign
    language is largely a property of the signer, not the sign, so a mirrored clip is
    usually a valid example of the same sign — but see the caution in TRAINING.md.
    """
    clip = np.array(clip, dtype=np.float32, copy=True)

    clip[:, 0:POSE_DIM:3] *= -1               # pose x
    # A mirror also turns the left shoulder into the right one: swap each paired joint,
    # or the result is a body no real signer has (left shoulder on the right side).
    pose = clip[:, :POSE_DIM].reshape(len(clip), -1, 3)
    for a, b in POSE_MIRROR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    left = slice(POSE_DIM, POSE_DIM + 1 + HAND_DIM)
    right = slice(POSE_DIM + 1 + HAND_DIM, FRAME_DIM)
    clip[:, left.start + 1: left.stop: 3] *= -1     # hand x, skipping presence flag
    clip[:, right.start + 1: right.stop: 3] *= -1

    swapped = np.array(clip, copy=True)
    swapped[:, left] = clip[:, right]
    swapped[:, right] = clip[:, left]
    return swapped

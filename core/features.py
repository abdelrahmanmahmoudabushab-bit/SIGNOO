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

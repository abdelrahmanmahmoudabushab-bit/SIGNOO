"""Converts recorded videos into landmark feature files.

    python ml/extract_landmarks.py data/raw --out data/processed

Walks the raw video tree, runs MediaPipe Holistic over each clip, normalises and
resamples it, and writes one .npy per clip plus a manifest for the training scripts.

Expects the layout described in docs/DATA_COLLECTION.md:

    data/raw/<signer_id>/<sign_id>/<clip>.mp4   (+ matching .json metadata)

Optional, off by default so existing recordings behave exactly as before:

    --max-side N   downscale frames so the longer side is N px before tracking
    --trim-idle    drop leading/trailing frames with no hand visible, matching what
                   the kiosk's segmenter hands to the classifier at inference
    --workers N    process N clips in parallel
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import HAND_DIM, POSE_DIM, normalise_frame, resample

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
# Resolved against the repository, not the working directory, so the script can be
# run from anywhere without re-downloading the model into a new place.
MODEL_PATH = Path(__file__).resolve().parent.parent / "vendor/models/holistic_landmarker.task"

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_MODEL_PATH = Path(__file__).resolve().parent.parent / "vendor/models/hand_landmarker.task"

LEFT_FLAG = POSE_DIM
RIGHT_FLAG = POSE_DIM + 1 + HAND_DIM

# MediaPipe pose landmark indices of the wrists; a detected hand is matched to the nearer one.
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
MAX_WRIST_GAP = 0.12        # image-normalised distance beyond which a hand is nobody's hand


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    """Downloads a MediaPipe model once, if it is not already present."""
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {path.name} → {path}")
    urllib.request.urlretrieve(url, path)
    return path


def _points(landmarks, n):
    """Tasks-API landmark list → (n, 3) array, or None when nothing was detected."""
    if not landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in landmarks[:n]], dtype=np.float32)


def trim_idle(frames):
    """Keeps first-to-last frame with a visible hand; None when no hand is ever seen.

    The kiosk only starts an utterance when a hand enters the frame and ends it after
    stillness, so a classifier trained on clips padded with empty lead-in would be
    scored on a distribution it never sees live.
    """
    bounds = idle_bounds(frames)
    return frames[bounds[0]: bounds[1]] if bounds else None


def idle_bounds(frames):
    """(first, last + 1) frame index with a visible hand, or None; trim_idle's rule, reusable
    for arrays that must be cut at exactly the same frames (the face features)."""
    present = [i for i, f in enumerate(frames) if f[LEFT_FLAG] > 0 or f[RIGHT_FLAG] > 0]
    return (present[0], present[-1] + 1) if present else None


def effective_stride(fps: float, stride: int = 1, target_fps: float = 0) -> int:
    """Every how-many-th frame to track.

    With `target_fps`, sample by time, not by frame count: a 120 fps phone clip and a 30 fps webcam clip
    then end up tracked at about the same rate, instead of one at 60 fps and the other at 15 (the same
    `--stride 2` meant very different things for them, and cost four times as much on the phone clip).
    """
    if target_fps:
        return max(1, int(round(fps / target_fps)))
    return max(1, int(stride))


def assign_hands(pose, detected, left, right):
    """Fills the hand slots Holistic left empty from a dedicated hand detector's results.

    Holistic finds hands only near where its pose model puts the wrists, and loses them
    when they are small, fast or blurred. A hand detector looks at the whole frame. Each
    detected hand goes to whichever *pose wrist* it is nearest, not by MediaPipe's own
    left/right label (whose convention depends on mirroring), so an assignment can never
    swap hands; a hand more than MAX_WRIST_GAP from both wrists, or one whose side is
    already filled by Holistic, is ignored.

    pose: (33, 3) normalised-image landmarks or None; detected: list of (21, 3) arrays.
    """
    if pose is None or not detected:
        return left, right
    wrists = {"left": pose[POSE_LEFT_WRIST, :2], "right": pose[POSE_RIGHT_WRIST, :2]}
    taken = {"left": left is not None, "right": right is not None}
    placed = {}
    pairs = sorted(((float(np.linalg.norm(h[0, :2] - wrists[side])), i, side)
                    for i, h in enumerate(detected) for side in wrists), key=lambda t: t[0])
    used = set()
    for gap, i, side in pairs:
        if gap > MAX_WRIST_GAP or i in used or side in placed or taken[side]:
            continue
        placed[side] = detected[i]
        used.add(i)
    return placed.get("left", left), placed.get("right", right)


_RTMW = None


def rtmw_detector():
    """RTMW whole-body pose (OpenMMLab, via rtmlib + ONNX Runtime), loaded once per process.

    Used only as a last-resort fallback for hands: on motion-blurred frames (a fast sign is mostly
    blur) it still places the 21 hand points, where MediaPipe finds nothing. On the 8 worst clips it
    found a hand in 48% of frames to MediaPipe-with-fallback's 12%. It is 2-D, so the hand points it
    supplies have z = 0, and it is several times slower, so it is opt-in (--rtmw-fallback).
    """
    global _RTMW
    if _RTMW is None:
        from rtmlib import Wholebody
        _RTMW = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    return _RTMW


def rtmw_hands(frame, detector):
    """[(21, 3) normalised-image hand arrays] RTMW is confident about in this BGR frame (z = 0)."""
    keypoints, scores = detector(frame)
    if not len(scores):
        return []
    person = int(np.argmax(scores.mean(axis=1)))
    height, width = frame.shape[:2]
    hands = []
    for lo, hi in ((91, 112), (112, 133)):                  # COCO-WholeBody: left hand, right hand
        if scores[person, lo:hi].mean() > 0.5:
            pts = np.zeros((N_HAND_POINTS, 3), dtype=np.float32)
            pts[:, 0] = keypoints[person, lo:hi, 0] / width
            pts[:, 1] = keypoints[person, lo:hi, 1] / height
            hands.append(pts)
    return hands


N_HAND_POINTS = 21


def landmarks_from_video(path, make_landmarker, cv2, max_side=0, make_hands=None, want_face=False, stride=1,
                         use_rtmw=False, info=None, target_fps=0):
    """Runs Holistic over one video; returns (frames, faces).

    Pass `info` (a dict) to learn more: it is filled with `fps`, `frames_read`, `stride`, and, when it
    already holds `"raw": True`, the full per-frame landmarks (`pose`, `left`, `right`, `face`, NaN where
    absent), the detector that supplied each hand (`hand_source`: 0 none, 1 Holistic, 2 hand detector,
    3 RTMW) and each frame's `timestamp_ms`. Keeping those means features can be recomputed later
    (different trim, gap policy, face features) without tracking the video again.

    `frames` are normalised landmark frames. `faces` is a parallel list of
    core.features.face_features vectors when `want_face`, else None. With `make_hands`
    (a factory for a HandLandmarker), frames where Holistic lost a hand get a second
    chance from the dedicated detector; see assign_hands.

    A fresh landmarker per clip, deliberately. In VIDEO mode MediaPipe tracks across
    frames and requires timestamps to increase, so a shared instance would both carry
    tracking state from one signer's clip into the next and reject the second clip's
    timestamps for restarting at zero.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = effective_stride(fps, stride, target_fps)
    frames, faces = [], ([] if want_face else None)
    keep_raw = bool(info is not None and info.get("raw"))
    raw = {k: [] for k in ("pose", "left", "right", "face", "hand_source", "timestamp_ms")} if keep_raw else None
    index = 0
    landmarker = make_landmarker()
    hands = make_hands() if make_hands else None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride > 1 and index % stride:
                index += 1              # decoded but not tracked: every clip is resampled to SEQ_LEN anyway
                continue
            if max_side and max(frame.shape[:2]) > max_side:
                scale = max_side / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            # Timestamps must increase; VIDEO mode uses them for tracking continuity.
            timestamp = int(index * 1000 / fps)
            result = landmarker.detect_for_video(image, timestamp)
            index += 1

            pose = _points(result.pose_landmarks, 33)
            left = _points(result.left_hand_landmarks, 21)
            right = _points(result.right_hand_landmarks, 21)
            from_holistic = (left is not None, right is not None)
            if hands is not None and pose is not None and (left is None or right is None):
                found = hands.detect_for_video(image, timestamp)
                detected = [_points(h, 21) for h in found.hand_landmarks]
                left, right = assign_hands(pose, detected, left, right)
            from_detector = (left is not None, right is not None)
            if use_rtmw and pose is not None and (left is None or right is None):
                left, right = assign_hands(pose, rtmw_hands(frame, rtmw_detector()), left, right)

            frames.append(normalise_frame(pose, left, right))
            face_points = _points(result.face_landmarks, 468)
            if want_face:
                from core.features import face_features
                faces.append(face_features(face_points))
            if keep_raw:
                nan = lambda n: np.full((n, 3), np.nan, dtype=np.float16)  # noqa: E731
                raw["pose"].append(pose.astype(np.float16) if pose is not None else nan(33))
                raw["left"].append(left.astype(np.float16) if left is not None else nan(21))
                raw["right"].append(right.astype(np.float16) if right is not None else nan(21))
                raw["face"].append(face_points.astype(np.float16) if face_points is not None else nan(468))
                raw["hand_source"].append([
                    1 if from_holistic[side] else 2 if from_detector[side] else 3 if (left, right)[side] is not None else 0
                    for side in (0, 1)])
                raw["timestamp_ms"].append(timestamp)
    finally:
        cap.release()
        landmarker.close()
        if hands is not None:
            hands.close()
    if info is not None:
        info.update(fps=float(fps), frames_read=index, stride=stride)
        if keep_raw:
            info["arrays"] = {k: np.asarray(v) for k, v in raw.items()}
    return frames, faces


def _process(job):
    """One clip, start to finish. Top-level so worker processes can run it; returns
    None on success or a short reason the clip was skipped."""
    video, target, model_path, max_side, trim, extra = job
    import cv2
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    make_hands = None
    if extra.get("hand_model"):
        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(extra["hand_model"])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        make_hands = lambda: vision.HandLandmarker.create_from_options(hand_options)  # noqa: E731

    def run(side):
        info = {"raw": extra.get("raw", False)}
        frames_, faces_ = landmarks_from_video(
            Path(video), lambda: vision.HolisticLandmarker.create_from_options(options), cv2, side,
            make_hands=make_hands, want_face=extra.get("face", False), stride=extra.get("stride", 1),
            use_rtmw=extra.get("rtmw", False), info=info, target_fps=extra.get("target_fps", 0))
        info["side"] = side
        return frames_, faces_, info

    def hand_share(fs):
        return float(np.mean([(f[LEFT_FLAG] > 0) or (f[RIGHT_FLAG] > 0) for f in fs])) if fs else 0.0

    frames, faces, info = run(max_side)
    # Small, fast or blurred hands are the usual loss. One more pass at a larger size often
    # finds them; the better of the two passes is kept, never a mixture.
    retry = extra.get("retry_side", 0)
    if frames and retry > max_side and hand_share(frames) < extra.get("retry_below", 0.6):
        frames2, faces2, info2 = run(retry)
        if hand_share(frames2) > hand_share(frames):
            frames, faces, info = frames2, faces2, info2
    if not frames:
        return "no frames"
    tracked = float(np.mean([f.any() for f in frames]))
    if tracked < 0.5:
        return f"body tracked in only {tracked:.0%} of frames"
    lo, hi = 0, len(frames)
    if trim:
        bounds = idle_bounds(frames)
        if bounds is None:
            return "no hand ever visible"
        lo, hi = bounds
    margin = int(extra.get("trim_margin", 0))
    if trim and margin:
        lo, hi = max(0, lo - margin), min(len(frames), hi + margin)       # keep a little onset and release
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, resample(frames[lo:hi]))
    if faces is not None:
        # Same frames, same resampling, as the landmarks. Saved as `<clip>.face` (a file
        # handle, so numpy adds no .npy), which keeps `*.npy` globs meaning "clips".
        with open(target.with_suffix(".face"), "wb") as handle:
            np.save(handle, resample(faces[lo:hi]))
    stride = int(info.get("stride", 1))
    fps = float(info.get("fps", 30.0))
    # The sign's real length, which resampling to SEQ_LEN throws away; also the lineage of the clip.
    meta = {
        "fps": fps, "stride": stride, "frames_read": int(info.get("frames_read", 0)),
        "tracked_frames": len(frames), "span": [lo, hi],
        "duration_s": round((hi - lo) * stride / fps, 4),
        "hand_share": round(hand_share(frames[lo:hi]), 4), "tracked_side": int(info.get("side", 0)),
        "extractor": {"hand_fallback": bool(extra.get("hand_model")), "rtmw": bool(extra.get("rtmw")),
                      "retry_side": int(extra.get("retry_side", 0)), "face": bool(extra.get("face")),
                      "trim_margin": margin, "version": 3},
    }
    target.with_suffix(".meta").write_text(json.dumps(meta), encoding="utf-8")
    if extra.get("raw") and "arrays" in info:
        np.savez_compressed(target.with_name(target.stem + ".raw.npz"), span=np.array([lo, hi]), **info["arrays"])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", help="root of the raw video tree")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--model", default=str(MODEL_PATH),
                    help="path to holistic_landmarker.task (downloaded if missing)")
    ap.add_argument("--max-side", type=int, default=0, help="downscale so the longer side is N px (0 = off)")
    ap.add_argument("--trim-idle", action="store_true", help="drop frames with no hand at the start/end")
    ap.add_argument("--workers", type=int, default=1, help="clips processed in parallel")
    ap.add_argument("--hand-fallback", action="store_true",
                    help="run a dedicated hand detector on frames where Holistic lost a hand")
    ap.add_argument("--retry-side", type=int, default=0,
                    help="re-run clips whose hands were tracked in under --retry-below of frames "
                         "at this larger size (keeps the better pass); 0 = off")
    ap.add_argument("--retry-below", type=float, default=0.6)
    ap.add_argument("--target-fps", type=float, default=0,
                    help="track about this many frames per second whatever the video's rate (overrides --stride); "
                         "15 matches the 30 fps clips already extracted at stride 2")
    ap.add_argument("--save-raw", action="store_true",
                    help="also save <clip>.raw.npz: every landmark MediaPipe returned (face included) with the "
                         "detector that supplied each hand, so features can be recomputed without re-tracking")
    ap.add_argument("--trim-margin", type=int, default=0,
                    help="with --trim-idle, keep this many tracked frames of context either side of the signing")
    ap.add_argument("--rtmw-fallback", action="store_true",
                    help="last-resort hands from RTMW (needs `pip install rtmlib onnxruntime`); slow, "
                         "2-D (z = 0), but finds motion-blurred hands MediaPipe loses")
    ap.add_argument("--stride", type=int, default=1,
                    help="track every Nth frame (2 halves the time; the clip is resampled to SEQ_LEN anyway)")
    ap.add_argument("--face", action="store_true",
                    help="also write <clip>.face: (SEQ_LEN, 12) face features (feature version 3)")
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as e:                    # keeps the rest of the pipeline usable
        raise SystemExit(
            f"missing dependency: {e.name}\n"
            "install with: pip install mediapipe opencv-python-headless\n"
            "on headless Linux also: apt-get install libegl1 libgles2 libgl1"
        )

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in raw.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"no videos found under {raw}")
    print(f"{len(videos)} videos under {raw}\n")

    model_path = ensure_model(Path(args.model))

    def rel_for(video):
        return Path(video.parent.parent.name) / video.parent.name / f"{video.stem}.npy"

    extra = {"face": args.face, "retry_side": args.retry_side, "retry_below": args.retry_below,
             "stride": args.stride, "rtmw": args.rtmw_fallback, "raw": args.save_raw, "trim_margin": args.trim_margin,
             "target_fps": args.target_fps}
    if args.hand_fallback:
        extra["hand_model"] = str(ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL))

    def wanted(video):
        target = out / rel_for(video)
        return args.overwrite or not target.exists() or (args.face and not target.with_suffix(".face").exists())

    jobs = [(str(v), str(out / rel_for(v)), str(model_path), args.max_side, args.trim_idle, extra)
            for v in videos if wanted(v)]

    skipped: dict[str, str] = {}
    if args.workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, (job, reason) in enumerate(zip(jobs, pool.map(_process, jobs)), 1):
                if reason:
                    skipped[job[0]] = reason
                if n % 25 == 0:
                    print(f"  processed {n}/{len(jobs)}", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            reason = _process(job)
            if reason:
                skipped[job[0]] = reason
            if n % 25 == 0:
                print(f"  processed {n}/{len(jobs)}", flush=True)

    manifest = []
    for video in videos:
        rel = rel_for(video)
        if not (out / rel).exists():
            continue
        # Carry the recording metadata through, when it exists.
        meta_path = video.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        manifest.append({
            "signer_id": meta.get("signer_id", video.parent.parent.name),
            "sign_id": meta.get("sign_id", video.parent.name),
            "label_ar": meta.get("label_ar", ""),
            "label_en": meta.get("label_en", ""),
            "handedness": meta.get("handedness", "right"),
            "source": str(video.relative_to(raw)),
            "features": str(rel),
        })

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(manifest)} clips → {out/'manifest.json'}")
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for video, why in list(skipped.items())[:10]:
            print(f"  {video} ({why})")
    print("\nNext: python ml/clean.py", out / "manifest.json")


if __name__ == "__main__":
    main()

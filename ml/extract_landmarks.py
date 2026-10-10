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

LEFT_FLAG = POSE_DIM
RIGHT_FLAG = POSE_DIM + 1 + HAND_DIM


def ensure_model(path=MODEL_PATH):
    """Downloads the Holistic model once, if it is not already present."""
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading holistic model → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
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
    present = [i for i, f in enumerate(frames) if f[LEFT_FLAG] > 0 or f[RIGHT_FLAG] > 0]
    return frames[present[0]: present[-1] + 1] if present else None


def landmarks_from_video(path, make_landmarker, cv2, max_side=0):
    """Runs Holistic over one video, returning a list of normalised frames.

    A fresh landmarker per clip, deliberately. In VIDEO mode MediaPipe tracks across
    frames and requires timestamps to increase, so a shared instance would both carry
    tracking state from one signer's clip into the next and reject the second clip's
    timestamps for restarting at zero.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    index = 0
    landmarker = make_landmarker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_side and max(frame.shape[:2]) > max_side:
                scale = max_side / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            # Timestamps must increase; VIDEO mode uses them for tracking continuity.
            result = landmarker.detect_for_video(image, int(index * 1000 / fps))
            index += 1

            frames.append(normalise_frame(
                _points(result.pose_landmarks, 33),
                _points(result.left_hand_landmarks, 21),
                _points(result.right_hand_landmarks, 21),
            ))
    finally:
        cap.release()
        landmarker.close()
    return frames


def _process(job):
    """One clip, start to finish. Top-level so worker processes can run it; returns
    None on success or a short reason the clip was skipped."""
    video, target, model_path, max_side, trim = job
    import cv2
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    frames = landmarks_from_video(
        Path(video), lambda: vision.HolisticLandmarker.create_from_options(options), cv2, max_side
    )
    if not frames:
        return "no frames"
    tracked = float(np.mean([f.any() for f in frames]))
    if tracked < 0.5:
        return f"body tracked in only {tracked:.0%} of frames"
    if trim:
        frames = trim_idle(frames)
        if frames is None:
            return "no hand ever visible"
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, resample(frames))
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

    jobs = [(str(v), str(out / rel_for(v)), str(model_path), args.max_side, args.trim_idle)
            for v in videos if args.overwrite or not (out / rel_for(v)).exists()]

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

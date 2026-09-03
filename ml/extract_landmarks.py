"""Converts recorded videos into landmark feature files.

    python ml/extract_landmarks.py data/raw --out data/processed

Walks the raw video tree, runs MediaPipe Holistic over each clip, normalises and
resamples it, and writes one .npy per clip plus a manifest for the training scripts.

Expects the layout described in docs/DATA_COLLECTION.md:

    data/raw/<signer_id>/<sign_id>/<clip>.mp4   (+ matching .json metadata)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import normalise_frame, resample

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
# Resolved against the repository, not the working directory, so the script can be
# run from anywhere without re-downloading the model into a new place.
MODEL_PATH = Path(__file__).resolve().parent.parent / "vendor/models/holistic_landmarker.task"


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


def landmarks_from_video(path, landmarker, cv2):
    """Runs Holistic over one video, returning a list of normalised frames."""
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", help="root of the raw video tree")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--model", default=str(MODEL_PATH),
                    help="path to holistic_landmarker.task (downloaded if missing)")
    args = ap.parse_args()

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
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
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    manifest, skipped = [], []
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, video in enumerate(videos, 1):
            signer_id = video.parent.parent.name
            sign_id = video.parent.name
            rel = Path(signer_id) / sign_id / f"{video.stem}.npy"
            target = out / rel

            if target.exists() and not args.overwrite:
                pass
            else:
                frames = landmarks_from_video(video, landmarker, cv2)
                if not frames:
                    skipped.append(f"{video} (no frames)")
                    continue
                tracked = float(np.mean([f.any() for f in frames]))
                if tracked < 0.5:
                    skipped.append(f"{video} (body tracked in only {tracked:.0%} of frames)")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, resample(frames))

            # Carry the recording metadata through, when it exists.
            meta_path = video.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            manifest.append({
                "signer_id": meta.get("signer_id", signer_id),
                "sign_id": meta.get("sign_id", sign_id),
                "label_ar": meta.get("label_ar", ""),
                "label_en": meta.get("label_en", ""),
                "handedness": meta.get("handedness", "right"),
                "source": str(video.relative_to(raw)),
                "features": str(rel),
            })
            print(f"  [{i}/{len(videos)}] {rel}", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(manifest)} clips → {out/'manifest.json'}")
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for s in skipped[:10]:
            print(f"  {s}")
    print("\nNext: python ml/clean.py", out / "manifest.json")


if __name__ == "__main__":
    main()

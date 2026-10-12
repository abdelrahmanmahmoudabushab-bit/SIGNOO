"""B2 Perception: browser-sent JPEG frames -> `core.contracts.Landmarks`
(SYSTEM_DESIGN.md B2).

Uses MediaPipe Holistic in **LIVE_STREAM** mode, not VIDEO mode — `ml/extract_landmarks.py`
uses VIDEO mode because it processes a finite recorded file where every frame is
already known; here frames arrive one at a time from a live camera over the
WebSocket, and LIVE_STREAM mode is the Tasks API's mode for exactly that: it is
asynchronous (`detect_async`, a `result_callback`) rather than blocking per call,
which matters once frames are arriving faster than inference can keep up (the
whole reason B1's single-slot buffer exists).

Runs every second processed frame (`perception_stride`), matching the 30 fps
budget in SYSTEM_DESIGN.md §4.1 — the largest single item there.

Camera capture itself lives in the browser (`getUserMedia`); this module never
touches `cv2.VideoCapture`. It degrades to a clear, typed error rather than a
crash when MediaPipe or its model file is unavailable (Windows dev machines
without the vendor model, headless CI, etc.) — see `PerceptionUnavailable`.
"""
from __future__ import annotations

import base64
import logging
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

from core.contracts import Landmarks
from kiosk.buffers import SingleSlotBuffer

logger = logging.getLogger("signoo.kiosk.perception")

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "vendor/models/holistic_landmarker.task"


class PerceptionUnavailable(RuntimeError):
    """MediaPipe, OpenCV, or the model file are not usable in this environment.

    Raised, never crashed past silently — SYSTEM_DESIGN.md §6's "camera
    disconnects -> visible out-of-service screen" failure mode applies equally to
    "perception cannot start at all".
    """


def _points(landmark_list, n: int) -> np.ndarray | None:
    if not landmark_list:
        return None
    return np.array([[p.x, p.y, p.z] for p in landmark_list[:n]], dtype=np.float32)


class HolisticPerception:
    """One instance per active WebSocket session — MediaPipe tracking state must
    not bleed between visitors, same reasoning as `ml/extract_landmarks.py`'s "a
    fresh landmarker per clip, deliberately".
    """

    def __init__(self, stride: int = 2, ensure_model: bool = True):
        self.stride = max(1, stride)
        self._frame_index = 0
        self._result_buffer: SingleSlotBuffer[Landmarks] = SingleSlotBuffer()
        self._landmarker = None
        self._start_ms = None
        self._lock = threading.Lock()
        self._cv2 = None
        self._build(ensure_model)

    def _build(self, ensure_model: bool) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise PerceptionUnavailable(
                f"missing dependency: {exc.name}. "
                "pip install mediapipe opencv-python-headless"
            ) from exc

        model_path = MODEL_PATH
        if not model_path.exists():
            if not ensure_model:
                raise PerceptionUnavailable(f"model file missing: {model_path}")
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as exc:
                raise PerceptionUnavailable(
                    f"holistic_landmarker.task not found at {model_path} and could "
                    f"not be downloaded ({exc}). Run scripts/fetch-vendor.sh or place "
                    "the file manually."
                ) from exc

        self._cv2 = cv2
        self._mp = mp

        def _on_result(result, output_image, timestamp_ms):
            landmarks = Landmarks(
                ts_ms=timestamp_ms,
                pose=_points(result.pose_landmarks, 33),
                left=_points(result.left_hand_landmarks, 21),
                right=_points(result.right_hand_landmarks, 21),
            )
            self._result_buffer.put(landmarks)   # newest-wins, same B1 discipline

        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            result_callback=_on_result,
        )
        try:
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        except Exception as exc:
            raise PerceptionUnavailable(f"could not start Holistic Landmarker: {exc}") from exc

    def submit_frame(self, jpeg_bytes: bytes, ts_ms: int) -> bool:
        """Decodes one JPEG and (every `stride`th call) sends it to MediaPipe.
        Returns False for frames skipped by the stride, so the caller can tell
        "skipped on purpose" apart from "perception is behind".
        """
        self._frame_index += 1
        if self._frame_index % self.stride != 0:
            return False

        cv2 = self._cv2
        array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise PerceptionUnavailable("could not decode frame as an image")

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        )
        with self._lock:
            if self._start_ms is None:
                self._start_ms = ts_ms
            # LIVE_STREAM requires strictly increasing timestamps.
            monotonic_ts = max(ts_ms - self._start_ms, self._frame_index)
            self._landmarker.detect_async(image, monotonic_ts)
        return True

    def latest(self) -> Landmarks | None:
        """Freshest available result — the perception half of the newest-wins
        discipline; `_result_buffer` drops anything not yet consumed.
        """
        return self._result_buffer.get()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, without decoding the image; None if
    this is not a JPEG. Lets the caller refuse an oversized image before any pixel
    buffer is allocated for it.
    """
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:                              # fill byte before a marker
            i += 1
        elif marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers with no payload
            i += 2
        elif marker in (0xD9, 0xDA):                    # image data with no frame header before it
            return None
        else:
            length = int.from_bytes(data[i + 2:i + 4], "big")
            if length < 2:
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # a start-of-frame
                if i + 9 > len(data):
                    return None
                return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
            i += 2 + length
    return None


def decode_base64_frame(data_url_or_b64: str) -> bytes:
    """Accepts either a bare base64 string or a `data:image/jpeg;base64,...` URL,
    since browsers commonly hand a canvas/video frame over as the latter.
    """
    if "," in data_url_or_b64[:64]:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    return base64.b64decode(data_url_or_b64)

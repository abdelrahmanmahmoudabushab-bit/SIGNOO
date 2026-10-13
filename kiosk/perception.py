"""B2 Perception: browser-sent JPEG frames -> `core.contracts.Landmarks`
(SYSTEM_DESIGN.md B2).

Uses MediaPipe Holistic in **LIVE_STREAM** mode, not VIDEO mode — `ml/extract_landmarks.py`
uses VIDEO mode because it processes a finite recorded file where every frame is
already known; here frames arrive one at a time from a live camera over the
WebSocket, and LIVE_STREAM mode is the Tasks API's mode for exactly that: it is
asynchronous (`detect_async`, a `result_callback`) rather than blocking per call,
which matters once frames are arriving faster than inference can keep up (the
whole reason B1's single-slot buffer exists).

Runs on every `perception_stride`th frame (1 = all of them; see kiosk/config.py for why).
MediaPipe is the largest single item in the latency budget (SYSTEM_DESIGN.md §4.1) and
on the Jetson's CPU it sustains about 10 frames a second.

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
import weakref
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


# Output streams that are legitimately absent on a frame: nobody's hands are up, or the
# person has stepped out of frame. `name.endswith("world_landmarks")` streams carry a
# LandmarkList, the rest a NormalizedLandmarkList.
_OPTIONAL_STREAMS = (
    "pose_landmarks", "pose_world_landmarks",
    "left_hand_landmarks", "left_hand_world_landmarks",
    "right_hand_landmarks", "right_hand_world_landmarks",
)


def fill_empty_streams(output_packets, make_empty):
    """Copy of MediaPipe's per-frame output packets with every empty optional stream
    replaced by `make_empty(stream_name)`, a packet holding an empty landmark list.

    Why: MediaPipe 0.10.18's holistic LIVE_STREAM callback checks only the *face* stream
    for emptiness, then reads the pose and both hand streams unconditionally. A person who
    is in view with their hands down is the normal state of a kiosk waiting for a visitor
    to sign, and it makes the C++ side hit `Check failed: holder_ != nullptr The packet is
    empty`, which is an abort: the whole backend process dies, not an exception we could
    catch. A filled stream reads as "nothing detected", which is the truth.
    """
    filled = dict(output_packets)
    for name in _OPTIONAL_STREAMS:
        packet = filled.get(name)
        if packet is not None and packet.is_empty():
            filled[name] = make_empty(name)
    return filled


def install_empty_packet_guard() -> bool:
    """Applies `fill_empty_streams` to the installed MediaPipe's result builder. Only for
    the 0.10.x line that has the bug (the Jetson ships 0.10.18); 1.x does not need it, and
    anything that cannot be imported is left alone. Returns whether the guard is in place."""
    try:
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.tasks.python.vision import holistic_landmarker as holistic
    except ImportError:
        return False
    if not mp.__version__.startswith("0.10."):
        return False
    if getattr(holistic, "_signoo_guarded", False):
        return True
    # The guard reaches into a private name. If a MediaPipe 0.10.x release renames it, say so
    # loudly and carry on unguarded, rather than failing to start a kiosk that may not need it.
    original = getattr(holistic, "_build_landmarker_result", None)
    packet_creator = getattr(holistic, "packet_creator", None)
    if original is None or packet_creator is None:
        logger.warning("mediapipe_empty_packet_guard_not_installed",
                       extra={"mediapipe": mp.__version__,
                              "why": "internal names changed; a hands-down frame may abort the backend"})
        return False

    def make_empty(name: str):
        proto = landmark_pb2.LandmarkList() if name.endswith("world_landmarks") \
            else landmark_pb2.NormalizedLandmarkList()
        return packet_creator.create_proto(proto)

    # `packets_callback` looks this name up in the module on every frame, so replacing the
    # module attribute is enough; nothing needs to be re-created.
    holistic._build_landmarker_result = lambda output_packets: original(
        fill_empty_streams(output_packets, make_empty))
    holistic._signoo_guarded = True
    logger.info("mediapipe_empty_packet_guard_installed", extra={"mediapipe": mp.__version__})
    return True


class _ResultNotifier:
    """Who to tell when a tracking result arrives. A separate small object, so that the
    tracker's callback holds this and not the perception: see the note on reference cycles
    where the callback is built."""

    def __init__(self) -> None:
        self.listener = None

    def fire(self) -> None:
        listener = self.listener
        if listener is None:
            return
        try:
            listener()
        except Exception:
            # This runs on MediaPipe's thread; an exception escaping here would land in
            # its C++ callback machinery. A listener that fails loses a notification.
            logger.exception("perception_listener_failed")


def _close_quietly(landmarker) -> None:
    try:
        landmarker.close()
    except Exception:
        logger.exception("landmarker_close_failed")


def _close_outside_the_collector(landmarker) -> None:
    """Runs when a `HolisticPerception` is freed without having been closed. It may be
    running inside the cycle collector, where closing deadlocks, so the close is handed to
    a thread of its own."""
    logger.warning("perception_dropped_without_close")
    threading.Thread(target=_close_quietly, args=(landmarker,), name="perception-close",
                     daemon=True).start()


class HolisticPerception:
    """One instance per active WebSocket session — MediaPipe tracking state must
    not bleed between visitors, same reasoning as `ml/extract_landmarks.py`'s "a
    fresh landmarker per clip, deliberately".
    """

    def __init__(self, stride: int = 1, ensure_model: bool = True):
        self.stride = max(1, stride)
        self._frame_index = 0
        self._result_buffer: SingleSlotBuffer[Landmarks] = SingleSlotBuffer()
        self._landmarker = None
        self._start_ms = None
        self._lock = threading.Lock()
        self._cv2 = None
        self.brightness: float | None = None   # 0 (black) .. 255 (white); None until a frame arrives
        self._notifier = _ResultNotifier()
        self._build(ensure_model)

    def set_listener(self, listener) -> None:
        """`listener()` is called, on MediaPipe's own thread, each time a tracking result
        has been put in the buffer; pass None to stop. It must only signal: the caller
        collects the result with `latest()` on its own thread.

        Without this a result sat in the buffer until the *next* frame happened to arrive
        and ask for it. That costs a frame of delay on every hand position, and when two
        results finish between two frames the older one is overwritten unread: measured
        2026-10-13, hand positions reached the browser 4 times a second from a tracker
        producing 7."""
        self._notifier.listener = listener

    @property
    def dropped_results(self) -> int:
        """Tracking results overwritten in the buffer before anyone read them."""
        return self._result_buffer.dropped

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
        install_empty_packet_guard()

        # The callback holds the buffer, not `self`. Holding `self` would make a cycle
        # (perception -> landmarker -> callback -> perception), so a perception that was
        # dropped without `close()` would wait for the garbage collector, and MediaPipe
        # deadlocks when its landmarker is freed that way, at whatever moment the collector
        # happens to run. Without the cycle it is freed the instant it is dropped.
        result_buffer = self._result_buffer
        notifier = self._notifier

        def _on_result(result, output_image, timestamp_ms):
            landmarks = Landmarks(
                ts_ms=timestamp_ms,
                pose=_points(result.pose_landmarks, 33),
                left=_points(result.left_hand_landmarks, 21),
                right=_points(result.right_hand_landmarks, 21),
            )
            result_buffer.put(landmarks)   # newest-wins, same B1 discipline
            notifier.fire()

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
        # Safety net for a perception that is dropped without `close()`. Measured: a landmarker
        # freed by Python's cycle collector deadlocks inside MediaPipe's own `__del__`, and the
        # collector can run at any moment, in any thread. The finalizer keeps the landmarker
        # alive (so the collector never frees it) and closes it on a thread of its own.
        self._finalizer = weakref.finalize(self, _close_outside_the_collector, self._landmarker)

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
        # FR-K5 lighting feedback: mean brightness of a sparse sample of the frame (every
        # 8th pixel each way, so it costs microseconds), smoothed over roughly ten frames so
        # one flash or one dark frame does not flip a warning on.
        brightness = float(image_bgr[::8, ::8].mean())
        self.brightness = brightness if self.brightness is None else 0.8 * self.brightness + 0.2 * brightness

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
        """Frees the landmarker. Safe to call twice, and never raises: it runs in cleanup
        paths, where an exception would skip whatever cleanup comes after it."""
        self._notifier.listener = None   # nobody is told about results from a closing tracker
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is None:
            return
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()   # closed here and now; the safety net must not close it again
        _close_quietly(landmarker)


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

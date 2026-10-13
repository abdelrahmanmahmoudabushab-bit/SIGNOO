"""The tracker's lifetime: started early, never leaked, never freed by the collector.

Found 2026-10-13: a MediaPipe landmarker that is freed by Python's cycle collector, rather
than closed, deadlocks inside MediaPipe's own `__del__`. It hung the test suite, and on a
kiosk it would hang the backend at whatever moment the collector next ran.
"""
import asyncio
import gc
import json
import time

import pytest

from kiosk.perception import HolisticPerception, PerceptionUnavailable


def _perception() -> HolisticPerception:
    try:
        return HolisticPerception(stride=1, ensure_model=False)
    except PerceptionUnavailable as exc:
        pytest.skip(f"MediaPipe or its model is not available here: {exc}")


def test_closing_twice_is_safe():
    perception = _perception()
    perception.close()
    perception.close()


def test_a_tracker_dropped_without_close_does_not_hang_the_collector():
    """The safety net: dropped inside a reference cycle, then collected. Before the fix
    this call never returned."""
    perception = _perception()
    cycle = [perception]
    cycle.append(cycle)
    del perception, cycle

    gc.collect()

    time.sleep(1.0)        # the hand-off close runs on its own thread
    gc.collect()           # and what it leaves behind is safe to collect too


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        pass


def _selectable_language(kiosk_main) -> str:
    for code, pack in kiosk_main.registry.packs.items():
        if pack.selectable:
            return code
    pytest.skip("no language with a loadable model in this checkout")


def test_the_tracker_starts_when_a_language_is_picked_not_at_consent():
    """So that signing can begin the moment the visitor agrees, instead of after the
    seconds MediaPipe takes to start."""
    import kiosk.main as kiosk_main
    _perception().close()                       # skip here if MediaPipe cannot run at all

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        await conn.handle({"type": "language", "lang": _selectable_language(kiosk_main)})
        assert conn._perception_task is not None
        await conn._perception_task
        started_before_consent = conn.perception is not None
        tracker = conn.perception
        await conn.handle({"type": "consent", "processing": True, "retention": False})
        same_tracker_after_consent = conn.perception is tracker
        conn.shut_down()
        return started_before_consent, same_tracker_after_consent

    assert asyncio.run(visit()) == (True, True)


def test_a_tracker_that_finishes_starting_after_the_page_left_is_closed_not_attached():
    import kiosk.main as kiosk_main
    _perception().close()

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        await conn.handle({"type": "language", "lang": _selectable_language(kiosk_main)})
        conn.shut_down()                        # the page goes away while the tracker is starting
        await conn._perception_task
        return conn.perception

    assert asyncio.run(visit()) is None


def test_a_tracker_started_for_a_visit_that_ended_is_not_handed_to_the_next_visitor():
    import kiosk.main as kiosk_main
    _perception().close()

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        await conn.handle({"type": "language", "lang": _selectable_language(kiosk_main)})
        conn._reset_for_next_visitor()          # the visit ends before the tracker is ready
        await conn._perception_task
        leftover = conn.perception
        conn.shut_down()
        return leftover

    assert asyncio.run(visit()) is None


def test_no_frame_reaches_the_tracker_before_consent():
    """Starting the tracker early must not mean watching early."""
    import kiosk.main as kiosk_main
    _perception().close()

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        await conn.handle({"type": "language", "lang": _selectable_language(kiosk_main)})
        await conn._perception_task
        submitted = []
        conn.perception.submit_frame = lambda jpeg, ts: submitted.append(ts)
        # A real, tiny JPEG header is enough: the frame is refused on session state first.
        import base64
        import cv2
        import numpy as np
        ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        assert ok
        await conn.handle({"type": "frame", "ts_ms": 1, "data": base64.b64encode(jpeg.tobytes()).decode()})
        conn.shut_down()
        return submitted

    assert asyncio.run(visit()) == []

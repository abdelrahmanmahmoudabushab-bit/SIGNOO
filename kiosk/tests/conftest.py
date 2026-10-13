"""The kiosk test suite runs against the synthetic-trained `liu` checkpoint — there is
no real signer data yet — so it opts in to the setting a deployment leaves off
(kiosk/config.py::allow_synthetic). Set before kiosk.config is first imported, because
CONFIG is built at import time."""
import os

import gc
import sys

import pytest

os.environ.setdefault("SIGNOO_ALLOW_SYNTHETIC", "1")


@pytest.fixture(autouse=True)
def _close_every_tracker_a_test_started():
    """Tests build `KioskConnection`s directly, without the WebSocket handler whose `finally`
    shuts them down. Picking a language starts the tracker, and a MediaPipe landmarker that
    is never closed deadlocks when the garbage collector frees it later, in the middle of
    some other test. The connection itself is gone by the time a test returns, but its
    tracker is not (it sits in a reference cycle with its own result callback), so the
    trackers are what is looked for and closed here."""
    yield
    perception = sys.modules.get("kiosk.perception")
    if perception is None:
        return
    for obj in gc.get_objects():
        if isinstance(obj, perception.HolisticPerception):
            obj.close()

"""B1 Capture: single-slot, newest-wins buffer — SYSTEM_DESIGN.md B1 is explicit
this must drop frames rather than queue them; this test is the proof.
"""
from kiosk.buffers import SingleSlotBuffer


def test_newest_wins_older_dropped():
    buf: SingleSlotBuffer[int] = SingleSlotBuffer()
    buf.put(1)
    buf.put(2)
    buf.put(3)                     # 1 and 2 are gone — never queued
    assert buf.get() == 3
    assert buf.dropped == 2


def test_get_empties_the_slot():
    buf: SingleSlotBuffer[str] = SingleSlotBuffer()
    assert buf.get() is None       # nothing put yet
    buf.put("frame")
    assert buf.get() == "frame"
    assert buf.get() is None       # consumed — not still sitting there

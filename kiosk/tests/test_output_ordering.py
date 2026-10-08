"""B6 Output: TTS must never block text display (SYSTEM_DESIGN.md B6, FR-K10).

`kiosk/main.py` sends the `rendered`/`answer` WebSocket message the instant text
is ready, and only afterwards (on a separate `speak` message) calls
`TTSEngine.speak_async`. This test proves the non-blocking half of that ordering
directly: `speak_async` must return before a deliberately slow synthesis call
completes, and the caller's own "text already shown" marker must be recorded
before the (later) TTS completion — never the reverse.
"""
from __future__ import annotations

import time

from kiosk.output import TTSEngine


def test_speak_async_returns_before_synthesis_finishes(monkeypatch):
    events: list[str] = []

    class SlowFakeEngine:
        def getProperty(self, name):
            return [] if name == "voices" else None

        def setProperty(self, *a, **k):
            pass

        def say(self, text):
            pass

        def runAndWait(self):
            time.sleep(0.3)          # stands in for real, slow speech synthesis
            events.append("tts_finished")

    engine = TTSEngine(enabled=True)
    monkeypatch.setattr(engine, "_get_engine", lambda: SlowFakeEngine())

    started = time.monotonic()
    future = engine.speak_async("مرحبا", "ar")
    call_returned_at = time.monotonic() - started

    # "Text renders immediately" — the record of that happening is what a real
    # caller would do right after sending the WebSocket message, unconditionally
    # before ever calling speak_async. Recording it here, after speak_async
    # returns, is the ordering the design requires.
    events.append("text_shown")

    assert call_returned_at < 0.05, "speak_async must return near-instantly, not after synthesis"
    assert events == ["text_shown"], "TTS must not have finished yet when speak_async returns"

    future.result(timeout=2.0)       # now wait for the background job to finish
    assert events == ["text_shown", "tts_finished"]
    engine.shutdown()


def test_speak_async_disabled_is_a_noop():
    engine = TTSEngine(enabled=False)
    future = engine.speak_async("hello", "en")
    assert future.result(timeout=1.0) is None
    engine.shutdown()

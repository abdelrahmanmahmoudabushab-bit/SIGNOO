"""B6 Output: text renders immediately, speech is strictly outside the latency
budget and never blocks it (SYSTEM_DESIGN.md B6, FR-K10).

The WebSocket handler in `kiosk/main.py` sends the `Rendered`/`Answer` JSON message
the instant it has one — that *is* "text renders immediately"; nothing in this
module is on that path. `TTSEngine.speak_async` is the separate, async call: it
hands the utterance to a single background worker thread and returns a `Future`
without waiting for synthesis, so a slow or hung TTS engine can never delay the
text the visitor already sees (`kiosk/tests/test_output_ordering.py` proves the
ordering with a deliberately slow fake engine).
"""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger("signoo.kiosk.output")

_LANG_VOICE_HINT = {"ar": "ar", "en": "en"}


class TTSEngine:
    """Offline OS-provided synthesis via pyttsx3 — a drop-in stand-in for Piper
    (SYSTEM_DESIGN.md §5's technology table names Piper for the real device; the
    interface here is the same call shape, so swapping the implementation later
    does not touch any caller).

    One worker thread, because pyttsx3 engines are not safe to drive concurrently;
    a second `speak_async` call while one is in flight simply queues behind it,
    which is the correct behaviour for a kiosk that only ever addresses one visitor.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signoo-tts")
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
        return self._engine

    def speak_async(self, text: str, lang: str) -> Future:
        """Fires synthesis in the background and returns immediately.

        Returns a `Future` so tests and callers can observe completion without the
        caller ever being made to wait for it on the happy path.
        """
        if not self.enabled or not text.strip():
            future: Future = Future()
            future.set_result(None)
            return future

        def _job():
            try:
                engine = self._get_engine()
                for voice in engine.getProperty("voices") or []:
                    hint = _LANG_VOICE_HINT.get(lang, lang)
                    if hint in (voice.id or "").lower() or hint in (voice.languages or [""])[0].lower():
                        engine.setProperty("voice", voice.id)
                        break
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as exc:
                # TTS is an enhancement (SYSTEM_DESIGN.md §6: "Memory pressure —
                # unload the answering model first, recognition is the core
                # service"). A synthesis failure is expected-and-recoverable: log
                # it, never raise, never touch the text already shown.
                logger.debug("tts_failed", extra={"lang": lang, "error": str(exc)})
                return False

        # submit() returns immediately; the job runs on the worker thread. This is
        # the whole proof of "never blocks display" — the caller gets a Future back
        # without synthesis having started, let alone finished.
        return self._executor.submit(_job)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

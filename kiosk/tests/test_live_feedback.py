"""What the visitor is told while signing (FR-K5, NFR-X1) and what the kiosk reports about
itself (B9): sign states, lighting, "I couldn't see your hands", live stats, and the rule
that a language not trained on real signing never reads as live.
"""
import asyncio
import base64
import json

import numpy as np
import pytest

from core.contracts import Landmarks
from kiosk.config import CONFIG, REPO_ROOT, SEGMENTATION_DEFAULTS
from kiosk.segmentation import Segmenter
from kiosk.session import LiveStats, SessionState, read_memory_mb, telemetry_snapshot


def _frame(ts_ms: int, hand: bool, shift: float = 0.0) -> Landmarks:
    """One tracked frame: a body, and a left hand (moved by `shift`) or no hands at all."""
    left = np.full((21, 3), 0.4 + shift, dtype=np.float32) if hand else None
    return Landmarks(ts_ms=ts_ms, pose=np.zeros((33, 3), dtype=np.float32), left=left, right=None)


def _feed(segmenter: Segmenter, pattern: list[bool], step_ms: int = 100):
    """Feeds one frame per entry (True = a hand is visible, moving a little each time)."""
    utterances = []
    for i, hand in enumerate(pattern):
        utterance = segmenter.feed(_frame(i * step_ms, hand, shift=0.05 * (i % 2)))
        if utterance is not None:
            utterances.append(utterance)
    return utterances


# ── B3: holes in a sign are counted, the end of a sign is not ───────────────────────

def test_a_gap_inside_a_word_is_counted_once_the_hands_come_back():
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)
    # six frames with a hand, two lost frames between each, then the hands leave for good
    pattern = [True, False, False] * 5 + [True] + [False] * 10

    (utterance,) = _feed(segmenter, pattern)

    assert len(utterance.landmarks) == 6
    assert utterance.frames_without_hands == 10        # five holes of two frames


def test_hands_leaving_at_the_end_of_a_sign_is_not_a_hole():
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)

    (utterance,) = _feed(segmenter, [True] * 8 + [False] * 10)

    assert len(utterance.landmarks) == 8
    assert utterance.frames_without_hands == 0


def test_the_count_does_not_carry_over_into_the_next_word():
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)
    holey = [True, False, False] * 5 + [True] + [False] * 10
    clean = [True] * 8 + [False] * 10

    first, second = _feed(segmenter, holey + clean)

    assert first.frames_without_hands == 10
    assert second.frames_without_hands == 0


def test_buffering_is_true_only_while_a_word_is_being_collected():
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)
    assert not segmenter.buffering                      # nobody there

    segmenter.feed(_frame(0, hand=True))
    assert segmenter.buffering                          # hands up: a word has started

    for i in range(1, 12):
        segmenter.feed(_frame(i * 100, hand=False))
    assert not segmenter.buffering                      # hands gone long enough: back to idle


# ── B3: how long a word is, is measured in time, not in frames ──────────────────────
# Found 2026-10-13: "at least 6 frames" is 0.2 s when the tracker runs at 30 frames a
# second and 0.6 s at the Jetson's 10, where ordinary quick signs were dropped silently.

def test_a_quick_sign_survives_a_slow_tracker():
    """Half a second of signing is five frames at 10 tracked frames a second."""
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)

    utterances = _feed(segmenter, [True] * 5 + [False] * 10, step_ms=100)

    assert [len(u.landmarks) for u in utterances] == [5]


def test_a_tracking_flicker_is_still_not_a_word_at_any_tracker_speed():
    slow = Segmenter(**SEGMENTATION_DEFAULTS)
    assert _feed(slow, [True] * 2 + [False] * 10, step_ms=100) == []       # two frames, 0.1 s

    fast = Segmenter(**SEGMENTATION_DEFAULTS)
    assert _feed(fast, [True] * 4 + [False] * 30, step_ms=33) == []        # four frames, but only 0.1 s


def test_the_same_sign_is_one_word_whether_tracked_fast_or_slowly():
    """Two seconds of signing: 60 frames at 30 a second, 20 at 10. The old frame limit
    (45) cut the fast one in two."""
    fast = _feed(Segmenter(**SEGMENTATION_DEFAULTS), [True] * 60 + [False] * 30, step_ms=33)
    slow = _feed(Segmenter(**SEGMENTATION_DEFAULTS), [True] * 20 + [False] * 10, step_ms=100)

    assert [len(u.landmarks) for u in fast] == [60]
    assert [len(u.landmarks) for u in slow] == [20]


def test_hands_that_never_pause_are_closed_after_the_time_limit():
    segmenter = Segmenter(**SEGMENTATION_DEFAULTS)
    seconds = CONFIG.max_utterance_ms / 1000

    utterances = _feed(segmenter, [True] * int(seconds * 10 + 5), step_ms=100)

    assert len(utterances) == 1
    assert utterances[0].t_end - utterances[0].t_start == CONFIG.max_utterance_ms


# ── the connection: what is sent to the screen ──────────────────────────────────────

class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        pass


class _ScriptedPerception:
    """Stands in for MediaPipe: each submitted frame yields the next scripted result."""

    def __init__(self, pattern: list[bool], brightness: float = 120.0):
        self._pattern = list(pattern)
        self._index = 0
        self._latest = None
        self.brightness = brightness

    def submit_frame(self, jpeg: bytes, ts_ms: int) -> bool:
        hand = self._pattern[self._index]
        self._latest = _frame(ts_ms, hand, shift=0.05 * (self._index % 2))
        self._index += 1
        return True

    def latest(self):
        latest, self._latest = self._latest, None
        return latest

    def set_listener(self, listener) -> None:
        self.listener = listener        # results here are ready at once; nothing to wake

    def close(self) -> None:
        pass


class _SlowPerception(_ScriptedPerception):
    """Like the real tracker: a frame's result is not ready when `submit_frame` returns. It
    arrives a little later, on another thread, which then calls the listener."""

    def submit_frame(self, jpeg: bytes, ts_ms: int) -> bool:
        import threading
        hand = self._pattern[self._index]
        result = _frame(ts_ms, hand, shift=0.05 * (self._index % 2))
        self._index += 1

        def deliver():
            self._latest = result
            self.listener()

        threading.Timer(0.05, deliver).start()
        return True


def _jpeg_b64() -> str:
    import cv2
    ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(jpeg.tobytes()).decode()


def _run_visit(pattern: list[bool], brightness: float = 120.0) -> list[dict]:
    """A consented visit on the default engine, fed `pattern`; returns every message sent."""
    import kiosk.main as kiosk_main

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        conn.perception = _ScriptedPerception(pattern, brightness)
        conn.session.consent_processing = True
        conn.session.state = SessionState.ACTIVE
        data = _jpeg_b64()
        for i in range(len(pattern)):
            await conn.handle({"type": "frame", "ts_ms": i * 100, "data": data})
        conn.perception = None
        conn.shut_down()
        return conn.ws.sent

    return asyncio.run(visit())


def _of(messages: list[dict], kind: str) -> list[dict]:
    return [m for m in messages if m["type"] == kind]


def test_a_sign_that_was_mostly_not_seen_is_refused_with_its_reason_and_not_classified(monkeypatch):
    import kiosk.main as kiosk_main
    classified = []
    recognizer = kiosk_main.registry.get(kiosk_main.DEFAULT_LANGUAGE).recognizer
    monkeypatch.setattr(recognizer, "classify", lambda utterance: classified.append(utterance))

    sent = _run_visit([True, False, False] * 5 + [True] + [False] * 10)

    (hypothesis,) = _of(sent, "hypothesis")
    assert (hypothesis["status"], hypothesis["reason"], hypothesis["gloss"]) == ("rejected", "hands_lost", None)
    assert classified == []                             # a guess was never even computed


def test_a_cleanly_seen_sign_is_classified_and_the_screen_is_told_each_stage():
    sent = _run_visit([True] * 8 + [False] * 10)

    (hypothesis,) = _of(sent, "hypothesis")
    assert hypothesis.get("reason") != "hands_lost"
    assert hypothesis["status"] in {"committed", "candidates", "rejected"}
    assert [m["state"] for m in _of(sent, "signing")] == ["signing", "recognising", "watching"]
    # "reading your sign" is on screen before the result is
    assert sent.index({"type": "signing", "state": "recognising"}) < sent.index(hypothesis)


def test_a_tracking_result_reaches_the_screen_without_waiting_for_the_next_frame():
    """Found 2026-10-13: results were only collected when the next frame arrived. With the
    browser waiting for a result before sending another frame, each side waited for the
    other until a timeout, and every second result was overwritten unread."""
    import kiosk.main as kiosk_main

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        conn.perception = _SlowPerception([True])
        conn.session.consent_processing = True
        conn.session.state = SessionState.ACTIVE
        await conn.handle({"type": "frame", "ts_ms": 0, "data": _jpeg_b64()})
        sent_at_once = len(_of(conn.ws.sent, "landmarks"))
        await asyncio.sleep(0.4)                # no second frame is ever sent
        conn.perception = None
        conn.shut_down()
        return sent_at_once, len(_of(conn.ws.sent, "landmarks"))

    assert asyncio.run(visit()) == (0, 1)


def test_every_result_of_a_steady_stream_is_handled_none_overwritten():
    """Ten frames, each answered 50 ms later while the next is already on its way."""
    import kiosk.main as kiosk_main

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        conn.perception = _SlowPerception([True] * 10)
        conn.session.consent_processing = True
        conn.session.state = SessionState.ACTIVE
        data = _jpeg_b64()
        for i in range(10):
            await conn.handle({"type": "frame", "ts_ms": i * 100, "data": data})
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)
        conn.perception = None
        conn.shut_down()
        return len(_of(conn.ws.sent, "landmarks"))

    assert asyncio.run(visit()) == 10


def test_status_messages_are_sent_when_they_change_not_with_every_frame():
    sent = _run_visit([True] * 5)

    assert len(_of(sent, "landmarks")) == 5             # the marker positions do come every frame
    assert _of(sent, "framing") == [{"type": "framing", "ok": True, "reason": None}]
    assert _of(sent, "signing") == [{"type": "signing", "state": "signing"}]


@pytest.mark.parametrize("brightness, reason", [
    (CONFIG.light_too_dark - 1, "too_dark"),
    (CONFIG.light_too_bright + 1, "too_bright"),
])
def test_lighting_the_camera_cannot_work_in_is_reported(brightness, reason):
    sent = _run_visit([True] * 3, brightness=brightness)

    assert _of(sent, "framing") == [{"type": "framing", "ok": False, "reason": reason}]


def test_a_recognition_never_puts_who_or_what_into_the_live_stats():
    import kiosk.main as kiosk_main
    before = kiosk_main.LIVE.snapshot()["outcomes"]

    _run_visit([True] * 8 + [False] * 10)

    after = kiosk_main.LIVE.snapshot()
    assert sum(after["outcomes"].values()) == sum(before.values()) + 1
    assert set(after) == {"tracker_fps", "frame_ms", "classify_ms", "outcomes"}   # counts and timings only
    assert set(after["outcomes"]) <= {"committed", "candidates", "rejected", "hands_lost"}


def _recognition_log(caplog, retention: bool) -> dict:
    """The `recognition` log record of one cleanly signed word, with or without retention."""
    import logging
    import kiosk.main as kiosk_main

    async def visit():
        conn = kiosk_main.KioskConnection(_FakeWS())
        pattern = [True] * 8 + [False] * 10
        conn.perception = _ScriptedPerception(pattern)
        conn.session.consent_processing = True
        conn.session.consent_retention = retention
        conn.session.state = SessionState.ACTIVE
        data = _jpeg_b64()
        for i in range(len(pattern)):
            await conn.handle({"type": "frame", "ts_ms": i * 100, "data": data})
        conn.perception = None
        conn.shut_down()

    with caplog.at_level(logging.INFO, logger="signoo.kiosk.main"):
        asyncio.run(visit())
    (record,) = [r for r in caplog.records if r.getMessage() == "recognition"]
    return record.__dict__


def test_what_a_visitor_signed_is_not_written_to_the_log_unless_they_agreed_to_retention(caplog):
    """NFR-S3: nothing of a session outlives it without the retention opt-in. The words are
    session content; the confidences are not."""
    record = _recognition_log(caplog, retention=False)

    assert record["gloss"] is None
    assert all(word is None for word, _ in record["top3"])
    assert record["words_logged"] is False
    assert len(record["top3"]) > 0 and all(0.0 <= c <= 1.0 for _, c in record["top3"])   # still tunable


def test_with_retention_the_log_names_the_words_so_a_session_can_be_judged(caplog):
    record = _recognition_log(caplog, retention=True)

    assert record["words_logged"] is True
    assert all(isinstance(word, str) for word, _ in record["top3"])


# ── B9: live stats ──────────────────────────────────────────────────────────────────

def test_tracker_rate_is_the_average_over_the_last_few_seconds():
    now = [100.0]
    stats = LiveStats(clock=lambda: now[0])
    for _ in range(50):                                 # ten results a second for five seconds
        now[0] += 0.1
        stats.note_tracker_result()

    assert stats.snapshot()["tracker_fps"] == 10.0

    now[0] += 60                                        # then nobody in front of the camera
    assert stats.snapshot()["tracker_fps"] is None      # "no visitor", not "stalled at 0"


def test_stage_timings_and_outcomes_are_reported():
    stats = LiveStats()
    stats.note_frame(0.004)
    stats.note_frame(0.006)
    stats.note_outcome("committed", classify_seconds=0.0123)
    stats.note_outcome("hands_lost")

    snapshot = stats.snapshot()

    assert snapshot["frame_ms"] == 5.0
    assert snapshot["classify_ms"] == 12.3
    assert snapshot["outcomes"] == {"committed": 1, "hands_lost": 1}


def test_memory_is_read_from_the_kernel_and_unknown_where_there_is_none(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t  524288 kB\nThreads:\t9\n")

    assert read_memory_mb(str(status)) == 512.0
    assert read_memory_mb(str(tmp_path / "missing")) is None
    reported = telemetry_snapshot()["memory_mb"]
    assert reported == "unavailable" or isinstance(reported, float)   # never a made-up number


def test_health_reports_the_pipeline_and_each_models_start_up_pass():
    from fastapi.testclient import TestClient
    import kiosk.main as kiosk_main

    with TestClient(kiosk_main.app) as client:          # entering runs the start-up warm-up
        health = client.get("/health").json()

    assert {"tracker_fps", "frame_ms", "classify_ms", "outcomes", "memory_mb"} <= set(health["telemetry"])
    loaded = {code for code, pack in kiosk_main.registry.packs.items() if pack.recognizer is not None}
    assert set(health["warm_up_s"]) == loaded
    assert all(isinstance(seconds, float) for seconds in health["warm_up_s"].values())   # none failed


# ── kiosk/live_check.py: how a clip played to the kiosk is scored ───────────────────

def _hyp(status: str, confidence: float, gloss=None, alternatives=(), reason=None) -> dict:
    return {"type": "hypothesis", "status": status, "confidence": confidence, "gloss": gloss,
            "alternatives": [list(a) for a in alternatives], "reason": reason}


@pytest.mark.parametrize("hypotheses, outcome", [
    ([], "no_word"),                                             # nothing reached the screen at all
    ([_hyp("committed", 0.9, "dog")], "committed_correct"),
    ([_hyp("committed", 0.9, "cat")], "committed_WRONG"),
    ([_hyp("candidates", 0.5, alternatives=[("cat", 0.5), ("dog", 0.3)])], "candidates_with_truth"),
    ([_hyp("candidates", 0.5, alternatives=[("cat", 0.5), ("cow", 0.3)])], "candidates"),
    ([_hyp("rejected", 0.1, reason="not_recognised")], "rejected"),
    ([_hyp("rejected", 0.0, reason="hands_lost")], "hands_lost"),
    # two results for one clip: the visitor is judged by the more confident one
    ([_hyp("rejected", 0.1, reason="not_recognised"), _hyp("committed", 0.9, "dog")], "committed_correct"),
])
def test_a_played_clip_is_scored_by_what_the_visitor_would_have_seen(hypotheses, outcome):
    from kiosk.live_check import judge

    assert judge("dog", hypotheses) == outcome


def test_the_live_check_paces_frames_exactly_as_the_browser_does():
    """It is only a stand-in for the browser while its numbers match assets/app.js."""
    import re
    from kiosk import live_check

    script = (REPO_ROOT / "assets" / "app.js").read_text(encoding="utf-8")

    def constant(name: str) -> float:
        return float(re.search(rf"const {name} = ([0-9.]+);", script).group(1))

    assert live_check.FRAME_INTERVAL_S == constant("FRAME_INTERVAL_MS") / 1000
    assert live_check.FRAME_REPLY_TIMEOUT_S == constant("FRAME_REPLY_TIMEOUT_MS") / 1000
    assert live_check.SEND_MAX_SIDE == constant("SEND_MAX_SIDE")
    assert f"toDataURL('image/jpeg', {live_check.JPEG_QUALITY / 100:g})" in script


# ── the non-negotiable: not trained on real signing means "coming soon" ─────────────

def test_no_language_trained_only_on_generated_data_is_marked_live_or_pilot():
    """CLAUDE.md: "A language card that isn't trained on real data says 'coming soon.'"
    The code already refuses to offer such a language (`LanguagePack.selectable`); this
    keeps the data from claiming otherwise."""
    entries = json.loads((REPO_ROOT / "core" / "languages.json").read_text(encoding="utf-8"))

    wrong = [e["code"] for e in entries if not e.get("real_data") and e["status"] != "coming_soon"]

    assert wrong == []


def test_every_card_in_the_page_starts_disabled_until_the_backend_enables_it():
    """If GET /languages cannot answer, the page must not leave a card pressable."""
    import re
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")

    cards = re.findall(r'<button class="lang-card"[^>]*>', html)

    assert cards and all("disabled" in card for card in cards)

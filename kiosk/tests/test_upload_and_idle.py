"""B10 -> B11 store-and-forward upload, anonymous signer ids, and the idle timeout
(FR-K13): camera frames and pings must not keep a walked-away visitor's session alive."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from kiosk.data_capture import CapturedItem, SessionCapture, flush_pending_uploads, write_pending_upload


def _capture(session_id: str = "11111111-2222-3333-4444-555555555555") -> SessionCapture:
    capture = SessionCapture(
        session_id=session_id, kiosk_id="kiosk-test", language="liu",
        consent_retention=True, started_at="2026-10-10T10:00:00+00:00",
    )
    capture.add(CapturedItem("u1", [[0.0] * 3], "marhaba", 0.9))
    capture.ended_at = "2026-10-10T10:01:00+00:00"
    return capture


class _Server:
    """Loopback stand-in for server/'s /ingest/sessions."""

    def __init__(self, status: int):
        outer = self
        outer.bodies: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.bodies.append(json.loads(self.rfile.read(length)))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_every_item_carries_an_anonymous_per_visit_signer_id():
    payload = _capture().to_payload()
    assert payload["items"][0]["signer_id"] == "visitor-11111111"
    other = _capture("99999999-0000-0000-0000-000000000000").to_payload()
    assert other["items"][0]["signer_id"] != payload["items"][0]["signer_id"]


def test_delivered_session_is_removed_from_the_queue(tmp_path):
    write_pending_upload(_capture(), tmp_path)
    server = _Server(201)
    try:
        assert flush_pending_uploads(tmp_path, server.url) == 1
    finally:
        server.close()
    assert list(tmp_path.glob("*.json")) == []
    assert server.bodies[0]["items"][0]["signer_id"] == "visitor-11111111"


def test_unreachable_server_keeps_the_session_queued(tmp_path):
    write_pending_upload(_capture(), tmp_path)
    assert flush_pending_uploads(tmp_path, "http://127.0.0.1:9", timeout_s=1.0) == 0
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_rejected_session_moves_aside_and_does_not_block_the_next(tmp_path):
    """A 4xx the server will never reconsider leaves the queue, but is kept (one file per
    session id, so NFR-S6 deletion can still find it) instead of being retried forever."""
    pending = tmp_path / "pending"
    write_pending_upload(_capture("aaaaaaaa-0000-0000-0000-000000000000"), pending)
    write_pending_upload(_capture("bbbbbbbb-0000-0000-0000-000000000000"), pending)
    server = _Server(422)
    try:
        assert flush_pending_uploads(pending, server.url) == 0
    finally:
        server.close()
    assert len(server.bodies) == 2           # both were attempted
    assert list(pending.glob("*.json")) == []
    kept = sorted(p.name for p in (tmp_path / "rejected_uploads").glob("*.json"))
    assert kept == ["aaaaaaaa-0000-0000-0000-000000000000.json", "bbbbbbbb-0000-0000-0000-000000000000.json"]


def test_server_error_keeps_the_session_queued_for_retry(tmp_path):
    """5xx and 'try later' refusals (token, rate limit) say nothing about the file."""
    for status in (500, 401, 429):
        pending = tmp_path / str(status)
        write_pending_upload(_capture(), pending)
        server = _Server(status)
        try:
            assert flush_pending_uploads(pending, server.url) == 0
        finally:
            server.close()
        assert len(list(pending.glob("*.json"))) == 1
        assert not (tmp_path / "rejected_uploads").exists()


def test_blank_server_url_sends_nothing(tmp_path):
    write_pending_upload(_capture(), tmp_path)
    assert flush_pending_uploads(tmp_path, "") == 0
    assert len(list(tmp_path.glob("*.json"))) == 1


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _connection():
    from kiosk.main import KioskConnection
    conn = KioskConnection(_FakeWS())
    conn.session.last_activity_at = time.time() - 30
    return conn


def test_ping_and_frames_do_not_reset_the_idle_clock():
    conn = _connection()
    before = conn.session.last_activity_at
    asyncio.run(conn.handle({"type": "ping"}))
    asyncio.run(conn.handle({"type": "frame", "ts_ms": 1, "data": ""}))
    assert conn.session.last_activity_at == before


def test_real_interaction_resets_the_idle_clock():
    conn = _connection()
    before = conn.session.last_activity_at
    asyncio.run(conn.handle({"type": "control", "action": "extend"}))
    assert conn.session.last_activity_at > before


def test_idle_active_session_is_ended_and_reset():
    from kiosk.config import CONFIG
    from kiosk.session import SessionState

    conn = _connection()
    old_id = conn.session.correlation_id
    conn.session.state = SessionState.ACTIVE
    conn.session.last_activity_at = time.time() - CONFIG.idle_timeout_s - 1
    asyncio.run(conn.check_idle())
    assert conn.session.correlation_id != old_id          # a fresh session for the next visitor
    assert conn.session.state == SessionState.ATTRACT
    assert any(m["type"] == "session_reset" for m in conn.ws.sent)

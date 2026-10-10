"""B9 -> B15 heartbeat: registers once, then reports; and the kiosk token travels with
every call the server protects."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from kiosk.data_capture import CapturedItem, SessionCapture, flush_pending_uploads, write_pending_upload
from kiosk.fleet import FleetReporter
from kiosk.session import read_temperature_c


class _Server:
    def __init__(self, status: int = 200):
        outer = self
        outer.calls: list[tuple[str, dict, dict]] = []
        outer.status = status

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append((self.path, body, dict(self.headers)))
                self.send_response(outer.status)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_first_tick_registers_then_heartbeats_with_the_token():
    server = _Server()
    try:
        reporter = FleetReporter(server.url, "kiosk-7", token="kiosk-secret")
        assert reporter.tick() is True
        assert reporter.tick() is True
    finally:
        server.close()
    paths = [call[0] for call in server.calls]
    assert paths == ["/fleet/kiosks/register", "/fleet/kiosks/kiosk-7/heartbeat",
                     "/fleet/kiosks/kiosk-7/heartbeat"]          # registered once, not every tick
    assert all(call[2].get("X-Kiosk-Token") == "kiosk-secret" for call in server.calls)
    assert isinstance(server.calls[1][1]["uptime_s"], float)


def test_an_unreachable_server_is_not_an_error_and_retries_registration():
    reporter = FleetReporter("http://127.0.0.1:9", "kiosk-7", timeout_s=1.0)
    assert reporter.tick() is False
    assert reporter.registered is False


def test_a_failed_heartbeat_forces_re_registration():
    server = _Server()
    try:
        reporter = FleetReporter(server.url, "kiosk-7")
        reporter.tick()
        server.status = 500
        assert reporter.tick() is False
    finally:
        server.close()
    assert reporter.registered is False


def test_temperature_is_the_hottest_zone_in_celsius(tmp_path):
    for name, millidegrees in (("thermal_zone0", "45000"), ("thermal_zone1", "61500")):
        zone = tmp_path / name
        zone.mkdir()
        (zone / "temp").write_text(millidegrees)
    assert read_temperature_c(str(tmp_path)) == 61.5


def test_no_thermal_zones_means_unknown_not_zero(tmp_path):
    assert read_temperature_c(str(tmp_path)) is None


def test_uploads_carry_the_kiosk_token(tmp_path):
    capture = SessionCapture("11111111-2222-3333-4444-555555555555", "k", "liu", True, "t")
    capture.add(CapturedItem("u", [[0.0]], "marhaba", 0.9))
    capture.ended_at = "t"
    write_pending_upload(capture, tmp_path)
    server = _Server(201)
    try:
        assert flush_pending_uploads(tmp_path, server.url, token="kiosk-secret") == 1
    finally:
        server.close()
    assert server.calls[0][2].get("X-Kiosk-Token") == "kiosk-secret"

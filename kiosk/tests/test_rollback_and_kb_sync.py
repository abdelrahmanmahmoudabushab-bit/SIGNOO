"""A rollback has to reach the kiosk, a language trained on generated data must not be
offered to a visitor, and a knowledge base edited on the server must reach the kiosk
without ever being able to break the one it is running."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kiosk.kb_sync import KnowledgeSync
from kiosk.languages import LanguageRegistry
from kiosk.model_sync import STATE_FILE
from kiosk.tests.test_languages import _world
from kiosk.tests.test_model_sync import _answer, _files, _new_model, _setup


# ── rollback: the kiosk leaves a model the server withdrew ──────────────────────

def test_a_withdrawn_model_is_replaced_by_the_factory_one(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        assert sync.tick()["asl"] == "installed"
        assert _answer(registry) == "cat"

        server.responses.clear()                     # the server rolled v2 back: nothing assigned now
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes["asl"] == "reverted"
    assert _answer(registry) == "dog"                # the shipped model answers again
    assert registry.get("asl").model_version is None
    assert json.loads((synced / STATE_FILE).read_text(encoding="utf-8")) == {}
    assert _files(synced) == [STATE_FILE]            # the withdrawn checkpoint is gone, not left to be reloaded


def test_a_kiosk_that_was_never_given_a_model_is_left_alone_by_an_empty_answer(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    try:
        outcomes = sync.tick()
    finally:
        server.close()
    assert set(outcomes.values()) == {"unassigned"}
    assert _answer(registry) == "dog"


def test_a_revert_that_cannot_complete_keeps_the_model_it_has(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        sync.tick()
        server.responses.clear()
        registry.get("asl").factory_model = tmp_path / "models" / "deleted.pt"   # the shipped file is gone
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes["asl"] == "failed"
    assert _answer(registry) == "cat"                # still serving: no model is worse than a pulled one
    assert registry.get("asl").model_version == "v2"
    assert "asl" in json.loads((synced / STATE_FILE).read_text(encoding="utf-8"))   # a restart is not told it reverted


# ── a language trained on generated data is not offered ─────────────────────────

def test_a_synthetic_only_language_is_not_selectable_unless_allowed(tmp_path):
    pack = _world(tmp_path).get("liu")               # real_data: false in the fixture
    assert pack.ready and pack.real_data is False

    pack.allow_synthetic = False
    assert not pack.selectable
    summary = pack.summary()
    assert summary["ready"] is False and summary["model_loaded"] is True    # the card stays "coming soon"

    pack.allow_synthetic = True
    assert pack.selectable and pack.summary()["ready"] is True


def test_a_language_trained_on_real_people_is_selectable_without_the_override(tmp_path):
    pack = _world(tmp_path).get("asl")               # real_data: true
    pack.allow_synthetic = False
    assert pack.selectable


# ── knowledge base distribution ─────────────────────────────────────────────────

def _entry(answer: str) -> dict:
    return {"id": answer, "site_id": "default", "keywords": [answer], "answer_ar": answer,
            "answer_en": answer, "updated_at": "2026-10-12T00:00:00"}


class _KbServer:
    """GET .../kb?language=xx: answers from `responses` {code: (status, body, headers)};
    a language not listed gets 204. Records the request headers."""

    def __init__(self):
        outer = self
        outer.responses: dict[str, tuple[int, bytes, dict]] = {}
        outer.seen: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append({"path": self.path, **dict(self.headers)})
                language = self.path.rsplit("language=", 1)[-1]
                status, body, headers = outer.responses.get(language, (204, b"", {}))
                if status == 200 and self.headers.get("If-None-Match") == headers.get("ETag"):
                    status, body = 304, b""
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def offer(self, language: str, entries: list[dict], etag: str = '"e1"') -> None:
        self.responses[language] = (200, json.dumps(entries).encode("utf-8"), {"ETag": etag})

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _kb_setup(tmp_path: Path):
    registry = _world(tmp_path, asl_kb=[_entry("original")])
    server = _KbServer()
    kb_dir = tmp_path / "synced" / "kb"
    return registry, server, kb_dir, KnowledgeSync(server.url, "kiosk-7", registry, kb_dir, "tok", timeout_s=5.0)


def _answers(registry, code="asl") -> list[str]:
    return [e.answer_en for e in registry.get(code).kb.entries]


def test_an_edited_knowledge_base_replaces_the_one_the_language_answers_from(tmp_path):
    registry, server, kb_dir, sync = _kb_setup(tmp_path)
    assert _answers(registry) == ["original"]
    server.offer("asl", [_entry("edited-on-server")])
    try:
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes["asl"] == "updated"
    assert _answers(registry) == ["edited-on-server"]
    assert _answers(registry, "liu") != ["edited-on-server"]            # only the language that was asked for
    assert (kb_dir / "asl.json").is_file() and (kb_dir / "asl.etag").read_text() == '"e1"'
    assert server.seen[0]["X-Kiosk-Token"] == "tok"


def test_an_unchanged_knowledge_base_is_not_downloaded_again(tmp_path):
    registry, server, kb_dir, sync = _kb_setup(tmp_path)
    server.offer("asl", [_entry("v1")])
    try:
        sync.tick()
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes["asl"] == "up_to_date"
    asked = [r for r in server.seen if r["path"].endswith("language=asl")]
    assert "If-None-Match" not in asked[0]            # nothing cached yet on the first ask
    assert asked[-1]["If-None-Match"] == '"e1"'       # the second ask offered what it already had


def test_an_empty_answer_from_the_server_never_blanks_the_knowledge_base(tmp_path):
    registry, server, kb_dir, sync = _kb_setup(tmp_path)
    try:
        outcomes = sync.tick()                                           # the server has nothing: 204
    finally:
        server.close()
    assert outcomes["asl"] == "unassigned"
    assert _answers(registry) == ["original"]
    assert not kb_dir.exists()


def test_a_malformed_download_leaves_the_working_knowledge_base_untouched(tmp_path):
    registry, server, kb_dir, sync = _kb_setup(tmp_path)
    broken = [{"id": "x", "keywords": ["x"]}]                           # missing fields B7 reads
    server.responses["asl"] = (200, json.dumps(broken).encode(), {"ETag": '"bad"'})
    try:
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes["asl"] == "rejected"
    assert _answers(registry) == ["original"]
    assert not (kb_dir / "asl.json").exists() and not (kb_dir / "asl.etag").exists()


def test_a_restart_comes_back_on_the_synced_knowledge_base(tmp_path):
    registry, server, kb_dir, sync = _kb_setup(tmp_path)
    server.offer("asl", [_entry("edited-on-server")])
    try:
        sync.tick()
    finally:
        server.close()

    restarted = LanguageRegistry(tmp_path / "languages.json", tmp_path, 0.55, 0.8, 3, kb_dir=kb_dir)

    assert _answers(restarted) == ["edited-on-server"]


def test_skipping_the_language_message_does_not_reach_an_unselectable_engine(monkeypatch):
    """A connection starts on the default language. A client that goes straight to consent
    must not be recognised by it when it is not offered."""
    from fastapi.testclient import TestClient

    import kiosk.main as kiosk_main

    pack = kiosk_main.registry.get(kiosk_main.DEFAULT_LANGUAGE)
    monkeypatch.setattr(pack, "real_data", False)
    monkeypatch.setattr(pack, "allow_synthetic", False)

    with TestClient(kiosk_main.app) as client:
        with client.websocket_connect("/ws/session") as ws:
            assert ws.receive_json()["type"] == "state"
            ws.send_json({"type": "consent", "processing": True, "retention": False})
            reply = ws.receive_json()
            assert reply["type"] == "error" and reply["code"] == "language_unavailable"

            # Declining processing is still allowed: it starts nothing.
            ws.send_json({"type": "consent", "processing": False, "retention": False})
            assert ws.receive_json()["type"] == "state"

        monkeypatch.setattr(pack, "allow_synthetic", True)          # the deployment opts in
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "consent", "processing": True, "retention": False})
            assert ws.receive_json()["session_state"] == "active"


def test_empty_hand_streams_are_filled_so_mediapipe_never_reads_an_empty_packet():
    """Found on the Jetson (MediaPipe 0.10.18): a person in view with no hands up leaves the
    hand streams empty, and reading one aborts the whole backend process."""
    from kiosk.perception import fill_empty_streams

    class Packet:
        def __init__(self, empty, label=""):
            self._empty, self.label = empty, label

        def is_empty(self):
            return self._empty

    face = Packet(False, "face")
    packets = {"face_landmarks": face, "pose_landmarks": Packet(False, "pose"),
               "left_hand_landmarks": Packet(True), "left_hand_world_landmarks": Packet(True),
               "right_hand_landmarks": Packet(False, "right"), "right_hand_world_landmarks": Packet(True)}
    made = []

    def make_empty(name):
        made.append(name)
        return Packet(False, f"filled:{name}")

    filled = fill_empty_streams(packets, make_empty)

    assert sorted(made) == ["left_hand_landmarks", "left_hand_world_landmarks", "right_hand_world_landmarks"]
    assert not any(p.is_empty() for p in filled.values())
    assert filled["pose_landmarks"].label == "pose" and filled["right_hand_landmarks"].label == "right"   # real data untouched
    assert packets["left_hand_landmarks"].is_empty()                       # the input is not modified


def test_one_dead_thermal_zone_does_not_hide_the_others_or_crash_health(tmp_path, monkeypatch):
    """Found on the Jetson: a zone with no sensor made read_text raise TypeError, which took
    /health down with a 500. The hottest readable zone must still be reported."""
    from kiosk.session import read_temperature_c

    for zone, millidegrees in (("thermal_zone0", "41500"), ("thermal_zone1", "55250"), ("thermal_zone2", "0")):
        (tmp_path / zone).mkdir()
        (tmp_path / zone / "temp").write_text(millidegrees)
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.parent.name == "thermal_zone2":
            raise TypeError("can't concat NoneType to bytes")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert read_temperature_c(str(tmp_path)) == 55.25


def test_the_frontend_is_always_revalidated_so_a_kiosk_never_runs_a_stale_page():
    """Found on the Jetson: a week-old cached app.js kept new language cards "coming soon"."""
    from fastapi.testclient import TestClient

    import kiosk.main as kiosk_main

    client = TestClient(kiosk_main.app)
    for path in ("/", "/assets/app.js", "/assets/vision.js", "/core/vocab.json"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache", path
    assert "cache-control" not in client.get("/health").headers      # live data is not a cached file


# ── one connection at a time ────────────────────────────────────────────────────

def test_a_second_browser_connection_replaces_the_first_and_frees_it():
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import kiosk.main as kiosk_main

    with TestClient(kiosk_main.app) as client:
        with client.websocket_connect("/ws/session") as first:
            assert first.receive_json()["type"] == "state"
            with client.websocket_connect("/ws/session") as second:
                assert second.receive_json()["type"] == "state"

                # The old page is told it has been replaced, not left holding a camera pipeline.
                closed = None
                try:
                    first.receive_json()
                except WebSocketDisconnect as exc:
                    closed = exc
                assert closed is not None and closed.code == kiosk_main.SUPERSEDED_CLOSE_CODE

                second.send_json({"type": "ping"})
                assert second.receive_json() == {"type": "pong"}      # the new one is unaffected

        # Both gone: nothing is left registered as the active visitor.
        assert kiosk_main._active_connection is None
        with client.websocket_connect("/ws/session") as third:
            assert third.receive_json()["type"] == "state"


def test_a_corrupt_synced_file_falls_back_to_the_shipped_seed(tmp_path):
    _world(tmp_path, asl_kb=[_entry("original")])
    kb_dir = tmp_path / "synced" / "kb"
    kb_dir.mkdir(parents=True)
    (kb_dir / "asl.json").write_text("{not json", encoding="utf-8")

    restarted = LanguageRegistry(tmp_path / "languages.json", tmp_path, 0.55, 0.8, 3, kb_dir=kb_dir)

    assert _answers(restarted) == ["original"]

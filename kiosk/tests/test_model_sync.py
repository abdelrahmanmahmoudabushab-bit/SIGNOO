"""B15 -> B4 model sync: a kiosk fetches the model the server rolled out to it, and a
bad download can never replace a working model."""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kiosk.fleet import FleetReporter
from kiosk.languages import LanguageRegistry
from kiosk.model_sync import STATE_FILE, ModelSync, read_installed
from kiosk.tests.test_languages import _checkpoint, _utterance, _world


class _Server:
    """Answers GET .../model per language from `responses`: {code: (status, body, headers)};
    a language not listed gets 204. POSTs (register, heartbeat) are recorded and accepted."""

    def __init__(self):
        outer = self
        outer.responses: dict[str, tuple[int, bytes, dict]] = {}
        outer.gets: list[tuple[str, dict]] = []
        outer.posts: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.gets.append((self.path, dict(self.headers)))
                language = self.path.rsplit("language=", 1)[-1]
                status, body, headers = outer.responses.get(language, (204, b"", {}))
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts.append((self.path, body))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def offer(self, language: str, body: bytes, version: str, sha256: str | None = None) -> None:
        self.responses[language] = (200, body, {
            "X-Model-Version": version,
            "X-Model-Sha256": sha256 or hashlib.sha256(body).hexdigest()})

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _new_model(tmp_path: Path, labels=("dog", "cat"), favoured: int = 1) -> bytes:
    """Checkpoint bytes for a model that answers `cat`; the factory American one answers `dog`."""
    path = tmp_path / "upload.pt"
    _checkpoint(path, list(labels), favoured)
    return path.read_bytes()


def _setup(tmp_path: Path, token: str = ""):
    registry = _world(tmp_path)
    server = _Server()
    synced = tmp_path / "synced"
    return registry, server, synced, ModelSync(server.url, "kiosk-7", registry, synced, token, timeout_s=5.0)


def _answer(registry, code: str = "asl") -> str:
    return registry.get(code).recognizer.classify(_utterance()).gloss


def _files(synced: Path) -> list[str]:
    return sorted(p.relative_to(synced).as_posix() for p in synced.rglob("*") if p.is_file())


def test_a_good_download_is_installed_and_the_pack_answers_with_the_new_model(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    old = registry.get("asl").recognizer
    assert _answer(registry) == "dog"
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        outcomes = sync.tick()
    finally:
        server.close()

    assert outcomes == {"liu": "unassigned", "asl": "installed", "ssl": "unassigned"}
    assert _answer(registry) == "cat"
    assert registry.get("asl").model_version == "v2"
    assert _answer(registry, "liu") == "marhaba"                 # installed only where it was requested
    assert registry.get("liu").model_version is None
    # A classify that started before the swap holds the old recognizer and finishes on it.
    assert old.classify(_utterance()).gloss == "dog"

    state = json.loads((synced / STATE_FILE).read_text(encoding="utf-8"))
    assert list(state) == ["asl"] and state["asl"]["version"] == "v2"
    assert (synced / state["asl"]["path"]).is_file()
    assert _files(synced) == sorted([STATE_FILE, state["asl"]["path"]])   # no temp file left behind


def test_the_new_recognizer_keeps_the_languages_own_bars(tmp_path):
    registry = _world(tmp_path, asl_confident=0.95)
    server = _Server()
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        ModelSync(server.url, "kiosk-7", registry, tmp_path / "synced").tick()
    finally:
        server.close()
    recognizer = registry.get("asl").recognizer
    assert (recognizer.reject_threshold, recognizer.confident_threshold, recognizer.top_k) == (0.55, 0.95, 3)


def test_a_wrong_sha_download_is_rejected_and_the_old_model_still_answers(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    old = registry.get("asl").recognizer
    server.offer("asl", _new_model(tmp_path), "v2", sha256="0" * 64)
    try:
        assert sync.tick()["asl"] == "rejected"
    finally:
        server.close()
    assert registry.get("asl").recognizer is old and _answer(registry) == "dog"
    assert registry.get("asl").model_version is None
    assert _files(synced) == []                                  # nothing was written at all


def test_a_truncated_download_is_rejected(tmp_path):
    """The hash is of the whole file; half of it does not match."""
    registry, server, synced, sync = _setup(tmp_path)
    body = _new_model(tmp_path)
    server.offer("asl", body[: len(body) // 2], "v2", sha256=hashlib.sha256(body).hexdigest())
    try:
        assert sync.tick()["asl"] == "rejected"
    finally:
        server.close()
    assert _answer(registry) == "dog" and _files(synced) == []


def test_a_body_that_is_not_a_checkpoint_is_rejected_and_the_old_model_still_answers(tmp_path):
    """Right hash, wrong content: first garbage, then a real checkpoint of the wrong shape."""
    registry, server, synced, sync = _setup(tmp_path)
    old = registry.get("asl").recognizer
    wrong_shape = tmp_path / "wrong.pt"
    _checkpoint(wrong_shape, ["dog", "cat"], 1)
    import torch
    checkpoint = torch.load(wrong_shape, weights_only=True)
    checkpoint["labels"] = {"dog": 0, "cat": 1, "bird": 2}       # head has 2 outputs, labels say 3
    torch.save(checkpoint, wrong_shape)
    try:
        for body in (b"this is not a torch checkpoint", wrong_shape.read_bytes()):
            server.offer("asl", body, "v2")
            assert sync.tick()["asl"] == "rejected"
            assert registry.get("asl").recognizer is old and _answer(registry) == "dog"
            assert registry.get("asl").model_version is None
            assert _files(synced) == []                          # the bad file was removed, nothing recorded
    finally:
        server.close()


def test_a_bad_download_does_not_disturb_an_already_installed_version(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        sync.tick()
        before = _files(synced)
        server.offer("asl", b"garbage", "v3")
        assert sync.tick()["asl"] == "rejected"
    finally:
        server.close()
    assert _answer(registry) == "cat" and registry.get("asl").model_version == "v2"
    assert _files(synced) == before
    assert read_installed(synced)["asl"]["version"] == "v2"


def test_204_and_304_change_nothing(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    old = registry.get("asl").recognizer
    server.responses["asl"] = (304, b"", {})
    try:
        assert sync.tick() == {"liu": "unassigned", "asl": "up_to_date", "ssl": "unassigned"}
    finally:
        server.close()
    assert registry.get("asl").recognizer is old and registry.get("liu").model_version is None
    assert _files(synced) == []


def test_other_statuses_change_nothing(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    old = registry.get("asl").recognizer
    try:
        for status in (404, 409, 500):
            server.responses["asl"] = (status, b"{}", {})
            assert sync.tick()["asl"] == "failed"
    finally:
        server.close()
    assert registry.get("asl").recognizer is old and _files(synced) == []


def test_an_unreachable_server_changes_nothing(tmp_path):
    registry = _world(tmp_path)
    old = registry.get("asl").recognizer
    sync = ModelSync("http://127.0.0.1:9", "kiosk-7", registry, tmp_path / "synced", timeout_s=1.0)
    assert sync.tick() == {"liu": "failed", "asl": "failed", "ssl": "failed"}
    assert registry.get("asl").recognizer is old and not (tmp_path / "synced").exists()


def test_the_token_and_the_installed_version_are_sent(tmp_path):
    registry, server, synced, sync = _setup(tmp_path, token="kiosk-secret")
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        sync.tick()
        first = {path: headers for path, headers in server.gets}
        server.gets.clear()
        sync.tick()
        second = {path: headers for path, headers in server.gets}
    finally:
        server.close()

    asl, liu = "/fleet/kiosks/kiosk-7/model?language=asl", "/fleet/kiosks/kiosk-7/model?language=liu"
    assert set(first) == {asl, liu, "/fleet/kiosks/kiosk-7/model?language=ssl"}   # `esl` has no model configured
    assert all(headers.get("X-Kiosk-Token") == "kiosk-secret" for headers in first.values())
    assert "If-None-Match" not in first[asl]                     # factory model: no version to name
    assert second[asl]["If-None-Match"] == '"v2"'
    assert "If-None-Match" not in second[liu]


def test_no_token_configured_means_no_token_header(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    try:
        sync.tick()
    finally:
        server.close()
    assert all("X-Kiosk-Token" not in headers for _, headers in server.gets)


def _restart(tmp_path: Path, synced: Path) -> LanguageRegistry:
    return LanguageRegistry(tmp_path / "languages.json", tmp_path, 0.55, 0.8, 3,
                            installed=read_installed(synced))


def test_the_installed_version_survives_a_restart(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        sync.tick()
    finally:
        server.close()

    fresh = _restart(tmp_path, synced)
    assert _answer(fresh) == "cat" and fresh.get("asl").model_version == "v2"
    assert _answer(fresh, "liu") == "marhaba" and fresh.get("liu").model_version is None
    assert fresh.model_versions() == {"asl": "v2"}


def test_an_installed_file_that_no_longer_loads_falls_back_to_the_configured_model(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        sync.tick()
    finally:
        server.close()
    read_installed(synced)["asl"]["path"].write_bytes(b"disk corruption")

    fresh = _restart(tmp_path, synced)
    assert fresh.get("asl").ready and _answer(fresh) == "dog"
    assert fresh.get("asl").model_version is None               # so the next sync asks for it again

    (synced / STATE_FILE).write_text("{not json", encoding="utf-8")
    assert read_installed(synced) == {}


def test_a_newer_version_replaces_the_previous_download_on_disk(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    try:
        server.offer("asl", _new_model(tmp_path), "v2")
        sync.tick()
        server.offer("asl", _new_model(tmp_path, favoured=0), "v3")
        sync.tick()
    finally:
        server.close()
    state = json.loads((synced / STATE_FILE).read_text(encoding="utf-8"))
    assert state["asl"]["version"] == "v3" and _answer(registry) == "dog"
    assert _files(synced) == sorted([STATE_FILE, state["asl"]["path"]])
    assert (tmp_path / "models" / "asl.pt").is_file()            # the factory checkpoint is never touched


def test_a_version_string_cannot_choose_where_the_file_lands(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("asl", _new_model(tmp_path), "../../evil/v9")
    try:
        assert sync.tick()["asl"] == "installed"
    finally:
        server.close()
    path = read_installed(synced)["asl"]["path"]
    assert path.parent == synced / "asl" and not (tmp_path / "evil").exists()
    assert registry.get("asl").model_version == "../../evil/v9"  # reported back verbatim


def test_a_language_that_never_came_up_gets_the_model_on_disk_for_the_next_restart(tmp_path):
    """`ssl` is configured but its factory checkpoint is missing, so it has no engine to
    swap into mid-run; the download is kept and a restart brings the language up on it."""
    registry, server, synced, sync = _setup(tmp_path)
    server.offer("ssl", _new_model(tmp_path, labels=("marhaba", "shukran")), "v1")
    try:
        assert sync.tick()["ssl"] == "installed"
    finally:
        server.close()
    assert not registry.get("ssl").ready

    fresh = _restart(tmp_path, synced)
    assert fresh.get("ssl").ready and _answer(fresh, "ssl") == "shukran"


def test_the_heartbeat_carries_model_versions(tmp_path):
    registry, server, synced, sync = _setup(tmp_path)
    reporter = FleetReporter(server.url, "kiosk-7", model_versions=registry.model_versions)
    server.offer("asl", _new_model(tmp_path), "v2")
    try:
        assert reporter.tick() is True
        sync.tick()
        assert reporter.tick() is True
    finally:
        server.close()
    beats = [body for path, body in server.posts if path.endswith("/heartbeat")]
    assert "model_versions" not in beats[0]                      # factory models only: nothing to list
    assert beats[1]["model_versions"] == {"asl": "v2"}

"""B15 -> B4: fetches the model the server has rolled out to this kiosk, per language.

Each tick asks the server, for every language that has a model configured, whether a
different version is assigned. A download replaces the running model only after its
sha256 matches the server's header AND it loads as a checkpoint; anything else leaves
the working model exactly as it was and removes the bad file.

What is installed is recorded in `installed.json` beside the downloaded checkpoints:

    {"asl": {"version": "v7", "path": "asl/v7-1a2b3c4d.pt"}}      # path relative to that file

so a restart comes back up on the installed version (kiosk/languages.py reads it).

Nothing here can affect a visitor: it runs off the event loop, every call is bounded by
a timeout, and a swap is one attribute assignment of a fully built Recognizer. A
classify already running holds its own reference to the old Recognizer and finishes on
it; no lock is needed, and none is taken.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from kiosk.recognition import Recognizer

logger = logging.getLogger("signoo.kiosk.model_sync")

STATE_FILE = "installed.json"
# A landmark model is a few MB. This only stops a wrong or hostile response from
# filling the Jetson's memory before the hash check can refuse it.
MAX_MODEL_BYTES = 512 * 1024 * 1024


def _read_state(models_dir: Path) -> dict[str, dict]:
    """The state file as written: {code: {"version": str, "path": str}}. Empty if
    nothing is recorded or it is unreadable (the kiosk then starts on its factory models)."""
    try:
        raw = json.loads((models_dir / STATE_FILE).read_text(encoding="utf-8"))
        root = models_dir.resolve()
        state = {}
        for code, entry in raw.items():
            version, path = str(entry["version"]), str(entry["path"])
            # These paths get loaded and later unlinked. One that leaves the models
            # directory (an absolute path, "..", a link) is not ours to touch.
            if (models_dir / path).resolve().is_relative_to(root):
                state[code] = {"version": version, "path": path}
            else:
                logger.warning("model_state_path_refused", extra={"language": code, "path": path})
        return state
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("model_state_unreadable", extra={"reason": str(exc)})
        return {}


def read_installed(models_dir: Path) -> dict[str, dict]:
    """{code: {"version": str, "path": absolute Path}} for kiosk/languages.py at startup."""
    return {code: {"version": entry["version"], "path": models_dir / entry["path"]}
            for code, entry in _read_state(models_dir).items()}


def _write_atomic(path: Path, data: bytes) -> None:
    """Temp file in the same directory, then os.replace: a power cut leaves either the
    old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


class ModelSync:
    def __init__(self, server_url: str, kiosk_id: str, registry, models_dir: Path,
                 token: str = "", timeout_s: float = 15.0):
        self.base = server_url.rstrip("/")
        self.kiosk_id = kiosk_id
        self.registry = registry
        self.models_dir = Path(models_dir)
        self.token = token
        self.timeout_s = timeout_s

    def tick(self) -> dict[str, str]:
        """One pass over every language with a model configured. Returns what happened
        per language: installed | up_to_date | unassigned | reverted | rejected | failed."""
        outcomes = {}
        for pack in self.registry.packs.values():
            if pack.bars is None:
                continue
            try:
                outcomes[pack.code] = self._sync(pack)
            except Exception:   # one language's failure must not stop the others or the heartbeat
                logger.exception("model_sync_error", extra={"language": pack.code})
                outcomes[pack.code] = "failed"
        return outcomes

    def _sync(self, pack) -> str:
        url = (f"{self.base}/fleet/kiosks/{urllib.parse.quote(self.kiosk_id, safe='')}/model?"
               + urllib.parse.urlencode({"language": pack.code}))
        headers = {}
        if self.token:
            headers["X-Kiosk-Token"] = self.token
        if pack.model_version:
            headers["If-None-Match"] = f'"{pack.model_version}"'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=self.timeout_s) as response:
                if response.status == 204:
                    # Nothing is assigned. If this kiosk is running a model the server
                    # delivered, the server has withdrawn it (a rollback): go back to the
                    # factory checkpoint rather than keep serving a model it pulled.
                    return self._revert_to_factory(pack) if pack.model_version else "unassigned"
                if response.status != 200:
                    logger.info("model_sync_failed", extra={"language": pack.code, "status": response.status})
                    return "failed"
                version = (response.headers.get("X-Model-Version") or "").strip()
                expected = (response.headers.get("X-Model-Sha256") or "").strip().lower()
                body = response.read(MAX_MODEL_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return "up_to_date"
            logger.info("model_sync_failed", extra={"language": pack.code, "status": exc.code})
            return "failed"
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            # offline, timeout, or the connection cut mid-body (IncompleteRead)
            logger.info("model_sync_failed", extra={"language": pack.code, "reason": str(exc)})
            return "failed"

        # Hash first: nothing is written or parsed until the bytes are the server's bytes.
        actual = hashlib.sha256(body).hexdigest()
        if not version or len(body) > MAX_MODEL_BYTES or actual != expected:
            logger.warning("model_download_rejected", extra={
                "language": pack.code, "version": version, "bytes": len(body),
                "expected_sha256": expected, "actual_sha256": actual})
            return "rejected"
        return self._install(pack, version, body, actual)

    def _revert_to_factory(self, pack) -> str:
        """Replaces a delivered model with the checkpoint the kiosk shipped with.

        Nothing changes unless the factory model loads and the change is recorded: a
        kiosk that cannot revert keeps serving what it has and tries again next tick,
        rather than ending up with no model or one that comes back after a restart.
        """
        if pack.factory_model is None or pack.bars is None:
            return "unassigned"
        reject, confident, top_k = pack.bars
        try:
            recognizer = Recognizer(pack.factory_model, reject, confident, top_k)
        except Exception as exc:   # the factory file is missing or unreadable
            logger.warning("model_revert_failed",
                           extra={"language": pack.code, "reason": str(exc)})
            return "failed"

        state = _read_state(self.models_dir)
        entry = state.pop(pack.code, None)
        try:
            _write_atomic(self.models_dir / STATE_FILE, json.dumps(state, indent=2).encode("utf-8"))
        except OSError as exc:
            logger.warning("model_revert_failed",
                           extra={"language": pack.code, "reason": str(exc)})
            return "failed"

        withdrawn = pack.model_version
        pack.model_version = None
        if pack.vocab is not None:
            pack.recognizer = recognizer
        if entry is not None:
            (self.models_dir / entry["path"]).unlink(missing_ok=True)
        logger.info("model_reverted", extra={"language": pack.code, "withdrawn_version": withdrawn})
        return "reverted"

    def _install(self, pack, version: str, body: bytes, sha256: str) -> str:
        # The version is the server's text: keep it out of the path except as a safe
        # name. The hash suffix keeps two versions that sanitise alike apart.
        name = re.sub(r"[^A-Za-z0-9._-]", "_", version).lstrip(".")[:64] or "model"
        relative = Path(pack.code) / f"{name}-{sha256[:8]}.pt"
        target = self.models_dir / relative
        running = pack.recognizer.model_path if pack.recognizer is not None else None
        reject, confident, top_k = pack.bars
        try:
            _write_atomic(target, body)
            recognizer = Recognizer(target, reject, confident, top_k)
        except Exception as exc:   # not only ModelUnavailable: wrong tensor shapes raise RuntimeError
            if target != running:
                target.unlink(missing_ok=True)
            logger.warning("model_install_rejected",
                           extra={"language": pack.code, "version": version, "reason": str(exc)})
            return "rejected"

        state = _read_state(self.models_dir)
        previous = self.models_dir / state[pack.code]["path"] if pack.code in state else None
        state[pack.code] = {"version": version, "path": relative.as_posix()}
        try:
            _write_atomic(self.models_dir / STATE_FILE, json.dumps(state, indent=2).encode("utf-8"))
        except OSError as exc:
            # Unrecorded, it would silently revert on restart; treat it as not installed.
            if target != running:
                target.unlink(missing_ok=True)
            logger.warning("model_install_rejected",
                           extra={"language": pack.code, "version": version, "reason": str(exc)})
            return "rejected"

        pack.model_version = version
        if pack.vocab is not None:
            pack.recognizer = recognizer      # the swap: in-flight classifies keep the old object
        # else: this language never came up (its factory files did not load), so there is
        # no engine to swap into. The model is on disk and recorded; a restart loads it.
        if previous is not None and previous != target:
            previous.unlink(missing_ok=True)  # only ever a file this module downloaded
        logger.info("model_installed", extra={"language": pack.code, "version": version,
                                              "live": pack.vocab is not None})
        return "installed"

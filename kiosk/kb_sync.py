"""B14 -> B7: fetches each language's knowledge base from the server.

The console edits the server's copy. A kiosk in another building does not share a disk
with the server, so it asks for the answers it should use, per language, the same way it
asks for a model (kiosk/model_sync.py). The server's copy is written under
`<synced_models_dir>/kb/<code>.json` and the language's `KnowledgeBase` is swapped to it;
the source-controlled seed file in `kiosk/data/` is never overwritten. A restart comes back
on the synced copy (kiosk/languages.py prefers it).

Nothing here can affect a visitor: it runs off the event loop, every call is bounded by a
timeout, a download that does not parse leaves the working knowledge base untouched, and a
"no entries" answer from the server never blanks one.
"""
from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from kiosk.dialogue import KBEntry, KnowledgeBase
from kiosk.model_sync import _write_atomic

logger = logging.getLogger("signoo.kiosk.kb_sync")

# A knowledge base is a few dozen short answers. This only stops a wrong response from
# filling the device's memory before it is parsed.
MAX_KB_BYTES = 8 * 1024 * 1024


class KnowledgeSync:
    def __init__(self, server_url: str, kiosk_id: str, registry, kb_dir: Path,
                 token: str = "", timeout_s: float = 15.0):
        self.base = server_url.rstrip("/")
        self.kiosk_id = kiosk_id
        self.registry = registry
        self.kb_dir = Path(kb_dir)
        self.token = token
        self.timeout_s = timeout_s

    def tick(self) -> dict[str, str]:
        """One pass over every language with a model configured. Returns what happened
        per language: updated | up_to_date | unassigned | rejected | failed."""
        outcomes = {}
        for pack in self.registry.packs.values():
            if pack.bars is None:
                continue
            try:
                outcomes[pack.code] = self._sync(pack)
            except Exception:   # one language's failure must not stop the others or the heartbeat
                logger.exception("kb_sync_error", extra={"language": pack.code})
                outcomes[pack.code] = "failed"
        return outcomes

    def _etag_path(self, code: str) -> Path:
        return self.kb_dir / f"{code}.etag"

    def _stored_etag(self, code: str) -> str | None:
        try:
            return self._etag_path(code).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _sync(self, pack) -> str:
        url = (f"{self.base}/fleet/kiosks/{urllib.parse.quote(self.kiosk_id, safe='')}/kb?"
               + urllib.parse.urlencode({"language": pack.code}))
        headers = {}
        if self.token:
            headers["X-Kiosk-Token"] = self.token
        # Only offered when the file it describes is still there to be served.
        stored = self._stored_etag(pack.code)
        if stored and (self.kb_dir / f"{pack.code}.json").is_file():
            headers["If-None-Match"] = stored
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=self.timeout_s) as response:
                if response.status == 204:
                    return "unassigned"      # the server has nothing for this language: keep ours
                if response.status != 200:
                    return "failed"
                etag = (response.headers.get("ETag") or "").strip()
                body = response.read(MAX_KB_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return "up_to_date"
            logger.info("kb_sync_failed", extra={"language": pack.code, "status": exc.code})
            return "failed"
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            logger.info("kb_sync_failed", extra={"language": pack.code, "reason": str(exc)})
            return "failed"

        if len(body) > MAX_KB_BYTES or not etag:
            logger.warning("kb_download_rejected", extra={"language": pack.code, "bytes": len(body)})
            return "rejected"
        try:
            raw = json.loads(body.decode("utf-8"))
            [KBEntry(**entry) for entry in raw]       # every entry must have exactly the fields B7 reads
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            logger.warning("kb_download_rejected", extra={"language": pack.code, "reason": str(exc)})
            return "rejected"

        target = self.kb_dir / f"{pack.code}.json"
        try:
            _write_atomic(target, body)
            knowledge_base = KnowledgeBase(target)
            _write_atomic(self._etag_path(pack.code), etag.encode("utf-8"))
        except Exception as exc:
            # Not recorded or not loadable: leave what is serving as it is, try again next tick.
            logger.warning("kb_install_rejected", extra={"language": pack.code, "reason": str(exc)})
            return "failed"
        pack.kb = knowledge_base       # one attribute assignment; a session already open keeps the old object
        logger.info("kb_updated", extra={"language": pack.code, "entries": len(raw)})
        return "updated"

"""B9 -> B15: tells the server this kiosk exists and is healthy.

Registers once, then sends a heartbeat on a timer. Every field is optional on the
wire and left out when unknown, because the server keeps the last known value for an
omitted field; guessing "not throttled" would hide a real overheating condition.

Nothing here can affect a visitor: every call is bounded by a short timeout and a
failure only means "try again next tick".
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from kiosk.session import read_temperature_c

logger = logging.getLogger("signoo.kiosk.fleet")


class FleetReporter:
    def __init__(self, server_url: str, kiosk_id: str, token: str = "", site_id: str = "default",
                 language: str = "liu", timeout_s: float = 5.0):
        self.base = server_url.rstrip("/")
        self.kiosk_id = kiosk_id
        self.token = token
        self.site_id = site_id
        self.language = language
        self.timeout_s = timeout_s
        self.started = time.monotonic()
        self.registered = False

    def _post(self, path: str, payload: dict) -> bool:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Kiosk-Token"] = self.token
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.info("fleet_call_failed", extra={"path": path, "reason": str(exc)})
            return False

    def tick(self) -> bool:
        """Registers if needed, then sends one heartbeat. True if the heartbeat landed."""
        if not self.registered:
            self.registered = self._post("/fleet/kiosks/register", {
                "kiosk_id": self.kiosk_id, "site_id": self.site_id, "language": self.language,
            })
            if not self.registered:
                return False
        beat: dict = {"uptime_s": round(time.monotonic() - self.started, 1)}
        temperature = read_temperature_c()
        if temperature is not None:
            beat["temperature_c"] = temperature
        delivered = self._post(f"/fleet/kiosks/{self.kiosk_id}/heartbeat", beat)
        if not delivered:
            self.registered = False       # server may have lost us (new database); re-register next tick
        return delivered

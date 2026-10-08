"""Manual, ad-hoc WebSocket walk-through against a running `kiosk.main:app` —
not part of the pytest suite (needs a live server + real camera to be fully
meaningful), but useful for confirming the wire protocol by hand during
development. See kiosk/README.md for how this was used to verify the consent
gate, the framing message, and control actions end to end.

    .venv/Scripts/python.exe -m uvicorn kiosk.main:app --port 8812 &
    .venv/Scripts/python.exe kiosk/tests/manual_ws_check.py ws://127.0.0.1:8812/ws/session
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys

import cv2
import numpy as np
import websockets


def _tiny_jpeg() -> str:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def main(url: str) -> None:
    async with websockets.connect(url) as ws:
        print("<-", await ws.recv())                                  # initial state

        await ws.send(json.dumps({"type": "language", "lang": "liu"}))
        print("<-", await ws.recv())

        await ws.send(json.dumps({"type": "consent", "processing": True, "retention": False}))
        print("<-", await ws.recv())

        # MediaPipe's LIVE_STREAM callback is asynchronous — a result for frame N
        # typically lands while frame N+1 is being handled, not synchronously
        # within the same message. Send several frames and drain whatever comes
        # back, rather than assuming a fixed reply count per frame.
        frame = _tiny_jpeg()
        for i in range(5):
            await ws.send(json.dumps({"type": "frame", "ts_ms": i * 130, "data": frame}))
            try:
                while True:
                    print("<-", await asyncio.wait_for(ws.recv(), timeout=0.5))
            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"type": "control", "action": "clear"}))
        print("<-", await ws.recv())

        await ws.send(json.dumps({"type": "ask"}))
        print("<-", await ws.recv())          # answer (refusal — nothing signed)

        await ws.send(json.dumps({"type": "control", "action": "end_session"}))
        print("<-", await ws.recv())          # session_reset


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8812/ws/session"))

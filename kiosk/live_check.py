"""Plays recorded sign videos to a kiosk backend over its real WebSocket, the way the
browser does, and reports what a visitor would have seen for each one.

    # against a backend this command starts itself
    python -m kiosk.live_check data/popsign/raw --language asl --signers 7002 --signs apple book

    # against a kiosk that is already running (for example the Jetson)
    python -m kiosk.live_check data/popsign/raw --language asl --signers 7002 \
        --url ws://192.168.100.60:8000/ws/session

How this differs from `kiosk/replay.py`: the replay tool drives the tracker, segmenter
and recogniser directly, so it measures them. This one goes through everything in
between as well: the JPEG the browser would send, its pacing ("do not send while the last
frame is unanswered"), the WebSocket, the consent gate, and when results are sent back.
Two faults found on 2026-10-13 were invisible to the unit tests and to the replay tool
and plain here: words dropped on a slow tracker, and results that waited for the next
frame (docs/LIVE_PIPELINE.md §15).

Against a running kiosk it uses that kiosk's own tracker, MediaPipe version and CPU, so
it is the way to put known clips through the real device. A kiosk serves one connection
at a time: this takes it over, so use it only when no visitor is at the kiosk. Consent is
sent with retention off, so nothing of the run is kept.

Videos are expected as `<root>/<signer>/<sign>/<clip>.mp4`; the folder names are the
ground truth. Use validation signers; the test signers stay untouched until a final model.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# What assets/app.js does. Keep these equal to it, or this stops being the browser.
SEND_MAX_SIDE = 480
JPEG_QUALITY = 60
FRAME_INTERVAL_S = 0.070
FRAME_REPLY_TIMEOUT_S = 0.200

HOLD_LAST_FRAME_S = 1.5     # the signer holds still, as at the end of a recording
EMPTY_SCENE_S = 1.0         # then steps out, so a word still open is closed

OUTCOMES = ("committed_correct", "committed_WRONG", "candidates_with_truth", "candidates",
            "rejected", "hands_lost", "no_word")


def judge(truth: str, hypotheses: list[dict]) -> str:
    """What the visitor ended up with for one clip, from its `hypothesis` messages.

    A clip is judged by its most confident result, as `kiosk/replay.py` does. "no_word"
    is the outcome with nothing on screen at all: the sign was never closed into a word.
    """
    if not hypotheses:
        return "no_word"
    best = max(hypotheses, key=lambda h: h.get("confidence", 0.0))
    if best["status"] == "committed":
        return "committed_correct" if best.get("gloss") == truth else "committed_WRONG"
    if best["status"] == "candidates":
        offered = [gloss for gloss, _ in best.get("alternatives", [])]
        return "candidates_with_truth" if truth in offered else "candidates"
    return "hands_lost" if best.get("reason") == "hands_lost" else "rejected"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_backend(port: int, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SIGNOO_SERVER_URL="")
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "kiosk.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def _health(http_base: str) -> dict:
    with urllib.request.urlopen(f"{http_base}/health", timeout=5) as response:
        return json.loads(response.read())


def _wait_until_healthy(http_base: str, seconds: float) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        try:
            return _health(http_base)
        except OSError:
            if time.monotonic() > deadline:
                raise SystemExit(f"no answer from {http_base}/health within {seconds:.0f} s")
            time.sleep(0.5)


def _frames(path: Path):
    """The clip's frames at the size the browser sends, then the held last frame, then an
    empty scene. Returns (frames, frames per second)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = SEND_MAX_SIDE / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        return [], fps
    frames += [frames[-1]] * int(HOLD_LAST_FRAME_S * fps)
    frames += [frames[0] * 0 + 90] * int(EMPTY_SCENE_S * fps)      # a plain grey wall
    return frames, fps


def play_clip(ws, path: Path) -> dict:
    """Sends one clip in real time with the browser's pacing; returns what came back."""
    import cv2

    frames, fps = _frames(path)
    messages: list[dict] = []
    sent = returned = with_hand = 0
    awaiting = False
    sent_at = 0.0
    start = time.monotonic()
    next_tick = start

    def drain(timeout: float) -> None:
        nonlocal awaiting, returned, with_hand
        while True:
            try:
                message = json.loads(ws.recv(timeout=timeout))
            except TimeoutError:
                return
            messages.append(message)
            if message["type"] == "landmarks":
                awaiting = False
                returned += 1
                with_hand += bool(message["left"] or message["right"])

    while frames:
        now = time.monotonic()
        index = int((now - start) * fps)       # the camera does not wait: late means skipped
        if index >= len(frames):
            break
        drain(0.001)
        if now >= next_tick:
            next_tick += FRAME_INTERVAL_S
            if not (awaiting and now - sent_at < FRAME_REPLY_TIMEOUT_S):
                ok, jpeg = cv2.imencode(".jpg", frames[index], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    ws.send(json.dumps({"type": "frame", "ts_ms": int(now * 1000),
                                        "data": base64.b64encode(jpeg.tobytes()).decode()}))
                    sent += 1
                    awaiting = True
                    sent_at = now
        time.sleep(0.002)
    drain(0.6)                                  # a word closed by the last frames is still on its way
    seconds = len(frames) / fps if frames else 0.0
    return {
        "seconds": seconds, "sent": sent, "returned": returned, "with_hand": with_hand,
        "hypotheses": [m for m in messages if m["type"] == "hypothesis"],
        "states": [m["state"] for m in messages if m["type"] == "signing"],
        "framing": [m["reason"] for m in messages if m["type"] == "framing" and m["reason"]],
        "errors": [m for m in messages if m["type"] == "error"],
    }


def _wait_for_state(ws, name: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            message = json.loads(ws.recv(timeout=0.2))
        except TimeoutError:
            continue
        if message.get("type") == "state" and message.get("session_state") == name:
            return
        if message.get("type") == "error":
            raise SystemExit(f"the kiosk refused: {message.get('code')}: {message.get('message')}")
    raise SystemExit(f"the kiosk did not reach the '{name}' state within {seconds:.0f} s")


def main() -> None:
    from websockets.sync.client import connect

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder of <signer>/<sign>/<clip>.mp4")
    ap.add_argument("--language", default="asl")
    ap.add_argument("--signers", nargs="*", help="only these signer folders (default: all)")
    ap.add_argument("--signs", nargs="*", help="only these sign folders (default: all)")
    ap.add_argument("--per-sign", type=int, default=1, help="clips per signer per sign (default 1)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips")
    ap.add_argument("--url", help="ws://host:port/ws/session of a running kiosk; "
                                  "default: start a backend here and use that")
    args = ap.parse_args()

    by_sign: dict[tuple[str, str], list[Path]] = {}
    for video in sorted(Path(args.root).rglob("*.mp4")):
        signer, sign = video.parent.parent.name, video.parent.name
        if args.signers and signer not in args.signers:
            continue
        if args.signs and sign not in args.signs:
            continue
        by_sign.setdefault((signer, sign), []).append(video)
    clips = [v for videos in by_sign.values() for v in videos[: args.per_sign]]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit("no clips matched")

    backend = None
    if args.url:
        ws_url = args.url
        http_base = "http" + ws_url[2:].rsplit("/ws/", 1)[0] if ws_url.startswith("ws") else None
    else:
        port = _free_port()
        log_path = Path(os.environ.get("TEMP", ".")) / f"signoo-live-check-{port}.log"
        backend = _start_backend(port, log_path)
        ws_url, http_base = f"ws://127.0.0.1:{port}/ws/session", f"http://127.0.0.1:{port}"
        print(f"started a kiosk backend on port {port} (its log: {log_path})")

    try:
        health = _wait_until_healthy(http_base, 120)
        print(f"kiosk is up; start-up pass per model (s): {health.get('warm_up_s')}")
        with connect(ws_url, max_size=None) as ws:
            ws.send(json.dumps({"type": "language", "lang": args.language}))
            _wait_for_state(ws, "language", 15)
            time.sleep(3.0)                     # a visitor reads the consent screen
            asked = time.monotonic()
            ws.send(json.dumps({"type": "consent", "processing": True, "retention": False}))
            _wait_for_state(ws, "active", 60)
            print(f"consent to active: {(time.monotonic() - asked) * 1000:.0f} ms\n")

            outcomes = Counter()
            sent_rates, returned_rates = [], []
            for n, clip in enumerate(clips, 1):
                truth = clip.parent.name
                result = play_clip(ws, clip)
                outcome = judge(truth, result["hypotheses"])
                outcomes[outcome] += 1
                if result["seconds"]:
                    sent_rates.append(result["sent"] / result["seconds"])
                    returned_rates.append(result["returned"] / result["seconds"])
                best = max(result["hypotheses"], key=lambda h: h.get("confidence", 0.0), default=None)
                said = (f"{best.get('gloss') or [g for g, _ in best.get('alternatives', [])]} "
                        f"({best.get('confidence', 0.0):.2f})") if best else "nothing"
                print(f"[{n}/{len(clips)}] {truth:10s} {outcome:22s} said={said}  "
                      f"sent={result['sent']} returned={result['returned']} with_hand={result['with_hand']} "
                      f"states={result['states']} framing={sorted(set(result['framing']))} "
                      f"{clip.parent.parent.name}", flush=True)
                for error in result["errors"]:
                    print(f"      error from the kiosk: {error.get('code')}: {error.get('message')}")
                ws.send(json.dumps({"type": "control", "action": "clear"}))
                time.sleep(0.2)
            ws.send(json.dumps({"type": "control", "action": "end_session"}))

        total = len(clips)
        print(f"\nover the kiosk's WebSocket ({args.language}), {total} clips")
        for key in OUTCOMES:
            if outcomes[key]:
                print(f"  {key:24s} {outcomes[key]:4d}  ({outcomes[key] / total:.0%})")
        reachable = outcomes["committed_correct"] + outcomes["candidates_with_truth"]
        print(f"  right word committed or offered: {reachable} of {total} ({reachable / total:.0%})")
        if sent_rates:
            print(f"  frames sent {min(sent_rates):.1f}-{max(sent_rates):.1f}/s, "
                  f"results returned {min(returned_rates):.1f}-{max(returned_rates):.1f}/s "
                  "(they should be close; a gap means results are being lost)")
        try:
            print(f"  kiosk telemetry: {_health(http_base)['telemetry']}")
        except OSError:
            pass
    finally:
        if backend is not None:
            backend.terminate()
            try:
                backend.wait(10)
            except subprocess.TimeoutExpired:
                backend.kill()


if __name__ == "__main__":
    main()

"""A stand-in for a physical kiosk, exercising B15 fleet control end to end.

No physical fleet exists yet (SYSTEM_DESIGN.md B15 "exercised with one simulated
local kiosk client" / delegated-weaving-quilt.md Phase 3). This script registers
one kiosk and heartbeats it a few times against a running `server/main.py`.

    python server/scripts/simulate_kiosk.py --base-url http://127.0.0.1:8000 \
        --kiosk-id kiosk-dev-01 --heartbeats 3 --interval 2
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--kiosk-id", default="kiosk-dev-01")
    ap.add_argument("--site-id", default="default")
    ap.add_argument("--language", default="liu")
    ap.add_argument("--heartbeats", type=int, default=3)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=10.0)

    resp = client.post("/fleet/kiosks/register", json={
        "kiosk_id": args.kiosk_id, "site_id": args.site_id, "language": args.language,
    })
    resp.raise_for_status()
    print(f"registered: {resp.json()}")

    for i in range(args.heartbeats):
        payload = {
            "current_model_version": None,
            # Non-Jetson hardware reports "unavailable" per SYSTEM_DESIGN.md B9 —
            # this simulator instead makes up a plausible number, which is a
            # documented simulation shortcut, not what real kiosk code should do.
            "temperature_c": round(40 + random.uniform(-2, 10), 1),
            "throttled": False,
            "uptime_s": (i + 1) * args.interval,
        }
        resp = client.post(f"/fleet/kiosks/{args.kiosk_id}/heartbeat", json=payload)
        resp.raise_for_status()
        print(f"heartbeat {i + 1}/{args.heartbeats}: {resp.json()}")
        if i < args.heartbeats - 1:
            time.sleep(args.interval)

    print("done")


if __name__ == "__main__":
    sys.exit(main())

"""Prints the credential to put on one kiosk, for a server running in `per_kiosk` mode.

    SIGNOO_KIOSK_TOKEN=<the server's master key> python -m server.scripts.kiosk_token kiosk-lobby-01

or, inside the running container:

    docker compose -f deploy/docker-compose.yml exec api python -m server.scripts.kiosk_token kiosk-lobby-01

Set the printed value as SIGNOO_KIOSK_TOKEN on that kiosk (with the same SIGNOO_KIOSK_ID).
It opens that kiosk's endpoints and no other's. The master key stays on the server: a kiosk
that held it could mint a credential for any id, which is what this mode exists to prevent.
"""
from __future__ import annotations

import re
import sys

from server.auth import kiosk_token_for
from server.config import settings
from server.schemas import SAFE_ID_PATTERN


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m server.scripts.kiosk_token <kiosk_id>", file=sys.stderr)
        return 2
    kiosk_id = argv[1]
    if not re.fullmatch(SAFE_ID_PATTERN, kiosk_id):
        print(f"kiosk id {kiosk_id!r} is not a plain id (letters, digits, '_', '-', '.')", file=sys.stderr)
        return 2
    if not settings.kiosk_token:
        print("SIGNOO_KIOSK_TOKEN (the master key) is not set", file=sys.stderr)
        return 1
    if settings.kiosk_token_mode != "per_kiosk":
        print("note: the server is in 'shared' mode, where every kiosk sends the master key itself; "
              "this token only works once SIGNOO_KIOSK_TOKEN_MODE=per_kiosk", file=sys.stderr)
    print(kiosk_token_for(kiosk_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/bin/sh
# Migrate the database, then start the API. In this order on purpose: the app's own
# init_db() creates tables directly, and tables made that way cannot be migrated afterwards.
set -eu

echo "applying database migrations"
cd /app
alembic -c server/alembic.ini upgrade head

# One worker is enough for a fleet of kiosks (each makes a request every 30 s), and a second
# one buys nothing here: training holds a per-language lock, so it would only repeat the
# scheduler's idle ticks. Raise it only if the console is slow, not for the kiosks.
exec uvicorn server.main:app --host 0.0.0.0 --port 8001 --workers "${SIGNOO_WORKERS:-1}" \
    --proxy-headers --forwarded-allow-ips="*"

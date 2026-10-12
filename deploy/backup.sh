#!/bin/bash
# Backs up what cannot be recreated: the database (reviewed data, model registry, users) and
# the `runs` volume (every trained checkpoint and the frozen benchmark).
#
#   deploy/backup.sh                       # writes to /var/backups/signoo
#   SIGNOO_BACKUP_REMOTE=spaces:my-bucket/signoo deploy/backup.sh   # also copies off the droplet
#
# Run it from cron (crontab -e):   17 3 * * *  /opt/signoo/deploy/backup.sh >> /var/log/signoo-backup.log 2>&1
#
# A backup that lives only on the droplet it protects is not a backup: set SIGNOO_BACKUP_REMOTE
# to an rclone remote (DigitalOcean Spaces is S3-compatible; `rclone config` -> "S3" -> "DigitalOcean").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -f "$HERE/docker-compose.yml" --env-file "$HERE/.env")
DEST="${SIGNOO_BACKUP_DIR:-/var/backups/signoo}"
KEEP_DAYS="${SIGNOO_BACKUP_KEEP_DAYS:-14}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$DEST"

"${COMPOSE[@]}" exec -T db pg_dump -U signoo -d signoo --no-owner | gzip > "$DEST/db-$STAMP.sql.gz"

# The runs volume, read through a throwaway container so the API need not stop.
docker run --rm -v signoo_runs:/data:ro -v "$DEST":/out alpine \
  tar -czf "/out/runs-$STAMP.tar.gz" -C /data .

# An empty or tiny dump means pg_dump failed quietly; refuse to call that a backup.
if [ "$(stat -c %s "$DEST/db-$STAMP.sql.gz")" -lt 200 ]; then
  echo "database dump is suspiciously small; keeping it but failing" >&2
  exit 1
fi

if [ -n "${SIGNOO_BACKUP_REMOTE:-}" ]; then
  rclone copy "$DEST/db-$STAMP.sql.gz" "$SIGNOO_BACKUP_REMOTE"
  rclone copy "$DEST/runs-$STAMP.tar.gz" "$SIGNOO_BACKUP_REMOTE"
fi

find "$DEST" -name 'db-*.sql.gz' -mtime +"$KEEP_DAYS" -delete
find "$DEST" -name 'runs-*.tar.gz' -mtime +"$KEEP_DAYS" -delete
echo "backup $STAMP ok"

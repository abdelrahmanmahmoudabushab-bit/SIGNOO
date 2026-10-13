#!/usr/bin/env bash
# Sends the kiosk to the Jetson, restarts it, and checks that it came back healthy.
#
#   scripts/deploy-jetson.sh             # deploy
#   scripts/deploy-jetson.sh --dry-run   # list what would be sent, change nothing
#
# What goes: kiosk/, core/, assets/, index.html, and the model files that
# core/languages.json names. What does not: ml/ (the Jetson keeps the model code its
# checkpoints were built with; send it deliberately with --with-ml), data/, tests'
# caches, and anything git ignores apart from those model files.
#
# Before anything is overwritten the Jetson's current copy is saved to
# ~/signoo-backups/<time>.tgz (the last five are kept) and the command to put it back is
# printed. A file deleted here is not deleted there.
#
# Settings (environment):
#   JETSON_HOST   default jetson@192.168.100.60
#   JETSON_DIR    default signoo        (relative to the remote home directory)
#   JETSON_PORT   default 8000          (the kiosk backend's port on the Jetson)
#
# Exit codes: 0 deployed and healthy · 2 the Jetson cannot be reached (nothing changed)
#             3 sent, but the kiosk did not come back healthy (its log tail is printed)
set -u
cd "$(dirname "$0")/.." || exit 1

HOST="${JETSON_HOST:-jetson@192.168.100.60}"
DIR="${JETSON_DIR:-signoo}"
PORT="${JETSON_PORT:-8000}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
DRY=0; WITH_ML=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --with-ml) WITH_ML=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

# ── what to send ────────────────────────────────────────────────────────────────
paths="kiosk core assets index.html"
[ "$WITH_ML" = 1 ] && paths="$paths ml"
list="$(mktemp)"; trap 'rm -f "$list"' EXIT
# Tracked files plus new ones git does not ignore; never a visitor's queued session.
{ git ls-files -- $paths; git ls-files --others --exclude-standard -- $paths; } \
  | grep -v -e '__pycache__' -e '^kiosk/data/pending_uploads/' -e '\.pyc$' | sort -u > "$list"
# Files git lists but that no longer exist (deleted, not yet committed) cannot be sent.
while read -r f; do [ -f "$f" ] && echo "$f"; done < "$list" > "$list.ok" && mv "$list.ok" "$list"

models="$(grep -o '"model": *"[^"]*"' core/languages.json | sed 's/.*"\([^"]*\)"$/\1/' | sort -u)"
for m in $models; do
  [ -f "$m" ] || { echo "core/languages.json names $m, which does not exist here. Nothing sent."; exit 1; }
done

echo "Jetson: $HOST:~/$DIR"
echo "Code:   $(wc -l < "$list" | tr -d ' ') files from: $paths"
echo "Models: $(echo $models | tr '\n' ' ')"
if [ "$DRY" = 1 ]; then
  echo; echo "--dry-run: nothing was sent. The files:"; cat "$list"; echo "$models"
  exit 0
fi

# ── is it there? ────────────────────────────────────────────────────────────────
if ! $SSH "$HOST" true 2>/dev/null; then
  echo
  echo "Cannot reach $HOST. Nothing was changed."
  echo "Check that the Jetson is powered, on this network, and still at that address"
  echo "(its address can change; set JETSON_HOST=user@address to use another)."
  exit 2
fi

# ── save what is there now ──────────────────────────────────────────────────────
stamp="$(date +%Y%m%d-%H%M%S)"
$SSH "$HOST" "mkdir -p ~/signoo-backups ~/$DIR && cd ~/$DIR && \
  tar -czf ~/signoo-backups/$stamp.tgz --exclude=__pycache__ --exclude=kiosk/data/pending_uploads \
      kiosk core assets index.html 2>/dev/null; \
  ls -1t ~/signoo-backups/*.tgz 2>/dev/null | tail -n +6 | xargs -r rm -f" \
  || { echo "Could not save a backup on the Jetson. Nothing was changed."; exit 3; }
echo "Backup: ~/signoo-backups/$stamp.tgz"

# ── send ────────────────────────────────────────────────────────────────────────
tar -cf - -T "$list" | $SSH "$HOST" "tar -xf - -C ~/$DIR" \
  || { echo "Sending the code failed part-way. Restore with the command at the end."; failed=1; }

for m in $models; do
  here="$(sha256sum "$m" | cut -d' ' -f1)"
  there="$($SSH "$HOST" "sha256sum ~/$DIR/$m 2>/dev/null | cut -d' ' -f1")"
  if [ "$here" = "$there" ]; then
    echo "Model $m: already there"
  else
    $SSH "$HOST" "mkdir -p ~/$DIR/$(dirname "$m")" && scp -q -o ConnectTimeout=6 "$m" "$HOST:$DIR/$m" \
      && echo "Model $m: sent" || { echo "Model $m: FAILED to send"; failed=1; }
  fi
done

# ── restart and check ───────────────────────────────────────────────────────────
restore="ssh $HOST 'cd ~/$DIR && tar -xzf ~/signoo-backups/$stamp.tgz && systemctl --user restart signoo-kiosk signoo-kiosk-app'"
if [ "${failed:-0}" = 1 ]; then
  echo; echo "Not restarted, because not everything arrived. To put the previous version back:"; echo "  $restore"
  exit 3
fi

$SSH "$HOST" "systemctl --user restart signoo-kiosk" || { echo "Could not restart signoo-kiosk."; exit 3; }
echo -n "Waiting for the kiosk backend"
health=""
for _ in $(seq 1 45); do
  health="$($SSH "$HOST" "curl -fsS -m 3 http://127.0.0.1:$PORT/health" 2>/dev/null)" && break
  echo -n "."; sleep 2
done
echo
if [ -z "$health" ]; then
  echo "The backend did not answer /health within 90 seconds. Its log:"
  $SSH "$HOST" "journalctl --user -u signoo-kiosk -n 30 --no-pager" 2>/dev/null
  echo; echo "To put the previous version back:"; echo "  $restore"
  exit 3
fi

# The page is served uncached, but the open window still holds the old one: restart the shell.
$SSH "$HOST" "systemctl --user restart signoo-kiosk-app" || echo "Note: could not restart the kiosk window (signoo-kiosk-app)."

echo "Healthy. Languages a visitor can press:"
$SSH "$HOST" "curl -fsS -m 3 http://127.0.0.1:$PORT/languages" 2>/dev/null \
  | tr '{' '\n' | grep '"ready": *true' | sed 's/.*"code": *"\([^"]*\)".*"status": *"\([^"]*\)".*/  \1 (\2)/'
echo "Start-up test pass per model (seconds): $(echo "$health" | grep -o '"warm_up_s": *{[^}]*}')"
echo
echo "To put the previous version back:"
echo "  $restore"

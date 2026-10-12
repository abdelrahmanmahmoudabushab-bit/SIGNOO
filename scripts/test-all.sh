#!/usr/bin/env bash
# Runs every test suite and the console type-check; exits non-zero if any fails.
#
#   scripts/test-all.sh          # all suites (the server's is slow, about 8 minutes)
#   scripts/test-all.sh fast     # skip the server suite's model-training tests
#
# Run it alone: test_gate.py trains two models in subprocesses and can fail with an
# out-of-memory error if another heavy process (landmark extraction, a dev server)
# is running at the same time.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
failed=0

run() {
  echo; echo "=== $1"; shift
  "$@" || { echo "FAILED: $*"; failed=1; }
}

run "kiosk"   .venv/Scripts/python.exe -m pytest kiosk/tests -q
run "ml"      .venv/Scripts/python.exe -m pytest ml/tests -q
if [ "$1" = "fast" ]; then
  run "server (fast, no training tests)" server/.venv/Scripts/python.exe -m pytest server/tests -q --ignore=server/tests/test_gate.py
else
  run "server" server/.venv/Scripts/python.exe -m pytest server/tests -q
fi
run "console tests" bash -c "cd console && npm test -- --run"
run "console types" bash -c "cd console && npx tsc -p tsconfig.app.json --noEmit"

echo
[ "$failed" = 0 ] && echo "ALL GREEN" || echo "SOMETHING FAILED"
exit "$failed"

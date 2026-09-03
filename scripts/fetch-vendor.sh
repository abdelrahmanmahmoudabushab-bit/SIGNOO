#!/usr/bin/env bash
# Downloads the hand tracker and its model into vendor/ so SIGNOO runs without
# an internet connection. Re-run to update. Safe to run more than once.
set -euo pipefail

VERSION="0.10.14"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$VENDOR/tasks-vision" "$VENDOR/models"

echo "→ downloading @mediapipe/tasks-vision@$VERSION (~20 MB)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
( cd "$TMP" && npm pack "@mediapipe/tasks-vision@$VERSION" --silent >/dev/null && tar -xzf ./*.tgz )
cp "$TMP/package/vision_bundle.mjs" "$VENDOR/tasks-vision/"
cp -r "$TMP/package/wasm" "$VENDOR/tasks-vision/"

echo "→ downloading hand landmark model (~8 MB)"
curl -fL --progress-bar "$MODEL_URL" -o "$VENDOR/models/hand_landmarker.task"

echo
echo "✓ done — SIGNOO will now use vendor/ and works offline."
du -sh "$VENDOR"

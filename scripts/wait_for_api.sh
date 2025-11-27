#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://127.0.0.1:8000/health}"
TIMEOUT="${2:-60}"
for i in $(seq 1 "$TIMEOUT"); do
  if curl -sSf "$URL" >/dev/null; then
    echo "API is up: $URL"
    exit 0
  fi
  sleep 1
  echo "waiting ($i/$TIMEOUT): $URL"
done
echo "Timeout waiting for $URL" >&2
exit 1

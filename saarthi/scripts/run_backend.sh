#!/usr/bin/env bash
# Start the SAARTHI Flask backend on :5000
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# activate the project venv
if [ -d "$REPO/.venv" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
else
  echo "venv not found at $REPO/.venv — run scripts/setup.sh first" >&2
  exit 1
fi

cd "$HERE/backend"
export FLASK_PORT="${FLASK_PORT:-5000}"
echo "SAARTHI backend -> http://localhost:${FLASK_PORT}"
exec python app.py

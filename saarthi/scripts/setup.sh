#!/usr/bin/env bash
# One-time setup: Python venv + backend deps + sample data + frontend deps.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

echo "==> Python venv"
PY="$(command -v python3.11 || command -v python3)"
[ -d "$REPO/.venv" ] || "$PY" -m venv "$REPO/.venv"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
python -m pip install --quiet --upgrade pip wheel setuptools
echo "==> Backend deps (this can take a few minutes)"
pip install --quiet -r "$HERE/backend/requirements.txt"

echo "==> Sample datasets"
python "$HERE/scripts/make_sample_data.py"

echo "==> Frontend deps"
if [ -d "$HOME/.local/opt/node-v20.18.0-linux-x64/bin" ]; then
  export PATH="$HOME/.local/opt/node-v20.18.0-linux-x64/bin:$PATH"
fi
if command -v npm >/dev/null 2>&1; then
  (cd "$HERE/frontend" && npm install)
else
  echo "  (node/npm not found — install Node 20+ to build the frontend)"
fi

echo "==> Done. Start with: scripts/run_backend.sh  and  scripts/run_frontend.sh"

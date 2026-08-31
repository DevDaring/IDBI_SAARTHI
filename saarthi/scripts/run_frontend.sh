#!/usr/bin/env bash
# Start the SAARTHI React/Vite dev server on :5173 (proxies /api -> :5000)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# put the user-local Node 20 on PATH if present
if [ -d "$HOME/.local/opt/node-v20.18.0-linux-x64/bin" ]; then
  export PATH="$HOME/.local/opt/node-v20.18.0-linux-x64/bin:$PATH"
fi
command -v node >/dev/null 2>&1 || { echo "node not found on PATH" >&2; exit 1; }

cd "$HERE/frontend"
[ -d node_modules ] || npm install
echo "SAARTHI frontend -> http://localhost:5173"
exec npm run dev -- --host

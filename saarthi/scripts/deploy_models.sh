#!/usr/bin/env bash
# Deploy trained models + updated backend into the live host at
# /home/Debz/Host/saarthi, then restart the service.
#
# Safe by design:
#   * backs up whatever is currently deployed before overwriting
#   * verifies the app imports cleanly BEFORE restarting the service
#   * rolls back and re-starts the old code if the health check fails
set -euo pipefail

REPO="/home/Debz/Hackathon/IDBI_Hackathon"
HOST="/home/Debz/Host/saarthi"
VENV="$REPO/.venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$HOST/.deploy_backup/$STAMP"

echo "== SAARTHI model deploy $STAMP =="

if [ ! -d "$HOST/backend" ]; then
  echo "!! host backend not found at $HOST/backend" >&2
  exit 1
fi

# ---------------------------------------------------------------- backup
mkdir -p "$BACKUP"
cp -a "$HOST/backend" "$BACKUP/backend"
[ -d "$HOST/models" ] && cp -a "$HOST/models" "$BACKUP/models" || true
echo "   backed up current deploy -> $BACKUP"

# ---------------------------------------------------------------- models
mkdir -p "$HOST/models"
shopt -s nullglob
for f in "$REPO"/saarthi/models/global_canonical.joblib \
         "$REPO"/saarthi/models/metrics.json \
         "$REPO"/saarthi/models/ablation_sequence.json \
         "$REPO"/saarthi/models/coles.pt \
         "$REPO"/saarthi/models/probe_results.json; do
  [ -f "$f" ] && cp -f "$f" "$HOST/models/" && echo "   model  $(basename "$f") ($(du -h "$f" | cut -f1))"
done
shopt -u nullglob

# ---------------------------------------------------------------- backend
for f in pipeline/pretrained.py pipeline/orchestrator.py app.py; do
  if [ -f "$REPO/saarthi/backend/$f" ]; then
    cp -f "$REPO/saarthi/backend/$f" "$HOST/backend/$f"
    echo "   code   $f"
  fi
done

# ------------------------------------------------------- pre-flight check
echo "-- import check (before restart)"
if ! ( cd "$HOST/backend" && SAARTHI_GLOBAL_MODEL="$HOST/models/global_canonical.joblib" \
        "$VENV" -c "
import sys
sys.path.insert(0, '.')
from pipeline import pretrained
print('   pretrained.info():', pretrained.info())
import app  # noqa: F401
print('   app imports OK')
" ); then
  echo "!! import check FAILED - rolling back, service untouched" >&2
  rm -rf "$HOST/backend"
  cp -a "$BACKUP/backend" "$HOST/backend"
  exit 1
fi

# ---------------------------------------------------------------- restart
echo "-- restarting saarthi service"
sudo systemctl restart saarthi
sleep 6

if curl -fsS --max-time 20 http://127.0.0.1:8080/api/health >/dev/null 2>&1; then
  echo "   health OK"
  curl -fsS --max-time 20 http://127.0.0.1:8080/api/pretrained 2>/dev/null | head -c 400 || true
  echo
  echo "== deploy complete =="
else
  echo "!! health check FAILED - rolling back" >&2
  rm -rf "$HOST/backend"
  cp -a "$BACKUP/backend" "$HOST/backend"
  sudo systemctl restart saarthi
  exit 1
fi

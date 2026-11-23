#!/usr/bin/env bash
# Lance le serveur Transcribe local sur http://localhost:7878
# Usage : ./run.sh   (puis ouvre l'URL dans ton navigateur)

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${TRANSCRIBE_VENV:-$HOME/Claude code/video-pipeline/bin/whisperx-env}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "✗ Python introuvable dans $VENV" >&2
  echo "   Définis TRANSCRIBE_VENV (voir README)." >&2
  exit 1
fi

URL="http://localhost:${PORT:-7878}"
echo "→ Démarrage sur $URL"

# Ouvre le navigateur après ~1.5s, le serveur a le temps de démarrer.
( sleep 1.5; open "$URL" ) &

cd "$HERE"
exec "$PY" server.py

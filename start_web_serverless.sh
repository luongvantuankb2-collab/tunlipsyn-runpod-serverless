#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/web_saas"

if [ ! -f .env ]; then
  cp .env.runpod.example .env
  echo "Created web_saas/.env from .env.runpod.example. Edit RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, PUBLIC_BASE_URL, WORKER_TOKEN."
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -x "$ROOT_DIR/.venv_web/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv_web/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python not found" >&2
    exit 1
  fi
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
APP_HOST="$(grep -E '^APP_HOST=' .env | tail -n 1 | cut -d= -f2- || true)"
APP_PORT="$(grep -E '^APP_PORT=' .env | tail -n 1 | cut -d= -f2- || true)"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8080}"

mkdir -p storage work ../logs

echo "Starting Tun Lipsync web/serverless backend on ${APP_HOST}:${APP_PORT}"
exec "$PYTHON_BIN" -m uvicorn backend.app:app --host "$APP_HOST" --port "$APP_PORT"

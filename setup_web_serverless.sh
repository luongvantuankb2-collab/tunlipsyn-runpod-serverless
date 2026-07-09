#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv_web/bin/python ]; then
  python3 -m venv .venv_web
fi

.venv_web/bin/python -m pip install --upgrade pip wheel
.venv_web/bin/python -m pip install -r web_saas/backend/requirements.txt

if [ ! -f web_saas/.env ]; then
  cp web_saas/.env.runpod.example web_saas/.env
fi

echo "Web serverless setup done."
echo "Edit: web_saas/.env"
echo "Run:  nohup bash start_web_serverless.sh > logs/web.log 2>&1 &"

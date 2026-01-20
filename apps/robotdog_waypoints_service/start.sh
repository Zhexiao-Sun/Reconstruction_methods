#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/openbayes/home/Reconstruction_methods}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

exec python -m uvicorn apps.robotdog_waypoints_service.app:app \
  --app-dir "${APP_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${WORKERS}"



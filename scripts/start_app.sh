#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export ALLCAPS_STATE_DIR="${ALLCAPS_STATE_DIR:-/var/data}"
mkdir -p "${ALLCAPS_STATE_DIR}"
APP_RUNTIME="$(printf '%s' "${ALLCAPS_RUNTIME:-web}" | tr '[:upper:]' '[:lower:]')"

SCHEDULER_PID=""
cleanup() {
  if [[ -n "${SCHEDULER_PID}" ]]; then
    kill "${SCHEDULER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

case "${ALLCAPS_SCHEDULER_ENABLED:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    python -m trump_workbench.scheduler &
    SCHEDULER_PID="$!"
    ;;
esac

case "$APP_RUNTIME" in
  web|fastapi|react)
    exec python -m uvicorn trump_workbench.api:app \
      --host 0.0.0.0 \
      --port "${PORT:-8000}"
    ;;
  streamlit)
    exec streamlit run app.py \
      --server.address 0.0.0.0 \
      --server.port "${PORT:-8501}" \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  *)
    echo "Unsupported ALLCAPS_RUNTIME '$APP_RUNTIME'. Use 'web' or 'streamlit'." >&2
    exit 2
    ;;
esac

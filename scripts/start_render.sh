#!/usr/bin/env bash
set -euo pipefail

export ALLCAPS_STATE_DIR="${ALLCAPS_STATE_DIR:-/var/data}"
mkdir -p "${ALLCAPS_STATE_DIR}"

SCHEDULER_PID=""
cleanup() {
  if [[ -n "${SCHEDULER_PID}" ]]; then
    kill "${SCHEDULER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

case "${ALLCAPS_SCHEDULER_ENABLED:-true}" in
  1|true|TRUE|yes|YES|on|ON)
    python -m trump_workbench.scheduler &
    SCHEDULER_PID="$!"
    ;;
esac

exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.headless true \
  --browser.gatherUsageStats false

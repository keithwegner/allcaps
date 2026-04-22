#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No Python interpreter found on PATH." >&2
  exit 1
fi

PORT="${STREAMLIT_SMOKE_PORT:-8513}"
STARTUP_TIMEOUT_SECONDS="${STREAMLIT_SMOKE_TIMEOUT_SECONDS:-45}"
LOG_FILE="${STREAMLIT_SMOKE_LOG_FILE:-${TMPDIR:-/tmp}/allcaps-streamlit-smoke.log}"

echo "==> Compile check"
"$PYTHON_BIN" -m py_compile app.py trump_workbench/*.py tests/*.py

echo "==> Test suite with coverage"
"$PYTHON_BIN" -m coverage erase
"$PYTHON_BIN" -m coverage run -m unittest discover -s tests -v

echo "==> Coverage gate"
"$PYTHON_BIN" -m coverage report --fail-under=90 -m
"$PYTHON_BIN" -m coverage xml

if [[ -f "$ROOT_DIR/frontend/package.json" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Frontend build"
    npm ci --prefix "$ROOT_DIR/frontend"
    npm run build --prefix "$ROOT_DIR/frontend"
    echo "==> Frontend unit tests with coverage"
    npm run test:coverage --prefix "$ROOT_DIR/frontend"
    echo "==> Frontend UI tests"
    npm exec --prefix "$ROOT_DIR/frontend" playwright install chromium
    npm run test:ui --prefix "$ROOT_DIR/frontend"
  else
    echo "npm is required for the frontend build but was not found on PATH." >&2
    exit 1
  fi
fi

echo "==> Streamlit smoke test"
rm -f "$LOG_FILE"
"$PYTHON_BIN" -m streamlit run app.py --server.headless true --server.port "$PORT" >"$LOG_FILE" 2>&1 &
app_pid=$!

cleanup() {
  if kill -0 "$app_pid" >/dev/null 2>&1; then
    kill "$app_pid" >/dev/null 2>&1 || true
    wait "$app_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" - "$PORT" "$STARTUP_TIMEOUT_SECONDS" "$LOG_FILE" <<'PY'
from __future__ import annotations

import pathlib
import sys
import time
import urllib.request

port = int(sys.argv[1])
timeout_seconds = int(sys.argv[2])
log_path = pathlib.Path(sys.argv[3])
url = f"http://127.0.0.1:{port}"
deadline = time.time() + timeout_seconds
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status < 500:
                print(f"Streamlit smoke check passed on {url}")
                sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
    time.sleep(1)

log_excerpt = ""
if log_path.exists():
    log_excerpt = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]

sys.stderr.write(
    f"Streamlit failed to start on {url} within {timeout_seconds} seconds. "
    f"Last error: {last_error}\n",
)
if log_excerpt:
    sys.stderr.write("\n--- Streamlit log tail ---\n")
    sys.stderr.write(log_excerpt)
    sys.stderr.write("\n")
sys.exit(1)
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export ALLCAPS_STATE_DIR="${ALLCAPS_STATE_DIR:-/var/data}"
export ALLCAPS_PUBLIC_MODE="${ALLCAPS_PUBLIC_MODE:-true}"
export ALLCAPS_AUTO_BOOTSTRAP_ON_START="${ALLCAPS_AUTO_BOOTSTRAP_ON_START:-false}"
export ALLCAPS_SCHEDULER_ENABLED="${ALLCAPS_SCHEDULER_ENABLED:-true}"

exec bash "$ROOT_DIR/scripts/start_app.sh"

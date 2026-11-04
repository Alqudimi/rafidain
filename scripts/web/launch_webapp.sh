#!/usr/bin/env bash
# Rafidain web launcher: install/sync, run /web + /live, and optionally tunnel.
# Usage: bash scripts/web/launch_webapp.sh [--engine mock|seedvc] [--no-tunnel]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$ROOT"
if [[ " ${*:-} " == *" --stop "* ]]; then
  exec uv run --python 3.11 python scripts/manage.py stop
fi
if [[ " ${*:-} " == *" --status "* ]]; then
  exec uv run --python 3.11 python scripts/manage.py status
fi
ENGINE="auto"; PORT="${RAFIDAIN_PORT:-8000}"; TUNNEL=1; MOCK=0
ARGS=()
while (($#)); do
  case "$1" in
    --engine) ENGINE="${2:?missing engine}"; shift 2 ;;
    --port) PORT="${2:?missing port}"; shift 2 ;;
    --no-tunnel) TUNNEL=0; shift ;;
    --mock) MOCK=1; shift ;;
    --no-models) shift ;;
    --api-key|--device|--source|--ref) shift 2 ;;
    -h|--help) sed -n '1,22p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ "$ENGINE" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ENGINE=seedvc
  else
    ENGINE=mock
  fi
  echo "[rafidain] engine=auto -> $ENGINE"
fi
make install ENGINE="$ENGINE"
if [[ "$MOCK" == 1 || "$ENGINE" == mock ]]; then
  make web-mock PORT="$PORT"
else
  make web PORT="$PORT"
fi
if [[ "$TUNNEL" == 1 ]]; then
  make tunnel PORT="$PORT"
  make tunnel-url
fi

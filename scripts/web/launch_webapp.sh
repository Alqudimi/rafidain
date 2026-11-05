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
  echo "[rafidain] بدء Cloudflare Quick Tunnel..."
  set +e
  make tunnel PORT="$PORT"
  TUNNEL_RC=$?
  set -e
  # manage.py normally writes .run/tunnel.url; the fallback also extracts a URL
  # from the log when cloudflared is slow to flush its first output.
  if [[ ! -s .run/tunnel.url && -f .run/tunnel.log ]]; then
    grep -Eo 'https://[A-Za-z0-9-]+\.trycloudflare\.com' .run/tunnel.log | tail -n1 > .run/tunnel.url || true
  fi
  for _ in $(seq 1 15); do
    [[ -s .run/tunnel.url ]] && break
    sleep 1
    grep -Eo 'https://[A-Za-z0-9-]+\.trycloudflare\.com' .run/tunnel.log 2>/dev/null | tail -n1 > .run/tunnel.url || true
  done
  if [[ -s .run/tunnel.url ]]; then
    PUBLIC_URL="$(cat .run/tunnel.url)"
    echo ""
    echo "[rafidain] الرابط العام المؤقت المجاني: $PUBLIC_URL"
    echo "[rafidain] لوحة التحكم: $PUBLIC_URL/web"
    echo "[rafidain] البث الحي: $PUBLIC_URL/live"
    echo "[rafidain] API: $PUBLIC_URL/docs"
  else
    echo "[rafidain] تعذر استخراج رابط Cloudflare (exit=$TUNNEL_RC). راجع: .run/tunnel.log" >&2
    exit "${TUNNEL_RC:-1}"
  fi
else
  echo "[rafidain] الرابط المحلي: http://127.0.0.1:$PORT/web"
  echo "[rafidain] النفق معطل بسبب --no-tunnel"
fi

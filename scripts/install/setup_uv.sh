#!/usr/bin/env bash
# ============================================================================
#  RAFIDAIN — unified uv installer (Linux / macOS / WSL / Google Colab)
#  رافدَين — مثبّت موحّد واحد لكل البيئات، مبني على uv والمكتشف الذكي
# ----------------------------------------------------------------------------
#  Bootstrap: install + detect + download + serve + tunnel, in one command.
#
#  USAGE:
#    bash scripts/install/setup_uv.sh                      # from the repo
#    curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_uv.sh | bash
#
#  OPTIONS:
#    --engine seedvc|mock   default: seedvc on GPU, mock otherwise
#    --port N               server port (default 8000)
#    --api-key KEY          fixed API key (default: random)
#    --no-auth              disable API auth (development only)
#    --no-models            skip the ~1.5 GB checkpoint pre-download
#    --no-tests             skip the test suite
#    --no-tunnel            skip the public trycloudflare.com tunnel
#    --no-serve             install only, do not start the server
#    --variant cpu|cu118|cu124|cu126   override torch wheel detection
#    --source URL|DIR       project source (default: this repo / auto-clone)
#    -h | --help
# ============================================================================
set -u -Eeuo pipefail

RAFIDAIN_GIT="https://github.com/Alqudimi/rafidain"
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)" || SCRIPT_DIR=""

# ------------------------------------------------------------------ helpers
if { [ -t 1 ] || [ -t 2 ]; } && [ -z "${NO_COLOR:-}" ]; then
  C_HDR=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
  C_ERR=$'\033[1;31m'; C_DIM=$'\033[2m'; C_NC=$'\033[0m'
else
  C_HDR=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_NC=""
fi
log()  { printf '%s\n' "${C_DIM}[..]${C_NC} $*"; }
ok()   { printf '%s\n' "${C_OK}[✓]${C_NC} $*"; }
warn() { printf '%s\n' "${C_WARN}[!]${C_NC} $*"; }
err()  { printf '%s\n' "${C_ERR}[✗]${C_NC} $*" >&2; }
die()  { err "$*"; exit 1; }
usage() { sed -n '2,35p' "$SCRIPT_PATH" 2>/dev/null | sed 's/^# \{0,1\}//'; }

# ----------------------------------------------------------------- arguments
ENGINE=""; PORT="${RAFIDAIN_PORT:-8000}"; API_KEY="${RAFIDAIN_API_KEY:-}"
NO_AUTH=0; NO_MODELS=0; NO_TESTS=0; NO_TUNNEL=0; NO_SERVE=0
VARIANT=""; SOURCE="${RAFIDAIN_SOURCE:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    ENGINE="${2:?}"; shift 2 ;;
    --port)      PORT="${2:?}"; shift 2 ;;
    --api-key)   API_KEY="${2:?}"; shift 2 ;;
    --variant)   VARIANT="${2:?}"; shift 2 ;;
    --source)    SOURCE="${2:?}"; shift 2 ;;
    --no-auth)   NO_AUTH=1; shift ;;
    --no-models) NO_MODELS=1; shift ;;
    --no-tests)  NO_TESTS=1; shift ;;
    --no-tunnel) NO_TUNNEL=1; shift ;;
    --no-serve)  NO_SERVE=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

printf '%s\n' "${C_HDR}"
printf '%s\n' '  ╭──────────────────────────────────────────────────────────╮'
printf '%s\n' '  │   رافدَين · Rafidain — unified uv installer              │'
printf '%s\n' '  │   realtime Iraqi voice conversion · Seed-VC tiny        │'
printf '%s\n' '  ╰──────────────────────────────────────────────────────────╯'
printf '%s\n' "${C_NC}"

# ------------------------------------------------------- [1/7] find project
is_project_dir() { [ -f "$1/pyproject.toml" ] && grep -q 'name = "rafidain"' "$1/pyproject.toml" 2>/dev/null; }
PROJECT=""
if [ -n "$SOURCE" ]; then
  if [ -d "$SOURCE" ]; then
    PROJECT="$(cd "$SOURCE" && pwd)"
  else
    log "git clone: $SOURCE"
    INSTALL_DIR="$PWD/rafidain"
    rm -rf "$INSTALL_DIR"
    git clone --depth 1 "$SOURCE" "$INSTALL_DIR" || die "clone failed: $SOURCE"
    PROJECT="$INSTALL_DIR"
  fi
elif is_project_dir "$PWD"; then
  PROJECT="$PWD"
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR/../.."; then
  PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR"; then
  PROJECT="$SCRIPT_DIR"
fi
if [ -z "$PROJECT" ]; then
  log "project not found nearby — cloning $RAFIDAIN_GIT"
  INSTALL_DIR="$PWD/rafidain"
  rm -rf "$INSTALL_DIR"
  git clone --depth 1 "$RAFIDAIN_GIT" "$INSTALL_DIR" || die "could not clone $RAFIDAIN_GIT"
  PROJECT="$INSTALL_DIR"
fi
ok "project: $PROJECT"
cd "$PROJECT"

# ------------------------------------------------------ [2/7] detect environment
ENV_JSON="$(python3 scripts/manage.py detect 2>/dev/null || true)"
OS_NAME="$(printf '%s' "$ENV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["os"])' 2>/dev/null || echo linux)"
IS_COLAB="$(printf '%s' "$ENV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["colab"])' 2>/dev/null || echo False)"
HAS_GPU="$(printf '%s' "$ENV_JSON" | python3 -c 'import json,sys; g=json.load(sys.stdin)["gpu"]; print(bool(g))' 2>/dev/null || echo False)"
VARIANT_AUTO="$(printf '%s' "$ENV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["torch_variant"])' 2>/dev/null || echo cpu)"
[ "$IS_COLAB" = "True" ] && ok "Google Colab detected" || log "environment: $OS_NAME"
[ "$HAS_GPU" = "True" ] && ok "GPU detected → torch extra: ${VARIANT}" || warn "no GPU → mock engine or CPU torch (latency above call budgets)"

if [ -z "$ENGINE" ]; then
  [ "$HAS_GPU" = "True" ] && ENGINE="seedvc" || ENGINE="mock"
fi
if [ -z "$VARIANT" ]; then
  VARIANT="$VARIANT_AUTO"
fi
command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg missing — voice uploads need it (apt-get install -y ffmpeg)"
ok "engine=$ENGINE · torch variant=$VARIANT · port=$PORT"

# ------------------------------------------------------------ [3/7] install
log "syncing the locked uv environment (this pins every dependency)…"
if [ "$ENGINE" = "mock" ]; then
  python3 scripts/manage.py install --engine mock
else
  VARG=""; [ "$VARIANT" != "auto" ] && VARG="--variant $VARIANT"
  # shellcheck disable=SC2086
  python3 scripts/manage.py install --engine seedvc $VARG
fi
ok "python environment ready (uv.lock, reproducible)"

# -------------------------------------------------- [4/7] seed-vc + models
python3 scripts/manage.py seedvc
if [ "$ENGINE" = "seedvc" ] && [ "$NO_MODELS" = 0 ]; then
  log "pre-downloading checkpoints (~1.5 GB incl. XLSR)…"
  python3 scripts/manage.py download || warn "checkpoint download failed — first start will retry"
else
  log "model download skipped (mock / --no-models)"
fi

# ------------------------------------------------------------------ [5/7] .env
if [ -f .env ]; then
  cp .env ".env.bak.$(date +%Y%m%d-%H%M%S)"
  log "existing .env backed up"
fi
if [ -z "$API_KEY" ]; then
  API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
fi
AUTH_LINE="RAFIDAIN__API_KEYS=[\"$API_KEY\"]"
[ "$NO_AUTH" = 1 ] && AUTH_LINE="RAFIDAIN__API_KEYS=[]"
cat > .env <<EOF
# Generated by setup_uv.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
RAFIDAIN__HOST=0.0.0.0
RAFIDAIN__PORT=$PORT
RAFIDAIN__LOG_LEVEL=INFO
RAFIDAIN__LOG_JSON=true
$AUTH_LINE
RAFIDAIN__CORS_ORIGINS=["*"]
RAFIDAIN__ENGINE__IMPL=$ENGINE
RAFIDAIN__ENGINE__DEVICE=auto
RAFIDAIN__ENGINE__FP16=true
SEED_VC_ROOT=./third_party/seed-vc
RAFIDAIN__ENGINE__CACHE_DIR=./checkpoints
RAFIDAIN__ENGINE__EAGER_LOAD=true
RAFIDAIN__ENGINE__WARMUP=true
RAFIDAIN__DEFAULT_PROFILE=balanced
RAFIDAIN__SESSION_TTL_S=300
RAFIDAIN__MAX_SESSIONS=64
RAFIDAIN__SEND_QUEUE_SIZE=64
RAFIDAIN__VAD=off
RAFIDAIN__DATA_DIR=./data
RAFIDAIN__REDIS_URL=
RAFIDAIN__WEBUI=true
EOF
[ "$NO_AUTH" = 1 ] && warn "authentication DISABLED (--no-auth) — never do this on a shared server"
ok ".env written (engine=$ENGINE)"

# ------------------------------------------------------------------ [6/7] tests
if [ "$NO_TESTS" = 0 ]; then
  log "running the test suite (mock engine — no GPU needed)…"
  if uv run --python 3.11 python -m pytest -q >/dev/null 2>&1; then
    ok "all tests passed"
  else
    warn "some tests failed — inspect: uv run pytest -q"
  fi
fi

# ------------------------------------------------------------------ [7/7] serve
if [ "$NO_SERVE" = 1 ]; then
  printf '%s\n' "${C_OK}  ✅ التثبيت مكتمل — شغّل لاحقاً:  make web ثم make tunnel للبث العام${C_NC}"
  exit 0
fi

log "starting the server + web UI…"
python3 scripts/manage.py stop >/dev/null 2>&1 || true
if [ "$ENGINE" = "mock" ]; then
  python3 scripts/manage.py web --mock --port "$PORT" --wait 180
else
  python3 scripts/manage.py web --port "$PORT" --wait 600
fi

TUNNEL_URL=""
if [ "$NO_TUNNEL" = 0 ]; then
  log "opening the free public tunnel (trycloudflare.com)…"
  if python3 scripts/manage.py tunnel start --port "$PORT"; then
    TUNNEL_URL="$(cat .run/tunnel.url 2>/dev/null || true)"
  else
    warn "tunnel unavailable — the server is still reachable locally"
  fi
fi

# -------------------------------------------------------------------- summary
printf '%s\n' ""
printf '%s\n' "${C_OK}  ✅ رافدَين جاهز — RAFIDAIN IS UP${C_NC}"
printf '%s\n' "  ─────────────────────────────────────────────────────────"
printf '  %-16s %s\n' "Local"        "http://127.0.0.1:$PORT"
printf '  %-16s %s\n' "لوحة التحكم"  "http://127.0.0.1:$PORT/web"
printf '  %-16s %s\n' "البث الحي"    "http://127.0.0.1:$PORT/live"
[ -n "$TUNNEL_URL" ] && printf '  %-16s %s\n' "الرابط العام" "$TUNNEL_URL"
if [ "$NO_AUTH" = 0 ]; then
  printf '  %-16s %s\n' "API key"    "$API_KEY"
fi
printf '  %-16s %s\n' "الأوامر"      "make status · make stop · make train ARGS='…'"
printf '%s\n' "${C_DIM}  ⚠ استنساخ صوت شخص يتطلب موافقته. النفق العام غير مصادق —"
printf '%s\n' "    أبقِ المفتاح سرياً. على Colab يُعاد التشغيل بإعادة السكربت نفسه.${C_NC}"

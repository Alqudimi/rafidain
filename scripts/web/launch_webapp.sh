#!/usr/bin/env bash
# ============================================================================
#  RAFIDAIN — Web App launcher (realtime voice-conversion control panel)
#  رافدَين — مشغّل لوحة التحكم: تحويل صوت لحظي من الميكروفون أو من ملف
# ----------------------------------------------------------------------------
#  Starts the API + the built-in web page and (by default) broadcasts it on a
#  free temporary public domain via Cloudflare (cloudflared quick tunnel).
#
#  USAGE (single command — script lives in the Alqudimi/rafidain repo):
#    curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/web/launch_webapp.sh | bash
#    curl -fsSL <url> | bash -s -- --port 8000 --engine mock --no-tunnel
#  Or locally:
#    bash scripts/web/launch_webapp.sh [--options]
#
#  OPTIONS:
#    --port N           API/port for the web page (default 8000 or .env)
#    --engine seedvc|mock  engine (default: seedvc when GPU + deps, else mock)
#    --device auto|cuda|cpu
#    --api-key KEY      fixed API key (default: reuse .env / generate one)
#    --no-auth          disable API authentication (development only)
#    --no-tunnel        do NOT open the public trycloudflare.com URL
#    --no-models        with --engine seedvc: skip checkpoint pre-download
#    --foreground       run attached to this shell (Ctrl+C stops)
#    --stop             stop a running webapp + tunnel
#    --status           show running state, URLs and pids
#    --skip-deps        do not install anything (use the env as-is)
#    --source SRC       project source: local dir / git URL (rarely needed)
#    --ref BRANCH       git branch when cloning (default main)
#    -h | --help        show this help
# ============================================================================
set -Eeuo pipefail

# ----------------------------------------------------------------- constants
RAFIDAIN_GIT="https://github.com/Alqudimi/rafidain"
UV_INSTALL_SH="https://astral.sh/uv/install.sh"
CLOUDFLARED_BASE="https://github.com/cloudflare/cloudflared/releases/latest/download"
READY_TIMEOUT_S="${RAFIDAIN_READY_TIMEOUT_S:-900}"

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
hr()   { printf '%s\n' "${C_DIM}──────────────────────────────────────────────────────────────${C_NC}"; }
usage() { sed -n '2,33p' "${BASH_SOURCE[0]:-$0}" 2>/dev/null | sed 's/^# \{0,1\}//'; }
spinner() {  # spinner <pid> <message>  (live progress while a pid runs)
  local pid="$1" msg="$2" spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r%s %s ' "${spin:i++%8:1}" "$msg" >&2
    sleep 0.12
  done
  printf '\r\033[K' >&2
}
trap 'rc=$?; [ "$rc" -ne 0 ] && { printf "\n"; err "launch failed (exit $rc)."; }' EXIT
STEP_N=0; STEP_T=6
step() { STEP_N=$((STEP_N+1)); printf '\n%s\n' "${C_HDR}━━━ [$STEP_N/$STEP_T] $* ━━━${C_NC}"; }
if [ -t 1 ]; then PIPV=(--progress-bar on); CURLV=(--progress-bar); else PIPV=(-q); CURLV=(-sS); fi

# ----------------------------------------------------------------- arguments
PORT="${RAFIDAIN_PORT:-}"; ENGINE=""; DEVICE=""; API_KEY="${RAFIDAIN_API_KEY:-}"
NO_AUTH=0; TUNNEL=1; NO_MODELS=0; FOREGROUND=0; ACTION="run"; SKIP_DEPS=0
SOURCE="${RAFIDAIN_SOURCE:-}"; REF="${RAFIDAIN_REF:-main}"
while [ $# -gt 0 ]; do
  case "$1" in
    --port)       PORT="${2:?}"; shift 2 ;;
    --engine)     ENGINE="${2:?}"; shift 2 ;;
    --device)     DEVICE="${2:?}"; shift 2 ;;
    --api-key)    API_KEY="${2:?}"; shift 2 ;;
    --source)     SOURCE="${2:?}"; shift 2 ;;
    --ref)        REF="${2:?}"; shift 2 ;;
    --no-auth)    NO_AUTH=1; shift ;;
    --no-tunnel)  TUNNEL=0; shift ;;
    --no-models)  NO_MODELS=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --skip-deps)  SKIP_DEPS=1; shift ;;
    --stop)       ACTION="stop"; shift ;;
    --status)     ACTION="status"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# ---------------------------------------------------------------- find project
is_project_dir() { [ -f "$1/pyproject.toml" ] && grep -q 'name = "rafidain"' "$1/pyproject.toml" 2>/dev/null; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
find_project() {
  local d
  for d in "$PWD" "$PWD/rafidain" "$PWD/rafidain-main" "$PWD/rafidain-master" \
           "$PWD/rafidain-backend" "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
    [ -n "$d" ] && is_project_dir "$d" && { printf '%s' "$d"; return 0; }
  done
  return 1
}
PROJECT="$(find_project || true)"
if [ -z "$PROJECT" ]; then
  log "project not found nearby — cloning $RAFIDAIN_GIT (ref: $REF)"
  command -v git >/dev/null 2>&1 || die "git not found — clone $RAFIDAIN_GIT manually and re-run from inside it"
  git clone --depth 1 --branch "$REF" "$RAFIDAIN_GIT" "$PWD/rafidain" >/dev/null 2>&1
  is_project_dir "$PWD/rafidain" && PROJECT="$PWD/rafidain" || die "clone failed — see instructions in README.md"
fi
cd "$PROJECT"

# ------------------------------------------------------------ stop / status
do_stop() {
  step "Stopping the Rafidain web app"
  for f in webapp.pid tunnel.pid; do
    if [ -f "$f" ]; then
      P="$(cat "$f")"
      if kill -0 "$P" 2>/dev/null; then kill "$P" 2>/dev/null && ok "stopped pid $P ($f)"; else warn "$f stale (pid $P not running)"; fi
      rm -f "$f"
    fi
  done
  pkill -f "rafidain.main" 2>/dev/null && ok "rafidain server stopped" || log "no rafidain server running"
  pkill -f "cloudflared tunnel" 2>/dev/null && ok "cloudflared tunnel stopped" || log "no tunnel running"
  ok "done"
  exit 0
}
do_status() {
  step "Rafidain web app status"
  for f in webapp.pid tunnel.pid; do
    if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
      ok "$f: running (pid $(cat "$f"))"
    else
      warn "$f: not running"
    fi
  done
  PORT_G="${PORT:-8000}"
  [ -z "$PORT_G" ] && PORT_G=8000
  for ep in healthz readyz; do
    R="$(curl -sf -m 3 "http://127.0.0.1:$PORT_G/$ep" 2>/dev/null || true)"
    [ -n "$R" ] && ok "/$ep → $R" || warn "/$ep unreachable (server down on port $PORT_G?)"
  done
  TU="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log 2>/dev/null | head -n1 || true)"
  [ -n "$TU" ] && ok "public tunnel: $TU" || log "no tunnel URL on record"
  exit 0
}
[ "$ACTION" = "stop" ] && do_stop
[ "$ACTION" = "status" ] && do_status

# ---------------------------------------------------------- [1/6] environment
step "Environment check"
command -v curl >/dev/null 2>&1 || die "curl is required"
IS_COLAB=0; [ -d /content ] && python3 -c "import google.colab" >/dev/null 2>&1 && IS_COLAB=1
[ "$IS_COLAB" = 1 ] && log "Google Colab runtime detected"

# ------------------------------------------------------------- [2/6] uv env
step "Python environment (uv-managed)"
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  log "uv not found — installing (standalone, no root needed)"
  if curl -fsSL --retry 2 "$UV_INSTALL_SH" -o /tmp/uv-install.sh 2>/dev/null && sh /tmp/uv-install.sh >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    UV_BIN="$(command -v uv || true)"
  fi
fi
VENV_PY="$PROJECT/.venv/bin/python"

env_ok() {  # env_ok <python-bin> — fast sanity probe
  [ -x "$1" ] && "$1" -c "import rafidain, fastapi, uvicorn, numpy, soxr" >/dev/null 2>&1
}

if [ "$SKIP_DEPS" = 1 ]; then
  log "--skip-deps: using the environment as-is"
elif env_ok "$VENV_PY"; then
  ok "existing .venv is healthy (uv: ${UV_BIN:-found later})"
elif [ -n "$UV_BIN" ]; then
  log "preparing .venv with uv ($("$UV_BIN" --version 2>/dev/null | head -1))"
  if [ ! -x "$VENV_PY" ]; then
    if "$UV_BIN" venv --python 3.11 "$PROJECT/.venv" >/dev/null 2>&1 \
       || "$UV_BIN" venv --python 3.10 "$PROJECT/.venv" >/dev/null 2>&1; then :; else
      # fall back to whatever python uv can find (downloads a managed build)
      "$UV_BIN" venv "$PROJECT/.venv" >/dev/null 2>&1 || true
    fi
  fi
  if [ -x "$VENV_PY" ]; then
    REQ="requirements.txt"; [ -f "$PROJECT/requirements.lock" ] && REQ="requirements.lock"
    log "uv pip install -r $REQ -e . (fast)"
    "$UV_BIN" pip install --python "$VENV_PY" -q -r "$PROJECT/$REQ" -e "$PROJECT" 2>&1 | tail -2 || true
  fi
else
  warn "uv unavailable — falling back to python3 -m venv + pip"
  if [ ! -x "$VENV_PY" ]; then
    command -v python3 >/dev/null 2>&1 || die "no python3 and no uv — install Python 3.10+ or uv"
    python3 -m venv "$PROJECT/.venv" || die "venv creation failed (python3-venv missing?)"
  fi
  "$VENV_PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
  "$VENV_PY" -m pip install --no-input "${PIPV[@]}" -r "$PROJECT/requirements.txt" -e "$PROJECT" \
    || die "dependency install failed — re-run or use scripts/install/setup_ubuntu.sh"
fi
env_ok "$VENV_PY" || die "environment is not usable even after install — run scripts/install/setup_ubuntu.sh first"
ok "python env ready: $VENV_PY"
PYBIN="$VENV_PY"

# ------------------------------------------------------- [3/6] engine choice
step "Engine selection"
GPU_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then GPU_PRESENT=1; fi
ML_OK=0; "$PYBIN" -c "import torch, librosa, soundfile" >/dev/null 2>&1 && ML_OK=1
if [ -z "$ENGINE" ]; then
  if [ "$GPU_PRESENT" = 1 ] && [ "$ML_OK" = 1 ]; then ENGINE="seedvc"; else ENGINE="mock"; fi
fi
if [ "$ENGINE" = "seedvc" ]; then
  [ "$ML_OK" = 1 ] || die "engine=seedvc needs torch/librosa/soundfile — run the installer or: make install-finetune"
  [ -d "$PROJECT/third_party/seed-vc" ] || warn "third_party/seed-vc missing — run: make seedvc (the server will try to import it)"
fi
if [ "$ENGINE" = "mock" ]; then
  warn "engine=mock — صوتك سيعود تقريباً كما هو (وضع تجريبي بلا نموذج)"
fi
[ -z "$DEVICE" ] && { [ "$GPU_PRESENT" = 1 ] && DEVICE="auto" || DEVICE="cpu"; }
ok "engine=$ENGINE device=$DEVICE"

# ------------------------------------------------------------- [4/6] config
step "Configuration (.env)"
if [ ! -f .env ]; then
  if [ -z "$API_KEY" ]; then API_KEY="$("$PYBIN" -c 'import secrets; print(secrets.token_hex(24))')"; fi
  AUTH_LINE="RAFIDAIN__API_KEYS=[\"$API_KEY\"]"
  [ "$NO_AUTH" = 1 ] && AUTH_LINE="RAFIDAIN__API_KEYS=[]"
  FP16="true"; [ "$DEVICE" = "cpu" ] && FP16="false"
  cat > .env <<EOF
# Generated by launch_webapp.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
RAFIDAIN__HOST=0.0.0.0
RAFIDAIN__PORT=${PORT:-8000}
RAFIDAIN__LOG_LEVEL=INFO
RAFIDAIN__CORS_ORIGINS=["*"]
$AUTH_LINE
RAFIDAIN__ENGINE__IMPL=$ENGINE
RAFIDAIN__ENGINE__DEVICE=$DEVICE
RAFIDAIN__ENGINE__FP16=$FP16
RAFIDAIN__ENGINE__CACHE_DIR=./checkpoints
RAFIDAIN__ENGINE__EAGER_LOAD=true
RAFIDAIN__ENGINE__WARMUP=true
RAFIDAIN__DEFAULT_PROFILE=balanced
RAFIDAIN__SESSION_TTL_S=600
RAFIDAIN__MAX_SESSIONS=64
RAFIDAIN__SEND_QUEUE_SIZE=64
RAFIDAIN__VAD=off
RAFIDAIN__DATA_DIR=./data
RAFIDAIN__REDIS_URL=
EOF
  NEW_ENV=1
  ok ".env generated (engine=$ENGINE)"
else
  ok ".env found — reusing it (delete .env to regenerate)"
  [ -n "$PORT" ] && warn "--port $PORT overrides .env via environment"
fi
[ -n "$PORT" ] && export RAFIDAIN__PORT="$PORT"
PORT="${PORT:-8000}"
if [ -z "${NEW_ENV:-}" ] && [ -n "$API_KEY" ]; then export RAFIDAIN__API_KEYS="[\"$API_KEY\"]"; fi
if [ "$NO_AUTH" = 1 ]; then export RAFIDAIN__API_KEYS="[]"; warn "authentication DISABLED (--no-auth)"; fi

if [ "$ENGINE" = "seedvc" ] && [ "$NO_MODELS" = 0 ] && [ ! -f checkpoints/.rafidain_models_ok ]; then
  log "pre-downloading checkpoints (skip next time with --no-models)…"
  "$PYBIN" scripts/download_models.py --cache-dir ./checkpoints && touch checkpoints/.rafidain_models_ok \
    || warn "checkpoint pre-download failed — the server will fetch on first start"
fi

# ------------------------------------------------------------- [5/6] server
step "Starting the server on port $PORT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [ "$FOREGROUND" = 1 ]; then
  log "foreground mode — open http://127.0.0.1:$PORT/ then Ctrl+C to stop"
  exec "$PYBIN" -m rafidain.main
fi
pkill -f "rafidain.main" 2>/dev/null || true; sleep 0.5
nohup "$PYBIN" -m rafidain.main > webapp.log 2>&1 < /dev/null &
echo $! > webapp.pid
log "server pid $(cat webapp.pid) — waiting for /healthz"
UP=0
for _ in $(seq 1 60); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && { UP=1; break; }
  kill -0 "$(cat webapp.pid)" 2>/dev/null || break
  sleep 1
done
[ "$UP" = 1 ] || { err "server did not start — last log lines:"; tail -n 25 webapp.log; exit 1; }
ok "/healthz OK — الصفحة تعمل على http://127.0.0.1:$PORT/"
log "waiting for the engine (/readyz; seedvc first load can take minutes)"
READY=0
for _ in $(seq 1 "$READY_TIMEOUT_S"); do
  R="$(curl -sf -m 2 "http://127.0.0.1:$PORT/readyz" 2>/dev/null | "$PYBIN" -c 'import json,sys; print(json.load(sys.stdin).get("ready"))' 2>/dev/null || echo "")"
  [ "$R" = "True" ] && { READY=1; break; }
  sleep 1
done
[ "$READY" = 1 ] && ok "/readyz OK — المحرك جاهز" || warn "engine still loading — الصفحة تعمل لكن البث ينتظر المحرك (tail -f webapp.log)"

# ------------------------------------------------------------- [6/6] tunnel
TUNNEL_URL=""
if [ "$TUNNEL" = 1 ]; then
  step "Public tunnel (Cloudflare quick tunnel — free, no account)"
  ARCH="$(uname -m)"; case "$ARCH" in
    x86_64) CF_BIN="cloudflared-linux-amd64" ;;
    aarch64|arm64) CF_BIN="cloudflared-linux-arm64" ;;
    *) CF_BIN="" ;;
  esac
  if command -v cloudflared >/dev/null 2>&1; then
    CF_CMD="cloudflared"
  elif [ -n "$CF_BIN" ]; then
    mkdir -p tools
    if [ ! -x tools/cloudflared ]; then
      log "downloading cloudflared ($ARCH)…"
      curl -fL --retry 3 ${CURLV[@]+"${CURLV[@]}"} -o tools/cloudflared "$CLOUDFLARED_BASE/$CF_BIN" \
        && chmod +x tools/cloudflared || { warn "cloudflared download failed"; CF_CMD=""; }
    fi
    [ -x tools/cloudflared ] && CF_CMD="tools/cloudflared"
  else
    CF_CMD=""
  fi
  if [ -n "${CF_CMD:-}" ]; then
    pkill -f "cloudflared tunnel" 2>/dev/null || true; sleep 0.5
    nohup "$CF_CMD" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > tunnel.log 2>&1 < /dev/null &
    echo $! > tunnel.pid
    log "tunnel pid $(cat tunnel.pid) — waiting for the public URL"
    for _ in $(seq 1 60); do
      TUNNEL_URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log 2>/dev/null | head -n1)"
      [ -n "$TUNNEL_URL" ] && break
      sleep 1
    done
    [ -n "$TUNNEL_URL" ] && ok "public URL: $TUNNEL_URL" || { warn "tunnel URL not found — see tunnel.log"; rm -f tunnel.pid; }
  else
    warn "cloudflared unavailable on this machine — public tunnel skipped (local URL still works)"
  fi
fi

# -------------------------------------------------------------------- summary
printf '\n'
hr
printf '%s\n' "${C_OK}  ✅ RAFIDAIN WEB APP IS UP  |  لوحة التحكم تعمل${C_NC}"
hr
printf '  %-14s %s\n' "الصفحة المحلية" "http://127.0.0.1:$PORT/"
[ -n "$TUNNEL_URL" ] && printf '  %-14s %s\n' "الصفحة العامة" "$TUNNEL_URL/"
printf '  %-14s %s\n' "Engine" "$ENGINE (device=$DEVICE)"
printf '  %-14s %s\n' "Docs" "http://127.0.0.1:$PORT/docs"
printf '  %-14s %s\n' "Logs" "tail -f $PROJECT/webapp.log"
printf '  %-14s %s\n' "Stop" "bash scripts/web/launch_webapp.sh --stop   (or: make stop)"
printf '  %-14s %s\n' "Status" "bash scripts/web/launch_webapp.sh --status"
[ "$IS_COLAB" = 1 ] && printf '%s\n' "${C_DIM}  Colab: افتح الرابط العام في متصفحك — الخادم يبقى شغالاً طالما الخلية تعمل${C_NC}"
hr
printf '%s\n' "${C_DIM}  ⚠ الرابط العام مؤقت (trycloudflare) — لا تشاركه إلا مع من تثق به.${C_NC}"
hr

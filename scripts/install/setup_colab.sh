#!/usr/bin/env bash
# ============================================================================
#  RAFIDAIN — Google Colab one-shot installer (curl-bash ready)
#  رافدَين — مثبّت كامل لبيئة Google Colab: تثبيت + تهيئة + تشغيل
# ----------------------------------------------------------------------------
#  Realtime Iraqi-dialect voice cloning on seed-uvit-tat-xlsr-tiny (Seed-VC).
#
#  USAGE — from a Colab cell (Runtime ▸ Change runtime type ▸ GPU T4 first):
#    A) Hosted on GitHub (after pushing rafidain-backend to your account):
#         !curl -fsSL https://raw.githubusercontent.com/<USER>/rafidain-backend/main/scripts/install/setup_colab.sh | bash
#       Passing options through the pipe:
#         !curl -fsSL <url> | bash -s -- --engine mock --no-tunnel
#    B) Local — project already extracted on the VM (e.g. /content/rafidain-backend):
#         !bash /content/rafidain-backend/scripts/install/setup_colab.sh
#    C) Explicit source — git URL, tarball URL, or local directory:
#         !bash setup_colab.sh --source https://github.com/<USER>/rafidain-backend
#         !bash setup_colab.sh --source https://host/rafidain-backend.tar.gz
#
#  OPTIONS:
#    --engine seedvc|mock  engine (default: seedvc when a GPU is present,
#                          mock otherwise — mock needs no ML stack at all)
#    --device auto|cuda|cpu
#    --port N              API port (default 8000)
#    --api-key KEY         use a fixed API key (default: generate a random one)
#    --no-auth             disable API authentication (development only)
#    --no-models           skip the checkpoint pre-download (~1.5 GB)
#    --no-tests            skip the test suite (42 tests, ~10 s)
#    --no-tunnel           skip the public trycloudflare.com tunnel
#    --foreground          run the server attached to this shell
#    --source SRC          project source: local dir, git URL, or tarball URL
#    --ref BRANCH          git branch when cloning (default main)
#    -h | --help           show this help and exit
# ============================================================================
set -Eeuo pipefail

# ----------------------------------------------------------------- constants
SEED_VC_GIT="https://github.com/Plachtaa/seed-vc"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
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
step() { STEP_N=$((STEP_N + 1)); printf '\n%s\n' "${C_HDR}━━━ [$STEP_N/$STEP_T] $* ━━━${C_NC}"; }
usage() { sed -n '2,40p' "${BASH_SOURCE[0]:-$0}" 2>/dev/null | sed 's/^# \{0,1\}//'; }
verle() {  # verle A B → success when version A <= B
  [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then
        printf "\n"; err "Setup failed (exit $rc, line $LINENO).";
        log "Debug re-run:  bash -x <script> <your-args>";
        log "Colab fix:     Runtime ▸ Change runtime type ▸ GPU (T4), then re-run.";
      fi' EXIT
STEP_N=0; STEP_T=10

# ----------------------------------------------------------------- arguments
SOURCE="${RAFIDAIN_SOURCE:-${RAFIDAIN_REPO_URL:-}}"
REF="${RAFIDAIN_REF:-main}"
ENGINE=""; DEVICE=""; PORT="${RAFIDAIN_PORT:-8000}"; API_KEY="${RAFIDAIN_API_KEY:-}"
NO_AUTH=0; NO_MODELS=0; NO_TESTS=0; NO_TUNNEL=0; FOREGROUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --engine)    ENGINE="${2:?}"; shift 2 ;;
    --device)    DEVICE="${2:?}"; shift 2 ;;
    --port)      PORT="${2:?}"; shift 2 ;;
    --api-key)   API_KEY="${2:?}"; shift 2 ;;
    --source)    SOURCE="${2:?}"; shift 2 ;;
    --ref)       REF="${2:?}"; shift 2 ;;
    --no-auth)   NO_AUTH=1; shift ;;
    --no-models) NO_MODELS=1; shift ;;
    --no-tests)  NO_TESTS=1; shift ;;
    --no-tunnel) NO_TUNNEL=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done
if [ -n "$ENGINE" ] && [ "$ENGINE" != "seedvc" ] && [ "$ENGINE" != "mock" ]; then
  die "--engine must be 'seedvc' or 'mock' (got: $ENGINE)"
fi

printf '%s\n' "${C_HDR}"
printf '%s\n' '  ██████╗  █████╗ ██████╗ ██╗██╗   ██╗ █████╗  ██████╗ ███████╗'
printf '%s\n' '  ██╔══██╗██╔══██╗██╔══██╗██║██║   ██║██╔══██╗██╔═══██╗██╔════╝'
printf '%s\n' '  ██████╔╝███████║██████╔╝██║██║   ██║███████║██║   ██║█████╗  '
printf '%s\n' '  ██╔══██╗██╔══██║██╔══██╗██║╚██╗ ██╔╝██╔══██║██║   ██║██╔══╝  '
printf '%s\n' '  ██║  ██║██║  ██║██████╔╝██║ ╚████╔╝ ██║  ██║╚██████╔║███████╗'
printf '%s\n' '  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝═══╝ ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝'
printf '%s\n' '   Realtime Iraqi voice cloning · seed-uvit-tat-xlsr-tiny · Google Colab'
printf '%s\n'"${C_NC}"

# ------------------------------------------------- [1/10] environment probe
step "Environment probe"
IS_COLAB=0
if [ -d /content ] && { [ -n "${COLAB_RELEASE_TAG:-}" ] || case "${PYTHONPATH:-}" in *"/env/python"*) true;; *) false;; esac; }; then
  IS_COLAB=1
fi
if [ -f /usr/lib/python3/dist-packages/google/colab ] 2>/dev/null; then IS_COLAB=1; fi
[ "$IS_COLAB" = 1 ] && ok "Google Colab detected" || log "not Colab — plain Linux mode"

for c in curl git python3; do command -v "$c" >/dev/null 2>&1 || die "required command not found: $c"; done
PYBIN="$(command -v python3)"
PYV="$("$PYBIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
verle "3.10" "$PYV" || die "need Python >= 3.10, found $PYV ($PYBIN)"
ok "Python $PYV ($PYBIN)"

GPU_PRESENT=0; TORCH_INDEX=""; CUDA_CAP=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_PRESENT=1
  CUDA_CAP="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)"
  [ -z "$CUDA_CAP" ] && CUDA_CAP="12.4"
  CMAJ="${CUDA_CAP%%.*}"; CMIN="${CUDA_CAP#*.}"
  # torch 2.6 ships cu118 / cu124 / cu126 wheels (cu121 does not exist for 2.6)
  if [ "$CMAJ" -ge 13 ] || { [ "$CMAJ" -eq 12 ] && [ "$CMIN" -ge 4 ]; }; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  elif [ "$CMAJ" -eq 11 ] && [ "$CMIN" -ge 8 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  else
    GPU_PRESENT=0; warn "driver CUDA $CUDA_CAP < 11.8 — GPU unusable for torch; using CPU"
  fi
fi
if [ "$GPU_PRESENT" = 1 ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  ok "GPU: ${GPU_NAME:-unknown} (driver CUDA ${CUDA_CAP} → ${TORCH_INDEX##*/})"
else
  warn "no usable GPU — for the real engine: Runtime ▸ Change runtime type ▸ T4 GPU"
fi

if [ -z "$ENGINE" ]; then
  [ "$GPU_PRESENT" = 1 ] && ENGINE="seedvc" || ENGINE="mock"
fi
if [ -z "$DEVICE" ]; then
  [ "$GPU_PRESENT" = 1 ] && DEVICE="auto"   || DEVICE="cpu"
fi
if [ "$ENGINE" = "seedvc" ] && [ "$GPU_PRESENT" = 0 ]; then
  warn "seedvc on CPU: install will proceed but streaming latency will exceed call budgets"
fi
ok "engine=$ENGINE device=$DEVICE port=$PORT"

# system bits (Colab normally has them; best effort)
if [ "$IS_COLAB" = 1 ] || [ "$(id -u)" = 0 ]; then
  MISS=""
  command -v ffmpeg >/dev/null 2>&1 || MISS="$MISS ffmpeg"
  command -v git   >/dev/null 2>&1 || MISS="$MISS git"
  if [ -n "$MISS" ] && command -v apt-get >/dev/null 2>&1; then
    log "installing system packages:$MISS"
    (export DEBIAN_FRONTEND=noninteractive
     apt-get update -qq >/dev/null 2>&1 || true
     apt-get install -y -qq $MISS >/dev/null 2>&1) || warn "apt install failed — continuing"
  fi
fi
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)" \
  || warn "ffmpeg missing — voice uploads will fail (decode layer needs it)"

PIP_FLAGS=(--no-input)
if "$PYBIN" -c 'import sysconfig,pathlib,sys
sys.exit(0 if (pathlib.Path(sysconfig.get_path("stdlib"))/"EXTERNALLY-MANAGED").exists() else 1)'; then
  PIP_FLAGS+=(--break-system-packages)   # non-Colab Debian-style system pip
  log "system pip is externally-managed — using --break-system-packages"
fi
ok "pip ready"

# ------------------------------------------------------ [2/10] project source
step "Locating the Rafidain project"
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=""
if [ -f "$SCRIPT_PATH" ]; then SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"; fi

is_project_dir() {  # is_project_dir <path>
  [ -f "$1/pyproject.toml" ] && grep -q 'name = "rafidain"' "$1/pyproject.toml" 2>/dev/null
}
PROJECT=""
if [ -n "$SOURCE" ]; then
  case "$SOURCE" in
    http://*|https://*|git@*)
      INSTALL_DIR="$PWD/rafidain-backend"
      case "$SOURCE" in
        *.tar.gz|*.tgz)
          log "downloading tarball: $SOURCE"
          TMPD="$(mktemp -d)"
          curl -fsSL --retry 3 -o "$TMPD/src.tgz" "$SOURCE"
          mkdir -p "$TMPD/x" && tar -xzf "$TMPD/src.tgz" -C "$TMPD/x"
          PROJ_TOML="$(find "$TMPD/x" -name pyproject.toml -type f 2>/dev/null | head -n1)"
          [ -n "$PROJ_TOML" ] || die "tarball does not contain a pyproject.toml"
          mv "$(dirname "$PROJ_TOML")" "$INSTALL_DIR"; rm -rf "$TMPD" ;;
        *)
          log "git clone: $SOURCE (ref: $REF)"
          rm -rf "$INSTALL_DIR"
          git clone --depth 1 --branch "$REF" "$SOURCE" "$INSTALL_DIR" ;;
      esac
      PROJECT="$INSTALL_DIR" ;;
    *)
      if [ -d "$SOURCE" ]; then
        PROJECT="$(cd "$SOURCE" && pwd)"
      elif [ -f "$SOURCE" ]; then
        case "$SOURCE" in
          *.tar.gz|*.tgz)
            log "extracting local tarball: $SOURCE"
            INSTALL_DIR="$PWD/rafidain-backend"
            TMPD="$(mktemp -d)"
            cp "$SOURCE" "$TMPD/src.tgz"
            mkdir -p "$TMPD/x" && tar -xzf "$TMPD/src.tgz" -C "$TMPD/x"
            PROJ_TOML="$(find "$TMPD/x" -name pyproject.toml -type f 2>/dev/null | head -n1)"
            [ -n "$PROJ_TOML" ] || die "tarball does not contain a pyproject.toml"
            rm -rf "$INSTALL_DIR"; mv "$(dirname "$PROJ_TOML")" "$INSTALL_DIR"; rm -rf "$TMPD"
            PROJECT="$INSTALL_DIR" ;;
          *) die "--source file must be a local .tar.gz/.tgz (or a directory / URL)" ;;
        esac
      else
        die "--source not found: $SOURCE"
      fi ;;
  esac
elif is_project_dir "$PWD"; then
  PROJECT="$PWD"
elif is_project_dir "$PWD/rafidain-backend"; then
  PROJECT="$PWD/rafidain-backend"
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR"; then
  PROJECT="$SCRIPT_DIR"
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR/../.."; then
  PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
[ -n "$PROJECT" ] || die "Rafidain project not found!
  Fix A (local zip):  upload + unzip rafidain-backend.zip in the notebook, then re-run.
  Fix B (git):        re-run with:  --source https://github.com/<you>/rafidain-backend
  Fix C (tarball):    re-run with:  --source https://…/rafidain-backend.tar.gz"
ok "project: $PROJECT"

# ---------------------------------------------------------- [3/10] core deps
step "Installing core Python dependencies"
cd "$PROJECT"
"$PYBIN" -m pip install "${PIP_FLAGS[@]}" -q -r requirements.txt
"$PYBIN" -m pip install "${PIP_FLAGS[@]}" -q -e ".[dev]"
ok "fastapi/uvicorn/pydantic/soxr/… + dev tooling installed"

# ------------------------------------------------------------ [4/10] ML stack
step "Installing the ML stack (engine=$ENGINE)"
KEEP_TORCH=0
if [ "$ENGINE" = "seedvc" ]; then
  HAVE_TORCH=0; TORCH_VER=""
  if "$PYBIN" -c 'import torch' >/dev/null 2>&1; then
    HAVE_TORCH=1
    TORCH_VER="$("$PYBIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "")"
  fi
  TORCH_CUDA_OK=0
  [ "$HAVE_TORCH" = 1 ] && TORCH_CUDA_OK="$("$PYBIN" -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo 0)"
  # >= 2.6 required: transformers' CVE-2025-32434 check refuses torch.load on .bin
  # checkpoints (XLSR ships pytorch_model.bin only) with older torch.
  if [ "$GPU_PRESENT" = 1 ] && [ "$TORCH_CUDA_OK" = 1 ] && [ -n "$TORCH_VER" ] && verle "2.6.0" "$TORCH_VER"; then
    KEEP_TORCH=1
    ok "keeping preinstalled torch $TORCH_VER (CUDA-capable, >= 2.6)"
  else
    # torch 2.6.0: supports py3.9-3.13, passes the transformers CVE gate, and
    # honours TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for Seed-VC's own .pth files.
    TORCH_PKGS=("torch==2.6.0" "torchaudio==2.6.0")
    if [ "$GPU_PRESENT" = 1 ]; then
      log "installing ${TORCH_PKGS[*]} from ${TORCH_INDEX##*/} (~2.5 GB, several minutes)"
      "$PYBIN" -m pip install "${PIP_FLAGS[@]}" "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX"
    else
      log "installing CPU wheels ${TORCH_PKGS[*]} (small)"
      "$PYBIN" -m pip install "${PIP_FLAGS[@]}" "${TORCH_PKGS[@]}" --index-url "$TORCH_CPU_INDEX"
    fi
    verle "$PYV" "3.12" && "$PYBIN" -m pip install "${PIP_FLAGS[@]}" -q "numpy==1.26.4"
  fi
  "$PYBIN" -m pip install "${PIP_FLAGS[@]}" -q "transformers>=4.40,<5" "librosa>=0.10" "soundfile>=0.12" "huggingface-hub>=0.23" "munch>=4.0" "einops>=0.7" "scipy>=1.10" "descript-audio-codec==1.0.0"
  ok "torch/transformers/librosa/soundfile ready"
else
  ok "mock engine — ML stack skipped entirely (no torch, no checkpoints)"
fi

# ------------------------------------------------------------- [5/10] Seed-VC
step "Fetching the Seed-VC repository (engine code + configs)"
if [ ! -f "third_party/seed-vc/modules/commons.py" ]; then
  rm -rf third_party/seed-vc
  log "git clone --depth 1 $SEED_VC_GIT"
  git clone --depth 1 "$SEED_VC_GIT" third_party/seed-vc
fi
[ -f "third_party/seed-vc/configs/hifigan.yml" ] || die "third_party/seed-vc/configs/hifigan.yml missing — clone failed?"
ok "third_party/seed-vc ready (modules + configs)"

# ---------------------------------------------------------- [6/10] checkpoints
step "Pre-downloading checkpoints (seed-uvit-tat-xlsr-tiny)"
if [ "$ENGINE" = "seedvc" ] && [ "$NO_MODELS" = 0 ]; then
  "$PYBIN" scripts/download_models.py --cache-dir ./checkpoints
  ok "checkpoints cached in $PROJECT/checkpoints (~1.5 GB incl. XLSR)"
elif [ "$ENGINE" = "seedvc" ]; then
  warn "--no-models: the server will download checkpoints on first start (slow)"
else
  log "mock engine — checkpoint download skipped"
fi

# ------------------------------------------------------------------ [7/10] .env
step "Writing configuration (.env)"
if [ -f .env ]; then
  cp .env ".env.bak.$(date +%Y%m%d-%H%M%S)"
  log "existing .env backed up"
fi
if [ -z "$API_KEY" ]; then
  API_KEY="$("$PYBIN" -c 'import secrets; print(secrets.token_hex(24))')"
fi
FP16="true"; [ "$DEVICE" = "cpu" ] && FP16="false"
AUTH_LINE="RAFIDAIN__API_KEYS=[\"$API_KEY\"]"
[ "$NO_AUTH" = 1 ] && AUTH_LINE="RAFIDAIN__API_KEYS=[]"
cat > .env <<EOF
# Generated by setup_colab.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
RAFIDAIN__HOST=0.0.0.0
RAFIDAIN__PORT=$PORT
RAFIDAIN__LOG_LEVEL=INFO
RAFIDAIN__LOG_JSON=true
$AUTH_LINE
RAFIDAIN__CORS_ORIGINS=["*"]
RAFIDAIN__ENGINE__IMPL=$ENGINE
RAFIDAIN__ENGINE__DEVICE=$DEVICE
RAFIDAIN__ENGINE__FP16=$FP16
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
EOF
[ "$NO_AUTH" = 1 ] && warn "authentication DISABLED (--no-auth) — never do this on a shared server"
ok ".env written (engine=$ENGINE, device=$DEVICE)"

cat > run_server.sh <<'EOF'
#!/usr/bin/env bash
# Generated by the Rafidain installer — starts the API server (Ctrl+C stops it).
cd "$(dirname "$0")"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # safe for torch>=2.6 checkpoint loading
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
exec "$PY" -m rafidain.main
EOF
chmod +x run_server.sh
ok "run_server.sh generated"

# ------------------------------------------------------------------ [8/10] tests
step "Test suite (mock engine — validates the whole stack)"
if [ "$NO_TESTS" = 0 ]; then
  set +e
  TEST_OUT="$("$PYBIN" -m pytest -q 2>&1 | tail -n 3)"
  TEST_RC=$?
  set -e
  printf '%s\n' "$TEST_OUT"
  [ "$TEST_RC" -eq 0 ] || die "test suite failed — inspect above and re-run"
  ok "all tests passed"
else
  log "tests skipped (--no-tests)"
fi

# --------------------------------------------------------------- [9/10] server
step "Starting the Rafidain server on port $PORT"
pkill -f "rafidain.main" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1
cd "$PROJECT"
if [ "$FOREGROUND" = 1 ]; then
  log "foreground mode — server output follows (Ctrl+C to stop)…"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  exec "$PYBIN" -m rafidain.main
fi
nohup "$PYBIN" -m rafidain.main > server.log 2>&1 &
echo $! > server.pid
log "pid $(cat server.pid); waiting for /healthz"
UP=0
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && { UP=1; break; }
  sleep 1
done
[ "$UP" = 1 ] || { err "server did not come up — last log lines:"; tail -n 30 server.log; exit 1; }
ok "/healthz OK"
log "waiting for /readyz (model load + warmup; first run can take minutes)"
READY=0
for _ in $(seq 1 "$READY_TIMEOUT_S"); do
  R="$(curl -sf "http://127.0.0.1:$PORT/readyz" 2>/dev/null \
       | "$PYBIN" -c 'import json,sys; print(json.load(sys.stdin).get("ready"))' 2>/dev/null || echo "")"
  [ "$R" = "True" ] && { READY=1; break; }
  sleep 1
done
if [ "$READY" = 1 ]; then
  ok "/readyz OK — engine loaded and warmed up"
else
  warn "engine still loading after ${READY_TIMEOUT_S}s — check: tail -f $PROJECT/server.log"
fi

# ------------------------------------------------------------------ [10/10] tunnel
TUNNEL_URL=""
step "Public tunnel (trycloudflare.com)"
if [ "$NO_TUNNEL" = 0 ]; then
  ARCH="$(uname -m)"; case "$ARCH" in
    x86_64) CF_BIN="cloudflared-linux-amd64" ;;
    aarch64|arm64) CF_BIN="cloudflared-linux-arm64" ;;
    *) CF_BIN="" ;;
  esac
  if [ -n "$CF_BIN" ]; then
    mkdir -p tools
    if [ ! -x "tools/cloudflared" ]; then
      log "downloading cloudflared ($ARCH)"
      curl -fsSL --retry 3 -o tools/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/$CF_BIN"
      chmod +x tools/cloudflared
    fi
    nohup tools/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
      > tunnel.log 2>&1 &
    echo $! > tunnel.pid
    log "tunnel starting (pid $(cat tunnel.pid))"
    for _ in $(seq 1 60); do
      TUNNEL_URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log 2>/dev/null | head -n1)"
      [ -n "$TUNNEL_URL" ] && break
      sleep 1
    done
    if [ -n "$TUNNEL_URL" ]; then
      ok "public URL: $TUNNEL_URL"
    else
      warn "no tunnel URL after 60s — see tunnel.log (server still reachable locally)"
    fi
  else
    warn "unsupported arch $ARCH for cloudflared — tunnel skipped"
  fi
else
  log "tunnel skipped (--no-tunnel)"
fi

# -------------------------------------------------------------------- summary
printf '\n'
hr
printf '%s\n' "${C_OK}  ✅ RAFIDAIN IS UP  |  رافدَين جاهز ومشغّل${C_NC}"
hr
printf '  %-12s %s\n' "Local URL"  "http://127.0.0.1:$PORT   (docs: /docs · metrics: /metrics)"
[ -n "$TUNNEL_URL" ] && printf '  %-12s %s\n' "Public URL"  "$TUNNEL_URL"
printf '  %-12s %s\n' "Engine"     "$ENGINE (device=$DEVICE, profile=balanced ≈ 430 ms/block on T4)"
if [ "$NO_AUTH" = 0 ]; then
  printf '  %-12s %s\n' "API key"  "$API_KEY"
  printf '  %-12s %s\n' "Auth"     "header  X-API-Key: <key>    ·    websocket  ?token=<key>"
fi
printf '  %-12s %s\n' "Project"    "$PROJECT"
printf '  %-12s %s\n' "Logs"       "server.log  tunnel.log"
printf '  %-12s %s\n' "Stop"       "kill \$(cat server.pid)  ·  pkill -f rafidain.main"
hr
printf '%s\n' "  TRY IT / جرّب الآن:"
printf '  %s\n' "  curl -s http://127.0.0.1:$PORT/healthz"
if [ "$NO_AUTH" = 0 ]; then
  printf '  %s\n' "  $PYBIN scripts/smoke_ws_client.py --url http://127.0.0.1:$PORT --seconds 3 --api-key $API_KEY"
  [ -n "$TUNNEL_URL" ] && printf '  %s\n' "  $PYBIN scripts/smoke_ws_client.py --url $TUNNEL_URL --seconds 3 --api-key $API_KEY"
else
  printf '  %s\n' "  $PYBIN scripts/smoke_ws_client.py --url http://127.0.0.1:$PORT --seconds 3"
fi
hr
printf '%s\n' "${C_DIM}  ⚠ The public tunnel is unauthenticated TLS — anyone with the URL can reach"
printf '%s\n' "    the API; keep the API key secret and never upload voices without consent."
printf '%s\n' "  ⚠ Colab recycles the VM: re-run this script to bring everything back up."
printf '%s\n' "  ℹ Iraqi-dialect fine-tuning path: python finetune/run_finetune.py --help${C_NC}"
hr

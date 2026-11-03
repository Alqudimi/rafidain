#!/usr/bin/env bash
# ============================================================================
#  RAFIDAIN — Ubuntu/Debian one-shot installer (curl-bash ready)
#  رافدَين — مثبّت كامل لبيئة أوبونتو: تثبيت + تهيئة + تشغيل
# ----------------------------------------------------------------------------
#  Realtime Iraqi-dialect voice cloning on seed-uvit-tat-xlsr-tiny (Seed-VC).
#
#  USAGE:
#    A) Hosted on GitHub (curl-bash — clones https://github.com/Alqudimi/rafidain):
#         curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_ubuntu.sh | bash
#       With options through the pipe:
#         curl -fsSL <url> | bash -s -- --engine seedvc --service
#    B) Local (script inside the cloned/extracted project):
#         bash rafidain/scripts/install/setup_ubuntu.sh
#    C) Explicit source:
#         bash setup_ubuntu.sh --source https://github.com/Alqudimi/rafidain
#
#  OPTIONS:
#    --engine seedvc|mock  engine (default: seedvc when a GPU is present,
#                          mock otherwise)
#    --device auto|cuda|cpu
#    --port N              API port (default 8000)
#    --api-key KEY         use a fixed API key (default: generate a random one)
#    --no-auth             disable API authentication (development only)
#    --no-models           skip the checkpoint pre-download (~1.5 GB)
#    --no-tests            skip the test suite
#    --service             install + start a systemd unit (needs root/sudo)
#    --tunnel              expose a public trycloudflare.com tunnel
#    --foreground          run the server attached to this shell
#    --skip-apt            do not touch apt (for restricted environments)
#    --source SRC          project source: local dir, git URL, or tarball URL
#    --ref BRANCH          git branch when cloning (default main)
#    -h | --help           show this help and exit
# ============================================================================
set -Eeuo pipefail

# ----------------------------------------------------------------- constants
SEED_VC_GIT="https://github.com/Plachtaa/seed-vc"
RAFIDAIN_GIT="https://github.com/Alqudimi/rafidain"
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
      fi' EXIT
# live progress on a terminal; quiet when piped/redirected (curl-bash safe)
if [ -t 1 ]; then PIPV=(--progress-bar on); CURLV=(--progress-bar); else PIPV=(-q); CURLV=(-sS); fi
STEP_N=0; STEP_T=10

# ----------------------------------------------------------------- arguments
SOURCE="${RAFIDAIN_SOURCE:-${RAFIDAIN_REPO_URL:-}}"
REF="${RAFIDAIN_REF:-main}"
ENGINE=""; DEVICE=""; PORT="${RAFIDAIN_PORT:-8000}"; API_KEY="${RAFIDAIN_API_KEY:-}"
NO_AUTH=0; NO_MODELS=0; NO_TESTS=0; TUNNEL=0; FOREGROUND=0; SERVICE=0; SKIP_APT=0
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
    --tunnel)    TUNNEL=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --service)   SERVICE=1; shift ;;
    --skip-apt)  SKIP_APT=1; shift ;;
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
printf '%s\n' '   Realtime Iraqi voice cloning · seed-uvit-tat-xlsr-tiny · Ubuntu/Debian'
printf '%s\n'"${C_NC}"

# ------------------------------------------------- [1/10] environment probe
step "Environment probe"
if [ -f /etc/os-release ]; then . /etc/os-release; else ID="linux"; fi
OS_ID="${ID:-linux}"; OS_VER="${VERSION_ID:-}"
log "OS: ${PRETTY_NAME:-unknown} (id=$OS_ID $OS_VER) · user=$(id -un) uid=$(id -u)"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  # curl-bash safe: a sudo password prompt would eat the piped script —
  # only use sudo when it works non-interactively (sudo -n).
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then SUDO="sudo -n"
  elif command -v sudo >/dev/null 2>&1; then
    warn "sudo needs a password — system package installs skipped (re-run as root, or with passwordless sudo, or install ffmpeg/python3 yourself)"; SKIP_APT=1
  else warn "not root and sudo missing — system package installs will be skipped"; SKIP_APT=1; fi
fi
command -v curl >/dev/null 2>&1 || {
  [ "$SKIP_APT" = 0 ] && { $SUDO apt-get update -qq >/dev/null 2>&1 || true
                           $SUDO apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 || true; }
  command -v curl >/dev/null 2>&1 || die "curl is required but not installable";
}

PYSYS="$(command -v python3 || true)"
PYV="0"
[ -n "$PYSYS" ] && PYV="$("$PYSYS" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)"

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
  warn "no usable GPU — the installer will default to the mock engine (see --engine seedvc for CPU)"
fi

if [ -z "$ENGINE" ]; then
  [ "$GPU_PRESENT" = 1 ] && ENGINE="seedvc" || ENGINE="mock"
fi
if [ -z "$DEVICE" ]; then
  [ "$GPU_PRESENT" = 1 ] && DEVICE="auto"   || DEVICE="cpu"
fi
if [ "$ENGINE" = "seedvc" ] && [ "$GPU_PRESENT" = 0 ]; then
  warn "seedvc on CPU: works, but streaming latency will exceed call budgets (mock is instant)"
fi
ok "engine=$ENGINE device=$DEVICE port=$PORT"

# ------------------------------------------------------- [2/10] system packages
step "System packages (apt)"
APT_MISSING=""
[ -n "$PYSYS" ] || APT_MISSING="$APT_MISSING python3"
command -v git    >/dev/null 2>&1 || APT_MISSING="$APT_MISSING git"
command -v ffmpeg >/dev/null 2>&1 || APT_MISSING="$APT_MISSING ffmpeg"
# 0 = fine · 1 = present but < 3.10 · 2 = missing entirely
PY_TOO_OLD=0
if [ -z "$PYSYS" ] || [ "$PYV" = "0" ]; then
  PY_TOO_OLD=2
elif ! verle "3.10" "$PYV"; then
  PY_TOO_OLD=1
fi
if [ "$SKIP_APT" = 1 ]; then
  [ -n "$APT_MISSING" ] && warn "--skip-apt: not installing:$APT_MISSING"
  [ "$PY_TOO_OLD" -ne 0 ] && warn "--skip-apt: system python may be unsuitable (${PYV:-none})"
  log "apt skipped by request"
elif [ -n "$APT_MISSING" ] || [ "$PY_TOO_OLD" -ne 0 ]; then
  log "installing system packages (curl git ffmpeg python3 python3-venv…)"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ca-certificates curl git ffmpeg python3 python3-venv software-properties-common >/dev/null
  PYSYS="$(command -v python3)"
  if [ -n "$PYSYS" ]; then
    PYV="$("$PYSYS" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)"
  fi
  if ! verle "3.10" "$PYV"; then
    # e.g. Ubuntu 20.04 ships 3.8 — pull a newer interpreter from deadsnakes
    if [ "$OS_ID" = "ubuntu" ]; then
      log "system Python ${PYV:-none} < 3.10 — installing python3.11 from deadsnakes PPA"
      $SUDO add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
      $SUDO apt-get update -qq
      $SUDO apt-get install -y -qq python3.11 python3.11-venv >/dev/null
      PYSYS="$(command -v python3.11)"
      PYV="$("$PYSYS" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)"
    else
      die "system Python ${PYV:-none} < 3.10 and this is not Ubuntu — install Python 3.10+ manually and re-run"
    fi
  fi
fi
verle "3.10" "$PYV" || die "Python >= 3.10 required, found ${PYV:-none}
  On Ubuntu:  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv
  Or re-run with --skip-apt if a venv-capable python is available under another name"
ok "Python $PYV ($PYSYS)"
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg present" \
  || warn "ffmpeg missing — voice uploads will fail without it"
command -v git >/dev/null 2>&1 && ok "git present" || warn "git missing — Seed-VC clone will fail"

# ------------------------------------------------------ [3/10] project source
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
      INSTALL_DIR="$PWD/rafidain"
      case "$SOURCE" in
        *.tar.gz|*.tgz)
          log "downloading tarball: $SOURCE"
          TMPD="$(mktemp -d)"
          curl -fsL "${CURLV[@]}" --retry 3 -o "$TMPD/src.tgz" "$SOURCE"
          mkdir -p "$TMPD/x" && tar -xzf "$TMPD/src.tgz" -C "$TMPD/x"
          PROJ_TOML="$(find "$TMPD/x" -name pyproject.toml -type f 2>/dev/null | head -n1)"
          [ -n "$PROJ_TOML" ] || die "tarball does not contain a pyproject.toml"
          rm -rf "$INSTALL_DIR"; mv "$(dirname "$PROJ_TOML")" "$INSTALL_DIR"; rm -rf "$TMPD" ;;
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
            INSTALL_DIR="$PWD/rafidain"
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
elif is_project_dir "$PWD/rafidain"; then
  PROJECT="$PWD/rafidain"
elif is_project_dir "$PWD/rafidain-backend"; then
  # legacy layout (pre-repo zips shipped the project under rafidain-backend/)
  PROJECT="$PWD/rafidain-backend"
elif is_project_dir "$PWD/rafidain-main" || is_project_dir "$PWD/rafidain-master"; then
  # GitHub "Download ZIP" extracts to <repo>-<branch>
  for _d in rafidain-main rafidain-master; do
    is_project_dir "$PWD/$_d" && PROJECT="$PWD/$_d" && break
  done
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR"; then
  PROJECT="$SCRIPT_DIR"
elif [ -n "$SCRIPT_DIR" ] && is_project_dir "$SCRIPT_DIR/../.."; then
  PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [ -z "$PROJECT" ]; then
  # curl-bash flow: nothing nearby → clone the official repository
  log "project not found nearby — cloning $RAFIDAIN_GIT (ref: $REF)"
  INSTALL_DIR="$PWD/rafidain"
  rm -rf "$INSTALL_DIR"
  if git clone --progress --depth 1 --branch "$REF" "$RAFIDAIN_GIT" "$INSTALL_DIR" 2>/dev/null \
     && is_project_dir "$INSTALL_DIR"; then
    PROJECT="$INSTALL_DIR"
  else
    rm -rf "$INSTALL_DIR"
    die "could not clone $RAFIDAIN_GIT
  Fix A (local):  clone this repo / extract its zip next to me, then re-run.
  Fix B (git):    re-run with:  --source https://github.com/Alqudimi/rafidain
  Fix C (tar):    re-run with:  --source https://…/rafidain.tar.gz"
  fi
fi
ok "project: $PROJECT"
cd "$PROJECT"

# ----------------------------------------- [4/10] virtualenv + core deps
step "Creating the virtualenv (.venv) + core dependencies"
VENV_PY="$PROJECT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  "$PYSYS" -m venv "$PROJECT/.venv"
fi
"$VENV_PY" -m pip install -q --upgrade pip wheel setuptools
"$VENV_PY" -m pip install -q --no-input -r requirements.txt
"$VENV_PY" -m pip install -q --no-input -e ".[dev]"
ok "virtualenv ready: $PROJECT/.venv (fastapi/uvicorn/pydantic/soxr/… + dev tooling)"
PYBIN="$VENV_PY"

# ------------------------------------------------------------ [5/10] ML stack
step "Installing the ML stack (engine=$ENGINE)"
if [ "$ENGINE" = "seedvc" ]; then
  # torch 2.6.0: supports py3.9-3.13, passes transformers' CVE-2025-32434 gate
  # (XLSR ships pytorch_model.bin only — newer transformers refuse torch<2.6),
  # and honours TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for Seed-VC's own .pth files.
  TORCH_PKGS=("torch==2.6.0" "torchaudio==2.6.0")
  if [ "$GPU_PRESENT" = 1 ]; then
    log "installing ${TORCH_PKGS[*]} from ${TORCH_INDEX##*/} (~2.5 GB, several minutes)"
    "$PYBIN" -m pip install --no-input "${PIPV[@]}" "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX"
  else
    log "installing CPU wheels ${TORCH_PKGS[*]} (small)"
    "$PYBIN" -m pip install --no-input "${PIPV[@]}" "${TORCH_PKGS[@]}" --index-url "$TORCH_CPU_INDEX"
  fi
  verle "$PYV" "3.12" && "$PYBIN" -m pip install -q --no-input "numpy==1.26.4"
  "$PYBIN" -m pip install --no-input "${PIPV[@]}" "transformers>=4.40,<5" "librosa>=0.10" "soundfile>=0.12" "huggingface-hub>=0.23" "munch>=4.0" "einops>=0.7" "scipy>=1.10" "descript-audio-codec==1.0.0"
  ok "torch/transformers/librosa/soundfile ready"
else
  ok "mock engine — ML stack skipped entirely (no torch, no checkpoints)"
fi

# ------------------------------------------------------------- [6/10] Seed-VC
step "Fetching the Seed-VC repository (engine code + configs)"
if [ ! -f "third_party/seed-vc/modules/commons.py" ]; then
  rm -rf third_party/seed-vc
  log "git clone --depth 1 $SEED_VC_GIT"
  git clone --progress --depth 1 "$SEED_VC_GIT" third_party/seed-vc
fi
[ -f "third_party/seed-vc/configs/hifigan.yml" ] || die "third_party/seed-vc/configs/hifigan.yml missing — clone failed?"
ok "third_party/seed-vc ready (modules + configs)"

# ---------------------------------------------------------- [7/10] checkpoints
step "Pre-downloading checkpoints (seed-uvit-tat-xlsr-tiny)"
if [ "$ENGINE" = "seedvc" ] && [ "$NO_MODELS" = 0 ]; then
  "$PYBIN" scripts/download_models.py --cache-dir ./checkpoints
  ok "checkpoints cached in $PROJECT/checkpoints (~1.5 GB incl. XLSR)"
elif [ "$ENGINE" = "seedvc" ]; then
  warn "--no-models: the server will download checkpoints on first start (slow)"
else
  log "mock engine — checkpoint download skipped"
fi

# ------------------------------------------------------------------ [8/10] .env
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
# Generated by setup_ubuntu.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
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
[ "$NO_AUTH" = 1 ] && warn "authentication DISABLED (--no-auth) — never do this on a public server"
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

# ------------------------------------------------------------------ [9/10] tests
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

# --------------------------------------------------------------- [10/10] run
step "Starting the Rafidain server on port $PORT"
pkill -f "rafidain.main" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1
cd "$PROJECT"
SERVICE_INSTALLED=0
if [ "$SERVICE" = 1 ]; then
  if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
    warn "--service needs root/sudo — falling back to a background process"
  else
    RUN_USER="$(id -un)"
    log "installing systemd unit (user=$RUN_USER)"
    $SUDO tee /etc/systemd/system/rafidain.service >/dev/null <<EOF
[Unit]
Description=Rafidain - realtime Iraqi voice-conversion backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT
Environment=TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
Environment=SEED_VC_ROOT=$PROJECT/third_party/seed-vc
ExecStart=$PROJECT/.venv/bin/python -m rafidain.main
Restart=on-failure
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable --now rafidain.service
    SERVICE_INSTALLED=1
    ok "systemd service 'rafidain' enabled and started"
  fi
fi
if [ "$FOREGROUND" = 1 ]; then
  log "foreground mode — server output follows (Ctrl+C to stop)…"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  exec "$PYBIN" -m rafidain.main
fi
if [ "$SERVICE_INSTALLED" = 0 ]; then
  nohup "$PYBIN" -m rafidain.main > server.log 2>&1 &
  echo $! > server.pid
  log "background pid $(cat server.pid); waiting for /healthz"
  UP=0
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && { UP=1; break; }
    sleep 1
  done
  [ "$UP" = 1 ] || { err "server did not come up — last log lines:"; tail -n 30 server.log; exit 1; }
  ok "/healthz OK"
else
  log "waiting for /healthz"
  UP=0
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && { UP=1; break; }
    sleep 1
  done
  [ "$UP" = 1 ] || { err "service did not come up — journalctl -u rafidain -n 50"; exit 1; }
  ok "/healthz OK"
fi
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

TUNNEL_URL=""
if [ "$TUNNEL" = 1 ]; then
  log "starting public tunnel (--tunnel)"
  ARCH="$(uname -m)"; case "$ARCH" in
    x86_64) CF_BIN="cloudflared-linux-amd64" ;;
    aarch64|arm64) CF_BIN="cloudflared-linux-arm64" ;;
    *) CF_BIN="" ;;
  esac
  if [ -n "$CF_BIN" ]; then
    mkdir -p tools
    if [ ! -x "tools/cloudflared" ]; then
      curl -fL "${CURLV[@]}" --retry 3 -o tools/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/$CF_BIN"
      chmod +x tools/cloudflared
    fi
    nohup tools/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
      > tunnel.log 2>&1 &
    echo $! > tunnel.pid
    for _ in $(seq 1 60); do
      TUNNEL_URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log 2>/dev/null | head -n1)"
      [ -n "$TUNNEL_URL" ] && break
      sleep 1
    done
    [ -n "$TUNNEL_URL" ] && ok "public URL: $TUNNEL_URL" || warn "tunnel URL not found — see tunnel.log"
  else
    warn "unsupported arch $ARCH — tunnel skipped"
  fi
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
if [ "$SERVICE_INSTALLED" = 1 ]; then
  printf '  %-12s %s\n' "Service"   "systemctl status rafidain · journalctl -u rafidain -f"
  printf '  %-12s %s\n' "Restart"   "sudo systemctl restart rafidain"
else
  printf '  %-12s %s\n' "Logs"      "tail -f $PROJECT/server.log"
  printf '  %-12s %s\n' "Stop"      "kill \$(cat $PROJECT/server.pid)"
fi
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
printf '%s\n' "${C_DIM}  ⚠ Put the service behind TLS (nginx/caddy) before exposing it publicly,"
printf '%s\n' "    restrict CORS, and never clone a voice without the speaker's consent."
printf '%s\n' "  ℹ Iraqi-dialect fine-tuning path: python finetune/run_finetune.py --help${C_NC}"
hr

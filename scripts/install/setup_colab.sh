#!/usr/bin/env bash
# Rafidain compatibility installer — uv-first and safe with `bash -u`.
set -Eeuo pipefail
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { echo "Usage: bash $0 [setup_uv.sh options]"; exit 0; }
RAFIDAIN_GIT="https://github.com/Alqudimi/rafidain"
SELF="${BASH_SOURCE[0]:-$0}"
SELF_DIR="$(cd -- "$(dirname -- "$SELF")" 2>/dev/null && pwd -P || pwd -P)"
ROOT="$(cd -- "$SELF_DIR/../.." 2>/dev/null && pwd -P || true)"
if [[ ! -f "$ROOT/scripts/install/setup_uv.sh" ]]; then
  ROOT="${RAFIDAIN_HOME:-$PWD/rafidain}"
  if [[ ! -f "$ROOT/scripts/install/setup_uv.sh" ]]; then
    command -v git >/dev/null 2>&1 || { echo "git مطلوب لتنزيل المشروع: $RAFIDAIN_GIT" >&2; exit 1; }
    rm -rf "$ROOT"
    git clone --depth 1 "$RAFIDAIN_GIT" "$ROOT"
  fi
fi
cd "$ROOT"
export RAFIDAIN_COLAB=1
exec bash "$ROOT/scripts/install/setup_uv.sh" --no-tunnel "$@"

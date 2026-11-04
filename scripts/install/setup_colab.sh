#!/usr/bin/env bash
# Rafidain Google Colab bootstrap — uv-first, Drive-aware entrypoint.
# Usage: bash scripts/install/setup_colab.sh [--engine seedvc|mock] [--no-models]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export RAFIDAIN_COLAB=1
# setup_uv detects /content and the mounted Drive; --no-tunnel is safer in notebooks.
exec bash scripts/install/setup_uv.sh --no-tunnel "$@"

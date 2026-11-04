#!/usr/bin/env bash
# Rafidain Ubuntu/WSL installer — uv-first compatibility entrypoint.
# Usage: bash scripts/install/setup_ubuntu.sh [setup_uv.sh options]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/install/setup_uv.sh "$@"

#!/usr/bin/env bash
# Deprecated compatibility name for Google Colab.
# It now uses uv, the locked extras, and the checkpoint/Drive-aware trainer.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export RAFIDAIN_COLAB=1
exec bash scripts/install/setup_colab.sh "$@"

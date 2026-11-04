#!/usr/bin/env bash
# Deprecated compatibility name. The old embedded pip/bootstrap script is gone.
# Use the repository's reproducible uv lifecycle instead.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec bash scripts/install/setup_uv.sh "$@"

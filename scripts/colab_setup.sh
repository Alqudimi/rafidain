#!/usr/bin/env bash
# Colab entrypoint: uv install + optional Seed-VC/model setup, no tunnel by default.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/install/setup_colab.sh "$@"

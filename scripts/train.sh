#!/usr/bin/env bash
# One-command training entrypoint; all Python execution is inside uv.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
exec uv run --python 3.11 python finetune/run_finetune.py "$@"

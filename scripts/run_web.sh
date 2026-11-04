#!/usr/bin/env bash
# One-command web control: uv sync, server, /web + /live, optional Cloudflare.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
exec bash scripts/web/launch_webapp.sh "$@"

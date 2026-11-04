# ---------------------------------------------------------------------------
# Rafidain — unified developer & operator entrypoints (uv-managed)
# ---------------------------------------------------------------------------
# Everything routes through scripts/manage.py, which auto-detects the
# system (Linux / macOS / Windows / WSL / Colab / Docker), the GPU and its
# CUDA version, bootstraps uv, and installs exactly the locked dependency
# set for the detected environment (uv.lock pins every torch variant).
#
# Windows users (no make): call the manager directly —
#   uv run scripts/manage.py install / web / tunnel / stop
# ---------------------------------------------------------------------------

.PHONY: help env-info install install-dev install-cpu install-cu118 \
        install-cu124 install-cu126 install-mock install-webui lock \
        seedvc download-models train train-resume serve serve-mock \
        web web-mock web-tunnel tunnel tunnel-stop tunnel-url tunnel-log \
        stop status test lint fmt typecheck bench smoke clean clean-all

PY := uv run --python 3.11 python
MANAGE := $(PY) scripts/manage.py
SEED_VC_REF ?= main
PORT ?= 8000

help:
	@echo "رافدَين — Rafidain unified commands (uv + smart auto-detection)"
	@echo ""
	@echo "  ## Setup & installation"
	@echo "  make env-info           smart environment report (OS/GPU/CUDA/uv/Colab)"
	@echo "  make install            install everything for THIS machine (auto GPU/CUDA)"
	@echo "  make install-mock       light stack only — mock engine, no torch, no models"
	@echo "  make install-dev        install + dev tooling (pytest/ruff/mypy)"
	@echo "  make install-cpu        force CPU torch wheels"
	@echo "  make install-cu118      force CUDA 11.8 torch wheels"
	@echo "  make install-cu124      force CUDA 12.4 torch wheels"
	@echo "  make install-cu126      force CUDA 12.6 torch wheels"
	@echo "  make lock               re-resolve and commit a new uv.lock"
	@echo ""
	@echo "  ## Models & training"
	@echo "  make seedvc             clone Seed-VC into third_party/ (engine + training)"
	@echo "  make download-models    pre-download Seed-VC checkpoints (~1.5 GB)"
	@echo "  make train ARGS=\"--dataset-dir ./data/iraqi --max-steps 2000\""
	@echo "  make train-resume ARGS=\"--dataset-dir ./data/iraqi --run-name iraqi_v1\""
	@echo "                          resume from the newest checkpoint (+ Google Drive on Colab)"
	@echo ""
	@echo "  ## Run & broadcast"
	@echo "  make serve              start the API server (background, real engine)"
	@echo "  make serve-mock         start the API server with the mock engine"
	@echo "  make web                server + Arabic RTL control panel (/web) + live page (/live)"
	@echo "  make web-mock           the same with the mock engine (no GPU needed)"
	@echo "  make web-tunnel         web + free public tunnel (trycloudflare.com)"
	@echo "  make tunnel             open the public tunnel for the running server"
	@echo "  make tunnel-stop        close the tunnel"
	@echo "  make tunnel-url         print the current public URL"
	@echo "  make stop               stop server + tunnel cleanly"
	@echo "  make status             full status: server/engine/tunnel/GPU"
	@echo ""
	@echo "  ## Quality"
	@echo "  make test               run unit + integration tests (mock engine)"
	@echo "  make smoke              full end-to-end smoke test (server + web + WS stream)"
	@echo "  make lint               ruff lint"
	@echo "  make fmt                ruff auto-format imports + style"
	@echo "  make typecheck          mypy"
	@echo "  make bench              latency/RTF benchmark of the streaming engine"
	@echo "  make clean              remove caches and build artefacts"

# ------------------------------------------------------------ setup targets
env-info:
	@$(MANAGE) detect

install:
	@$(MANAGE) install

install-dev:
	@$(MANAGE) install --dev

install-mock:
	@$(MANAGE) install --engine mock

install-cpu:
	@$(MANAGE) install --variant cpu

install-cu118:
	@$(MANAGE) install --variant cu118

install-cu124:
	@$(MANAGE) install --variant cu124

install-cu126:
	@$(MANAGE) install --variant cu126

install-webui:
	@uv sync --python 3.11 --frozen --extra webui

lock:
	uv lock

# ---------------------------------------------------------- models & train
seedvc:
	@$(MANAGE) seedvc

download-models:
	@$(MANAGE) download

train:
	@$(MANAGE) train -- $(ARGS)

train-resume:
	@$(MANAGE) train -- --resume always $(ARGS)

# --------------------------------------------------------------- run & web
serve:
	@$(MANAGE) serve --port $(PORT)

serve-mock:
	@$(MANAGE) serve --port $(PORT) --mock

web:
	@$(MANAGE) web --port $(PORT)

web-mock:
	@$(MANAGE) web --port $(PORT) --mock

web-tunnel:
	@$(MANAGE) web --port $(PORT) --tunnel

tunnel:
	@$(MANAGE) tunnel start --port $(PORT)

tunnel-stop:
	@$(MANAGE) tunnel stop

tunnel-url:
	@$(MANAGE) tunnel url

tunnel-log:
	@$(MANAGE) tunnel log

stop:
	@$(MANAGE) stop

status:
	@$(MANAGE) status

# ------------------------------------------------------------------ quality
test:
	$(PY) -m pytest -q

smoke:
	bash scripts/smoke_e2e.sh

lint:
	$(PY) -m ruff check rafidain tests finetune evaluation scripts

fmt:
	$(PY) -m ruff check --fix rafidain tests finetune evaluation scripts

typecheck:
	$(PY) -m mypy rafidain

bench:
	$(PY) evaluation/benchmark_stream.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-all: clean
	rm -rf .run tools data smoke_output.*

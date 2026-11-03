# ---------------------------------------------------------------------------
# Rafidain — developer entrypoints
# ---------------------------------------------------------------------------
.PHONY: help install install-dev seedvc serve serve-mock test lint fmt typecheck \
        download-models bench docker-build docker-up clean

SEED_VC_REF ?= main

help:
	@echo "Rafidain — realtime Iraqi voice-conversion backend"
	@echo "  make install          install core runtime deps (light, no torch)"
	@echo "  make install-dev      install runtime + dev tooling (editable)"
	@echo "  make seedvc           clone Seed-VC into third_party/ (engine + training)"
	@echo "  make serve            run the API with the real Seed-VC engine"
	@echo "  make serve-mock       run the API with the mock engine (no ML deps)"
	@echo "  make test             run unit + integration tests (mock engine)"
	@echo "  make lint             ruff lint"
	@echo "  make fmt              ruff auto-format imports + style"
	@echo "  make typecheck        mypy"
	@echo "  make download-models  pre-download Seed-VC checkpoints from HuggingFace"
	@echo "  make bench            latency/RTF benchmark of the streaming engine"
	@echo "  make docker-up        build & start with docker compose (GPU)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

seedvc:
	@mkdir -p third_party
	git clone --depth 1 https://github.com/Plachtaa/seed-vc third_party/seed-vc
	@echo "Seed-VC cloned at third_party/seed-vc (ref: $(SEED_VC_REF))"
	@echo "Install its requirements separately: pip install -r requirements-finetune.txt"

serve:
	uvicorn rafidain.main:app --host $${RAFIDAIN__HOST:-0.0.0.0} --port $${RAFIDAIN__PORT:-8000}

serve-mock:
	RAFIDAIN__ENGINE__IMPL=mock uvicorn rafidain.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check rafidain tests finetune evaluation scripts

fmt:
	ruff check --fix rafidain tests finetune evaluation scripts

typecheck:
	mypy rafidain

download-models:
	python scripts/download_models.py

bench:
	python evaluation/benchmark_stream.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

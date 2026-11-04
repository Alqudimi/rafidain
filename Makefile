SHELL := /bin/bash
UV ?= uv
PYTHON ?= 3.11
ARGS ?=
PORT ?= 8000
.PHONY: help install install-dev seedvc download-models prepare train serve serve-mock web web-mock tunnel tunnel-stop tunnel-url status stop test lint fmt typecheck bench docker-build docker-up clean
help:
	@printf '%s\n' 'Rafidain (uv-managed)' '  make install [ENGINE=seedvc]' '  make install-dev' '  make seedvc' '  make download-models' '  make prepare ARGS="--input-dir ./raw_iraqi"' '  make train ARGS="--dataset-dir ./datasets/iraqi --run-name iraqi_v1 --gdrive auto"' '  make web | make web-mock | make tunnel | make stop' '  make test lint typecheck bench'
install:
	$(UV) run --python $(PYTHON) python scripts/manage.py install --engine $(if $(ENGINE),$(ENGINE),auto)
install-dev:
	$(UV) sync --python $(PYTHON) --frozen --group dev
seedvc:
	$(UV) run --python $(PYTHON) python scripts/manage.py seedvc
download-models:
	$(UV) run --python $(PYTHON) python scripts/manage.py download $(ARGS)
prepare:
	$(UV) run --python $(PYTHON) python finetune/prepare_dataset.py $(ARGS)
train:
	$(UV) run --python $(PYTHON) python finetune/run_finetune.py $(ARGS)
serve:
	$(UV) run --python $(PYTHON) python scripts/manage.py serve --port $(PORT)
serve-mock:
	$(UV) run --python $(PYTHON) python scripts/manage.py serve --mock --port $(PORT)
web:
	$(UV) run --python $(PYTHON) python scripts/manage.py web --port $(PORT)
web-mock:
	$(UV) run --python $(PYTHON) python scripts/manage.py web --mock --port $(PORT)
tunnel:
	$(UV) run --python $(PYTHON) python scripts/manage.py tunnel start --port $(PORT)
tunnel-stop:
	$(UV) run --python $(PYTHON) python scripts/manage.py tunnel stop
tunnel-url:
	$(UV) run --python $(PYTHON) python scripts/manage.py tunnel url
status:
	$(UV) run --python $(PYTHON) python scripts/manage.py status
stop:
	$(UV) run --python $(PYTHON) python scripts/manage.py stop
test:
	$(UV) run --python $(PYTHON) pytest -q
lint:
	$(UV) run --python $(PYTHON) ruff check rafidain tests finetune evaluation scripts
fmt:
	$(UV) run --python $(PYTHON) ruff check --fix rafidain tests finetune evaluation scripts
typecheck:
	$(UV) run --python $(PYTHON) mypy rafidain
bench:
	$(UV) run --python $(PYTHON) python evaluation/benchmark_stream.py
docker-build:
	docker compose -f docker/docker-compose.yml build
docker-up:
	docker compose -f docker/docker-compose.yml up -d
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .run
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

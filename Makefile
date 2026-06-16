PKG := twinrl_onco
SMOKE := flightplans/experiment/_smoke.toml

.PHONY: help install dev lint format type test smoke train docker clean

help:
	@echo "install  - pip install the package"
	@echo "dev      - install with dev tooling"
	@echo "lint     - ruff + isort + black checks"
	@echo "format   - apply isort + black"
	@echo "type     - mypy strict over $(PKG)"
	@echo "test     - run the checkride suite"
	@echo "smoke    - two-step training run on the smoke plan"
	@echo "train    - main-plan training run"
	@echo "docker   - build the container image"
	@echo "clean    - remove caches and run artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	isort --check-only .
	black --check .

format:
	isort .
	black .

type:
	python -m mypy $(PKG)

test:
	pytest -q

smoke:
	python -m $(PKG).panel.console train --config $(SMOKE) --checkpoint runs/smoke.pt

train:
	python -m $(PKG).panel.console train --config flightplans/experiment/main.toml

docker:
	docker build -t $(PKG):0.1.0 .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

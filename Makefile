# SPDX-License-Identifier: MIT
#
# Voidface top-level Makefile.
#
# The Makefile does not do the work itself. It delegates to uv, ruff, mypy,
# pytest, and the scripts in scripts/. Every target is a thin wrapper.

.DEFAULT_GOAL := help

PY      := uv run python
UV      := uv
RUFF    := uv run ruff
MYPY    := uv run mypy
PYTEST  := uv run pytest

.PHONY: help
help:
	@echo "Voidface build targets:"
	@echo ""
	@echo "  make install         install runtime + dev dependencies via uv"
	@echo "  make lint            ruff (src/ tools/ tests/ scripts/) + mypy"
	@echo "                       (src/voidface + tools/cli/voidface_cli)"
	@echo "  make format          ruff format the source tree"
	@echo "  make test            run unit + integration tests"
	@echo "  make bench           run ASR / perceptual benchmarks"
	@echo "  make docs            build documentation"
	@echo "  make check-licenses  verify every source file has an SPDX header"
	@echo "  make check-format    verify no unformatted files are checked in"
	@echo "  make clean           remove build artifacts and caches"
	@echo "  make distclean       clean + remove .venv and model caches"
	@echo ""

.PHONY: install
install:
	$(UV) sync --all-extras

.PHONY: lint
lint:
	$(RUFF) check src/ tools/ tests/ scripts/
	$(MYPY) src/voidface
	$(MYPY) tools/cli/voidface_cli

.PHONY: format
format:
	$(RUFF) format src/ tools/ tests/ scripts/
	$(RUFF) check --fix src/ tools/ tests/ scripts/

.PHONY: test
test:
	$(PYTEST) tests/

.PHONY: test-unit
test-unit:
	$(PYTEST) tests/unit/

.PHONY: test-integration
test-integration:
	$(PYTEST) tests/integration/

.PHONY: bench
bench:
	$(PY) scripts/benchmark.py

.PHONY: docs
docs:
	@echo "Documentation build TBD — see Documentation/ for source."

.PHONY: check-licenses
check-licenses:
	bash scripts/check-license-headers.sh

.PHONY: check-format
check-format:
	$(RUFF) format --check src/ tools/ tests/ scripts/
	$(RUFF) check src/ tools/ tests/ scripts/

.PHONY: download-models
download-models:
	bash scripts/download-models.sh

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov

.PHONY: distclean
distclean: clean
	rm -rf .venv models/downloaded .cache .huggingface

# Makefile for Vishwamai development

.PHONY: help clean test lint format docs build install dev publish

help:
	@echo "Available commands:"
	@echo "  make clean      - Remove build artifacts and cache files"
	@echo "  make test      - Run tests with pytest"
	@echo "  make lint      - Run linting checks"
	@echo "  make format    - Format code with black and isort"
	@echo "  make docs      - Build documentation"
	@echo "  make build     - Build package distribution"
	@echo "  make install   - Install package in development mode"
	@echo "  make dev       - Setup development environment"
	@echo "  make publish   - Publish package to PyPI"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/_build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete

test:
	python -m pytest tests/ \
		--cov=vishwamai \
		--cov-report=term-missing \
		--cov-report=html:coverage \
		-v

lint:
	flake8 vishwamai/
	mypy vishwamai/
	pylint vishwamai/
	black --check vishwamai/
	isort --check-only vishwamai/

format:
	black vishwamai/
	isort vishwamai/

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

build: clean
	python -m build
	twine check dist/*

install:
	pip install -e ".[dev]"

dev: install
	pre-commit install
	python -m pip install --upgrade pip
	pip install -r requirements.txt

publish: build
	twine upload dist/*

# Development environment setup
setup-env:
	python -m venv venv
	. venv/bin/activate && make dev

# TPU-specific commands
tpu-setup:
	pip install torch-xla
	pip install cloud-tpu-client

# Training commands
train:
	python scripts/train_model.py \
		--model-config configs/model_config.yaml \
		--training-config configs/training_config.yaml \
		--data-config configs/data_config.yaml

evaluate:
	python scripts/evaluate_model.py \
		--model-path models/vishwamai \
		--data-dir data/test \
		--output-dir results

# Data preprocessing
preprocess:
	python scripts/preprocess_data.py \
		--config configs/data_config.yaml \
		--input-dir data/raw \
		--output-dir data/processed

# Model export
export:
	python scripts/export_model.py \
		--model-path models/vishwamai \
		--output-dir exported_model \
		--format all

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/train_model.py
	snakeviz profile.stats

# Memory profiling
memcheck:
	mprof run scripts/train_model.py
	mprof plot

# Continuous testing
watch-tests:
	ptw tests/ -- -v

# Generate documentation
docs-serve: docs
	cd docs/_build/html && python -m http.server 8000

# Type checking
typecheck:
	mypy vishwamai/ --strict

# Security check
security:
	bandit -r vishwamai/
	safety check

# Benchmark
benchmark:
	python -m pytest tests/benchmarks --benchmark-only

# Docker commands
docker-build:
	docker build -t vishwamai .

docker-run:
	docker run -it --rm vishwamai

# Cleanup old branches and caches
cleanup:
	git fetch -p
	git branch -vv | grep ': gone]' | awk '{print $$1}' | xargs git branch -D
	pip cache purge
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage

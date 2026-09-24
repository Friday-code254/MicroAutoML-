.PHONY: install install-dev test test-unit test-integration test-regression lint format clean run-demo

# Install core dependencies
install:
	pip install -e .

# Install with dev tools
install-dev:
	pip install -e ".[dev,dashboard]"

# Run all tests
test:
	pytest tests/ -v --tb=short

# Unit tests only (fast)
test-unit:
	pytest tests/unit/ -v --tb=short

# Integration tests
test-integration:
	pytest tests/integration/ -v --tb=short

# Regression tests (CI-safe, < 60s target)
test-regression:
	pytest tests/regression/ -v --tb=short --timeout=60

# Lint
lint:
	ruff check automl/ tests/

# Format
format:
	ruff format automl/ tests/

# Remove build artifacts and cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Demo run on iris (requires Phase 2+ to be complete)
run-demo:
	python -c "from sklearn.datasets import load_iris; import pandas as pd; df = pd.DataFrame(load_iris(as_frame=True).frame); df.to_csv('/tmp/iris_demo.csv', index=False)" && \
	microautoml run --data /tmp/iris_demo.csv --target target --budget 120 --config configs/development.yaml

# Profile only demo
run-profile:
	python -c "from sklearn.datasets import load_iris; import pandas as pd; df = pd.DataFrame(load_iris(as_frame=True).frame); df.to_csv('/tmp/iris_demo.csv', index=False)" && \
	microautoml profile --data /tmp/iris_demo.csv --target target

# Run benchmark suite (requires Phase 11+)
benchmark:
	python benchmark/run.py --config configs/benchmark.yaml

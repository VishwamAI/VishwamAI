# Vishwamai Tests

This directory contains tests for the Vishwamai project, with a particular focus on precision options and model performance.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and configurations
├── pytest.ini          # PyTest configuration
├── requirements.txt    # Test-specific dependencies
├── test_precision.py  # Precision-related tests
└── README.md         # This file
```

## Prerequisites

Install test dependencies:

```bash
pip install -r tests/requirements.txt
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=vishwamai

# Run with parallel execution
pytest -n auto
```

### Precision-Specific Tests

```bash
# Run only precision tests
pytest -m precision

# Run precision tests with specific CUDA device
CUDA_VISIBLE_DEVICES=0 pytest -m precision

# Run precision benchmarks
pytest -m "precision and benchmark"
```

### GPU Tests

```bash
# Run GPU-specific tests
pytest -m gpu

# Skip GPU tests
pytest -m "not gpu"
```

## Test Categories

### Precision Tests (`test_precision.py`)

Tests for different precision modes:
- FP16 (Half Precision)
- FP32 (Single Precision)
- FP64 (Double Precision)
- BF16 (Brain Float)
- Mixed Precision Training

Features tested:
- Precision conversion
- Memory usage
- Numerical accuracy
- Training stability
- Gradient scaling
- Mixed precision training flow

### Benchmark Tests

Performance benchmarks for different configurations:
- Forward/backward pass timing
- Memory consumption
- Training throughput
- GPU utilization

## Configuration

### Test Fixtures (`conftest.py`)

Shared fixtures for:
- Model configurations
- Data generation
- Memory tracking
- Precision settings
- GPU device management

### PyTest Settings (`pytest.ini`)

Key settings:
- Test discovery paths
- Logging configuration
- Coverage settings
- Benchmark options
- Custom markers
- Environment variables

## Test Markers

Use these markers to run specific test groups:

```bash
# Precision tests
@pytest.mark.precision

# GPU-required tests
@pytest.mark.gpu

# CPU-only tests
@pytest.mark.cpu

# Benchmark tests
@pytest.mark.benchmark

# Distributed tests
@pytest.mark.distributed

# Slow tests
@pytest.mark.slow
```

## Environment Variables

Important environment variables for testing:

```bash
# GPU Selection
export CUDA_VISIBLE_DEVICES=0

# Architecture Support
export TORCH_CUDA_ARCH_LIST="7.0+PTX"

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Coverage Reports

Generate coverage reports:

```bash
# Terminal report
pytest --cov=vishwamai --cov-report=term-missing

# HTML report
pytest --cov=vishwamai --cov-report=html

# XML report for CI
pytest --cov=vishwamai --cov-report=xml
```

Coverage exclusions:
- Test files
- Setup scripts
- Documentation
- Example scripts

## Benchmarking

Run performance benchmarks:

```bash
# Full benchmark suite
pytest --benchmark-only

# Save benchmark results
pytest --benchmark-only --benchmark-autosave

# Compare with previous results
pytest-benchmark compare
```

## Troubleshooting

### Common Issues

1. CUDA Out of Memory:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   pytest -m precision
   ```

2. Test Timeouts:
   ```bash
   pytest --timeout=300
   ```

3. GPU Not Found:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Run CPU-only tests
   pytest -m "not gpu"
   ```

### Debug Logging

Enable debug logging:
```bash
pytest --log-cli-level=DEBUG
```

## Adding New Tests

1. Add appropriate markers
2. Use shared fixtures from `conftest.py`
3. Follow existing test patterns
4. Include performance considerations
5. Add documentation

## Continuous Integration

The test suite is integrated with CI/CD:
- Runs on pull requests
- Generates coverage reports
- Runs benchmark comparisons
- Enforces test quality standards

See [CI Configuration](../.github/workflows) for details.

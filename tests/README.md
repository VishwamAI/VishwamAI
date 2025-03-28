# VishwamAI Kernel Tests

This directory contains the test suite for VishwamAI kernels across different hardware platforms (TPU, GPU, CPU).

## Setup

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Platform-specific setup:

### TPU Setup
```bash
# Install TPU support
pip install "jax[tpu]>=0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Configure TPU access
export TPU_NAME="local"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

### GPU Setup
```bash
# Install CUDA support
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES="0"
```

## Running Tests

### Using the Test Runner

```bash
# Run all tests
python tests/run_tests.py

# Run platform-specific tests
python tests/run_tests.py --platform tpu
python tests/run_tests.py --platform gpu
python tests/run_tests.py --platform cpu

# Run with benchmarks
python tests/run_tests.py --benchmark

# Run specific test types
python tests/run_tests.py --test-type unit
python tests/run_tests.py --test-type integration

# Run with verbose output
python tests/run_tests.py -v
```

### Using pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_kernels.py

# Run tests with specific marker
pytest -m tpu tests/
pytest -m gpu tests/

# Run tests in parallel
pytest -n auto tests/

# Generate coverage report
pytest --cov=vishwamai.kernels tests/
```

## Test Organization

### Directory Structure
```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # pytest configuration and fixtures
├── requirements-test.txt # Test dependencies
├── run_tests.py         # Test runner script
└── test_kernels.py      # Main test suite
```

### Test Categories

1. Matrix Operations
- Basic matrix multiplication
- Hybrid matrix operations
- Sparse matrix operations

2. Attention Mechanisms
- Flash attention
- Multi-head attention
- Sliding window attention

3. Layer Operations
- Layer normalization
- Expert parallelism
- Fusion patterns

4. Integration Tests
- End-to-end kernel pipelines
- Cross-platform operations
- Memory management

## Performance Benchmarks

Run benchmarks with:
```bash
python tests/run_tests.py --benchmark
```

### Benchmark Categories

1. Speed Tests
- Operation latency
- Throughput measurements
- Scaling behavior

2. Memory Tests
- Memory usage patterns
- Peak memory consumption
- Memory bandwidth

3. Platform-Specific Tests
- TPU MXU utilization
- GPU SM occupancy
- CPU cache performance

## Test Coverage

Generate coverage report:
```bash
pytest --cov=vishwamai.kernels --cov-report=html tests/
```

Expected coverage areas:
- Core kernel operations
- Platform-specific optimizations
- Memory management
- Error handling

## Adding New Tests

1. Test Structure:
```python
@pytest.mark.platform  # tpu, gpu, or cpu
def test_new_feature():
    """Test docstring describing purpose."""
    # Test setup
    data = prepare_test_data()
    
    # Test execution
    result = run_operation(data)
    
    # Test validation
    validate_result(result)
```

2. Performance Test:
```python
@pytest.mark.benchmark
def test_performance(benchmark):
    """Benchmark docstring."""
    benchmark(operation_to_test, test_data)
```

## Debugging Tests

### Common Issues

1. Platform Availability
```python
# Check platform availability
if not jax.devices('tpu'):
    pytest.skip("No TPU available")
```

2. Memory Management
```python
# Clear device memory
@pytest.fixture(autouse=True)
def clear_memory():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

3. Device Placement
```python
# Ensure correct device placement
data = place_on_device(data, platform)
```

### Troubleshooting

1. Memory Issues:
- Use `--memprof` flag for memory profiling
- Check for memory leaks with `pytest-leaks`
- Monitor device memory usage

2. Performance Issues:
- Use `--benchmark-only` for focused testing
- Profile with `--benchmark-histogram`
- Check operation fusion with XLA debug flags

3. Platform Issues:
- Verify platform availability
- Check device configurations
- Validate environment variables

## Contributing

1. Adding Tests:
- Follow existing test structure
- Include docstrings and type hints
- Add appropriate markers
- Update test documentation

2. Running CI:
- Ensure all tests pass
- Check coverage requirements
- Verify cross-platform compatibility
- Run performance benchmarks

3. Code Review:
- Follow style guide
- Include test cases
- Document changes
- Update requirements if needed

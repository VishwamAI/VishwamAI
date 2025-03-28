# VishwamAI Kernel Tests

Test suite for VishwamAI kernels across different hardware platforms (TPU, GPU, CPU).

## Quick Start

1. Install test dependencies:
```bash
# For CPU-only testing
pip install -e "tests/.[dev]"

# For GPU testing
pip install -e "tests/.[gpu,dev]"

# For TPU testing
pip install -e "tests/.[tpu,dev]"

# For all features (including benchmarks)
pip install -e "tests/.[all]"
```

2. Run tests:
```bash
# Using test runner (recommended)
./tests/run_tests.py                   # All available platforms
./tests/run_tests.py --platform cpu    # CPU only
./tests/run_tests.py --platform gpu    # GPU only
./tests/run_tests.py --platform tpu    # TPU only

# Using pytest directly
pytest tests/                          # Run all tests
pytest -m gpu tests/                   # Run GPU tests only
pytest -m tpu tests/                   # Run TPU tests only
```

## Test Organization

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # pytest configuration and fixtures
├── requirements-test.txt # Test dependencies
├── run_tests.py         # Test runner script
├── setup.py            # Test package setup
└── test_kernels.py      # Main test suite
```

## Platform Support

### CPU
- Always available
- Vectorized operations with NumPy
- Multi-threading support
- SIMD optimizations

### GPU
- Requires CUDA-capable GPU
- PyTorch backend
- Tensor core optimizations
- Automatic mixed precision

### TPU
- Requires TPU access
- JAX/XLA backend
- SPMD support
- BFloat16 optimization

## Running Benchmarks

```bash
# Run all benchmarks
./tests/run_tests.py --benchmark

# Platform-specific benchmarks
./tests/run_tests.py --platform gpu --benchmark
./tests/run_tests.py --platform tpu --benchmark

# Generate benchmark report
pytest --benchmark-only --benchmark-json=report.json tests/
```

## Test Types

### Unit Tests
```bash
./tests/run_tests.py --test-type unit
```

### Integration Tests
```bash
./tests/run_tests.py --test-type integration
```

### Performance Tests
```bash
# Run benchmarks with detailed profiling
./tests/run_tests.py --benchmark --verbose
```

## Common Issues

### TPU Not Found
```
Error: TPU platform not available
```
Solution:
1. Check TPU environment variables
2. Verify JAX TPU installation
3. Check TPU connectivity

### CUDA Not Available
```
Error: GPU platform not available
```
Solution:
1. Check CUDA installation
2. Verify GPU drivers
3. Check PyTorch CUDA support

### Memory Issues
```
CUDA out of memory
```
Solution:
1. Reduce batch size
2. Clear GPU cache
3. Use gradient checkpointing

## Development

### Adding New Tests

1. Create test function:
```python
@pytest.mark.platform  # tpu, gpu, or cpu
def test_new_feature():
    """Test docstring."""
    # Test implementation
```

2. Add to test suite:
```python
# In test_kernels.py
class TestNewFeature:
    def test_operation(self):
        pass
```

### Running Tests in Development

```bash
# Run with verbose output
./tests/run_tests.py -v

# Run specific test
pytest tests/test_kernels.py::TestClass::test_function

# Run with debugger
pytest --pdb tests/
```

### Code Style

```bash
# Format code
black tests/

# Check types
mypy tests/

# Sort imports
isort tests/
```

## CI/CD Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      platform: [cpu, gpu]
  steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: |
        pip install -e ".[${matrix.platform},dev]"
        ./tests/run_tests.py --platform ${{ matrix.platform }}
```

## Contributing

1. Follow test structure
2. Add appropriate markers
3. Include performance tests
4. Update documentation
5. Run full test suite

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

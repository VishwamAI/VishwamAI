l# Quick Testing Guide for VishwamAI

## Quick Start

1. Setup Test Environment:
```bash
# For TPU testing
./setup_test_env.sh tpu

# For GPU testing
./setup_test_env.sh gpu

# For CPU testing
./setup_test_env.sh cpu

# For all platforms
./setup_test_env.sh all
```

2. Run Tests:
```bash
# TPU tests
./tests/run_tests.py --platform tpu

# GPU tests
./tests/run_tests.py --platform gpu

# CPU tests
./tests/run_tests.py --platform cpu
```

## Common Testing Commands

### Test Specific Features
```bash
# Test matrix operations
pytest tests/test_kernels.py::TestMatrixOperations

# Test attention mechanisms
pytest tests/test_kernels.py::TestAttentionMechanisms

# Test sparse operations
pytest tests/test_kernels.py::TestSparseOperations
```

### Run Benchmarks
```bash
# Full benchmark suite
./tests/run_tests.py --benchmark

# Platform-specific benchmarks
./tests/run_tests.py --platform tpu --benchmark
./tests/run_tests.py --platform gpu --benchmark
./tests/run_tests.py --platform cpu --benchmark

# Generate benchmark report
pytest --benchmark-only --benchmark-json=bench.json tests/
```

### Integration Tests
```bash
# Run all integration tests
./tests/run_tests.py --test-type integration

# Platform-specific integration tests
./tests/run_tests.py --platform tpu --test-type integration
```

## Quick Environment Checks

### TPU Environment
```bash
# Check TPU availability
python3 -c "import jax; print(jax.devices('tpu'))"

# Verify TPU configuration
echo $TPU_NAME
echo $XRT_TPU_CONFIG

# Test TPU memory
python3 -c "import jax; import jax.numpy as jnp; x = jnp.ones((8192, 8192))"
```

### GPU Environment
```bash
# Check CUDA availability
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

### CPU Environment
```bash
# Check CPU threads
python3 -c "import os; print(f'CPU Threads: {os.cpu_count()}')"

# Check NumPy configuration
python3 -c "import numpy; numpy.show_config()"
```

## Common Issues & Solutions

### TPU Issues

1. TPU Not Found
```bash
# Solution: Check TPU setup
gcloud compute tpus tpu-vm describe vishwamai-tpu --zone=us-central1-a
source setup_test_env.sh tpu
```

2. JAX Import Error
```bash
# Solution: Reinstall JAX TPU
pip uninstall jax jaxlib
pip install "jax[tpu]>=0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### GPU Issues

1. CUDA Out of Memory
```bash
# Solution: Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

2. GPU Not Detected
```bash
# Solution: Check CUDA setup
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
```

### Test Runner Issues

1. Test Discovery Failed
```bash
# Solution: Check pytest configuration
pytest --collect-only tests/

# Update PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2. Test Timeouts
```bash
# Solution: Increase timeout in pytest.ini or command line
pytest --timeout=300 tests/
```

## Performance Testing

### Memory Profiling
```bash
# Profile TPU memory
./tests/run_tests.py --platform tpu --benchmark --memprof

# Profile GPU memory
./tests/run_tests.py --platform gpu --benchmark --memprof
```

### Throughput Testing
```bash
# Test operation throughput
pytest --benchmark-only --benchmark-min-rounds=100 tests/test_kernels.py::TestMatrixOperations
```

### Load Testing
```bash
# Run parallel tests
pytest -n auto tests/

# Run with different batch sizes
./tests/run_tests.py --benchmark --batch-sizes="1,8,32,128"
```

## Development Flow

1. Before Committing Changes:
```bash
# Run format checks
black tests/
isort tests/
flake8 tests/

# Run type checks
mypy tests/

# Run quick tests
./tests/run_tests.py --platform cpu  # Fastest platform for basic checks
```

2. Full Validation:
```bash
# Run complete test suite
./tests/run_tests.py --platform all --test-type all --benchmark
```

3. Generate Reports:
```bash
# Coverage report
pytest --cov=vishwamai.kernels --cov-report=html tests/

# Performance report
pytest --benchmark-only --benchmark-json=report.json tests/
python -m pytest_benchmark.utils compare report.json

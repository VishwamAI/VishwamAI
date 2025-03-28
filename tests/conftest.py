"""PyTest configuration for VishwamAI kernel tests."""

import os
import pytest
import jax
import torch
import numpy as np
from typing import Dict, Any

def pytest_configure(config):
    """Configure test environment."""
    # Configure JAX platform
    if jax.devices('tpu'):
        jax.config.update('jax_platform_name', 'tpu')
    elif torch.cuda.is_available():
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configure pytest markers
    config.addinivalue_line(
        "markers", 
        "gpu: mark test to run only on GPU"
    )
    config.addinivalue_line(
        "markers", 
        "tpu: mark test to run only on TPU"
    )
    config.addinivalue_line(
        "markers", 
        "integration: mark as integration test"
    )

def pytest_runtest_setup(item):
    """Set up test environment before each test."""
    # Skip GPU tests if CUDA is not available
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Test requires GPU")
    
    # Skip TPU tests if TPU is not available
    if "tpu" in item.keywords and not jax.devices('tpu'):
        pytest.skip("Test requires TPU")

@pytest.fixture(scope="session")
def device_info() -> Dict[str, Any]:
    """Provide device information for tests."""
    info = {
        "cpu_threads": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "tpu_available": bool(jax.devices('tpu')),
        "platform": jax.config.jax_platform_name,
    }
    
    if info["gpu_available"]:
        info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        })
    
    if info["tpu_available"]:
        info.update({
            "tpu_devices": len(jax.devices('tpu')),
            "tpu_platform": jax.local_devices()[0].platform,
        })
    
    return info

@pytest.fixture
def dtype_config() -> Dict[str, Any]:
    """Configure data types for different platforms."""
    return {
        "tpu": {
            "default": jax.numpy.bfloat16,
            "compute": jax.numpy.float32,
        },
        "gpu": {
            "default": torch.float16,
            "compute": torch.float32,
        },
        "cpu": {
            "default": np.float32,
            "compute": np.float64,
        }
    }

@pytest.fixture(autouse=True)
def clear_device_memory():
    """Clear device memory before each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # JAX/TPU memory is managed automatically

@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 10,
        "test_iterations": 100,
        "sizes": [128, 512, 1024, 2048],
        "batch_sizes": [1, 8, 32],
        "timeout": 300,  # seconds
    }

def pytest_report_header(config):
    """Add system information to test report header."""
    return [
        "Test Environment:",
        f"Python: {pytest.config.python_version}",
        f"JAX: {jax.__version__}",
        f"PyTorch: {torch.__version__}",
        f"NumPy: {np.__version__}",
        f"Platform: {jax.config.jax_platform_name}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"TPU available: {bool(jax.devices('tpu'))}",
    ]

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance summary to test report."""
    if hasattr(terminalreporter, 'performance_data'):
        terminalreporter.write_sep("=", "Performance Summary")
        for platform, data in terminalreporter.performance_data.items():
            terminalreporter.write_line(f"\n{platform} Performance:")
            for test, metrics in data.items():
                terminalreporter.write_line(f"  {test}:")
                for metric, value in metrics.items():
                    terminalreporter.write_line(f"    {metric}: {value}")

@pytest.fixture
def performance_logger(request):
    """Log performance metrics during tests."""
    def log_metric(platform: str, test_name: str, metric: str, value: float):
        if not hasattr(request.config._reporter, 'performance_data'):
            request.config._reporter.performance_data = {}
        
        platform_data = request.config._reporter.performance_data.setdefault(platform, {})
        test_data = platform_data.setdefault(test_name, {})
        test_data[metric] = value
    
    return log_metric

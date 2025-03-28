"""PyTest configuration for VishwamAI kernel tests."""

import os
import sys
import pytest
import numpy as np
from typing import Dict, Any

# Import platform-specific libraries with error handling
PLATFORMS = {
    'cpu': True,  # CPU is always available
    'gpu': False,  # Will be updated during initialization
    'tpu': False  # Will be updated during initialization
}

try:
    import torch
    PLATFORMS['gpu'] = torch.cuda.is_available()
except ImportError:
    pass

try:
    import jax
    if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
        try:
            jax.devices('tpu')
            PLATFORMS['tpu'] = True
        except RuntimeError:
            pass
except ImportError:
    pass

def pytest_configure(config):
    """Configure test environment."""
    # Set random seeds
    np.random.seed(42)
    
    if PLATFORMS['gpu']:
        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        except Exception:
            pass

    # Configure platform-specific settings
    if PLATFORMS['tpu']:
        try:
            jax.config.update('jax_platform_name', 'tpu')
        except Exception:
            pass
    elif PLATFORMS['gpu']:
        try:
            jax.config.update('jax_platform_name', 'gpu')
        except Exception:
            pass
    else:
        try:
            jax.config.update('jax_platform_name', 'cpu')
        except Exception:
            pass
    
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
    if "gpu" in item.keywords and not PLATFORMS['gpu']:
        pytest.skip("Test requires GPU")
    
    # Skip TPU tests if TPU is not available
    if "tpu" in item.keywords and not PLATFORMS['tpu']:
        pytest.skip("Test requires TPU")

@pytest.fixture(scope="session")
def device_info() -> Dict[str, Any]:
    """Provide device information for tests."""
    info = {
        "cpu_threads": os.cpu_count(),
        "gpu_available": PLATFORMS['gpu'],
        "tpu_available": PLATFORMS['tpu'],
        "platform": "cpu"  # default
    }
    
    if info["gpu_available"]:
        info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "platform": "gpu"
        })
    
    if info["tpu_available"]:
        info.update({
            "tpu_devices": len(jax.devices('tpu')),
            "tpu_platform": jax.local_devices()[0].platform,
            "platform": "tpu"
        })
    
    return info

@pytest.fixture
def dtype_config() -> Dict[str, Any]:
    """Configure data types for different platforms."""
    config = {
        "cpu": {
            "default": np.float32,
            "compute": np.float64,
        }
    }
    
    if PLATFORMS['tpu']:
        config["tpu"] = {
            "default": "bfloat16",
            "compute": "float32",
        }
    
    if PLATFORMS['gpu']:
        config["gpu"] = {
            "default": torch.float16,
            "compute": torch.float32,
        }
    
    return config

@pytest.fixture(autouse=True)
def clear_device_memory():
    """Clear device memory before each test."""
    yield
    if PLATFORMS['gpu']:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
    headers = [
        "Test Environment:",
        f"Python: {'.'.join(map(str, sys.version_info[:3]))}",
        f"NumPy: {np.__version__}",
        f"Available Platforms: {[k for k, v in PLATFORMS.items() if v]}"
    ]
    
    if PLATFORMS['gpu']:
        headers.extend([
            f"PyTorch: {torch.__version__}",
            f"CUDA: {torch.version.cuda}"
        ])
    
    if PLATFORMS['tpu']:
        headers.append(f"JAX: {jax.__version__}")
    
    return headers

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

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance summary to test report."""
    if hasattr(terminalreporter, 'performance_data'):
        terminalreporter.write_sep("=", "Performance Summary")
        
        for platform, data in terminalreporter.performance_data.items():
            if platform in PLATFORMS and PLATFORMS[platform]:
                terminalreporter.write_line(f"\n{platform.upper()} Performance:")
                for test, metrics in data.items():
                    terminalreporter.write_line(f"  {test}:")
                    for metric, value in metrics.items():
                        terminalreporter.write_line(f"    {metric}: {value}")

"""Test suite for VishwamAI kernels."""

from typing import Dict

# Map platforms to their test requirements
PLATFORM_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "tpu": {
        "jax": "jax[tpu]>=0.4.1",
        "pytest": "pytest>=7.0.0",
        "pytest-xdist": "pytest-xdist>=3.0.0",
        "pytest-benchmark": "pytest-benchmark>=4.0.0"
    },
    "gpu": {
        "torch": "torch>=2.0.0",
        "pytest": "pytest>=7.0.0",
        "pytest-xdist": "pytest-xdist>=3.0.0",
        "pytest-benchmark": "pytest-benchmark>=4.0.0"
    },
    "cpu": {
        "numpy": "numpy>=1.20.0",
        "pytest": "pytest>=7.0.0",
        "pytest-xdist": "pytest-xdist>=3.0.0",
        "pytest-benchmark": "pytest-benchmark>=4.0.0"
    }
}

"""
Deep Efficient Parallelism (DeepEP) optimizations for GPU operations.
Provides efficient parallel execution and memory management capabilities.
"""

from .buffer import Buffer
from .utils import get_num_sms, get_best_configs

__all__ = [
    'Buffer',
    'get_num_sms',
    'get_best_configs'
]

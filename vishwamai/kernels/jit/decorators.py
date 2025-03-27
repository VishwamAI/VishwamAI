"""Decorators for JIT kernel compilation."""

import functools
from typing import Callable, Optional
from .jit_manager import KernelPlatform, register_kernel, no_jit

def kernel_decorator(platform: KernelPlatform):
    """Base kernel decorator factory."""
    def decorator(name: str):
        def wrapper(fn: Callable):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                if hasattr(fn, '_no_jit'):
                    return fn(*args, **kwargs)
                return register_kernel(name, platform, fn)(*args, **kwargs)
            return wrapped
        return wrapper
    return decorator

# Platform-specific decorators
tpu_kernel = kernel_decorator(KernelPlatform.TPU)
gpu_kernel = kernel_decorator(KernelPlatform.GPU)
triton_kernel = kernel_decorator(KernelPlatform.TRITON)
cpu_kernel = kernel_decorator(KernelPlatform.CPU)
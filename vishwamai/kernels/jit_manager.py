"""JIT compilation manager for VishwamAI kernels."""

import jax
import jax.numpy as jnp
import functools
from enum import Enum
from typing import Any, Callable, Dict, Optional

class KernelPlatform(Enum):
    """Supported kernel platforms."""
    TPU = "tpu"
    GPU = "gpu"
    CPU = "cpu"
    TRITON = "triton"

class JITManager:
    """Manages JIT compilation of kernels."""
    
    def __init__(self):
        self.kernels: Dict[str, Dict[KernelPlatform, Callable]] = {}
        self.current_platform = self._detect_platform()
    
    def _detect_platform(self) -> KernelPlatform:
        """Detect current execution platform."""
        backend = jax.lib.xla_bridge.get_backend().platform
        if backend == "tpu":
            return KernelPlatform.TPU
        elif backend == "gpu":
            return KernelPlatform.GPU
        else:
            return KernelPlatform.CPU

    def register(
        self,
        name: str,
        platform: KernelPlatform,
        kernel_fn: Callable
    ) -> None:
        """Register a kernel for a specific platform."""
        if name not in self.kernels:
            self.kernels[name] = {}
        self.kernels[name][platform] = kernel_fn
    
    def get(
        self,
        name: str,
        platform: Optional[KernelPlatform] = None
    ) -> Callable:
        """Get the appropriate kernel for current platform."""
        if platform is None:
            platform = self.current_platform
            
        if name not in self.kernels:
            raise KeyError(f"No kernel registered with name: {name}")
            
        if platform not in self.kernels[name]:
            # Fall back to CPU if platform-specific version not found
            platform = KernelPlatform.CPU
            if platform not in self.kernels[name]:
                raise KeyError(
                    f"No kernel found for platform {platform} with name: {name}"
                )
                
        return self.kernels[name][platform]

# Global JIT manager instance
_manager = JITManager()

def get_manager() -> JITManager:
    """Get the global JIT manager instance."""
    return _manager

def register_kernel(
    name: str,
    platform: KernelPlatform,
    kernel_fn: Callable
) -> Callable:
    """Register a kernel with the global manager."""
    _manager.register(name, platform, kernel_fn)
    return kernel_fn

def get_kernel(
    name: str,
    platform: Optional[KernelPlatform] = None
) -> Callable:
    """Get a kernel from the global manager."""
    return _manager.get(name, platform)

# Decorator utilities
def tpu_kernel(name: str):
    """Register a TPU kernel."""
    def decorator(fn):
        return register_kernel(name, KernelPlatform.TPU, jax.jit(fn))
    return decorator

def gpu_kernel(name: str):
    """Register a GPU kernel."""
    def decorator(fn):
        return register_kernel(name, KernelPlatform.GPU, jax.jit(fn))
    return decorator

def triton_kernel(name: str):
    """Register a Triton kernel."""
    def decorator(fn):
        return register_kernel(name, KernelPlatform.TRITON, fn)
    return decorator

def cpu_kernel(name: str):
    """Register a CPU kernel."""
    def decorator(fn):
        return register_kernel(name, KernelPlatform.CPU, fn)
    return decorator

def no_jit(fn: Callable) -> Callable:
    """Mark a function to not be JIT compiled."""
    fn._no_jit = True
    return fn

# Common kernel functions
@tpu_kernel("matmul")
def matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Basic matrix multiplication."""
    return jnp.matmul(a, b)

@tpu_kernel("layer_norm")
def layer_norm(
    x: jnp.ndarray,
    scale: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    eps: float = 1e-6
) -> jnp.ndarray:
    """Basic layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    
    x = (x - mean) / jnp.sqrt(var + eps)
    if scale is not None:
        x = x * scale
    if bias is not None:
        x = x + bias
    return x

@tpu_kernel("gelu")
def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """GELU activation function."""
    return jax.nn.gelu(x)

@tpu_kernel("flash_attention")
def flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Basic flash attention implementation."""
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    scores = jnp.einsum('...qd,...kd->...qk', q, k) * scale
    
    if mask is not None:
        scores = jnp.where(mask, scores, -1e10)
        
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('...qk,...kd->...qd', weights, v)
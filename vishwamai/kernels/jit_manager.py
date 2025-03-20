"""JIT compilation manager for VishwamAI kernels on TPU and GPU."""

import os
import sys
import tempfile
import hashlib
import logging
import importlib
import functools
import inspect
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Try to import backends
try:
    from .tpu import tpu_kernels
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

try:
    from .gpu import cuda_kernels
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from .gpu import triton_kernels
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Cache directory for JIT kernels
def get_cache_dir() -> str:
    """Get the kernel cache directory with proper permissions."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".vishwamai_kernel_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

class KernelPlatform:
    """Enum-like class to represent supported kernel platforms."""
    TPU = "tpu"
    GPU = "gpu"
    GPU_TRITON = "gpu_triton"
    CPU = "cpu"
    
    @staticmethod
    def detect_platform() -> str:
        """Detect the current execution platform."""
        if not jax.local_devices():
            return KernelPlatform.CPU
            
        platform = jax.local_devices()[0].platform
        
        if platform == "tpu" and TPU_AVAILABLE:
            return KernelPlatform.TPU
        elif platform == "gpu":
            if TRITON_AVAILABLE:
                return KernelPlatform.GPU_TRITON
            elif CUDA_AVAILABLE:
                return KernelPlatform.GPU
        
        return KernelPlatform.CPU

class JITManager:
    """Manager for just-in-time compiling kernels for appropriate hardware."""
    
    def __init__(self, platform: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the JIT compilation manager.
        
        Args:
            platform: Target platform (tpu, gpu, gpu_triton, or cpu)
            cache_dir: Directory for caching compiled kernels
        """
        self.platform = platform or KernelPlatform.detect_platform()
        self.cache_dir = cache_dir or get_cache_dir()
        self._kernel_registry = {}
        self._compiled_kernels = {}
        
        # Create platform-specific subdirectories
        self.platform_cache = os.path.join(self.cache_dir, self.platform)
        os.makedirs(self.platform_cache, exist_ok=True)
        
        # Import appropriate backend
        if self.platform == KernelPlatform.TPU and TPU_AVAILABLE:
            self.backend = tpu_kernels
            logger.info("Using TPU optimized kernels")
        elif self.platform == KernelPlatform.GPU_TRITON and TRITON_AVAILABLE:
            self.backend = triton_kernels
            logger.info("Using GPU optimized kernels with Triton")
        elif self.platform == KernelPlatform.GPU and CUDA_AVAILABLE:
            self.backend = cuda_kernels
            logger.info("Using GPU optimized kernels with CUDA")
        else:
            self.backend = None
            logger.warning(f"No optimized backend available for platform {self.platform}, falling back to default JAX implementations")
    
    def register_kernel(self, name: str, 
                        tpu_impl: Optional[Callable] = None,
                        gpu_impl: Optional[Callable] = None,
                        triton_impl: Optional[Callable] = None,
                        cpu_impl: Optional[Callable] = None,
                        fallback_impl: Optional[Callable] = None):
        """
        Register implementations for a kernel across different platforms.
        
        Args:
            name: Kernel name identifier
            tpu_impl: TPU-specific implementation
            gpu_impl: GPU-specific implementation (CUDA)
            triton_impl: Triton-based GPU implementation
            cpu_impl: CPU-specific implementation
            fallback_impl: Fallback implementation if platform-specific one is unavailable
        """
        self._kernel_registry[name] = {
            KernelPlatform.TPU: tpu_impl,
            KernelPlatform.GPU: gpu_impl,
            KernelPlatform.GPU_TRITON: triton_impl,
            KernelPlatform.CPU: cpu_impl,
            "fallback": fallback_impl or (lambda *args, **kwargs: None)
        }
    
    def get_kernel(self, name: str) -> Callable:
        """
        Get the appropriate kernel implementation for the current platform.
        
        Args:
            name: Kernel name identifier
            
        Returns:
            Compiled kernel function
        
        Raises:
            ValueError: If kernel is not registered
        """
        if name not in self._kernel_registry:
            raise ValueError(f"Kernel {name} is not registered")
            
        # Check if we've already compiled this kernel
        if name in self._compiled_kernels:
            return self._compiled_kernels[name]
            
        # Get the appropriate implementation
        impl = self._kernel_registry[name][self.platform]
            
        # If no platform-specific implementation, use fallback
        if impl is None:
            impl = self._kernel_registry[name]["fallback"]
            logger.warning(f"No {self.platform} implementation for kernel {name}, using fallback")
            
        # Compile the kernel if needed
        if hasattr(impl, '_is_compiled') and impl._is_compiled:
            compiled_fn = impl
        else:
            compiled_fn = self._compile_kernel(name, impl)
            
        # Cache the compiled kernel
        self._compiled_kernels[name] = compiled_fn
        return compiled_fn
    
    def _compile_kernel(self, name: str, impl: Callable) -> Callable:
        """
        Compile a kernel implementation for the current platform.
        
        Args:
            name: Kernel name
            impl: Kernel implementation function
            
        Returns:
            Compiled kernel function
        """
        # Hash the function source and arguments to generate a cache key
        try:
            source_code = inspect.getsource(impl)
            signature = str(inspect.signature(impl))
            key = hashlib.md5((source_code + signature + self.platform).encode()).hexdigest()
            cache_path = os.path.join(self.platform_cache, f"{name}_{key}.py")
            
            # Check if we have a cached version
            if os.path.exists(cache_path):
                # Load the cached module
                spec = importlib.util.spec_from_file_location(f"{name}_{key}", cache_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                compiled_fn = getattr(module, "kernel_impl")
                compiled_fn._is_compiled = True
                return compiled_fn
        except (TypeError, OSError) as e:
            # If we can't hash or cache the function, we'll just compile it without caching
            logger.warning(f"Unable to cache kernel {name}: {str(e)}")
        
        # Compile for the specific platform
        if self.platform == KernelPlatform.TPU:
            compiled_fn = jax.jit(impl)
        elif self.platform in (KernelPlatform.GPU, KernelPlatform.GPU_TRITON):
            # For GPU kernels
            if hasattr(impl, '_no_jit') and impl._no_jit:
                # Some kernels (like those using Triton) manage their own compilation
                compiled_fn = impl
            else:
                compiled_fn = jax.jit(impl, backend="cuda")
        else:
            # Default CPU compilation
            compiled_fn = jax.jit(impl)
            
        compiled_fn._is_compiled = True
        
        # Try to save to cache if we have a cache key
        try:
            if 'cache_path' in locals():
                with open(cache_path, 'w') as f:
                    f.write(f"# Cached kernel: {name} for {self.platform}\n")
                    f.write(f"import jax\n")
                    f.write(f"import jax.numpy as jnp\n\n")
                    f.write(f"{source_code}\n\n")
                    f.write(f"kernel_impl = {impl.__name__}\n")
                    f.write(f"kernel_impl._is_compiled = True\n")
        except Exception as e:
            logger.warning(f"Failed to cache kernel {name}: {str(e)}")
            
        return compiled_fn
    
    def load_backend_kernels(self):
        """Load all kernels from the current backend."""
        if self.backend is None:
            return
        
        if hasattr(self.backend, 'optimized_kernels'):
            for name, kernel_fn in self.backend.optimized_kernels.items():
                # Register with the appropriate platform
                if self.platform == KernelPlatform.TPU:
                    self.register_kernel(name, tpu_impl=kernel_fn, fallback_impl=kernel_fn)
                elif self.platform == KernelPlatform.GPU:
                    self.register_kernel(name, gpu_impl=kernel_fn, fallback_impl=kernel_fn)
                elif self.platform == KernelPlatform.GPU_TRITON:
                    self.register_kernel(name, triton_impl=kernel_fn, fallback_impl=kernel_fn)
                else:
                    self.register_kernel(name, fallback_impl=kernel_fn)

# Platform-specific kernel decorators
def no_jit(func):
    """Mark a function to skip JIT compilation."""
    func._no_jit = True
    return func

def tpu_kernel(func):
    """Decorator to mark a function as a TPU kernel implementation."""
    func._kernel_platform = KernelPlatform.TPU
    return func

def gpu_kernel(func):
    """Decorator to mark a function as a GPU kernel implementation."""
    func._kernel_platform = KernelPlatform.GPU
    return func

def triton_kernel(func):
    """Decorator to mark a function as a Triton kernel implementation."""
    func._kernel_platform = KernelPlatform.GPU_TRITON
    func._no_jit = True  # Triton manages its own compilation
    return func

def cpu_kernel(func):
    """Decorator to mark a function as a CPU kernel implementation."""
    func._kernel_platform = KernelPlatform.CPU
    return func

# Create a global manager instance for easy access
_default_manager = None

def get_manager() -> JITManager:
    """Get the default JIT manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = JITManager()
        _default_manager.load_backend_kernels()
    return _default_manager

def register_kernel(name: str, 
                    tpu_impl: Optional[Callable] = None,
                    gpu_impl: Optional[Callable] = None,
                    triton_impl: Optional[Callable] = None,
                    cpu_impl: Optional[Callable] = None,
                    fallback_impl: Optional[Callable] = None):
    """
    Register a kernel with the default manager.
    
    Args:
        name: Kernel name identifier
        tpu_impl: TPU-specific implementation
        gpu_impl: GPU-specific implementation (CUDA)
        triton_impl: Triton-based GPU implementation
        cpu_impl: CPU-specific implementation
        fallback_impl: Fallback implementation if platform-specific one is unavailable
    """
    manager = get_manager()
    manager.register_kernel(
        name, tpu_impl, gpu_impl, triton_impl, cpu_impl, fallback_impl
    )

def get_kernel(name: str) -> Callable:
    """
    Get a kernel from the default manager.
    
    Args:
        name: Kernel name identifier
    
    Returns:
        Compiled kernel function
    """
    manager = get_manager()
    return manager.get_kernel(name)

# Common kernel functions available through the interface
def matmul(x, y, transpose_a=False, transpose_b=False):
    """Platform-optimized matrix multiplication."""
    fn = get_kernel("matmul")
    return fn(x, y, transpose_a=transpose_a, transpose_b=transpose_b)

def layer_norm(x, weight=None, bias=None, eps=1e-5):
    """Platform-optimized layer normalization."""
    fn = get_kernel("layer_norm")
    return fn(x, weight=weight, bias=bias, eps=eps)

def gelu(x):
    """Platform-optimized GELU activation function."""
    fn = get_kernel("gelu")
    return fn(x)

def flash_attention(q, k, v, mask=None, causal=True, scale=None):
    """Platform-optimized flash attention implementation."""
    # Different backends might have slightly different APIs, handle them here
    if get_manager().platform == KernelPlatform.TPU:
        fn = get_kernel("attention")
        return fn(q, k, v, causal=causal, scale=scale)
    elif get_manager().platform == KernelPlatform.GPU_TRITON:
        fn = get_kernel("fused_attention")
        return fn(q, k, v, mask=mask, causal=causal, scale=scale)
    else:
        fn = get_kernel("flash_attention")
        return fn(q, k, v, mask=mask, causal=causal, scale=scale)
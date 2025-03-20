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
from jaxlib.xla_extension import DeviceArray
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

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
    CPU = "cpu"
    
    @staticmethod
    def detect_platform() -> str:
        """Detect the current execution platform."""
        if jax.devices()[0].platform == "tpu":
            return KernelPlatform.TPU
        elif jax.devices()[0].platform == "gpu":
            return KernelPlatform.GPU
        else:
            return KernelPlatform.CPU

class JITManager:
    """Manager for just-in-time compiling kernels for appropriate hardware."""
    
    def __init__(self, platform: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the JIT compilation manager.
        
        Args:
            platform: Target platform (tpu, gpu, or cpu)
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
        if self.platform == KernelPlatform.TPU:
            try:
                from ..tpu import tpu_kernels
                self.backend = tpu_kernels
            except ImportError:
                logger.warning("TPU kernel module not found, falling back to JAX defaults")
                self.backend = None
        elif self.platform == KernelPlatform.GPU:
            try:
                # Try to import Triton first, then fall back to standard CUDA
                try:
                    from ..gpu import triton_kernels
                    self.backend = triton_kernels
                    self.use_triton = True
                except ImportError:
                    from ..gpu import cuda_kernels
                    self.backend = cuda_kernels
                    self.use_triton = False
            except ImportError:
                logger.warning("GPU kernel module not found, falling back to JAX defaults")
                self.backend = None
                self.use_triton = False
        else:
            self.backend = None
    
    def register_kernel(self, name: str, 
                        tpu_impl: Optional[Callable] = None,
                        gpu_impl: Optional[Callable] = None,
                        cpu_impl: Optional[Callable] = None,
                        triton_impl: Optional[Callable] = None,
                        fallback_impl: Optional[Callable] = None):
        """
        Register implementations for a kernel across different platforms.
        
        Args:
            name: Kernel name identifier
            tpu_impl: TPU-specific implementation
            gpu_impl: GPU-specific implementation (CUDA)
            cpu_impl: CPU-specific implementation
            triton_impl: Triton-based GPU implementation
            fallback_impl: Fallback implementation if platform-specific one is unavailable
        """
        self._kernel_registry[name] = {
            KernelPlatform.TPU: tpu_impl,
            KernelPlatform.GPU: gpu_impl,
            KernelPlatform.CPU: cpu_impl,
            "triton": triton_impl,
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
        impl = None
        if self.platform == KernelPlatform.GPU and self.use_triton:
            impl = self._kernel_registry[name]["triton"]
            
        if impl is None:
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
        
        # Compile for the specific platform
        if self.platform == KernelPlatform.TPU:
            compiled_fn = jax.jit(impl)
        elif self.platform == KernelPlatform.GPU:
            if self.use_triton:
                # For Triton kernels, they handle their own compilation
                compiled_fn = impl
            else:
                # For CUDA kernels via JAX/XLA
                compiled_fn = jax.jit(impl, backend="cuda")
        else:
            # Default CPU compilation
            compiled_fn = jax.jit(impl)
            
        compiled_fn._is_compiled = True
        
        # Save to cache
        try:
            with open(cache_path, 'w') as f:
                f.write(f"# Cached kernel: {name} for {self.platform}\n")
                f.write(f"import jax\n")
                f.write(f"import jax.numpy as jnp\n\n")
                f.write(f"{inspect.getsource(impl)}\n\n")
                f.write(f"kernel_impl = {impl.__name__}\n")
                f.write(f"kernel_impl._is_compiled = True\n")
        except Exception as e:
            logger.warning(f"Failed to cache kernel {name}: {str(e)}")
            
        return compiled_fn
    
    @staticmethod
    def kernel(platform: Optional[str] = None):
        """
        Decorator to mark a function as a platform-specific kernel implementation.
        
        Args:
            platform: Target platform (tpu, gpu, cpu, or triton)
            
        Returns:
            Decorated function
        """
        def decorator(func):
            func._kernel_platform = platform
            func._is_kernel = True
            return func
        return decorator
    
    @classmethod
    def auto_register_kernels(cls, module) -> 'JITManager':
        """
        Auto-register all kernels defined in a module.
        
        Args:
            module: Python module containing kernel implementations
        
        Returns:
            JITManager instance with registered kernels
        """
        manager = cls()
        
        # Find all functions with _is_kernel attribute
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if hasattr(func, '_is_kernel'):
                platform = getattr(func, '_kernel_platform', None)
                
                # Initialize empty kernel registration if needed
                if name not in manager._kernel_registry:
                    manager.register_kernel(name)
                    
                # Register for the specified platform
                if platform == KernelPlatform.TPU:
                    manager._kernel_registry[name][KernelPlatform.TPU] = func
                elif platform == KernelPlatform.GPU:
                    manager._kernel_registry[name][KernelPlatform.GPU] = func
                elif platform == KernelPlatform.CPU:
                    manager._kernel_registry[name][KernelPlatform.CPU] = func
                elif platform == "triton":
                    manager._kernel_registry[name]["triton"] = func
                else:
                    # Default to fallback
                    manager._kernel_registry[name]["fallback"] = func
                    
        return manager

# Global instance for convenience
default_manager = JITManager()

def register_kernel(name: str, 
                    tpu_impl: Optional[Callable] = None,
                    gpu_impl: Optional[Callable] = None,
                    cpu_impl: Optional[Callable] = None,
                    triton_impl: Optional[Callable] = None,
                    fallback_impl: Optional[Callable] = None):
    """
    Register a kernel with the default manager.
    
    Args:
        name: Kernel name identifier
        tpu_impl: TPU-specific implementation
        gpu_impl: GPU-specific implementation (CUDA)
        cpu_impl: CPU-specific implementation
        triton_impl: Triton-based GPU implementation
        fallback_impl: Fallback implementation if platform-specific one is unavailable
    """
    default_manager.register_kernel(
        name, tpu_impl, gpu_impl, cpu_impl, triton_impl, fallback_impl
    )

def get_kernel(name: str) -> Callable:
    """
    Get a kernel from the default manager.
    
    Args:
        name: Kernel name identifier
    
    Returns:
        Compiled kernel function
    """
    return default_manager.get_kernel(name)

# Platform-specific kernel decorators
def tpu_kernel(func):
    """Decorator to mark a function as a TPU kernel implementation."""
    func._kernel_platform = KernelPlatform.TPU
    func._is_kernel = True
    return func

def gpu_kernel(func):
    """Decorator to mark a function as a GPU kernel implementation."""
    func._kernel_platform = KernelPlatform.GPU
    func._is_kernel = True
    return func

def cpu_kernel(func):
    """Decorator to mark a function as a CPU kernel implementation."""
    func._kernel_platform = KernelPlatform.CPU
    func._is_kernel = True
    return func

def triton_kernel(func):
    """Decorator to mark a function as a Triton kernel implementation."""
    func._kernel_platform = "triton"
    func._is_kernel = True
    return func
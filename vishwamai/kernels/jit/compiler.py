"""JIT compilation system for kernel optimization."""

import os
import hashlib
from typing import Dict, Any, Optional, Callable, Union
import jax
import jax.numpy as jnp
from jax.experimental import maps
import torch
from functools import partial, lru_cache

from ..core.kernel import KernelConfig, HardwareType

class JITCache:
    """Cache for compiled kernels."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/vishwamai/kernels")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
        
    def get_cache_key(self, code: str, config: Dict[str, Any]) -> str:
        """Generate cache key from code and config."""
        config_str = str(sorted(config.items()))
        content = f"{code}{config_str}".encode()
        return hashlib.sha256(content).hexdigest()
        
    def get(self, key: str) -> Optional[Callable]:
        """Get cached kernel if available."""
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, "rb") as f:
                self.memory_cache[key] = pickle.load(f)
            return self.memory_cache[key]
        return None
        
    def put(self, key: str, kernel: Callable):
        """Cache compiled kernel."""
        self.memory_cache[key] = kernel
        
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(kernel, f)

class KernelCompiler:
    """Compile kernels for different hardware targets."""
    
    def __init__(self):
        self.cache = JITCache()
        
    def compile_for_tpu(self,
                       kernel_fn: Callable,
                       static_argnums: Optional[tuple] = None,
                       **jit_options) -> Callable:
        """Compile kernel for TPU execution."""
        
        # Create optimized TPU version
        @partial(jax.jit,
                static_argnums=static_argnums,
                backend="tpu",
                **jit_options)
        def tpu_kernel(*args, **kwargs):
            return kernel_fn(*args, **kwargs)
            
        # Add optional SPMD compilation
        def wrapped_kernel(*args, **kwargs):
            if maps.thread_resources.env.physical_mesh.size > 1:
                return maps.xmap(
                    tpu_kernel,
                    in_axes=("batch", "hidden"),
                    out_axes=("batch", "hidden"),
                    axis_resources={"batch": "x", "hidden": "y"}
                )(*args, **kwargs)
            return tpu_kernel(*args, **kwargs)
            
        return wrapped_kernel
        
    def compile_for_gpu(self,
                       kernel_fn: Callable,
                       input_spec: Optional[Dict[str, torch.dtype]] = None,
                       **compile_options) -> Callable:
        """Compile kernel for GPU execution."""
        
        # Create TorchScript version
        def gpu_kernel(*args, **kwargs):
            # Convert inputs to correct device/dtype
            if input_spec:
                args = [arg.to(dtype=input_spec[f"arg_{i}"]) 
                       for i, arg in enumerate(args)]
                for k, v in kwargs.items():
                    if k in input_spec:
                        kwargs[k] = v.to(dtype=input_spec[k])
                        
            with torch.cuda.amp.autocast():
                return kernel_fn(*args, **kwargs)
                
        # JIT compile with TorchScript
        return torch.jit.script(gpu_kernel, **compile_options)
        
    @lru_cache(maxsize=1024)
    def get_or_compile(self,
                      kernel_fn: Callable,
                      config: KernelConfig,
                      kernel_name: str,
                      static_argnums: Optional[tuple] = None,
                      input_spec: Optional[Dict[str, Any]] = None,
                      **compile_options) -> Callable:
        """Get cached kernel or compile new one."""
        
        # Generate cache key
        key = self.cache.get_cache_key(
            kernel_name,
            {
                "config": config.__dict__,
                "static_argnums": static_argnums,
                "input_spec": input_spec,
                "compile_options": compile_options
            }
        )
        
        # Check cache first
        cached = self.cache.get(key)
        if cached is not None:
            return cached
            
        # Compile for target hardware
        if config.hardware == HardwareType.TPU:
            kernel = self.compile_for_tpu(
                kernel_fn,
                static_argnums=static_argnums,
                **compile_options
            )
        elif config.hardware == HardwareType.GPU:
            kernel = self.compile_for_gpu(
                kernel_fn,
                input_spec=input_spec,
                **compile_options
            )
        else:
            kernel = kernel_fn # CPU fallback
            
        # Cache compiled kernel
        self.cache.put(key, kernel)
        return kernel

# Global compiler instance
_compiler = KernelCompiler()

def get_compiler() -> KernelCompiler:
    """Get global kernel compiler instance."""
    return _compiler

"""Kernel compilation and profiling utilities."""

from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import jax
import jax.numpy as jnp
import torch
import numpy as np
from dataclasses import dataclass

from vishwamai.kernels.core.kernel import KernelConfig, HardwareType

@dataclass
class KernelProfile:
    """Profile data for a kernel execution."""
    exec_time: float
    memory_usage: int
    compute_intensity: float

class KernelCompiler:
    """Kernel compilation and caching."""
    
    def __init__(self):
        self.compiled_kernels: Dict[Tuple[str, str], Any] = {}
        
    def get_or_compile(self,
                       kernel_fn: Callable,
                       config: KernelConfig,
                       kernel_name: str,
                       static_argnums: Optional[Tuple[int, ...]] = None,
                       input_spec: Optional[Dict[str, Any]] = None) -> Any:
        """Get or compile kernel function."""
        cache_key = (kernel_name, str(config))
        
        if cache_key in self.compiled_kernels:
            return self.compiled_kernels[cache_key]
            
        # Compile kernel
        if config.hardware == HardwareType.TPU:
            compiled = self._compile_tpu(kernel_fn, static_argnums)
        elif config.hardware == HardwareType.GPU:
            compiled = self._compile_gpu(kernel_fn, input_spec)
        else:
            compiled = kernel_fn
            
        # Cache result
        self.compiled_kernels[cache_key] = compiled
        return compiled
        
    def _compile_tpu(self,
                     kernel_fn: Callable,
                     static_argnums: Optional[Tuple[int, ...]] = None) -> Any:
        """Compile kernel for TPU execution."""
        return jax.jit(kernel_fn, static_argnums=static_argnums)
        
    def _compile_gpu(self,
                     kernel_fn: Callable,
                     input_spec: Optional[Dict[str, Any]] = None) -> Any:
        """Compile kernel for GPU execution."""
        if input_spec is None:
            return kernel_fn
            
        # Create dummy inputs
        dummy_inputs = [
            torch.zeros(1, dtype=dtype, device="cuda")
            for dtype in input_spec.values()
        ]
        
        # Compile with TorchScript
        return torch.jit.trace(kernel_fn, dummy_inputs)

class KernelProfiler:
    """Profile kernel execution."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        
    def profile_kernel(self,
                       kernel_fn: Callable,
                       sample_inputs: Dict[str, Any],
                       num_warmup: int = 5,
                       num_runs: int = 10) -> KernelProfile:
        """Profile kernel execution time."""
        # Warmup
        for _ in range(num_warmup):
            kernel_fn(**sample_inputs)
            
        # Time execution
        start_time = time.perf_counter()
        for _ in range(num_runs):
            kernel_fn(**sample_inputs)
        end_time = time.perf_counter()
        
        # Estimate memory and compute
        memory_usage = self._estimate_memory(sample_inputs)
        compute_intensity = self._estimate_compute(kernel_fn, sample_inputs)
        
        return KernelProfile(
            exec_time=(end_time - start_time) / num_runs,
            memory_usage=memory_usage,
            compute_intensity=compute_intensity
        )
        
    def _estimate_memory(self, inputs: Dict[str, Any]) -> int:
        """Estimate memory usage of inputs."""
        total_memory = 0
        for tensor in inputs.values():
            if isinstance(tensor, (jnp.ndarray, np.ndarray)):
                total_memory += tensor.nbytes
            elif isinstance(tensor, torch.Tensor):
                total_memory += tensor.element_size() * tensor.nelement()
        return total_memory
        
    def _estimate_compute(self, kernel_fn: Callable, inputs: Dict[str, Any]) -> float:
        """Estimate computational intensity."""
        # For now return a simple ratio of compute to memory
        total_memory = self._estimate_memory(inputs)
        if total_memory == 0:
            return 0.0
        return self.profile_kernel(kernel_fn, inputs, num_warmup=1, num_runs=1).exec_time / total_memory

def get_compiler() -> KernelCompiler:
    """Get the global kernel compiler instance."""
    if not hasattr(get_compiler, "_instance"):
        get_compiler._instance = KernelCompiler()
    return get_compiler._instance
"""TPU kernel profiling utilities."""

import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class KernelProfile:
    """Profile data for a kernel execution."""
    name: str
    execution_time: float
    memory_used: int
    flops: int
    tpu_utilization: float
    input_shapes: List[tuple]
    output_shapes: List[tuple]

class TPUKernelProfiler:
    """Profiles TPU kernel execution."""
    
    def __init__(self):
        self.profiles: Dict[str, List[KernelProfile]] = defaultdict(list)
        self.current_kernel: Optional[str] = None
        self.start_time: Optional[float] = None
        
    def start_profile(self, kernel_name: str) -> None:
        """Start profiling a kernel."""
        self.current_kernel = kernel_name
        self.start_time = time.perf_counter()
        
    def end_profile(
        self,
        memory_used: int,
        flops: int,
        input_shapes: List[tuple],
        output_shapes: List[tuple]
    ) -> None:
        """End profiling current kernel."""
        if self.current_kernel is None or self.start_time is None:
            return
            
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        
        # Get TPU utilization
        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            tpu_utilization = stats["compute_time"] / (stats["compute_time"] + stats["idle_time"])
        else:
            tpu_utilization = 0.0
            
        profile = KernelProfile(
            name=self.current_kernel,
            execution_time=execution_time,
            memory_used=memory_used,
            flops=flops,
            tpu_utilization=tpu_utilization,
            input_shapes=input_shapes,
            output_shapes=output_shapes
        )
        
        self.profiles[self.current_kernel].append(profile)
        self.current_kernel = None
        self.start_time = None
        
    def profile_kernel(
        self,
        kernel_fn: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Profile a kernel function execution."""
        kernel_name = kernel_fn.__name__
        self.start_profile(kernel_name)
        
        # Get input shapes
        input_shapes = []
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                input_shapes.append(arg.shape)
                
        # Execute kernel
        result = kernel_fn(*args, **kwargs)
        
        # Get output shapes
        output_shapes = []
        if isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, (jnp.ndarray, np.ndarray)):
                    output_shapes.append(r.shape)
        elif isinstance(result, (jnp.ndarray, np.ndarray)):
            output_shapes.append(result.shape)
            
        # Estimate memory and FLOPs
        memory_used = self._estimate_memory_usage(input_shapes, output_shapes)
        flops = self._estimate_flops(kernel_name, input_shapes, output_shapes)
        
        self.end_profile(memory_used, flops, input_shapes, output_shapes)
        return result
        
    def _estimate_memory_usage(
        self,
        input_shapes: List[tuple],
        output_shapes: List[tuple]
    ) -> int:
        """Estimate memory usage from shapes."""
        total_elements = 0
        
        # Input arrays
        for shape in input_shapes:
            total_elements += np.prod(shape)
            
        # Output arrays
        for shape in output_shapes:
            total_elements += np.prod(shape)
            
        # Assume bfloat16 (2 bytes per element)
        return total_elements * 2
        
    def _estimate_flops(
        self,
        kernel_name: str,
        input_shapes: List[tuple],
        output_shapes: List[tuple]
    ) -> int:
        """Estimate FLOPs for common operations."""
        if "matmul" in kernel_name.lower():
            # Matrix multiplication
            if len(input_shapes) >= 2:
                m, n = input_shapes[0][-2:]
                _, k = input_shapes[1][-2:]
                return m * n * k * 2  # multiply-add counts as 2 FLOPs
        elif "conv" in kernel_name.lower():
            # Convolution
            if len(input_shapes) >= 2:
                n, c, h, w = input_shapes[0]
                f, _, kh, kw = input_shapes[1]
                oh = (h - kh + 1)
                ow = (w - kw + 1)
                return n * f * oh * ow * c * kh * kw * 2
                
        # Default estimate
        return sum(np.prod(shape) for shape in output_shapes)
        
    def get_kernel_stats(self, kernel_name: str) -> Dict[str, float]:
        """Get statistics for a kernel."""
        if kernel_name not in self.profiles:
            return {}
            
        profiles = self.profiles[kernel_name]
        times = [p.execution_time for p in profiles]
        memory = [p.memory_used for p in profiles]
        flops = [p.flops for p in profiles]
        util = [p.tpu_utilization for p in profiles]
        
        return {
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "avg_memory": np.mean(memory),
            "avg_flops": np.mean(flops),
            "avg_tpu_util": np.mean(util),
            "num_calls": len(profiles)
        }
        
    def print_summary(self, top_k: int = 10) -> None:
        """Print profiling summary for top K kernels."""
        # Sort kernels by average execution time
        kernel_times = []
        for name, profiles in self.profiles.items():
            avg_time = np.mean([p.execution_time for p in profiles])
            kernel_times.append((name, avg_time))
            
        kernel_times.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTPU Kernel Profiling Summary")
        print("-" * 80)
        print(f"{'Kernel Name':<30} {'Avg Time (ms)':<12} {'Avg Memory (MB)':<14} "
              f"{'Avg GFLOPs':<10} {'TPU Util %':<10} {'Calls':<8}")
        print("-" * 80)
        
        for name, _ in kernel_times[:top_k]:
            stats = self.get_kernel_stats(name)
            print(f"{name:<30} "
                  f"{stats['avg_time']*1000:>11.2f} "
                  f"{stats['avg_memory']/1e6:>13.1f} "
                  f"{stats['avg_flops']/1e9:>9.1f} "
                  f"{stats['avg_tpu_util']*100:>9.1f} "
                  f"{stats['num_calls']:>8}")
                  
    def reset(self) -> None:
        """Reset all profiling data."""
        self.profiles.clear()
        self.current_kernel = None
        self.start_time = None
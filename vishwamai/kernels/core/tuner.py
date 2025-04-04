"""Kernel autotuning and optimization."""

from typing import Dict, Any, List, Optional, Tuple, Callable
import dataclasses
import time
import jax
import jax.numpy as jnp
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from vishwamai.kernels.core.compiler import get_compiler, KernelProfiler
from vishwamai.kernels.core.kernel import KernelConfig, HardwareType
from vishwamai.kernels.core.memory import MemoryManager
from vishwamai.kernels.core.shapes import DynamicShapeOptimizer

@dataclasses.dataclass
class TuningConfig:
    """Configuration for kernel autotuning."""
    num_trials: int = 10
    num_warmup: int = 5
    min_block_size: int = 32
    max_block_size: int = 256
    search_parallel: bool = True
    timeout_seconds: float = 60.0
    block_sizes: List[int] = dataclasses.field(default_factory=lambda: [128, 256, 512])
    batch_sizes: List[int] = dataclasses.field(default_factory=lambda: [1, 8, 16, 32])
    num_warps: List[int] = dataclasses.field(default_factory=lambda: [4, 8, 16, 32])
    precision_modes: List[str] = dataclasses.field(default_factory=lambda: ["fp32", "bf16"])

@dataclasses.dataclass 
class TuningResult:
    """Results from kernel autotuning."""
    best_config: Dict[str, Any]
    timing_ms: float
    throughput: float
    memory_usage: int
    compute_utilization: float

class KernelTuner:
    """Automated kernel performance tuning."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.compiler = get_compiler()
        self.profiler = KernelProfiler(config)
        self.memory_manager = MemoryManager(config)
        self.shape_optimizer = DynamicShapeOptimizer(config)
        
    def tune_kernel(self,
                   kernel_fn: Callable,
                   sample_inputs: Dict[str, Any],
                   tuning_config: TuningConfig) -> TuningResult:
        """Find optimal kernel configuration."""
        if self.config.hardware == HardwareType.TPU:
            return self._tune_tpu_kernel(kernel_fn, sample_inputs, tuning_config)
        elif self.config.hardware == HardwareType.GPU:
            return self._tune_gpu_kernel(kernel_fn, sample_inputs, tuning_config)
        else:
            return self._tune_cpu_kernel(kernel_fn, sample_inputs, tuning_config)
            
    def _tune_tpu_kernel(self,
                        kernel_fn: Callable,
                        sample_inputs: Dict[str, Any],
                        tuning_config: TuningConfig) -> TuningResult:
        """Tune kernel for TPU execution."""
        best_result = None
        best_timing = float('inf')
        
        # Try different configurations
        for block_size in tuning_config.block_sizes:
            for batch_size in tuning_config.batch_sizes:
                for precision in tuning_config.precision_modes:
                    # Create test config
                    test_config = self.config.__class__(
                        hardware=HardwareType.TPU,
                        block_size=block_size,
                        batch_size=batch_size,
                        precision=precision
                    )
                    
                    # Compile with test config
                    compiled = self.compiler.get_or_compile(
                        kernel_fn,
                        test_config,
                        kernel_fn.__name__,
                        static_argnums=(1,)  # Assume second arg is static
                    )
                    
                    # Profile execution
                    profile = self.profiler.profile_kernel(
                        compiled,
                        sample_inputs,
                        num_warmup=tuning_config.num_warmup,
                        num_runs=tuning_config.num_trials
                    )
                    
                    # Check if best so far
                    if profile.exec_time < best_timing:
                        best_timing = profile.exec_time
                        best_result = TuningResult(
                            best_config={
                                "block_size": block_size,
                                "batch_size": batch_size,
                                "precision": precision
                            },
                            timing_ms=profile.exec_time * 1000,
                            throughput=1.0 / profile.exec_time,
                            memory_usage=profile.memory_usage,
                            compute_utilization=profile.compute_intensity
                        )
                        
        return best_result
        
    def _tune_gpu_kernel(self,
                        kernel_fn: Callable,
                        sample_inputs: Dict[str, Any],
                        tuning_config: TuningConfig) -> TuningResult:
        """Tune kernel for GPU execution."""
        best_result = None
        best_timing = float('inf')
        
        # Try different configurations
        for block_size in tuning_config.block_sizes:
            for num_warps in tuning_config.num_warps:
                for precision in tuning_config.precision_modes:
                    # Create test config
                    test_config = self.config.__class__(
                        hardware=HardwareType.GPU,
                        block_size=block_size,
                        num_warps=num_warps,
                        precision=precision
                    )
                    
                    # Compile with test config using JAX's jit
                    compiled = jax.jit(
                        kernel_fn,
                        static_argnums=(1,),  # Assume second arg is static
                        backend='gpu'
                    )
                    
                    # Profile execution
                    profile = self.profiler.profile_kernel(
                        compiled,
                        sample_inputs,
                        num_warmup=tuning_config.num_warmup,
                        num_runs=tuning_config.num_trials
                    )
                    
                    # Check if best so far
                    if profile.exec_time < best_timing:
                        best_timing = profile.exec_time
                        best_result = TuningResult(
                            best_config={
                                "block_size": block_size,
                                "num_warps": num_warps,
                                "precision": precision
                            },
                            timing_ms=profile.exec_time * 1000,
                            throughput=1.0 / profile.exec_time,
                            memory_usage=profile.memory_usage,
                            compute_utilization=profile.compute_intensity
                        )
                        
        return best_result
        
    def _tune_cpu_kernel(self,
                        kernel_fn: Callable,
                        sample_inputs: Dict[str, Any],
                        tuning_config: TuningConfig) -> TuningResult:
        """Tune kernel for CPU execution."""
        best_result = None
        best_timing = float('inf')
        
        # Try different configurations
        for block_size in tuning_config.block_sizes:
            for batch_size in tuning_config.batch_sizes:
                # Create test config
                test_config = self.config.__class__(
                    hardware=HardwareType.CPU,
                    block_size=block_size,
                    batch_size=batch_size
                )
                
                # Compile with JAX jit for CPU
                compiled = jax.jit(
                    kernel_fn,
                    static_argnums=(1,),
                    backend='cpu'
                )
                
                # Profile execution
                profile = self.profiler.profile_kernel(
                    compiled,
                    sample_inputs,
                    num_warmup=tuning_config.num_warmup,
                    num_runs=tuning_config.num_trials
                )
                
                # Check if best so far
                if profile.exec_time < best_timing:
                    best_timing = profile.exec_time
                    best_result = TuningResult(
                        best_config={
                            "block_size": block_size,
                            "batch_size": batch_size
                        },
                        timing_ms=profile.exec_time * 1000,
                        throughput=1.0 / profile.exec_time,
                        memory_usage=profile.memory_usage,
                        compute_utilization=profile.compute_intensity
                    )
                    
        return best_result

class AutotuneManager:
    """Manage kernel autotuning process."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.tuner = KernelTuner(config)
        self.tuning_history: Dict[str, List[TuningResult]] = {}
        
    def autotune_kernel(self,
                       kernel_fn: Callable,
                       sample_inputs: Dict[str, Any],
                       search_space: Optional[Dict[str, List[Any]]] = None) -> KernelConfig:
        """Automatically tune kernel parameters."""
        if search_space is None:
            # Use default search space
            if self.config.hardware == HardwareType.TPU:
                search_space = {
                    "block_sizes": [128, 256, 512],
                    "batch_sizes": [1, 8, 16, 32],
                    "precision_modes": ["bf16", "fp32"]
                }
            elif self.config.hardware == HardwareType.GPU:
                search_space = {
                    "block_sizes": [32, 64, 128],
                    "num_warps": [4, 8, 16, 32],
                    "precision_modes": ["fp16", "fp32"]
                }
            else:
                search_space = {
                    "block_sizes": [16, 32, 64],
                    "batch_sizes": [1, 4, 8, 16]
                }
                
        # Create tuning config
        tuning_config = TuningConfig(
            block_sizes=search_space["block_sizes"],
            batch_sizes=search_space.get("batch_sizes", [1]),
            num_warps=search_space.get("num_warps", [8]),
            precision_modes=search_space.get("precision_modes", ["fp32"])
        )
        
        # Run tuning
        result = self.tuner.tune_kernel(
            kernel_fn,
            sample_inputs,
            tuning_config
        )
        
        # Save history
        if kernel_fn.__name__ not in self.tuning_history:
            self.tuning_history[kernel_fn.__name__] = []
        self.tuning_history[kernel_fn.__name__].append(result)
        
        # Create optimized config
        return self.config.__class__(
            hardware=self.config.hardware,
            **result.best_config
        )
        
    def get_tuning_history(self,
                          kernel_name: str) -> Optional[List[TuningResult]]:
        """Get tuning history for kernel."""
        return self.tuning_history.get(kernel_name)
        
    def plot_tuning_results(self,
                           kernel_name: str) -> Optional[Dict[str, Any]]:
        """Plot tuning results for kernel."""
        history = self.get_tuning_history(kernel_name)
        if not history:
            return None
            
        # Extract metrics
        configs = [h.best_config for h in history]
        timings = [h.timing_ms for h in history]
        throughputs = [h.throughput for h in history]
        memory_usage = [h.memory_usage for h in history]
        
        return {
            "configs": configs,
            "timings_ms": timings,
            "throughputs": throughputs,
            "memory_usage": memory_usage
        }
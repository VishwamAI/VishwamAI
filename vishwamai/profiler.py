"""TPU performance profiling and monitoring utilities"""

import jax
import jax.numpy as jnp
from contextlib import contextmanager, Tuple
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np
from collections import defaultdict
import os
import json

class TPUProfiler:
    """TPU performance profiling and monitoring."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: Optional[str] = None
    ):
        self.config = config
        self.log_dir = log_dir or "tpu_profiles"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.current_step = 0
        self.start_time = time.time()
        
        # Initialize XLA profiler if available
        self.xla_profiler = None
        try:
            from jax.profiler import TraceAnnotation, trace
            self.xla_profiler = trace
        except:
            pass
    
    @contextmanager
    def profile_region(self, name: str):
        """Profile a specific code region."""
        start = time.time()
        if self.xla_profiler:
            with self.xla_profiler(name):
                yield
        else:
            yield
        duration = time.time() - start
        self.metrics[f"region_{name}_time"].append(duration)
    
    def start_step(self):
        """Start profiling a training step."""
        self.step_start = time.time()
        self._reset_step_metrics()
    
    def end_step(self):
        """End profiling a training step."""
        duration = time.time() - self.step_start
        self.metrics["step_time"].append(duration)
        self._save_step_metrics()
        self.current_step += 1
    
    def _reset_step_metrics(self):
        """Reset per-step metrics."""
        self.step_metrics = {
            "compute_time": 0.0,
            "communication_time": 0.0,
            "memory_used": 0.0,
            "tpu_utilization": 0.0
        }
    
    def _save_step_metrics(self):
        """Save metrics for current step."""
        for k, v in self.step_metrics.items():
            self.metrics[k].append(v)
        
        # Save to disk periodically
        if self.current_step % 100 == 0:
            self.save_metrics()
    
    def record_flops(
        self,
        computation: Any
    ):
        """Record FLOPs for a JAX computation."""
        try:
            flops = jax.jit(computation).lower().compile().cost_analysis()["flops"]
            self.metrics["flops"].append(flops)
        except:
            pass
    
    def record_memory(
        self,
        computation: Any
    ):
        """Record memory usage for a JAX computation."""
        try:
            mem_usage = jax.jit(computation).lower().compile().cost_analysis()["bytes_accessed"]
            self.metrics["memory_accessed"].append(mem_usage)
        except:
            pass
    
    def record_communication(
        self,
        bytes_sent: int
    ):
        """Record communication volume."""
        self.metrics["communication_volume"].append(bytes_sent)
    
    def measure_tpu_utilization(self):
        """Measure TPU core utilization."""
        devices = jax.devices()
        total_util = 0.0
        
        for device in devices:
            try:
                # This is a placeholder - actual TPU utilization measurement
                # would require platform-specific APIs
                util = 0.8  # Example utilization
                total_util += util
            except:
                pass
        
        avg_util = total_util / len(devices)
        self.step_metrics["tpu_utilization"] = avg_util
    
    def record_batch_time(
        self,
        batch_size: int,
        duration: float
    ):
        """Record processing time for a batch."""
        self.metrics["batch_size"].append(batch_size)
        self.metrics["batch_time"].append(duration)
        
        # Calculate throughput
        examples_per_second = batch_size / duration
        self.metrics["throughput"].append(examples_per_second)
    
    def optimize_batch_size(
        self,
        initial_batch_size: int,
        step_size: int = 8,
        target_duration: float = 0.1,
        max_trials: int = 5
    ) -> int:
        """Find optimal batch size based on profiling."""
        current_batch_size = initial_batch_size
        best_batch_size = initial_batch_size
        best_throughput = 0
        
        for _ in range(max_trials):
            # Measure throughput with current batch size
            start = time.time()
            # Simulate batch processing
            time.sleep(0.01)  # Placeholder for actual computation
            duration = time.time() - start
            
            throughput = current_batch_size / duration
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = current_batch_size
            
            # Adjust batch size
            if duration > target_duration:
                current_batch_size -= step_size
            else:
                current_batch_size += step_size
        
        return best_batch_size
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of collected metrics."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
        
        # Calculate overall statistics
        total_time = time.time() - self.start_time
        total_steps = self.current_step
        
        summary.update({
            "total_time": total_time,
            "total_steps": total_steps,
            "steps_per_second": total_steps / total_time if total_time > 0 else 0
        })
        
        return summary
    
    def save_metrics(self):
        """Save metrics to disk."""
        metrics_file = os.path.join(
            self.log_dir,
            f"tpu_metrics_step_{self.current_step}.json"
        )
        
        with open(metrics_file, "w") as f:
            json.dump(self.get_metrics_summary(), f, indent=2)
    
    def get_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        metrics = self.get_metrics_summary()
        
        # Check TPU utilization
        avg_util = metrics.get("tpu_utilization_mean", 0)
        if avg_util < 0.7:
            recommendations.append(
                "Low TPU utilization detected. Consider:"
                "\n- Increasing batch size"
                "\n- Reducing host-device communication"
                "\n- Using larger model parallel size"
            )
        
        # Check step time variance
        step_time_std = metrics.get("step_time_std", 0)
        step_time_mean = metrics.get("step_time_mean", 1)
        if step_time_std / step_time_mean > 0.2:
            recommendations.append(
                "High step time variance detected. Consider:"
                "\n- Checking for load imbalance"
                "\n- Reducing variable-length sequences"
                "\n- Using gradient accumulation"
            )
        
        # Check communication overhead
        comm_time = metrics.get("communication_time_mean", 0)
        compute_time = metrics.get("compute_time_mean", 1)
        if comm_time / compute_time > 0.3:
            recommendations.append(
                "High communication overhead detected. Consider:"
                "\n- Using gradient accumulation"
                "\n- Increasing model parallel size"
                "\n- Optimizing device mesh layout"
            )
        
        return recommendations
    
    def profile_memory_usage(
        self,
        computation: Any,
        input_shapes: Dict[str, Tuple[int, ...]],
        log_details: bool = True
    ) -> Dict[str, Any]:
        """Profile memory usage of a computation."""
        try:
            # Compile computation
            compiled = jax.jit(computation).lower().compile()
            
            # Get memory analysis
            memory_analysis = compiled.memory_analysis()
            
            results = {
                "peak_memory": memory_analysis.get("peak_bytes", 0),
                "persistent_memory": memory_analysis.get("persistent_bytes", 0),
                "input_output_memory": memory_analysis.get("input_output_bytes", 0)
            }
            
            if log_details:
                print("\nMemory Profile:")
                print(f"Peak Memory: {results['peak_memory'] / 1e9:.2f} GB")
                print(f"Persistent Memory: {results['persistent_memory'] / 1e9:.2f} GB")
                print(f"Input/Output Memory: {results['input_output_memory'] / 1e9:.2f} GB")
            
            return results
            
        except Exception as e:
            print(f"Memory profiling failed: {e}")
            return {}

    def add_detailed_metrics(self):
        """Add detailed metrics for TPU performance."""
        self.metrics["compute_efficiency"] = []
        self.metrics["memory_bandwidth"] = []
        self.metrics["latency"] = []
        self.metrics["energy_consumption"] = []

    def record_detailed_metrics(
        self,
        computation: Any
    ):
        """Record detailed metrics for a JAX computation."""
        try:
            analysis = jax.jit(computation).lower().compile().cost_analysis()
            self.metrics["compute_efficiency"].append(analysis.get("compute_efficiency", 0))
            self.metrics["memory_bandwidth"].append(analysis.get("memory_bandwidth", 0))
            self.metrics["latency"].append(analysis.get("latency", 0))
            self.metrics["energy_consumption"].append(analysis.get("energy_consumption", 0))
        except:
            pass

    def get_detailed_metrics_summary(self) -> Dict[str, float]:
        """Get summary of detailed metrics."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
        
        return summary

    def get_actionable_recommendations(self) -> List[str]:
        """Generate actionable recommendations for optimizing TPU performance."""
        recommendations = []
        metrics = self.get_detailed_metrics_summary()
        
        # Check compute efficiency
        compute_efficiency = metrics.get("compute_efficiency_mean", 0)
        if compute_efficiency < 0.7:
            recommendations.append(
                "Low compute efficiency detected. Consider:"
                "\n- Optimizing kernel implementations"
                "\n- Reducing data transfer overhead"
                "\n- Using mixed precision training"
            )
        
        # Check memory bandwidth
        memory_bandwidth = metrics.get("memory_bandwidth_mean", 0)
        if memory_bandwidth < 0.5:
            recommendations.append(
                "Low memory bandwidth utilization detected. Consider:"
                "\n- Optimizing memory access patterns"
                "\n- Using larger batch sizes"
                "\n- Reducing memory fragmentation"
            )
        
        # Check latency
        latency = metrics.get("latency_mean", 0)
        if latency > 1.0:
            recommendations.append(
                "High latency detected. Consider:"
                "\n- Reducing synchronization points"
                "\n- Using asynchronous operations"
                "\n- Optimizing data pipeline"
            )
        
        # Check energy consumption
        energy_consumption = metrics.get("energy_consumption_mean", 0)
        if energy_consumption > 100:
            recommendations.append(
                "High energy consumption detected. Consider:"
                "\n- Using energy-efficient kernels"
                "\n- Reducing unnecessary computations"
                "\n- Optimizing power management settings"
            )
        
        return recommendations

def create_profiler(
    config: Dict[str, Any],
    log_dir: Optional[str] = None
) -> TPUProfiler:
    """Create TPU profiler instance."""
    profiler = TPUProfiler(config=config, log_dir=log_dir)
    profiler.add_detailed_metrics()
    return profiler

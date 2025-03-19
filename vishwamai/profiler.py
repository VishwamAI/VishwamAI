"""TPU performance profiling and monitoring utilities"""

import jax
import jax.numpy as jnp
from contextlib import contextmanager
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
        self.last_trace_time = time.time()
        
        # Initialize XLA profiler if available
        self.xla_profiler = None
        try:
            from jax.profiler import TraceAnnotation, trace
            self.xla_profiler = trace
        except:
            pass

        # Set up memory tracking
        self.peak_memory = 0
        self.memory_history = []
        self.dtype = getattr(config, 'dtype', jnp.float32)
        
    @contextmanager
    def profile_region(self, name: str):
        """Profile a specific code region."""
        start = time.time()
        trace_key = f"{name}_{self.current_step}"
        
        if self.xla_profiler:
            with self.xla_profiler(trace_key):
                yield
        else:
            yield
            
        duration = time.time() - start
        self.metrics[f"region_{name}_time"].append(duration)
        
        # Track detailed metrics for this region
        self._track_region_metrics(name, duration)
    
    def _track_region_metrics(self, name: str, duration: float):
        """Track detailed metrics for a profiled region."""
        # Track computation intensity
        if hasattr(self, 'last_flops'):
            flops_rate = self.last_flops / duration if duration > 0 else 0
            self.metrics[f"{name}_flops_per_second"].append(flops_rate)
        
        # Track memory efficiency
        if hasattr(self, 'last_memory_accessed'):
            memory_bw = self.last_memory_accessed / duration if duration > 0 else 0
            self.metrics[f"{name}_memory_bandwidth"].append(memory_bw)
            
        # Track TPU utilization for this region
        self.measure_tpu_utilization()
        
    def measure_tpu_utilization(self):
        """Measure TPU core utilization with enhanced metrics."""
        devices = jax.devices()
        total_util = 0.0
        detailed_utils = []
        
        for device in devices:
            try:
                # This is a placeholder - actual TPU utilization measurement
                # would require platform-specific APIs
                compute_util = 0.85  # Example compute utilization
                memory_util = 0.75  # Example memory utilization
                
                detailed_utils.append({
                    'device_id': device.id,
                    'compute_util': compute_util,
                    'memory_util': memory_util
                })
                
                total_util += compute_util
            except:
                pass
        
        avg_util = total_util / len(devices)
        self.step_metrics["tpu_utilization"] = avg_util
        self.metrics["detailed_utilization"].append(detailed_utils)
        
    def record_flops(self, computation: Any):
        """Record FLOPs for a JAX computation."""
        try:
            flops = jax.jit(computation).lower().compile().cost_analysis()["flops"]
            self.metrics["flops"].append(flops)
            self.last_flops = flops
        except:
            pass
            
    def record_memory(self, computation: Any):
        """Record memory usage for a JAX computation."""
        try:
            mem_usage = jax.jit(computation).lower().compile().cost_analysis()["bytes_accessed"]
            self.metrics["memory_accessed"].append(mem_usage)
            self.last_memory_accessed = mem_usage
            
            # Track peak memory
            if mem_usage > self.peak_memory:
                self.peak_memory = mem_usage
        except:
            pass
            
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of collected metrics with enhanced statistics."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
                
                # Add percentile statistics
                summary[f"{metric}_p50"] = float(np.percentile(values, 50))
                summary[f"{metric}_p95"] = float(np.percentile(values, 95))
                summary[f"{metric}_p99"] = float(np.percentile(values, 99))
        
        # Calculate overall statistics
        total_time = time.time() - self.start_time
        total_steps = self.current_step
        
        summary.update({
            "total_time": total_time,
            "total_steps": total_steps,
            "steps_per_second": total_steps / total_time if total_time > 0 else 0,
            "peak_memory_gb": self.peak_memory / 1e9,
            "average_tpu_utilization": np.mean(self.metrics["tpu_utilization"]) if self.metrics["tpu_utilization"] else 0
        })
        
        return summary
        
    def get_performance_recommendations(self) -> List[str]:
        """Generate detailed performance optimization recommendations."""
        recommendations = []
        metrics = self.get_metrics_summary()
        
        # Check TPU utilization
        avg_util = metrics.get("tpu_utilization_mean", 0)
        if avg_util < 0.7:
            recommendations.append(
                "Low TPU utilization detected. Consider:\n"
                "- Increasing batch size\n"
                "- Using gradient accumulation\n"
                "- Enabling model parallelism\n"
                "- Reducing host-device transfers"
            )
        
        # Check memory efficiency
        mem_util = metrics.get("memory_efficiency_mean", 0)
        if mem_util < 0.6:
            recommendations.append(
                "Memory efficiency can be improved:\n"
                "- Enable gradient checkpointing\n"
                "- Use mixed precision training\n"
                "- Optimize attention implementation\n"
                "- Consider using FlashAttention"
            )
        
        # Check computation efficiency
        comp_util = metrics.get("compute_efficiency_mean", 0)
        if comp_util < 0.8:
            recommendations.append(
                "Computation efficiency can be improved:\n"
                "- Use optimal tensor layouts\n"
                "- Enable JIT compilation\n"
                "- Optimize data preprocessing\n"
                "- Consider using TPU-specific kernels"
            )
        
        # Check step time variance
        step_time_std = metrics.get("step_time_std", 0)
        step_time_mean = metrics.get("step_time_mean", 1)
        if step_time_std / step_time_mean > 0.2:
            recommendations.append(
                "High step time variance detected:\n"
                "- Check for load imbalance\n"
                "- Optimize data pipeline\n"
                "- Consider using static shapes\n"
                "- Profile host-device transfers"
            )
            
        return recommendations

    def save_metrics(self):
        """Save metrics to disk with enhanced data."""
        metrics_file = os.path.join(
            self.log_dir,
            f"tpu_metrics_step_{self.current_step}.json"
        )
        
        summary = self.get_metrics_summary()
        
        # Add additional metadata
        summary.update({
            "timestamp": time.time(),
            "total_training_time": time.time() - self.start_time,
            "device_count": jax.device_count(),
            "peak_memory_gb": self.peak_memory / 1e9,
            "recommendations": self.get_performance_recommendations()
        })
        
        with open(metrics_file, "w") as f:
            json.dump(summary, f, indent=2)
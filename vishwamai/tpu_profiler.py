"""
TPU Memory Profiling Utility for VishwamAI

This module provides tools to profile and analyze memory usage on TPUs,
helping optimize training with limited resources.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from functools import wraps, partial
import threading

import numpy as np
import jax
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from google.cloud import profiler
    HAS_GCP_PROFILER = True
except ImportError:
    HAS_GCP_PROFILER = False

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot with detailed breakdown."""
    timestamp: float
    total_gb: float
    used_gb: float
    peak_gb: float
    per_device_gb: Optional[List[float]] = None
    compilation_cache_gb: Optional[float] = None
    operation_name: str = ""
    parameters_gb: Optional[float] = None
    gradients_gb: Optional[float] = None
    optimizer_states_gb: Optional[float] = None

class TPUMemoryTracker:
    """Tracks TPU memory usage during training."""
    
    def __init__(self, 
                 log_interval_sec: int = 5, 
                 enable_detailed_profiling: bool = False,
                 output_dir: str = "./memory_profile"):
        """
        Initialize memory tracker.
        
        Args:
            log_interval_sec: How often to log memory usage (seconds)
            enable_detailed_profiling: Whether to enable JAX's detailed profiling
            output_dir: Where to save memory profiles
        """
        self.log_interval_sec = log_interval_sec
        self.enable_detailed_profiling = enable_detailed_profiling
        self.output_dir = output_dir
        self.snapshots: List[MemorySnapshot] = []
        self._thread = None
        self._stop_event = threading.Event()
        self._peak_memory = 0
        self._start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to get initial TPU info
        self.tpu_devices = jax.devices("tpu")
        self.device_count = len(self.tpu_devices) if self.tpu_devices else 0
        
        if self.device_count == 0:
            logger.warning("No TPU devices found, memory tracking will use CPU memory")
    
    def _get_memory_usage(self) -> MemorySnapshot:
        """Get current memory usage across TPU devices."""
        timestamp = time.time() - self._start_time
        used_gb = 0
        per_device_gb = []
        compilation_cache_gb = None
        
        # Try to get TPU-specific memory metrics
        try:
            # Try to access TPU memory info from JAX
            from jax.experimental.compilation_cache import compilation_cache as cc
            cc_info = cc.get_cache_info()
            if "tpu_memory_usage" in cc_info:
                # JAX reports memory in bytes, convert to GB
                used_bytes = cc_info["tpu_memory_usage"]
                used_gb = used_bytes / (1024**3)
                compilation_cache_gb = cc_info.get("compilation_cache_size", 0) / (1024**3)
        except:
            pass
            
        # If we couldn't get TPU memory, fall back to process memory
        if used_gb == 0 and HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            used_gb = process.memory_info().rss / (1024**3)
        
        # Try to get per-device memory if available
        try:
            if self.device_count > 0:
                # This is experimental and may not work on all TPU configurations
                per_device_gb = [
                    dev.memory_stats().get("bytes_in_use", 0) / (1024**3)
                    for dev in self.tpu_devices
                ]
        except:
            pass
        
        # Update peak memory
        self._peak_memory = max(self._peak_memory, used_gb)
        
        return MemorySnapshot(
            timestamp=timestamp,
            total_gb=self._get_total_memory(),
            used_gb=used_gb,
            peak_gb=self._peak_memory,
            per_device_gb=per_device_gb if per_device_gb else None,
            compilation_cache_gb=compilation_cache_gb,
            operation_name=""
        )
    
    def _get_total_memory(self) -> float:
        """Get total available TPU memory."""
        # Try to get TPU memory capacity
        try:
            if self.device_count > 0:
                # This is experimental and may not work on all TPU configurations
                return sum(
                    dev.memory_stats().get("bytes_limit", 0) / (1024**3)
                    for dev in self.tpu_devices
                )
        except:
            pass
            
        # Fall back to system memory if TPU memory not available
        if HAS_PSUTIL:
            return psutil.virtual_memory().total / (1024**3)
        
        return 0  # Unknown
    
    def _monitoring_thread(self):
        """Background thread to periodically log memory usage."""
        while not self._stop_event.is_set():
            try:
                snapshot = self._get_memory_usage()
                self.snapshots.append(snapshot)
                
                # Log memory usage
                logger.info(
                    f"Memory: {snapshot.used_gb:.2f}GB used, "
                    f"{snapshot.peak_gb:.2f}GB peak, "
                    f"{(snapshot.used_gb/max(1, snapshot.total_gb))*100:.1f}% utilization"
                )
                
                if snapshot.per_device_gb:
                    device_usage = ", ".join(
                        f"TPU{i}: {gb:.2f}GB" for i, gb in enumerate(snapshot.per_device_gb)
                    )
                    logger.info(f"Per-device memory: {device_usage}")
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring thread: {e}")
                
            time.sleep(self.log_interval_sec)
    
    def start(self):
        """Start memory tracking."""
        if self._thread is not None:
            logger.warning("Memory tracking is already running")
            return
            
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_thread, daemon=True)
        self._thread.start()
        
        # Enable profiler if requested
        if self.enable_detailed_profiling and HAS_GCP_PROFILER:
            try:
                profiler.start(
                    service_name="vishwamai-training",
                    host_name="",
                    logdir=os.path.join(self.output_dir, "profiler")
                )
                logger.info("GCP profiler started")
            except Exception as e:
                logger.error(f"Failed to start GCP profiler: {e}")
                
        logger.info("Memory tracking started")
    
    def stop(self):
        """Stop memory tracking."""
        if self._thread is None:
            logger.warning("Memory tracking is not running")
            return
            
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        
        # Disable profiler if it was enabled
        if self.enable_detailed_profiling and HAS_GCP_PROFILER:
            try:
                profiler.stop()
                logger.info("GCP profiler stopped")
            except Exception as e:
                logger.error(f"Failed to stop GCP profiler: {e}")
                
        logger.info(
            f"Memory tracking stopped. Peak usage: {self._peak_memory:.2f}GB, "
            f"Snapshots collected: {len(self.snapshots)}"
        )
    
    def record_operation(self, name: str):
        """Record named operation for memory tracking."""
        snapshot = self._get_memory_usage()
        snapshot.operation_name = name
        self.snapshots.append(snapshot)
        logger.info(f"Operation '{name}': {snapshot.used_gb:.2f}GB used")
        return snapshot
    
    def save_report(self, filename_prefix: str = "memory_profile"):
        """Save memory profiling report."""
        if not self.snapshots:
            logger.warning("No memory snapshots to save")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw data as JSON
        json_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "snapshots": [
                        {
                            "timestamp": s.timestamp,
                            "total_gb": s.total_gb,
                            "used_gb": s.used_gb,
                            "peak_gb": s.peak_gb,
                            "operation_name": s.operation_name,
                            "per_device_gb": s.per_device_gb,
                            "compilation_cache_gb": s.compilation_cache_gb
                        }
                        for s in self.snapshots
                    ],
                    "device_count": self.device_count,
                    "peak_memory_gb": self._peak_memory,
                    "total_duration": self.snapshots[-1].timestamp if self.snapshots else 0
                },
                f,
                indent=2
            )
            
        # Generate plots
        self._generate_plots(filename_prefix, timestamp)
        
        logger.info(f"Memory profile saved to {self.output_dir}/{filename_prefix}_{timestamp}.*")
        return json_path
    
    def _generate_plots(self, filename_prefix: str, timestamp: str):
        """Generate memory usage plots."""
        try:
            # Extract data
            timestamps = [s.timestamp for s in self.snapshots]
            memory_usage = [s.used_gb for s in self.snapshots]
            peak_memory = [s.peak_gb for s in self.snapshots]
            operations = [(s.timestamp, s.operation_name) for s in self.snapshots if s.operation_name]
            
            # Plot memory over time
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, memory_usage, label="Memory Usage (GB)", linewidth=2)
            plt.plot(timestamps, peak_memory, label="Peak Memory (GB)", linestyle="--")
            
            # Add vertical lines for operations
            for op_time, op_name in operations:
                plt.axvline(x=op_time, color="gray", linestyle=":", alpha=0.7)
                plt.text(op_time, max(memory_usage) * 0.9, op_name, 
                        rotation=90, verticalalignment='top')
            
            plt.title("TPU Memory Usage Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Memory (GB)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format y-axis to show GB
            def gb_formatter(x, pos):
                return f"{x:.1f} GB"
            plt.gca().yaxis.set_major_formatter(FuncFormatter(gb_formatter))
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            # Generate per-device plot if data is available
            if any(s.per_device_gb for s in self.snapshots):
                plt.figure(figsize=(12, 6))
                
                # Extract per-device data - handle cases where some snapshots might not have it
                device_data = {}
                for s in self.snapshots:
                    if s.per_device_gb:
                        for i, mem in enumerate(s.per_device_gb):
                            if i not in device_data:
                                device_data[i] = []
                            # Pad with None for missing data points
                            while len(device_data[i]) < len(device_data[0]):
                                device_data[i].append(None)
                            device_data[i].append(mem)
                
                # Plot each device
                for device_id, mem_data in device_data.items():
                    plt.plot(timestamps[:len(mem_data)], mem_data, 
                             label=f"TPU Device {device_id}", marker="o", markersize=3)
                
                plt.title("Per-Device TPU Memory Usage")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Memory (GB)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save the per-device plot
                dev_plot_path = os.path.join(self.output_dir, f"{filename_prefix}_devices_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(dev_plot_path, dpi=100)
                plt.close()
                
        except Exception as e:
            logger.error(f"Failed to generate memory usage plots: {e}")

def profile_func(label: str = None):
    """
    Decorator to profile memory usage of a function.
    
    Example:
        @profile_func("Forward pass")
        def forward_pass(x):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the current memory tracker or create a temporary one
            tracker = getattr(wrapper, "_tracker", None)
            temp_tracker = tracker is None
            if temp_tracker:
                tracker = TPUMemoryTracker()
                
            # Record start memory
            func_name = label or func.__name__
            start_label = f"{func_name} (start)"
            start_snapshot = tracker.record_operation(start_label)
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            finally:
                # Record end memory
                end_label = f"{func_name} (end)"
                end_snapshot = tracker.record_operation(end_label)
                
                # Log memory delta
                memory_delta = end_snapshot.used_gb - start_snapshot.used_gb
                logger.info(
                    f"Memory delta for {func_name}: {memory_delta:.2f}GB "
                    f"({'-' if memory_delta < 0 else '+'}{abs(memory_delta):.2f}GB)"
                )
            
            return result
        
        # Allow attaching a tracker to the function
        def attach_tracker(tracker):
            wrapper._tracker = tracker
            return wrapper
        
        wrapper.attach_tracker = attach_tracker
        return wrapper
    
    # Handle case where decorator is used without arguments
    if callable(label):
        func = label
        label = func.__name__
        return decorator(func)
    
    return decorator

def profile_model_memory(
    model_apply_fn: Callable,
    input_shapes: Dict[str, Tuple],
    param_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Profile memory usage of a model.
    
    Args:
        model_apply_fn: Model's apply function
        input_shapes: Dictionary of input shapes
        param_count: Number of parameters if known
        
    Returns:
        Dictionary of memory usage statistics
    """
    # Create dummy inputs
    inputs = {
        name: jnp.ones(shape, dtype=jnp.float32)
        for name, shape in input_shapes.items()
    }
    
    # Create memory tracker
    tracker = TPUMemoryTracker(log_interval_sec=1)
    tracker.start()
    
    try:
        # Initial memory usage
        init_mem = tracker.record_operation("Initial")
        
        # Forward pass
        tracker.record_operation("Forward pass start")
        _ = model_apply_fn(inputs)
        forward_mem = tracker.record_operation("Forward pass end")
        
        # Try a backward pass if supported
        try:
            grad_fn = jax.grad(lambda params: model_apply_fn(inputs, params).sum())
            tracker.record_operation("Backward pass start")
            _ = grad_fn({})
            backward_mem = tracker.record_operation("Backward pass end")
            backward_delta = backward_mem.used_gb - forward_mem.used_gb
        except:
            backward_delta = 0
        
        # Calculate memory per parameter
        if param_count:
            bytes_per_param = (forward_mem.used_gb * 1024**3) / param_count
            params_gb = param_count * 4 / (1024**3)  # Assume 4 bytes per parameter
        else:
            bytes_per_param = 0
            params_gb = 0
        
        # Memory stats
        forward_delta = forward_mem.used_gb - init_mem.used_gb
        
        return {
            "total_parameters": param_count,
            "parameters_memory_gb": params_gb,
            "forward_pass_memory_gb": forward_delta,
            "backward_pass_memory_gb": backward_delta,
            "total_memory_gb": forward_mem.used_gb,
            "bytes_per_parameter": bytes_per_param,
            "peak_memory_gb": tracker._peak_memory,
        }
    
    finally:
        tracker.stop()
        tracker.save_report("model_profile")

def optimize_batch_size(
    train_step_fn: Callable,
    sample_batch_fn: Callable,
    start_batch_size: int = 16,
    max_batch_size: int = 128,
    memory_threshold: float = 0.85,
    patience: int = 3
) -> int:
    """
    Find optimal batch size for TPU training.
    
    Args:
        train_step_fn: Function that takes a batch and performs a training step
        sample_batch_fn: Function that takes a batch size and returns a sample batch
        start_batch_size: Starting batch size to try
        max_batch_size: Maximum batch size to try
        memory_threshold: Memory utilization threshold (0.0-1.0)
        patience: How many successful steps to run at each batch size
    
    Returns:
        Optimal batch size
    """
    logger.info(f"Finding optimal batch size (start={start_batch_size}, max={max_batch_size})")
    
    current_batch_size = start_batch_size
    optimal_batch_size = current_batch_size
    
    tracker = TPUMemoryTracker(log_interval_sec=1)
    total_memory = tracker._get_total_memory()
    
    while current_batch_size <= max_batch_size:
        logger.info(f"Testing batch size: {current_batch_size}")
        try:
            # Run multiple steps to account for compilation overhead
            for i in range(patience):
                batch = sample_batch_fn(current_batch_size)
                tracker.start()
                train_step_fn(batch)
                tracker.stop()
                
                # Check memory usage
                if tracker._peak_memory / total_memory > memory_threshold:
                    raise RuntimeError(f"Memory threshold exceeded: {tracker._peak_memory:.2f}GB/{total_memory:.2f}GB")
            
            # This batch size worked
            logger.info(f"Batch size {current_batch_size} successful")
            optimal_batch_size = current_batch_size
            
            # Try a larger batch size
            current_batch_size *= 2
            
        except Exception as e:
            logger.info(f"Batch size {current_batch_size} failed: {e}")
            break
    
    # If we exceeded max_batch_size, use the last successful one
    optimal_batch_size = min(optimal_batch_size, max_batch_size)
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def analyze_model_scaling(
    model_fn: Callable,
    base_config: Dict[str, Any],
    scaling_factors: List[float],
    input_shape: Tuple[int, ...],
    max_memory_gb: Optional[float] = None
) -> Dict[str, List]:
    """
    Analyze how model memory scales with size.
    
    Args:
        model_fn: Function that takes a config and returns a model
        base_config: Base configuration dictionary
        scaling_factors: List of scaling factors to try
        input_shape: Shape of input tensor
        max_memory_gb: Maximum memory in GB (stops scaling when exceeded)
    
    Returns:
        Dictionary of scaling metrics
    """
    results = {
        "scaling_factors": [],
        "parameter_count": [],
        "memory_usage_gb": [],
        "forward_memory_gb": [],
        "backward_memory_gb": []
    }
    
    for factor in scaling_factors:
        # Scale the configuration
        config = dict(base_config)
        config["hidden_size"] = int(config["hidden_size"] * factor)
        if "intermediate_size" in config:
            config["intermediate_size"] = int(config["intermediate_size"] * factor)
        
        logger.info(f"Testing scaling factor {factor:.2f} (hidden_size={config['hidden_size']})")
        
        try:
            # Create model
            model = model_fn(config)
            
            # Count parameters
            param_count = sum(p.size for p in jax.tree_leaves(model.params))
            
            # Profile memory
            profile = profile_model_memory(
                model.apply,
                {"input_ids": input_shape},
                param_count
            )
            
            # Record results
            results["scaling_factors"].append(factor)
            results["parameter_count"].append(param_count)
            results["memory_usage_gb"].append(profile["total_memory_gb"])
            results["forward_memory_gb"].append(profile["forward_pass_memory_gb"])
            results["backward_memory_gb"].append(profile["backward_pass_memory_gb"])
            
            logger.info(f"Factor {factor:.2f}: {param_count:,} params, {profile['total_memory_gb']:.2f}GB memory")
            
            # Stop if we exceed max memory
            if max_memory_gb and profile["total_memory_gb"] > max_memory_gb:
                logger.info(f"Exceeding maximum memory ({profile['total_memory_gb']:.2f}GB > {max_memory_gb:.2f}GB), stopping scaling analysis")
                break
                
        except Exception as e:
            logger.error(f"Error testing scaling factor {factor}: {e}")
            break
    
    # Generate scaling plot
    if results["scaling_factors"]:
        try:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(results["scaling_factors"], results["parameter_count"], marker='o')
            plt.title("Parameter Count vs. Scale")
            plt.xlabel("Scaling Factor")
            plt.ylabel("Parameters")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(results["scaling_factors"], results["memory_usage_gb"], marker='o', label="Total Memory")
            plt.plot(results["scaling_factors"], results["forward_memory_gb"], marker='s', label="Forward Pass")
            if any(m > 0 for m in results["backward_memory_gb"]):
                plt.plot(results["scaling_factors"], results["backward_memory_gb"], marker='^', label="Backward Pass")
            plt.title("Memory Usage vs. Scale")
            plt.xlabel("Scaling Factor")
            plt.ylabel("Memory (GB)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("model_scaling_analysis.png", dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating scaling plot: {e}")
    
    return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = TPUMemoryTracker(output_dir="./memory_profiles")
    tracker.start()
    
    # Simulate some operations
    time.sleep(1)
    tracker.record_operation("Model initialization")
    time.sleep(1)
    tracker.record_operation("Training step 1")
    time.sleep(1)
    tracker.record_operation("Training step 2")
    
    tracker.stop()
    tracker.save_report("example_profile")
# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/profiling_tools.py

import torch
import time
import logging
from typing import Dict, List, Tuple, Union, Optional
from torch.profiler import profile, record_function, ProfilerActivity
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, make_jaxpr
    from jax.experimental import host_callback
    import flax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VishwamAIProfiler:
    """A class to profile VishwamAI models for performance bottlenecks."""
    
    def __init__(self, model: Union[torch.nn.Module, "flax.linen.Module"], device: str = "auto"):
        """
        Initialize the VishwamAIProfiler with a model and target device.
        
        Args:
            model: The VishwamAI model (PyTorch or Flax).
            device: The device to run the model on ("auto", "tpu", "gpu", or "cpu").
        """
        self.model = model
        self.memory_optimizer = MemoryOptimizer(model, device)
        self.device_type = self.memory_optimizer.device_type
        self.device = self.memory_optimizer.device
        logger.info(f"Initialized VishwamAIProfiler with model on device: {self.device_type}")

    def profile_model(self, input_tensor: Union[torch.Tensor, "jnp.ndarray"], 
                     num_steps: int = 10) -> Dict[str, any]:
        """Profile the model with device-specific profiling tools."""
        if self.device_type == "tpu":
            return self._profile_tpu(input_tensor, num_steps)
        else:
            return self._profile_gpu(input_tensor, num_steps)

    def _profile_tpu(self, input_tensor: "jnp.ndarray", num_steps: int) -> Dict[str, any]:
        """TPU-specific profiling using JAX tools."""
        profiling_stats = {}
        
        # Compile and analyze computation graph
        def compile_fn(x):
            return self.model.apply({"params": self.model.params}, x)
        jaxpr = make_jaxpr(compile_fn)(input_tensor)
        profiling_stats["computation_graph"] = str(jaxpr)
        
        # Profile execution time and memory
        times = []
        memory_usage = []
        
        def profile_step(step):
            start_time = time.time()
            output = compile_fn(input_tensor)
            step_time = time.time() - start_time
            
            # Get TPU memory stats
            mem_stats = jax.device_get(jax.device_memory_profile(self.device))
            memory_usage.append({
                "used_bytes": mem_stats.used_bytes,
                "peak_bytes": mem_stats.peak_bytes
            })
            times.append(step_time)
            return output

        # Warmup
        profile_step(0)
        
        # Profile steps
        for step in range(num_steps):
            profile_step(step)
            jax.clear_caches()
        
        profiling_stats.update({
            "avg_step_time_ms": (sum(times) / len(times)) * 1000,
            "min_step_time_ms": min(times) * 1000,
            "max_step_time_ms": max(times) * 1000,
            "avg_memory_usage_mb": sum(m["used_bytes"] for m in memory_usage) / len(memory_usage) / 1024**2,
            "peak_memory_usage_mb": max(m["peak_bytes"] for m in memory_usage) / 1024**2
        })
        
        logger.info("TPU profiling completed. Summary:")
        logger.info(f"Average step time: {profiling_stats['avg_step_time_ms']:.2f} ms")
        logger.info(f"Peak memory usage: {profiling_stats['peak_memory_usage_mb']:.2f} MB")
        
        return profiling_stats

    def _profile_gpu(self, input_tensor: torch.Tensor, num_steps: int) -> Dict[str, any]:
        """GPU-specific profiling using PyTorch tools."""
        self.model.eval()
        activities = [ProfilerActivity.CPU]
        if self.device_type == "gpu":
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                for _ in range(num_steps):
                    with torch.no_grad():
                        _ = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
                    self.memory_optimizer.clear_device_memory()
        
        # Extract profiling stats
        profiling_stats = {
            "avg_cpu_time_ms": prof.key_averages().table(sort_by="cpu_time_total"),
            "memory_usage": self.memory_optimizer.get_memory_usage(),
            "events": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        }
        
        logger.info("GPU profiling completed. Summary:")
        logger.info(profiling_stats["avg_cpu_time_ms"])
        logger.info(f"Memory usage: {profiling_stats['memory_usage']}")
        return profiling_stats

    def layer_wise_profiling(self, input_tensor: Union[torch.Tensor, "jnp.ndarray"]) -> List[Tuple[str, float, float]]:
        """Profile the model layer-wise with device-specific implementations."""
        if self.device_type == "tpu":
            return self._layer_wise_profiling_tpu(input_tensor)
        else:
            return self._layer_wise_profiling_gpu(input_tensor)

    def _layer_wise_profiling_tpu(self, input_tensor: "jnp.ndarray") -> List[Tuple[str, float, float]]:
        """TPU-specific layer-wise profiling."""
        layer_stats = []
        
        def profile_layer(layer_name):
            def wrapped_layer(inputs, layer_fn):
                start_time = time.time()
                mem_before = jax.device_get(jax.device_memory_profile(self.device)).used_bytes
                
                outputs = layer_fn(inputs)
                
                mem_after = jax.device_get(jax.device_memory_profile(self.device)).used_bytes
                latency_ms = (time.time() - start_time) * 1000
                memory_mb = (mem_after - mem_before) / 1024**2
                
                host_callback.id_tap(lambda x, _: layer_stats.append((layer_name, x[0], x[1])), 
                                  (latency_ms, memory_mb))
                return outputs
            return wrapped_layer

        # Profile each layer
        for name, layer in self.model.named_modules():
            if isinstance(layer, (flax.linen.Dense, flax.linen.Conv)):
                wrapped = profile_layer(name)(input_tensor, layer)
                _ = jit(wrapped)()
                jax.clear_caches()

        # Sort by latency
        layer_stats.sort(key=lambda x: x[1], reverse=True)
        logger.info("Layer-wise profiling completed (TPU):")
        for layer_name, latency, memory in layer_stats[:5]:
            logger.info(f"Layer {layer_name}: Latency = {latency:.2f} ms, Memory = {memory:.2f} MB")
        
        return layer_stats

    def _layer_wise_profiling_gpu(self, input_tensor: torch.Tensor) -> List[Tuple[str, float, float]]:
        """GPU-specific layer-wise profiling."""
        self.model.eval()
        layer_stats = []
        
        def profile_layer(module, input, output, layer_name):
            start_time = time.time()
            memory_before = torch.cuda.memory_allocated(self.device) / 1024**2 if self.device_type == "gpu" else 0
            
            output = module(input[0] if isinstance(input, tuple) else input)
            
            memory_after = torch.cuda.memory_allocated(self.device) / 1024**2 if self.device_type == "gpu" else 0
            latency_ms = (time.time() - start_time) * 1000
            memory_mb = memory_after - memory_before
            layer_stats.append((layer_name, latency_ms, memory_mb))
            return output

        # Register hooks for each layer
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention)):
                handle = module.register_forward_hook(
                    lambda m, i, o, n=name: profile_layer(m, i, o, n)
                )
                handles.append(handle)
        
        # Run inference
        with torch.no_grad():
            _ = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Sort by latency
        layer_stats.sort(key=lambda x: x[1], reverse=True)
        logger.info("Layer-wise profiling completed (GPU):")
        for layer_name, latency, memory in layer_stats[:5]:
            logger.info(f"Layer {layer_name}: Latency = {latency:.2f} ms, Memory = {memory:.2f} MB")
        
        return layer_stats

if __name__ == "__main__":
    # Example usage
    from vishwamai.models.cot_model import CoTModel
    from vishwamai.models.transformer import VishwamAITransformer
    
    # Initialize model and profiler
    transformer = VishwamAITransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6)
    model = CoTModel(transformer)
    profiler = VishwamAIProfiler(model)
    
    # Profile model
    if profiler.device_type == "tpu":
        input_tensor = jnp.ones((16, 100, 512))  # batch_size, seq_len, d_model
    else:
        input_tensor = torch.randn(16, 100, 512).to(profiler.device)
        
    profiling_stats = profiler.profile_model(input_tensor)
    layer_stats = profiler.layer_wise_profiling(input_tensor)
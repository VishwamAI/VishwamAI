# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/performance_tuning.py

import torch
import time
import logging
from typing import Dict, Optional, Union, Tuple
from vishwamai.optimisation.memory_optimization import MemoryOptimizer
import flax
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, grad, value_and_grad
    import optax
    from flax.training import train_state
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTuner:
    """A class to tune the performance of VishwamAI models."""
    
    def __init__(self, model: Union[torch.nn.Module, "flax.linen.Module"], device: str = "auto"):
        """
        Initialize the PerformanceTuner with a model and target device.
        
        Args:
            model: The VishwamAI model (PyTorch or Flax).
            device: The device to run the model on ("auto", "tpu", "gpu", or "cpu").
        """
        self.model = model
        self.memory_optimizer = MemoryOptimizer(model, device)
        self.device_type = self.memory_optimizer.device_type
        self.device = self.memory_optimizer.device
        logger.info(f"Initialized PerformanceTuner with model on device: {self.device_type}")

    def enable_hardware_optimizations(self):
        """Enable hardware-specific optimizations."""
        if self.device_type == "tpu":
            # TPU-specific optimizations
            if HAS_JAX:
                jax.config.update('jax_default_matmul_precision', 'bfloat16')
                jax.config.update('jax_enable_x64', False)
                logger.info("Enabled TPU optimizations (bfloat16, disabled x64)")
        else:
            # GPU-specific optimizations
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            self.memory_optimizer.enable_mixed_precision()
            logger.info("Enabled GPU optimizations (cuDNN, AMP)")

    def tune_batch_size(self, input_shape: Tuple, max_batch_size: int = 128, tolerance: float = 0.1) -> int:
        """Find optimal batch size for the current device."""
        optimal_batch_size = 1
        best_throughput = 0.0

        if self.device_type == "tpu":
            dummy_input = jnp.ones((1, *input_shape))
            
            @jit
            def measure_step(batch):
                return self.model.apply(self.model.params, batch)

            for batch_size in range(1, max_batch_size + 1, 1):
                try:
                    batch = jnp.repeat(dummy_input, batch_size, axis=0)
                    start_time = time.time()
                    _ = measure_step(batch)
                    elapsed_time = time.time() - start_time
                    
                    throughput = batch_size / elapsed_time
                    logger.info(f"TPU batch size {batch_size}: Throughput = {throughput:.2f} samples/sec")
                    
                    if throughput > best_throughput * (1 - tolerance):
                        best_throughput = throughput
                        optimal_batch_size = batch_size
                    else:
                        break
                except Exception as e:
                    logger.warning(f"TPU batch size {batch_size} failed: {str(e)}")
                    break
        else:
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            
            for batch_size in range(1, max_batch_size + 1, 1):
                try:
                    batch = dummy_input.repeat(batch_size, 1, 1)
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.memory_optimizer.forward_with_mixed_precision(self.model, batch)
                    elapsed_time = time.time() - start_time
                    
                    throughput = batch_size / elapsed_time
                    logger.info(f"GPU batch size {batch_size}: Throughput = {throughput:.2f} samples/sec")
                    
                    if throughput > best_throughput * (1 - tolerance):
                        best_throughput = throughput
                        optimal_batch_size = batch_size
                    else:
                        break
                except RuntimeError as e:
                    logger.warning(f"GPU batch size {batch_size} failed: {str(e)}")
                    break

        logger.info(f"Optimal batch size: {optimal_batch_size} with throughput {best_throughput:.2f} samples/sec")
        return optimal_batch_size

    def optimize_inference(self, input_tensor: Union[torch.Tensor, "jnp.ndarray"], use_jit: bool = True) -> Union[torch.Tensor, "jnp.ndarray"]:
        """Optimize inference with device-specific techniques."""
        self.enable_hardware_optimizations()
        
        if self.device_type == "tpu":
            if use_jit:
                @jit
                def optimized_forward(params, x):
                    return self.model.apply(params, x)
                return optimized_forward(self.model.params, input_tensor)
            return self.model.apply(self.model.params, input_tensor)
        else:
            if use_jit:
                try:
                    self.model = torch.jit.trace(self.model, input_tensor)
                    logger.info("Applied TorchScript JIT compilation")
                except Exception as e:
                    logger.warning(f"Failed to apply JIT: {str(e)}")
            
            with torch.no_grad():
                return self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)

    def measure_latency(self, input_tensor: Union[torch.Tensor, "jnp.ndarray"], num_trials: int = 10) -> Dict[str, float]:
        """Measure model latency with device-specific implementations."""
        latencies = []
        
        if self.device_type == "tpu":
            # TPU warmup
            _ = self.model.apply(self.model.params, input_tensor)
            
            for _ in range(num_trials):
                start_time = time.time()
                _ = self.model.apply(self.model.params, input_tensor)
                latencies.append((time.time() - start_time) * 1000)
                jax.clear_caches()
        else:
            with torch.no_grad():
                # GPU/CPU warmup
                _ = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
                
                for _ in range(num_trials):
                    start_time = time.time()
                    _ = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
                    latencies.append((time.time() - start_time) * 1000)
                    self.memory_optimizer.clear_device_memory()

        stats = {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        }
        logger.info(f"Latency stats ({self.device_type}): {stats}")
        return stats

if __name__ == "__main__":
    # Example usage
    from vishwamai.models.cot_model import CoTModel
    from vishwamai.models.transformer import VishwamAITransformer
    
    # Mock transformer and model
    transformer = VishwamAITransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6)
    model = CoTModel(transformer)
    
    # Initialize tuner
    tuner = PerformanceTuner(model)
    
    # Tune batch size
    input_shape = (100, 512)  # Example input shape (sequence_length, d_model)
    optimal_batch_size = tuner.tune_batch_size(input_shape)
    
    # Measure latency
    dummy_input = torch.randn(optimal_batch_size, *input_shape).to(tuner.device)
    latency_stats = tuner.measure_latency(dummy_input)
    print(latency_stats)
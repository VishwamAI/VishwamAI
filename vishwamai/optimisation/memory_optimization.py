# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/memory_optimization.py

import torch
import psutil
import os
from typing import Dict, Optional, Callable, Union
import logging
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

try:
    import jax
    import jax.numpy as jnp
    from jax import random, device_put, device_get
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
import flax
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """A class to handle memory optimization techniques for VishwamAI models."""
    
    def __init__(self, model: Union[torch.nn.Module, "flax.linen.Module"], device: str = "auto"):
        """
        Initialize the MemoryOptimizer with a model and target device.
        
        Args:
            model: The VishwamAI model (PyTorch or Flax).
            device: The device to run the model on ("auto", "tpu", "gpu", or "cpu").
        """
        self.model = model
        self.device_type = self._detect_device(device)
        
        if self.device_type == "tpu":
            if not HAS_JAX:
                raise ImportError("JAX is required for TPU support")
            self.device = jax.devices("tpu")[0]
        else:
            self.device = device if torch.cuda.is_available() else "cpu"
            if isinstance(model, torch.nn.Module):
                self.model.to(self.device)
        
        logger.info(f"Initialized MemoryOptimizer with model on device: {self.device_type}")

    def _detect_device(self, device: str) -> str:
        """Detect the optimal available device."""
        if device != "auto":
            return device
            
        if HAS_JAX:
            try:
                jax.devices("tpu")
                return "tpu"
            except RuntimeError:
                pass
                
        return "gpu" if torch.cuda.is_available() else "cpu"

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if self.device_type == "tpu":
            # JAX/Flax gradient checkpointing
            if hasattr(self.model, "use_scan"):
                self.model.use_scan = True
                logger.info("Enabled JAX scan-based gradient checkpointing")
        else:
            # PyTorch gradient checkpointing
            try:
                self.model.gradient_checkpointing = True
                logger.info("Enabled PyTorch gradient checkpointing")
            except AttributeError:
                logger.warning("Model does not support gradient checkpointing")

    def enable_mixed_precision(self, scaler: Optional[Union[torch.cuda.amp.GradScaler, optax.GradientTransformation]] = None):
        """Enable mixed precision with device-specific implementation."""
        if self.device_type == "tpu":
            # TPU bfloat16 mixed precision
            if hasattr(self.model, "precision"):
                self.model.precision = jnp.bfloat16
            self.scaler = scaler if scaler else optax.adaptive_grad_clip(1.0)
        else:
            # GPU float16 mixed precision
            self.scaler = scaler if scaler else torch.cuda.amp.GradScaler()
        logger.info(f"Enabled mixed precision for {self.device_type}")

    def forward_with_mixed_precision(self, forward_fn: Callable, *args, **kwargs):
        """Run forward pass with mixed precision using appropriate implementation."""
        if self.device_type == "tpu":
            # TPU mixed precision forward
            @jax.jit
            def mixed_precision_forward(*jax_args):
                with jax.default_matmul_precision('bfloat16'):
                    return forward_fn(*jax_args)
            args = [device_put(arg, self.device) for arg in args]
            output = mixed_precision_forward(*args)
            return device_get(output)
        else:
            # GPU mixed precision forward
            with autocast():
                return forward_fn(*args, **kwargs)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {
            "system_memory_used_mb": psutil.Process(os.getpid()).memory_info().rss / 1024**2
        }
        
        if self.device_type == "tpu":
            try:
                # Get TPU memory stats
                tpu_memory = jax.device_get(jax.device_memory_profile(self.device))
                memory_stats.update({
                    "tpu_memory_used_mb": tpu_memory.used_bytes / 1024**2,
                    "tpu_memory_total_mb": tpu_memory.total_bytes / 1024**2
                })
            except:
                logger.warning("Could not get TPU memory statistics")
        elif self.device_type == "gpu":
            memory_stats.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2
            })
        
        logger.info(f"Memory usage: {memory_stats}")
        return memory_stats

    def clear_device_memory(self):
        """Clear device memory caches."""
        if self.device_type == "tpu":
            jax.clear_caches()
            logger.info("Cleared JAX/TPU memory caches")
        elif self.device_type == "gpu":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")

def optimize_memory_for_batch(model: Union[torch.nn.Module, "flax.linen.Module"], 
                            batch: Union[torch.Tensor, "jnp.ndarray"],
                            max_batch_size: int,
                            device: str = "auto") -> list:
    """Split batch processing for memory constraints with device-specific handling."""
    optimizer = MemoryOptimizer(model, device)
    batch_size = batch.shape[0]
    outputs = []
    
    for i in range(0, batch_size, max_batch_size):
        if optimizer.device_type == "tpu":
            sub_batch = jax.lax.dynamic_slice(batch, (i, 0), (max_batch_size, batch.shape[1]))
            with jax.default_matmul_precision('bfloat16'):
                output = optimizer.forward_with_mixed_precision(model, sub_batch)
        else:
            sub_batch = batch[i:i + max_batch_size]
            with torch.no_grad():
                output = optimizer.forward_with_mixed_precision(model, sub_batch)
        
        outputs.append(output)
        optimizer.clear_device_memory()
    
    return outputs

if __name__ == "__main__":
    # Example usage
    from vishwamai.models.cot_model import CoTModel
    from vishwamai.models.transformer import VishwamAITransformer
    
    # Mock transformer and model
    transformer = VishwamAITransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6)
    model = CoTModel(transformer)
    
    # Initialize optimizer
    optimizer = MemoryOptimizer(model)
    
    # Enable optimizations
    optimizer.enable_gradient_checkpointing()
    optimizer.enable_mixed_precision()
    
    # Check memory usage
    memory_stats = optimizer.get_memory_usage()
    print(memory_stats)
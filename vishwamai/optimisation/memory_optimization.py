# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/memory_optimization.py

import torch
import psutil
import os
from typing import Dict, Optional, Callable
import logging
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """A class to handle memory optimization techniques for VishwamAI models."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Initialize the MemoryOptimizer with a model and target device.
        
        Args:
            model: The VishwamAI model (e.g., CoTModel, ToTModel).
            device: The device to run the model on (default: "cuda").
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Initialized MemoryOptimizer with model on device: {self.device}")

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing to reduce memory usage by trading computation.
        """
        try:
            self.model.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing for the model.")
        except AttributeError:
            logger.warning("Model does not support gradient checkpointing. Skipping.")

    def enable_mixed_precision(self, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Enable mixed precision training/inference to reduce memory usage.
        
        Args:
            scaler: Optional GradScaler for mixed precision training.
        """
        self.scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()
        logger.info("Enabled mixed precision training/inference.")

    def forward_with_mixed_precision(self, forward_fn: Callable, *args, **kwargs):
        """
        Run a forward pass with mixed precision.
        
        Args:
            forward_fn: The forward function to execute.
            *args, **kwargs: Arguments to pass to the forward function.
        
        Returns:
            The output of the forward function.
        """
        with autocast():
            output = forward_fn(*args, **kwargs)
        return output

    def checkpointed_forward(self, module: torch.nn.Module, *inputs):
        """
        Run a forward pass with gradient checkpointing for a specific module.
        
        Args:
            module: The module to checkpoint.
            *inputs: Inputs to the module.
        
        Returns:
            The output of the module.
        """
        return checkpoint(module, *inputs)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get the current memory usage of the system and GPU.
        
        Returns:
            A dictionary with memory usage statistics (in MB).
        """
        memory_stats = {}
        
        # System memory
        process = psutil.Process(os.getpid())
        memory_stats["system_memory_used_mb"] = process.memory_info().rss / 1024**2
        
        # GPU memory (if applicable)
        if self.device != "cpu":
            memory_stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(self.device) / 1024**2
            memory_stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(self.device) / 1024**2
        
        logger.info(f"Memory usage: {memory_stats}")
        return memory_stats

    def clear_gpu_memory(self):
        """
        Clear GPU memory caches to free up space.
        """
        if self.device != "cpu":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache.")
        else:
            logger.warning("No GPU device detected. Skipping GPU memory clearing.")

def optimize_memory_for_batch(model: torch.nn.Module, batch: torch.Tensor, max_batch_size: int) -> list:
    """
    Split a batch into smaller chunks to fit within memory constraints.
    
    Args:
        model: The VishwamAI model.
        batch: The input batch tensor.
        max_batch_size: Maximum batch size that fits in memory.
    
    Returns:
        A list of outputs for each sub-batch.
    """
    batch_size = batch.shape[0]
    outputs = []
    
    for i in range(0, batch_size, max_batch_size):
        sub_batch = batch[i:i + max_batch_size]
        optimizer = MemoryOptimizer(model)
        
        with torch.no_grad():
            output = optimizer.forward_with_mixed_precision(model, sub_batch)
        outputs.append(output)
        
        optimizer.clear_gpu_memory()
    
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
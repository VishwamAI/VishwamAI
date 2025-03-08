# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/profiling_tools.py

import torch
import time
import logging
from typing import Dict, List, Tuple
from torch.profiler import profile, record_function, ProfilerActivity
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VishwamAIProfiler:
    """A class to profile VishwamAI models for performance bottlenecks."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Initialize the VishwamAIProfiler with a model and target device.
        
        Args:
            model: The VishwamAI model (e.g., CoTModel, ToTModel).
            device: The device to run the model on (default: "cuda").
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.memory_optimizer = MemoryOptimizer(self.model, self.device)
        logger.info(f"Initialized VishwamAIProfiler with model on device: {self.device}")

    def profile_model(self, input_tensor: torch.Tensor, num_steps: int = 10) -> Dict[str, any]:
        """
        Profile the model for CPU/GPU usage, memory, and latency.
        
        Args:
            input_tensor: Input tensor for profiling.
            num_steps: Number of profiling steps (default: 10).
        
        Returns:
            A dictionary with profiling results.
        """
        self.model.eval()
        activities = [ProfilerActivity.CPU]
        if self.device != "cpu":
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
                    self.memory_optimizer.clear_gpu_memory()
        
        # Extract profiling stats
        profiling_stats = {
            "avg_cpu_time_ms": prof.key_averages().table(sort_by="cpu_time_total"),
            "memory_usage": self.memory_optimizer.get_memory_usage(),
            "events": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        }
        
        logger.info("Profiling completed. Summary:")
        logger.info(profiling_stats["avg_cpu_time_ms"])
        logger.info(f"Memory usage: {profiling_stats['memory_usage']}")
        return profiling_stats

    def layer_wise_profiling(self, input_tensor: torch.Tensor) -> List[Tuple[str, float, float]]:
        """
        Profile the model layer-wise to identify bottlenecks.
        
        Args:
            input_tensor: Input tensor for profiling.
        
        Returns:
            A list of tuples (layer_name, latency_ms, memory_mb).
        """
        self.model.eval()
        layer_stats = []
        
        # Hook to measure latency and memory for each layer
        def profile_layer(module, input, output, layer_name):
            start_time = time.time()
            memory_before = torch.cuda.memory_allocated(self.device) / 1024**2 if self.device != "cpu" else 0
            output = module(input[0] if isinstance(input, tuple) else input)
            memory_after = torch.cuda.memory_allocated(self.device) / 1024**2 if self.device != "cpu" else 0
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
        logger.info("Layer-wise profiling completed:")
        for layer_name, latency, memory in layer_stats[:5]:
            logger.info(f"Layer {layer_name}: Latency = {latency:.2f} ms, Memory = {memory:.2f} MB")
        
        return layer_stats

if __name__ == "__main__":
    # Example usage
    from vishwamai.models.cot_model import CoTModel
    from vishwamai.models.transformer import VishwamAITransformer
    
    # Mock transformer and model
    transformer = VishwamAITransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6)
    model = CoTModel(transformer)
    
    # Initialize profiler
    profiler = VishwamAIProfiler(model)
    
    # Profile the model
    dummy_input = torch.randn(16, 100, 512).to(profiler.device)  # batch_size, seq_len, d_model
    profiling_stats = profiler.profile_model(dummy_input)
    
    # Layer-wise profiling
    layer_stats = profiler.layer_wise_profiling(dummy_input)
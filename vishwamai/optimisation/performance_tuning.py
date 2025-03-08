# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/performance_tuning.py

import torch
import time
import logging
from typing import Dict, Optional
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTuner:
    """A class to tune the performance of VishwamAI models."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Initialize the PerformanceTuner with a model and target device.
        
        Args:
            model: The VishwamAI model (e.g., CoTModel, ToTModel).
            device: The device to run the model on (default: "cuda").
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.memory_optimizer = MemoryOptimizer(self.model, self.device)
        logger.info(f"Initialized PerformanceTuner with model on device: {self.device}")

    def enable_hardware_optimizations(self):
        """
        Enable hardware-specific optimizations like cuDNN and AMP.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.memory_optimizer.enable_mixed_precision()
        logger.info("Enabled hardware optimizations (cuDNN, AMP).")

    def tune_batch_size(self, input_shape: tuple, max_batch_size: int = 128, tolerance: float = 0.1) -> int:
        """
        Find the optimal batch size that maximizes throughput without OOM errors.
        
        Args:
            input_shape: Shape of the input tensor (excluding batch size).
            max_batch_size: Maximum batch size to test.
            tolerance: Acceptable performance drop tolerance (default: 0.1).
        
        Returns:
            The optimal batch size.
        """
        optimal_batch_size = 1
        best_throughput = 0.0
        
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        for batch_size in range(1, max_batch_size + 1, 1):
            try:
                # Create a batch of the current size
                batch = dummy_input.repeat(batch_size, 1, 1)
                
                # Measure throughput
                self.model.eval()
                with torch.no_grad():
                    start_time = time.time()
                    _ = self.memory_optimizer.forward_with_mixed_precision(self.model, batch)
                    self.memory_optimizer.clear_gpu_memory()
                    elapsed_time = time.time() - start_time
                
                throughput = batch_size / elapsed_time
                logger.info(f"Batch size {batch_size}: Throughput = {throughput:.2f} samples/sec")
                
                if throughput > best_throughput * (1 - tolerance):
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                else:
                    break
                
            except RuntimeError as e:
                logger.warning(f"Batch size {batch_size} failed (likely OOM): {str(e)}")
                break
        
        logger.info(f"Optimal batch size: {optimal_batch_size} with throughput {best_throughput:.2f} samples/sec")
        return optimal_batch_size

    def optimize_inference(self, input_tensor: torch.Tensor, use_jit: bool = True) -> torch.Tensor:
        """
        Optimize inference speed for the given input tensor.
        
        Args:
            input_tensor: Input tensor for inference.
            use_jit: Whether to use TorchScript JIT compilation (default: True).
        
        Returns:
            The output tensor.
        """
        self.model.eval()
        
        # Enable hardware optimizations
        self.enable_hardware_optimizations()
        
        # Optionally use TorchScript for JIT compilation
        if use_jit:
            try:
                self.model = torch.jit.trace(self.model, input_tensor)
                logger.info("Applied TorchScript JIT compilation for inference.")
            except Exception as e:
                logger.warning(f"Failed to apply JIT compilation: {str(e)}")
        
        # Run inference with mixed precision
        with torch.no_grad():
            output = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
        
        return output

    def measure_latency(self, input_tensor: torch.Tensor, num_trials: int = 10) -> Dict[str, float]:
        """
        Measure the average latency of the model for the given input.
        
        Args:
            input_tensor: Input tensor for inference.
            num_trials: Number of trials to average over (default: 10).
        
        Returns:
            A dictionary with latency statistics (in milliseconds).
        """
        self.model.eval()
        latencies = []
        
        with torch.no_grad():
            # Warm-up run
            self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
            
            for _ in range(num_trials):
                start_time = time.time()
                _ = self.memory_optimizer.forward_with_mixed_precision(self.model, input_tensor)
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
                self.memory_optimizer.clear_gpu_memory()
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        latency_stats = {
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
        }
        logger.info(f"Latency stats: {latency_stats}")
        return latency_stats

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
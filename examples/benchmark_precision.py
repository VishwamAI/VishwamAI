"""
Benchmark different precision modes
"""
import time
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List
import logging

from vishwamai.model import VishwamaiModel
from vishwamai.config import ModelConfig, PrecisionConfig, PrecisionMode
from vishwamai.utils.t4_utils import get_memory_stats, profile_memory_usage

logger = logging.getLogger(__name__)

class PrecisionBenchmark:
    """Benchmark different precision modes"""
    def __init__(
        self,
        model_sizes: List[int] = [768, 1024, 2048],  # Hidden sizes
        sequence_length: int = 1024,
        batch_size: int = 16,
        num_warmup: int = 5,
        num_iter: int = 100,
        output_dir: str = "benchmark_results"
    ):
        self.model_sizes = model_sizes
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def _create_model(
        self,
        hidden_size: int,
        precision_mode: PrecisionMode
    ) -> VishwamaiModel:
        """Create model with specified size and precision"""
        precision_config = PrecisionConfig(mode=precision_mode)
        model_config = ModelConfig(
            hidden_size=hidden_size,
            num_layers=8,
            num_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
            precision=precision_config
        )
        return VishwamaiModel(model_config)
        
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input tensor"""
        return torch.randint(
            0, 32000,
            (self.batch_size, self.sequence_length),
            device="cuda"
        )
        
    def benchmark_forward(
        self,
        model: VishwamaiModel,
        input_ids: torch.Tensor,
        precision_mode: PrecisionMode
    ) -> Dict[str, float]:
        """Benchmark forward pass"""
        # Warmup
        for _ in range(self.num_warmup):
            with torch.no_grad():
                _ = model(input_ids)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        memory_start = get_memory_stats()
        
        for _ in range(self.num_iter):
            with torch.no_grad():
                _ = model(input_ids)
        
        torch.cuda.synchronize()
        end_time = time.time()
        memory_end = get_memory_stats()
        
        time_per_iter = (end_time - start_time) / self.num_iter * 1000  # ms
        memory_used = memory_end["allocated"] - memory_start["allocated"]
        
        return {
            "forward_time_ms": time_per_iter,
            "memory_mb": memory_used,
            "memory_start": memory_start["allocated"],
            "memory_end": memory_end["allocated"]
        }
        
    def benchmark_backward(
        self,
        model: VishwamaiModel,
        input_ids: torch.Tensor,
        precision_mode: PrecisionMode
    ) -> Dict[str, float]:
        """Benchmark backward pass"""
        # Warmup
        for _ in range(self.num_warmup):
            outputs = model(input_ids)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        memory_start = get_memory_stats()
        
        for _ in range(self.num_iter):
            outputs = model(input_ids)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()
            
        torch.cuda.synchronize()
        end_time = time.time()
        memory_end = get_memory_stats()
        
        time_per_iter = (end_time - start_time) / self.num_iter * 1000  # ms
        memory_used = memory_end["allocated"] - memory_start["allocated"]
        
        return {
            "backward_time_ms": time_per_iter,
            "memory_mb": memory_used,
            "memory_start": memory_start["allocated"],
            "memory_end": memory_end["allocated"]
        }
        
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks for all configurations"""
        precision_modes = [
            PrecisionMode.FP16,
            PrecisionMode.FP32,
            PrecisionMode.FP64,
            PrecisionMode.BF16
        ]
        
        results = {}
        
        for hidden_size in self.model_sizes:
            results[hidden_size] = {}
            
            for mode in precision_modes:
                logger.info(f"Benchmarking hidden_size={hidden_size}, mode={mode.value}")
                
                # Create model and move to GPU
                model = self._create_model(hidden_size, mode)
                model = model.cuda()
                input_ids = self._create_dummy_input()
                
                # Run benchmarks
                forward_metrics = self.benchmark_forward(model, input_ids, mode)
                backward_metrics = self.benchmark_backward(model, input_ids, mode)
                
                # Profile memory
                memory_profile = profile_memory_usage(model, input_ids)
                
                results[hidden_size][mode.value] = {
                    **forward_metrics,
                    **backward_metrics,
                    "memory_profile": memory_profile,
                    "model_size": sum(p.numel() for p in model.parameters()),
                    "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                }
                
                # Clear GPU memory
                del model
                torch.cuda.empty_cache()
                
        self.results = results
        return results
        
    def save_results(self, filename: str = "precision_benchmark_results.json"):
        """Save benchmark results"""
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {
            str(k): {
                mode: {
                    metric: float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
                    for metric, value in metrics.items()
                }
                for mode, metrics in v.items()
            }
            for k, v in self.results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Benchmark results saved to {output_path}")
        
    def plot_results(self, filename: str = "precision_benchmark_plots.png"):
        """Plot benchmark results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib and seaborn required for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data for plotting
        sizes = []
        modes = []
        forward_times = []
        backward_times = []
        memory_usage = []
        model_sizes = []
        
        for size, size_results in self.results.items():
            for mode, metrics in size_results.items():
                sizes.append(size)
                modes.append(mode)
                forward_times.append(metrics["forward_time_ms"])
                backward_times.append(metrics["backward_time_ms"])
                memory_usage.append(metrics["memory_mb"])
                model_sizes.append(metrics["model_size_mb"])
                
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "Hidden Size": sizes,
            "Precision": modes,
            "Forward Time (ms)": forward_times,
            "Backward Time (ms)": backward_times,
            "Memory Usage (MB)": memory_usage,
            "Model Size (MB)": model_sizes
        })
        
        # Plot results
        sns.barplot(data=df, x="Hidden Size", y="Forward Time (ms)",
                   hue="Precision", ax=axes[0,0])
        sns.barplot(data=df, x="Hidden Size", y="Backward Time (ms)",
                   hue="Precision", ax=axes[0,1])
        sns.barplot(data=df, x="Hidden Size", y="Memory Usage (MB)",
                   hue="Precision", ax=axes[1,0])
        sns.barplot(data=df, x="Hidden Size", y="Model Size (MB)",
                   hue="Precision", ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    benchmark = PrecisionBenchmark(
        model_sizes=[768, 1024, 2048],
        sequence_length=1024,
        batch_size=16
    )
    
    results = benchmark.run_benchmarks()
    benchmark.save_results()
    benchmark.plot_results()

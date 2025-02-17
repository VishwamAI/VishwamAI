import os
import time
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Generator
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
from torch.cuda import nvtx
import tvm
from tvm import relay
from opentelemetry import trace, metrics
from prometheus_client import Counter, Histogram, Gauge
from stable_baselines3 import PPO

from .MoE import MoEConfig, create_moe_layer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Metrics for tracking inference performance."""
    latency: float
    throughput: float
    gpu_utilization: float
    memory_usage: float
    batch_size: int
    expert_usage: Optional[Dict[str, float]] = None
    token_routing_efficiency: Optional[float] = None

class KVCache:
    """Optimized KV cache with pruning capabilities."""
    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.cache = {}
        self.access_counts = {}
        self.last_access = {}
        
    def prune(self, threshold: float = 0.1):
        """Prune least used cache entries."""
        total_accesses = sum(self.access_counts.values())
        
        # Calculate access frequencies
        frequencies = {
            k: count / total_accesses 
            for k, count in self.access_counts.items()
        }
        
        # Remove entries below threshold
        for key, freq in frequencies.items():
            if freq < threshold:
                del self.cache[key]
                del self.access_counts[key]
                del self.last_access[key]
                
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached value with access tracking."""
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.last_access[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: torch.Tensor):
        """Add or update cache entry."""
        self.cache[key] = value
        self.access_counts[key] = 1
        self.last_access[key] = time.time()
        
        if len(self.cache) > self.max_size:
            self.prune()

class AutoscalingPolicy:
    """AI-driven autoscaling using PPO with MoE awareness."""
    def __init__(self, config: Dict[str, Any]):
        self.metrics_history = []
        self.current_batch_size = config.get("batch_size", {}).get("min", 1)
        self.rl_model = self._initialize_rl_model(config)
        
        # MoE-specific settings
        self.expert_capacity = config.get("expert_capacity", 32)
        self.min_active_experts = config.get("min_active_experts", 2)
        
    def _initialize_rl_model(self, config: Dict[str, Any]) -> PPO:
        """Initialize PPO model for autoscaling."""
        try:
            return PPO("MlpPolicy", 
                      "InferenceEnv", 
                      verbose=1, 
                      tensorboard_log="./tensorboard/")
        except Exception as e:
            logger.warning(f"Failed to initialize RL model: {e}. Using default scaling.")
            return None
    
    def _get_current_state(self) -> np.ndarray:
        """Get current state including MoE metrics."""
        if not self.metrics_history:
            return np.zeros(6)  # Default state
        latest_metrics = self.metrics_history[-1]
        expert_usage = latest_metrics.expert_usage or {}
        avg_expert_usage = np.mean(list(expert_usage.values())) if expert_usage else 0
        
        return np.array([
            latest_metrics.latency,
            latest_metrics.throughput,
            latest_metrics.gpu_utilization,
            latest_metrics.memory_usage,
            avg_expert_usage,
            latest_metrics.token_routing_efficiency or 0
        ])
    
    def _action_to_config(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert RL action to scaling configuration."""
        return {
            "batch_size": max(1, min(32, int(action[0] * 32))),
            "cache_size": int(action[1] * 8192),
            "expert_capacity": int(action[2] * self.expert_capacity)
        }

class HardwareOptimizer:
    """Hardware-specific optimizations using TVM."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tvm_target = self._get_tvm_target()
        self.enable_moe_fusion = config.get("optimization", {}).get("moe_fusion", True)
        
    def optimize_model(self, model: torch.nn.Module, input_shape: List[int]):
        """Optimize model using TVM with MoE support."""
        try:
            traced_model = torch.jit.trace(model, torch.randn(*input_shape))
            mod, params = relay.frontend.from_pytorch(traced_model, input_shape)
            
            if self.enable_moe_fusion:
                # Apply MoE-specific optimizations
                mod = self._apply_moe_optimizations(mod)
            
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=self.tvm_target, params=params)
            return lib
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def _apply_moe_optimizations(self, mod):
        """Apply MoE-specific optimizations."""
        try:
            # Custom pass to fuse MoE operations
            from tvm.relay import transform
            seq = transform.Sequential([
                transform.FuseMoEPatterns(),
                transform.MoELayoutOptimization(),
                transform.FoldConstant()
            ])
            return seq(mod)
        except Exception as e:
            logger.warning(f"MoE optimization failed: {e}")
            return mod

class InferenceEngine:
    """Main inference engine with MoE support."""
    def __init__(self, config_path: str = "configs/inference_config.yaml"):
        self.config = self._load_config(config_path)
        self.hardware_optimizer = HardwareOptimizer(self.config["inference"]["hardware"])
        self.autoscaling = AutoscalingPolicy(self.config["inference"]["scaling"])
        self.current_batch_size = 1
        
        # Initialize caches
        self.kv_cache = KVCache(
            max_size=self.config["inference"]["pipeline"]["cache"]["size"]
        )
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config["inference"]["optimization"]["threading"]["inter_op"]
        )
        
        # MoE-specific settings
        self.expert_parallelism = self.config["inference"]["optimization"].get(
            "expert_parallelism", True
        )
    
    def _init_metrics(self):
        """Initialize metrics including MoE-specific ones."""
        # Existing metrics
        self.latency_histogram = Histogram(
            "inference_latency_seconds", 
            "Inference request latency"
        )
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage"
        )
        self.throughput = Counter(
            "inference_requests_total",
            "Total number of inference requests"
        )
        
        # MoE-specific metrics
        self.expert_utilization = Gauge(
            "expert_utilization_percent",
            "Expert utilization percentage",
            ["expert_id"]
        )
        self.routing_efficiency = Gauge(
            "token_routing_efficiency",
            "Token routing efficiency"
        )
        self.cache_hit_rate = Gauge(
            "kv_cache_hit_rate",
            "KV cache hit rate"
        )
    
    def optimize_model(self, model: torch.nn.Module, input_shape: List[int]) -> torch.nn.Module:
        """Optimize model for inference with MoE support."""
        # Apply base optimizations
        if self.config["inference"]["engine"]["quantization"]["enabled"]:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Apply TVM optimizations
        optimized_lib = self.hardware_optimizer.optimize_model(model, input_shape)
        
        # Enable expert parallelism if configured
        if self.expert_parallelism:
            self._enable_expert_parallelism(model)
        
        return optimized_lib
    
    def _enable_expert_parallelism(self, model: torch.nn.Module):
        """Enable parallel execution of experts."""
        for module in model.modules():
            if hasattr(module, "experts"):
                # Set experts for parallel execution
                module.parallel_experts = True
    
    @torch.cuda.amp.autocast()
    def _run_inference(
        self, 
        model: torch.nn.Module, 
        inputs: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Run inference with automatic mixed precision and MoE tracking."""
        with nvtx.range("inference"):
            if batch_size:
                inputs = inputs.chunk(batch_size)
                outputs = []
                expert_usage = {}
                
                for batch in inputs:
                    with self.latency_histogram.time():
                        output = model(batch)
                        
                        # Collect MoE statistics
                        self._update_moe_metrics(model)
                        
                    outputs.append(output)
                return torch.cat(outputs)
            else:
                with self.latency_histogram.time():
                    output = model(inputs)
                    self._update_moe_metrics(model)
                    return output
    
    def _update_moe_metrics(self, model: torch.nn.Module):
        """Update MoE-specific metrics."""
        for name, module in model.named_modules():
            if hasattr(module, "gate"):
                # Update expert utilization metrics
                if hasattr(module.gate, "expert_counts"):
                    counts = module.gate.expert_counts.detach().cpu()
                    total = counts.sum().item()
                    for i, count in enumerate(counts):
                        util = (count / total * 100).item()
                        self.expert_utilization.labels(expert_id=i).set(util)
                
                # Update routing efficiency
                if hasattr(module.gate, "routing_efficiency"):
                    self.routing_efficiency.set(
                        module.gate.routing_efficiency.item()
                    )
    
    def run(
        self, 
        model: torch.nn.Module,
        inputs: Union[torch.Tensor, np.ndarray],
        stream: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """Run inference with automatic optimization."""
        start_time = time.time()
        
        # Convert inputs if necessary
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).cuda()
        
        # Get optimal configuration from autoscaler
        metrics = InferenceMetrics(
            latency=time.time() - start_time,
            throughput=self.throughput._value.get(),
            gpu_utilization=torch.cuda.utilization(),
            memory_usage=torch.cuda.memory_allocated() / 1024**3,
            batch_size=self.current_batch_size,
            expert_usage=self._get_expert_usage(model),
            token_routing_efficiency=self._get_routing_efficiency(model)
        )
        
        scaling_config = self.autoscaling.update(metrics)
        batch_size = scaling_config.get("batch_size", None)
        self.current_batch_size = batch_size if batch_size else self.current_batch_size
        
        # Update KV cache configuration
        if "cache_size" in scaling_config:
            self.kv_cache.max_size = scaling_config["cache_size"]
        
        # Run inference
        try:
            with torch.inference_mode():
                outputs = self._run_inference(model, inputs, batch_size)
                
            # Update metrics
            self._update_metrics(start_time)
            self.throughput.inc()
            
            return outputs.cpu().numpy() if isinstance(inputs, np.ndarray) else outputs
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError("Inference failed") from e

    def _get_expert_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """Get expert usage statistics."""
        usage = {}
        for name, module in model.named_modules():
            if hasattr(module, "gate") and hasattr(module.gate, "expert_counts"):
                counts = module.gate.expert_counts.detach().cpu()
                total = counts.sum().item()
                for i, count in enumerate(counts):
                    usage[f"{name}_expert_{i}"] = count.item() / total
        return usage

    def _get_routing_efficiency(self, model: torch.nn.Module) -> Optional[float]:
        """Get token routing efficiency."""
        efficiencies = []
        for module in model.modules():
            if hasattr(module, "gate") and hasattr(module.gate, "routing_efficiency"):
                efficiencies.append(module.gate.routing_efficiency.item())
        return np.mean(efficiencies) if efficiencies else None

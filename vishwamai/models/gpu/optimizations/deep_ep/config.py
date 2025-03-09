"""
Configuration module for DeepEP (Expert Parallelism)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Config:
    """Configuration class for DeepEP"""
    num_experts: int = 8
    expert_capacity: int = 4
    num_groups: int = 1
    dispatch_batch_size: Optional[int] = None
    cache_dir: str = "/tmp/vishwamai/deepep_cache"

def get_optimal_dispatch_config(
    batch_size: int,
    num_experts: int,
    device_memory: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate optimal dispatch configuration based on available resources
    """
    if device_memory is None:
        import torch
        if torch.cuda.is_available():
            device_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            device_memory = 8 * (1024 ** 3)  # Default 8GB

    optimal_batch_size = min(batch_size, device_memory // (4 * num_experts))
    
    return {
        "batch_size": optimal_batch_size,
        "num_experts": num_experts,
        "expert_capacity": max(1, optimal_batch_size // num_experts)
    }
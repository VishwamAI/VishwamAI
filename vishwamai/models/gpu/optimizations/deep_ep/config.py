"""
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
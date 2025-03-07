"""
Utility functions for Deep Efficient Parallelism (DeepEP).
"""

import torch
import math

def get_num_sms() -> int:
    """Get number of streaming multiprocessors (SMs) on current GPU"""
    if not torch.cuda.is_available():
        return 1
        
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count

def get_best_configs(shape: tuple, device_type: str = "gpu") -> dict:
    """
    Get optimal configurations for parallel operations
    
    Args:
        shape: Input tensor shape
        device_type: Target device type ("gpu" or "cpu")
        
    Returns:
        dict: Configuration parameters
    """
    if device_type == "cpu":
        return {
            "block_size": 32,
            "num_warps": 4,
            "num_stages": 2
        }
        
    # Get GPU capabilities
    num_sms = get_num_sms()
    
    # Scale block size based on tensor dimensions
    max_dim = max(shape)
    if max_dim <= 512:
        block_size = 32
    elif max_dim <= 2048:
        block_size = 64
    else:
        block_size = 128
        
    # Adjust warp count based on SMs
    num_warps = min(8, max(2, num_sms // 4))
    
    return {
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": 3,
        "num_sms": num_sms
    }

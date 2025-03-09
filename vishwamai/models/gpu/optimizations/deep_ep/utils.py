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

def set_num_sms(num_sms: int) -> None:
    """
    Set the number of streaming multiprocessors (SMs) to use.
    This is useful for testing or when you want to limit GPU utilization.
    
    Args:
        num_sms: Number of SMs to use
    """
    if not torch.cuda.is_available():
        return
        
    if num_sms <= 0:
        raise ValueError("Number of SMs must be positive")
        
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if num_sms > props.multi_processor_count:
        raise ValueError(f"Requested {num_sms} SMs but device only has {props.multi_processor_count}")
        
    # Store the value in a module-level variable
    global _configured_num_sms
    _configured_num_sms = num_sms

_configured_num_sms = None  # Module-level variable to store configured SM count

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

def init_expert_parallel() -> None:
    """
    Initialize expert parallelism for distributed processing.
    This function sets up the necessary environment and configurations
    for expert parallelism using DeepEP.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Expert parallelism requires CUDA-enabled GPUs.")
    
    # Get the number of streaming multiprocessors (SMs)
    num_sms = get_num_sms()
    
    # Set the number of SMs to use for expert parallelism
    set_num_sms(num_sms)
    
    # Additional initialization steps can be added here if needed
    print(f"Expert parallelism initialized with {num_sms} SMs.")

"""
T4 GPU utility functions
"""
import torch
from typing import Dict, Any

def get_device_capabilities() -> Dict[str, Any]:
    """
    Get the capabilities of the current device, particularly for T4 GPUs
    """
    capabilities = {
        "flash_attention": False,
        "tensor_cores": False,
        "bfloat16": False,
        "amp": False,
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        capabilities.update({
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "tensor_cores": props.major >= 7,  # Tensor Cores available from compute capability 7.0
            "bfloat16": hasattr(torch, 'bfloat16') and props.major >= 8,  # BF16 from compute capability 8.0
            "amp": True,  # Automatic Mixed Precision is supported on all modern GPUs
        })
        
        # Check for Flash Attention support
        try:
            import flash_attn
            capabilities["flash_attention"] = True
        except ImportError:
            pass

    return capabilities

def get_optimal_dtype(device_capabilities: Dict[str, Any]) -> torch.dtype:
    """
    Get the optimal dtype based on device capabilities
    """
    if not device_capabilities["cuda_available"]:
        return torch.float32
        
    if device_capabilities["bfloat16"]:
        return torch.bfloat16
    elif device_capabilities["amp"]:
        return torch.float16
    else:
        return torch.float32

def optimize_model_for_t4(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply T4-specific optimizations to a model
    """
    capabilities = get_device_capabilities()
    
    if capabilities["cuda_available"]:
        # Enable gradient checkpointing if available
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Move model to GPU
        model = model.cuda()
        
        # Set optimal dtype
        dtype = get_optimal_dtype(capabilities)
        if dtype in [torch.float16, torch.bfloat16]:
            model = model.to(dtype)
            
    return model

def enable_t4_optimizations() -> None:
    """
    Enable T4-specific PyTorch optimizations
    """
    if torch.cuda.is_available():
        # Enable TF32 for better performance with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking and deterministic algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set optimal memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
        
def get_memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0,
            "cached_gb": 0,
            "free_gb": 0,
            "total_gb": 0
        }
        
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cached_gb": torch.cuda.memory_reserved() / 1e9,
        "free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_reserved()) / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
    }

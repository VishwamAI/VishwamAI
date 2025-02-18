import torch
from typing import Tuple, Optional
from .gpu_config import GPUConfig, get_gpu_capabilities, optimize_gpu_settings, adjust_model_config

def initialize_training_setup(model_args) -> Tuple[GPUConfig, torch.device]:
    """Initialize training setup with GPU optimizations."""
    # Get GPU configuration
    gpu_config = get_gpu_capabilities()
    print(f"Detected GPU: {gpu_config.device_name}")
    print(f"Available memory: {gpu_config.memory_available:.1f} GB")
    
    # Apply GPU optimizations
    optimize_gpu_settings(gpu_config)
    
    # Adjust model configuration
    adjust_model_config(model_args, gpu_config)
    
    # Set device
    device = torch.device("cuda")
    
    return gpu_config, device

def setup_memory_optimizations(model, gpu_config: GPUConfig) -> None:
    """Apply memory optimizations to model."""
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    
    # Enable memory efficient attention if available
    if hasattr(model, "enable_mem_efficient_attention"):
        model.enable_mem_efficient_attention()
    
    # Set optimal dtype
    if gpu_config.supports_bf16:
        model.to(dtype=torch.bfloat16)
    elif gpu_config.supports_fp16:
        model.to(dtype=torch.float16)

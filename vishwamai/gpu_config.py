import torch
import math
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class GPUConfig:
    device_name: str
    memory_total: float
    memory_available: float
    compute_capability: Tuple[int, int]
    supports_bf16: bool
    supports_fp16: bool
    optimal_batch_size: int
    optimal_chunk_size: int

def get_gpu_capabilities() -> GPUConfig:
    """Get detailed GPU capabilities and recommended settings."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required.")
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    # Check compute capability for feature support
    compute_capability = (props.major, props.minor)
    supports_bf16 = compute_capability >= (8, 0)  # Ampere and newer
    supports_fp16 = True  # All modern GPUs support FP16
    
    # Calculate memory metrics
    memory_total = props.total_memory / (1024**3)  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
    memory_available = memory_total - memory_reserved
    
    # Calculate optimal batch and chunk sizes based on available memory
    if 'T4' in props.name:
        optimal_batch_size = min(2, max(1, int(memory_available * 0.3)))
        optimal_chunk_size = min(1024, max(128, int(memory_available * 256)))
    elif 'V100' in props.name:
        optimal_batch_size = min(4, max(1, int(memory_available * 0.4)))
        optimal_chunk_size = min(2048, max(256, int(memory_available * 512)))
    elif 'A100' in props.name:
        optimal_batch_size = min(8, max(2, int(memory_available * 0.5)))
        optimal_chunk_size = min(4096, max(512, int(memory_available * 1024)))
    else:
        optimal_batch_size = max(1, int(memory_available * 0.2))
        optimal_chunk_size = min(512, max(64, int(memory_available * 128)))
    
    return GPUConfig(
        device_name=props.name,
        memory_total=memory_total,
        memory_available=memory_available,
        compute_capability=compute_capability,
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
        optimal_batch_size=optimal_batch_size,
        optimal_chunk_size=optimal_chunk_size
    )

def optimize_gpu_settings(config: GPUConfig) -> None:
    """Apply optimal GPU settings based on configuration."""
    # Enable TF32 if available (on Ampere+)
    if config.compute_capability >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Set memory allocation settings
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.95)

def adjust_model_config(model_args, gpu_config: GPUConfig) -> None:
    """Adjust model configuration based on GPU capabilities."""
    # Update batch size
    model_args.max_batch_size = gpu_config.optimal_batch_size
    
    # Adjust sequence length based on available memory
    model_args.max_seq_len = min(
        model_args.max_seq_len,
        gpu_config.optimal_chunk_size
    )
    
    # Scale model dimensions based on memory
    mem_factor = gpu_config.memory_available / 16.0  # Normalize to 16GB baseline
    model_args.dim = min(
        model_args.dim,
        int(1024 * math.sqrt(mem_factor))
    )
    
    # Adjust number of layers
    model_args.n_layers = min(
        model_args.n_layers,
        int(12 * math.sqrt(mem_factor))
    )
    
    # Set precision based on hardware support
    if gpu_config.supports_bf16:
        model_args.dtype = "bfloat16"
    elif gpu_config.supports_fp16:
        model_args.dtype = "float16"
    else:
        model_args.dtype = "float32"

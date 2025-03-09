# /home/kasinadhsarma/VishwamAI/vishwamai/optimisation/optimization_utils.py

import torch
import logging
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for model optimization settings."""
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 32
    max_gradient_norm: float = 1.0
    use_dynamic_padding: bool = True
    use_kernel_injection: bool = True
    optimization_level: int = 3  # 0-3, higher means more aggressive optimization

@dataclass
class TPUConfig:
    """TPU-specific optimization settings."""
    bfloat16: bool = True
    circular_buffer: bool = True
    remat_scan: bool = True
    fused_attention: bool = True
    kernel_fusion: bool = True
    max_devices_per_host: int = 8
    prefetch_depth: int = 2
    spmd_pass: bool = True

@dataclass
class GPUConfig:
    """GPU-specific optimization settings."""
    amp_dtype: torch.dtype = torch.float16
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    kernel_fusion: bool = True
    memory_efficient_attention: bool = True
    fused_adam: bool = True
    fused_layernorm: bool = True

def get_device_config(device_type: str) -> Union[TPUConfig, GPUConfig]:
    """Get device-specific optimization configuration."""
    if device_type == "tpu":
        return TPUConfig()
    return GPUConfig()

def optimize_kernel_launch(device_type: str, config: OptimizationConfig) -> Dict[str, Any]:
    """Configure kernel launch parameters for optimal performance."""
    if device_type == "tpu":
        return {
            "block_size": 128,
            "num_warps": 4,
            "num_stages": 3,
            "use_immutable": True,
            "spmd_mode": True if config.optimization_level >= 2 else False,
            "auto_tune": True if config.optimization_level >= 3 else False
        }
    else:  # GPU
        return {
            "block_size": 256,
            "num_warps": 8,
            "num_stages": 2,
            "shared_memory": True,
            "dynamic_smem": True if config.optimization_level >= 2 else False,
            "auto_tune": True if config.optimization_level >= 3 else False
        }

def optimize_attention_config(device_type: str, hidden_size: int) -> Dict[str, Any]:
    """Get optimized attention configuration based on device and model size."""
    if device_type == "tpu":
        return {
            "query_chunk_size": 1024,
            "key_chunk_size": 4096,
            "fused_qkv": True,
            "recompute_attention": True,
            "use_flash_attention": False,  # Flash attention not optimal for TPU
            "attention_dtype": "bfloat16",
            "kernel_fusion": True
        }
    else:  # GPU
        return {
            "query_chunk_size": 2048,
            "key_chunk_size": 8192,
            "fused_qkv": True,
            "use_flash_attention": True,
            "attention_dtype": "float16",
            "kernel_fusion": True,
            "memory_efficient": True
        }

def optimize_data_layout(device_type: str, batch_size: int, seq_length: int, hidden_size: int) -> Dict[str, Any]:
    """Optimize data layout for specific device architecture."""
    if device_type == "tpu":
        # TPU prefers certain data layouts for optimal performance
        return {
            "batch_parallel": True,
            "sequence_parallel": seq_length >= 512,
            "hidden_parallel": hidden_size >= 1024,
            "preferred_layout": "BHLD",  # Batch, Heads, Length, Dim
            "padding_multiple": 128,
            "use_circular_buffer": True
        }
    else:  # GPU
        return {
            "batch_parallel": True,
            "sequence_parallel": seq_length >= 1024,
            "hidden_parallel": hidden_size >= 2048,
            "preferred_layout": "BHLK",  # Batch, Heads, Length, Key
            "padding_multiple": 8,
            "use_register_memory": True
        }

def create_optimizer_config(device_type: str, 
                          learning_rate: float,
                          weight_decay: float,
                          optimizer_type: str = "adam") -> Dict[str, Any]:
    """Create device-optimized optimizer configuration."""
    if device_type == "tpu":
        # TPU-optimized JAX/Optax optimizer config
        if optimizer_type.lower() == "adam":
            return {
                "optimizer": optax.adam,
                "learning_rate": learning_rate,
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "weight_decay": weight_decay,
                "mask": None,
                "use_fused": True
            }
    else:  # GPU
        # GPU-optimized PyTorch optimizer config
        if optimizer_type.lower() == "adam":
            return {
                "optimizer": "FusedAdam" if torch.cuda.is_available() else "Adam",
                "lr": learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": weight_decay,
                "amsgrad": False,
                "use_fused": True if torch.cuda.is_available() else False
            }
    
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_optimal_training_config(model_size: int, device_type: str, batch_size: int) -> Dict[str, Any]:
    """Get optimal training configuration based on model size and device."""
    base_config = {
        "optimizer": create_optimizer_config(
            device_type=device_type,
            learning_rate=1e-4,
            weight_decay=0.01
        ),
        "mixed_precision": True,
        "gradient_checkpointing": model_size >= 1e9,  # Enable for models > 1B params
        "batch_size": batch_size,
        "gradient_clipping": 1.0
    }
    
    if device_type == "tpu":
        base_config.update({
            "precision": "bfloat16",
            "spmd_pass": True,
            "circular_buffer": True,
            "kernel_fusion": True,
            "remat_policy": "save_memory" if model_size >= 1e9 else "save_time"
        })
    else:  # GPU
        base_config.update({
            "precision": "float16",
            "cudnn_benchmark": True,
            "fused_adam": True,
            "fused_layernorm": True,
            "memory_efficient_attention": True,
            "scaled_dot_product": True
        })
    
    return base_config

if __name__ == "__main__":
    # Example usage
    device_type = "tpu" if HAS_JAX else "gpu"
    config = OptimizationConfig()
    
    # Get device-specific configurations
    device_config = get_device_config(device_type)
    kernel_config = optimize_kernel_launch(device_type, config)
    attention_config = optimize_attention_config(device_type, hidden_size=768)
    data_layout = optimize_data_layout(device_type, batch_size=32, seq_length=512, hidden_size=768)
    
    print(f"Device type: {device_type}")
    print(f"Device config: {device_config}")
    print(f"Kernel config: {kernel_config}")
    print(f"Attention config: {attention_config}")
    print(f"Data layout: {data_layout}")
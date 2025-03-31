"""TPU pipeline and kernel fusion configurations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax
from jax import lax

@dataclass
class TPUPipelineConfig:
    """Configuration for TPU pipeline optimization."""
    
    # Pipeline stages
    num_micro_batches: int = 8  # Number of micro-batches for pipeline parallelism
    pipeline_stages: int = 4     # Number of pipeline stages
    stage_placement: str = "auto" # How to place stages across TPU cores
    
    # Memory optimizations  
    rematerialize: bool = True   # Whether to rematerialize activations
    remat_granularity: int = 2   # How many layers to group for rematerialization
    preserve_rng_state: bool = True  # Preserve RNG state across pipeline stages
    
    # Kernel fusion settings
    block_size: int = 128        # TPU MXU block size
    fuse_attention: bool = True  # Fuse attention operations
    fuse_layernorm: bool = True  # Fuse layer normalization
    fuse_relu: bool = True       # Fuse ReLU with preceding operation
    
    # Precision settings
    use_bfloat16: bool = True    # Use bfloat16 precision
    use_fp8: bool = False        # Enable FP8 for certain operations
    matmul_precision: Any = lax.Precision.HIGHEST
    
    # Prefetching
    num_prefetch: int = 2        # Number of steps to prefetch
    prefetch_to_device: bool = True  # Prefetch directly to TPU device memory
    
    # Compilation settings
    xla_flags: Optional[Dict[str, Any]] = None  # Custom XLA flags
    
    def __post_init__(self):
        if self.block_size % 128 != 0:
            raise ValueError("block_size must be multiple of 128 for TPU")
            
        # Set default XLA flags if not provided
        if self.xla_flags is None:
            self.xla_flags = {
                "xla_enable_fast_math": True,
                "xla_cpu_enable_fast_math": True,
                "xla_gpu_enable_fast_math": True,
                "xla_gpu_autotune_level": 4,
                "xla_force_host_platform_device_count": "1",
            }

def create_pipeline_mesh(num_cores: int, config: TPUPipelineConfig):
    """Create optimal device mesh for pipeline parallelism."""
    devices = jax.devices()
    
    if len(devices) < num_cores:
        raise ValueError(f"Requested {num_cores} cores but only {len(devices)} available")
    
    # Create 2D mesh for data and pipeline parallelism
    mesh_shape = (num_cores // config.pipeline_stages, config.pipeline_stages)
    device_mesh = jax.device_mesh(devices[:num_cores], mesh_shape)
    
    return device_mesh

def configure_tpu_pipeline(config: TPUPipelineConfig):
    """Configure JAX for optimal TPU pipeline performance."""
    
    # Set XLA flags
    for flag, value in config.xla_flags.items():
        jax.config.update(flag, value)
    
    # Configure precision
    if config.use_bfloat16:
        jax.config.update("jax_default_dtype_bits", "bfloat16")
    
    # Configure rematerialization
    if config.rematerialize:
        jax.config.update("jax_remat_opt_level", config.remat_granularity)
    
    # Configure prefetching
    if config.prefetch_to_device:
        jax.config.update("jax_prefetch_to_device", config.num_prefetch)
    
    return config

def get_optimal_pipeline_config(
    model_size: str = "large",
    num_cores: int = 8
) -> TPUPipelineConfig:
    """Get optimal pipeline configuration based on model size."""
    
    configs = {
        "small": TPUPipelineConfig(
            num_micro_batches=4,
            pipeline_stages=2,
            block_size=128,
            use_fp8=False
        ),
        "medium": TPUPipelineConfig(
            num_micro_batches=8,
            pipeline_stages=4,
            block_size=256,
            use_fp8=False
        ),
        "large": TPUPipelineConfig(
            num_micro_batches=16,
            pipeline_stages=8,
            block_size=512,
            use_fp8=True
        ),
        "xl": TPUPipelineConfig(
            num_micro_batches=32,
            pipeline_stages=16,
            block_size=1024,
            use_fp8=True
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
        
    config = configs[model_size]
    
    # Adjust pipeline stages based on available cores
    config.pipeline_stages = min(config.pipeline_stages, num_cores)
    
    return config
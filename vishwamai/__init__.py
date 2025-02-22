"""
Vishwamai - Efficient T4-Optimized Machine Learning Model
"""
from .model import (
    # Core model and components
    VishwamaiModel,
    TransformerLayer,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    TokenEmbedding,
    PositionalEmbedding,
    AxialPositionalEmbedding,
    FeedForward,
    RMSNorm,
    QueryGenerator,
    
    # Model creation functions
    create_model,
    create_tiny_model,
    create_base_model,
    create_large_model,
    create_expert_model,
    create_parallel_model,
    create_from_pretrained,
    
    # Configuration
    ModelArgs,
    UnifiedConfig,
    ExpertConfig,
    ParallelConfig,
    AdvancedMLPConfig,
    AdvancedTransformerConfig,
    ParallelLinearConfig,
    
    # Preset configurations
    VISHWAMAI_TINY,
    VISHWAMAI_BASE,
    VISHWAMAI_LARGE,
    VISHWAMAI_EXPERT,
    VISHWAMAI_PARALLEL
)

from .utils import (
    enable_t4_optimizations,
    get_device_capabilities,
    get_memory_stats
)

from .config import (
    ModelArgs,
    PretrainedConfig,
    TrainingConfig
)

__version__ = "0.1.0"

# Convenience imports for common use cases
def create_default_model(**kwargs):
    """Create a default model with recommended settings"""
    return create_model(config=VISHWAMAI_BASE, **kwargs)

def create_efficient_model(**kwargs):
    """Create a model optimized for efficiency"""
    config = VISHWAMAI_BASE.update(
        use_flash_attention=True,
        use_kernel_optimizations=True,
        unified=UnifiedConfig(
            transformer=dict(
                fused_qkv=True,
                fused_mlp=True,
                use_memory_efficient_attention=True
            )
        )
    )
    return create_model(config=config, **kwargs)

def create_distributed_model(num_gpus: int, **kwargs):
    """Create a model for distributed training"""
    config = VISHWAMAI_PARALLEL.update(
        unified=UnifiedConfig(
            parallel=dict(
                tensor_parallel_size=num_gpus,
                sequence_parallel=True
            )
        )
    )
    return create_model(config=config, **kwargs)

__all__ = [
    # Core model
    "VishwamaiModel",
    "TransformerLayer",
    "MultiHeadAttention",
    "SelfAttention",
    "CrossAttention",
    "TokenEmbedding",
    "PositionalEmbedding",
    "AxialPositionalEmbedding",
    "FeedForward",
    "RMSNorm",
    "QueryGenerator",
    
    # Model creation
    "create_model",
    "create_tiny_model",
    "create_base_model",
    "create_large_model",
    "create_expert_model",
    "create_parallel_model",
    "create_from_pretrained",
    "create_default_model",
    "create_efficient_model",
    "create_distributed_model",
    
    # Configuration
    "ModelArgs",
    "UnifiedConfig",
    "ExpertConfig",
    "ParallelConfig",
    "AdvancedMLPConfig",
    "AdvancedTransformerConfig",
    "ParallelLinearConfig",
    "PretrainedConfig",
    "TrainingConfig",
    
    # Preset configurations
    "VISHWAMAI_TINY",
    "VISHWAMAI_BASE",
    "VISHWAMAI_LARGE",
    "VISHWAMAI_EXPERT",
    "VISHWAMAI_PARALLEL",
    
    # Utilities
    "enable_t4_optimizations",
    "get_device_capabilities",
    "get_memory_stats"
]

"""
Model components and factories for Vishwamai
"""
from .attention import (
    MultiHeadAttention,
    SelfAttention,
    CrossAttention
)
from .embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    AxialPositionalEmbedding,
    apply_rotary_embeddings
)
from .feed_forward import FeedForward
from .layer_norm import RMSNorm
from .vishwamai_model import (
    VishwamaiModel,
    TransformerLayer,
    QueryGenerator
)
from .factory import (
    ModelType,
    ModelArchitecture,
    ModelSize,
    get_model_config,
    create_model,
    load_pretrained,
    SIZE_CONFIGS,
    PRESET_CONFIGS
)

from ..config.model_args import (
    ModelArgs,
    PretrainedConfig,
    TrainingConfig
)

__all__ = [
    # Main model and components
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
    
    # Configuration
    "ModelArgs",
    "PretrainedConfig",
    "TrainingConfig",
    
    # Factory and types
    "ModelType",
    "ModelArchitecture",
    "ModelSize",
    "get_model_config",
    "create_model",
    "load_pretrained",
    
    # Utilities
    "apply_rotary_embeddings"
]

# Preset configurations for different model sizes
VISHWAMAI_TINY = ModelArgs(
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=6,
    max_position_embeddings=2048
)

VISHWAMAI_SMALL = ModelArgs(
    hidden_size=512,
    intermediate_size=2048,
    num_attention_heads=8,
    num_hidden_layers=12,
    max_position_embeddings=2048
)

VISHWAMAI_BASE = ModelArgs(
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    max_position_embeddings=4096
)

VISHWAMAI_LARGE = ModelArgs(
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    num_hidden_layers=24,
    max_position_embeddings=4096
)

VISHWAMAI_XL = ModelArgs(
    hidden_size=2048,
    intermediate_size=8192,
    num_attention_heads=32,
    num_hidden_layers=32,
    max_position_embeddings=8192
)

# Specialized model configurations
VISHWAMAI_EXPERT = ModelArgs(
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    num_hidden_layers=24,
    max_position_embeddings=4096,
    num_experts=8,
    num_experts_per_token=2,
    expert_capacity=32
)

VISHWAMAI_QUANTIZED = ModelArgs(
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    max_position_embeddings=4096,
    dtype="fp8",
    use_kernel_optimizations=True
)

VISHWAMAI_EFFICIENT = ModelArgs(
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    max_position_embeddings=4096,
    use_flash_attention=True,
    gradient_checkpointing=True,
    use_kernel_optimizations=True
)

# Extended sequence model configurations
VISHWAMAI_EXTENDED = ModelArgs(
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    num_hidden_layers=24,
    max_position_embeddings=16384,
    rope_scaling={
        "type": "dynamic",
        "factor": 2.0
    }
)

# Default configuration
DEFAULT_CONFIG = VISHWAMAI_BASE

# Version
__version__ = "0.1.0"

# Convenience functions
def create_tiny_model(**kwargs):
    """Create a tiny Vishwamai model"""
    config = VISHWAMAI_TINY.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_small_model(**kwargs):
    """Create a small Vishwamai model"""
    config = VISHWAMAI_SMALL.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_base_model(**kwargs):
    """Create a base Vishwamai model"""
    config = VISHWAMAI_BASE.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_large_model(**kwargs):
    """Create a large Vishwamai model"""
    config = VISHWAMAI_LARGE.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_xl_model(**kwargs):
    """Create an XL Vishwamai model"""
    config = VISHWAMAI_XL.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_expert_model(**kwargs):
    """Create an expert Vishwamai model"""
    config = VISHWAMAI_EXPERT.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_quantized_model(**kwargs):
    """Create a quantized Vishwamai model"""
    config = VISHWAMAI_QUANTIZED.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_efficient_model(**kwargs):
    """Create an efficient Vishwamai model"""
    config = VISHWAMAI_EFFICIENT.copy()
    config.update(**kwargs)
    return create_model(config=config)

def create_extended_model(**kwargs):
    """Create an extended sequence Vishwamai model"""
    config = VISHWAMAI_EXTENDED.copy()
    config.update(**kwargs)
    return create_model(config=config)

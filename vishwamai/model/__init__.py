"""VishwamAI model architecture and components."""

from .transformer import (
    VishwamAIModel,
    TransformerConfig,
    TransformerLayer,
    TransformerBlock,
    MoEMLABlock,
    LayerCache
)

from .moe import (
    ExpertLayer,
    ExpertRouter,
    GatingMechanism,
    MoELayer
)

from .mla import (
    MLABlock,
    MLALayerManager,
    MultiLayerAttention,
    MLAResidual
)

from .attention import (
    MultiHeadSelfAttention,
    CrossAttention,
    FlashAttention
)

from .embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    RotaryPositionalEmbedding
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Main model
    'VishwamAIModel',
    'TransformerConfig',
    'TransformerLayer',
    'TransformerBlock',
    'MoEMLABlock',
    'LayerCache',
    
    # MoE components
    'ExpertLayer',
    'ExpertRouter',
    'GatingMechanism',
    'MoELayer',
    
    # MLA components
    'MLABlock',
    'MLALayerManager',
    'MultiLayerAttention',
    'MLAResidual',
    
    # Attention mechanisms
    'MultiHeadSelfAttention',
    'CrossAttention',
    'FlashAttention',
    
    # Embedding layers
    'TokenEmbedding',
    'PositionalEncoding',
    'RotaryPositionalEmbedding',
    
    # Version
    '__version__'
]

# Device and backend configuration
import jax
import jax.numpy as jnp

# Set default dtype
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_dtype_bits", 32)

# Function to get default device
def get_device():
    """Get default device (TPU/GPU/CPU)."""
    return jax.devices()[0]

# Function to get model configuration
def get_config(config_path: str = None) -> TransformerConfig:
    """Get model configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        TransformerConfig instance
    """
    if config_path is None:
        # Return default configuration
        return TransformerConfig()
    return TransformerConfig.from_yaml(config_path)
    
# Function to create model instance
def create_model(config: TransformerConfig = None) -> VishwamAIModel:
    """Create VishwamAI model instance.
    
    Args:
        config: Optional model configuration
        
    Returns:
        VishwamAIModel instance
    """
    if config is None:
        config = get_config()
    return VishwamAIModel(config)

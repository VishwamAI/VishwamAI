"""Multi-Level Attention module components."""

from typing import Optional, Dict, Any, Type, Union

import torch
import torch.nn as nn

from .attention import MLAAttention
from .layer_manager import MLALayerManager
from .residual import MLAResidual, AdaptiveResidual
from .mla_block import MLABlock

def create_mla_block(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> MLABlock:
    """Create MLA block from configuration.
    
    Args:
        config: Dictionary containing MLA configuration
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        Configured MLA block
    """
    # Extract required params
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    
    # Create MLA block
    return MLABlock(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_attention_levels=config.get("num_attention_levels", 3),
        intermediate_size=config.get("intermediate_size", 4 * hidden_size),
        activation=config.get("activation", "gelu"),
        attention_dropout_prob=config.get("attention_dropout_prob", 0.1),
        hidden_dropout_prob=config.get("hidden_dropout_prob", 0.1),
        layer_norm_eps=config.get("layer_norm_eps", 1e-5),
        use_adaptive_residual=config.get("use_adaptive_residual", True),
        use_layer_scale=config.get("use_layer_scale", True),
        layer_scale_init=config.get("layer_scale_init", 0.1),
        level_scale_factor=config.get("level_scale_factor", 0.5),
        device=device,
        dtype=dtype,
    )

def create_nested_mla_blocks(
    config: Dict[str, Any],
    num_blocks: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> nn.ModuleList:
    """Create stack of MLA blocks.
    
    Args:
        config: Dictionary containing MLA configuration
        num_blocks: Number of MLA blocks to create
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        ModuleList containing MLA blocks
    """
    return nn.ModuleList([
        create_mla_block(config, device=device, dtype=dtype)
        for _ in range(num_blocks)
    ])

def get_residual_handler(
    name: str
) -> Type[Union[MLAResidual, AdaptiveResidual]]:
    """Get residual handler class by name.
    
    Args:
        name: Name of residual handler ('standard' or 'adaptive')
        
    Returns:
        Residual handler class
    """
    handlers = {
        "standard": MLAResidual,
        "adaptive": AdaptiveResidual
    }
    
    if name not in handlers:
        raise ValueError(
            f"Unknown residual handler: {name}. "
            f"Available options are: {list(handlers.keys())}"
        )
        
    return handlers[name]

def compute_attention_pattern(
    num_layers: int,
    num_levels: int,
    adaptive: bool = True
) -> Dict[str, Any]:
    """Compute MLA attention pattern configuration.
    
    Args:
        num_layers: Total number of layers
        num_levels: Number of attention levels
        adaptive: Whether to use adaptive computation
        
    Returns:
        Dictionary containing attention pattern configuration
    """
    # Compute layer distribution
    layer_manager = MLALayerManager(
        num_layers=num_layers,
        hidden_size=1,  # Dummy value
        num_attention_levels=num_levels,
        adaptive_computation=adaptive
    )
    
    layer_config = layer_manager.get_layer_config()
    
    return {
        "num_layers": num_layers,
        "num_levels": num_levels,
        "layer_counts": layer_config["layer_counts"],
        "cumulative_layers": layer_config["cumulative_layers"],
        "adaptive": adaptive
    }

def init_mla_weights(
    module: nn.Module,
    initializer_range: float = 0.02,
    layer_scale_init: float = 0.1
):
    """Initialize MLA module weights.
    
    Args:
        module: Module to initialize
        initializer_range: Normal distribution standard deviation
        layer_scale_init: Initial value for layer scale parameters
    """
    if isinstance(module, (MLAAttention, MLABlock)):
        # Initialize attention/MLA weights using normal distribution
        for param in module.parameters():
            if param.ndim > 1:
                nn.init.normal_(param, std=initializer_range)
                
    elif isinstance(module, (MLAResidual, AdaptiveResidual)):
        # Initialize layer scale parameters
        if module.layer_scale is not None:
            for param in module.layer_scale:
                nn.init.constant_(param, layer_scale_init)
                
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
        
    elif isinstance(module, nn.Linear):
        # Initialize linear layers
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

__all__ = [
    # Main components
    "MLABlock",
    "MLAAttention",
    "MLALayerManager",
    "MLAResidual",
    "AdaptiveResidual",
    
    # Factory functions
    "create_mla_block",
    "create_nested_mla_blocks",
    "get_residual_handler",
    
    # Utility functions  
    "compute_attention_pattern",
    "init_mla_weights",
]

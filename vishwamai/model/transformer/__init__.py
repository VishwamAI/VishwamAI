"""Transformer model components."""

from typing import Optional, Dict, Any, Type, Union

import torch
import torch.nn as nn

from .config import (
    TransformerConfig,
    MoEMLAConfig,
    create_config_from_file,
    SMALL_CONFIG,
    BASE_CONFIG,
    LARGE_CONFIG,
    XL_CONFIG
)
from .model import TransformerModel, MoEMLATransformerModel
from .block import TransformerBlock, ParallelTransformerBlock
from .layer import TransformerLayer, PreNormTransformerLayer, get_transformer_layer
from .moe_mla_block import MoEMLABlock

def create_transformer_model(
    config: Union[Dict[str, Any], TransformerConfig],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> TransformerModel:
    """Create transformer model from configuration.
    
    Args:
        config: Dictionary or configuration object
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        Transformer model instance
    """
    if isinstance(config, dict):
        # Convert dict to config object
        if config.get('model_type') == 'moe_mla_transformer':
            config = MoEMLAConfig(**config)
        else:
            config = TransformerConfig(**config)
            
    # Create appropriate model
    if isinstance(config, MoEMLAConfig):
        return MoEMLATransformerModel(
            config=config,
            device=device,
            dtype=dtype
        )
    else:
        return TransformerModel(
            config=config,
            device=device,
            dtype=dtype
        )

def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Union[Dict[str, Any], TransformerConfig]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True
) -> TransformerModel:
    """Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional model configuration
        device: Device to load model on
        dtype: Data type for parameters
        strict: Whether to strictly enforce loading all weights
        
    Returns:
        Loaded model instance
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint or use provided config
    if config is None:
        if 'config' not in checkpoint:
            raise ValueError("No configuration found in checkpoint")
        config = checkpoint['config']
        
    # Create model
    model = create_transformer_model(
        config=config,
        device=device,
        dtype=dtype
    )
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=strict
    )
    
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")
        
    return model

def get_pretrained_config(
    model_size: str,
    model_type: str = "transformer"
) -> TransformerConfig:
    """Get configuration for pretrained model size.
    
    Args:
        model_size: Size of model ('small', 'base', 'large', or 'xl')
        model_type: Type of model ('transformer' or 'moe_mla_transformer')
        
    Returns:
        Model configuration
    """
    configs = {
        "small": SMALL_CONFIG,
        "base": BASE_CONFIG,
        "large": LARGE_CONFIG,
        "xl": XL_CONFIG
    }
    
    if model_size not in configs:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available options are: {list(configs.keys())}"
        )
        
    config_dict = configs[model_size]
    
    if model_type == "moe_mla_transformer":
        return MoEMLAConfig(**config_dict)
    else:
        return TransformerConfig(**config_dict)

def get_block_class(name: str) -> Type[Union[TransformerBlock, ParallelTransformerBlock]]:
    """Get transformer block class by name.
    
    Args:
        name: Block class name ('standard' or 'parallel')
        
    Returns:
        Block class
    """
    blocks = {
        "standard": TransformerBlock,
        "parallel": ParallelTransformerBlock
    }
    
    if name not in blocks:
        raise ValueError(
            f"Unknown block type: {name}. "
            f"Available options are: {list(blocks.keys())}"
        )
        
    return blocks[name]

def get_layer_class(name: str) -> Type[Union[TransformerLayer, PreNormTransformerLayer]]:
    """Get transformer layer class by name.
    
    Args:
        name: Layer class name ('standard' or 'pre_norm')
        
    Returns:
        Layer class
    """
    layers = {
        "standard": TransformerLayer,
        "pre_norm": PreNormTransformerLayer
    }
    
    if name not in layers:
        raise ValueError(
            f"Unknown layer type: {name}. "
            f"Available options are: {list(layers.keys())}"
        )
        
    return layers[name]

__all__ = [
    # Main components
    "TransformerModel",
    "MoEMLATransformerModel",
    "TransformerConfig",
    "MoEMLAConfig",
    "TransformerBlock",
    "ParallelTransformerBlock",
    "TransformerLayer",
    "PreNormTransformerLayer",
    "MoEMLABlock",
    
    # Factory functions
    "create_transformer_model",
    "load_model_from_checkpoint",
    "create_config_from_file",
    "get_pretrained_config",
    "get_block_class",
    "get_layer_class",
    "get_transformer_layer",
    
    # Configurations
    "SMALL_CONFIG",
    "BASE_CONFIG", 
    "LARGE_CONFIG",
    "XL_CONFIG"
]

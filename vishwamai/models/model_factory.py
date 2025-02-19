"""
Model factory for creating and loading VishwamAI models.
"""

import torch
from typing import Optional, Dict, Any, Union
import os
import json

from vishwamai.utils.config import ModelConfig
from vishwamai.models.tokenizer import Tokenizer
from vishwamai.models.Transformer import Transformer
from vishwamai.utils.fp8_cast_bf16 import convert_to_fp8, convert_to_bf16

def create_model(
    config: Union[ModelConfig, Dict[str, Any]],
    pretrained_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    use_cache: bool = True
) -> Transformer:
    """
    Create a new VishwamAI model.
    
    Args:
        config: Model configuration
        pretrained_path: Optional path to pretrained weights
        device: Device to place model on
        dtype: Data type for model parameters
        use_cache: Whether to use cached key/value states
        
    Returns:
        Initialized model
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create model
    model = Transformer(config)
    model.to(device=device, dtype=dtype)
    
    # Load pretrained weights if provided
    if pretrained_path is not None:
        load_pretrained_weights(model, pretrained_path)
        
    model.eval() if not model.training else model.train()
    return model

def load_pretrained_weights(
    model: Transformer,
    pretrained_path: str,
    strict: bool = True
) -> None:
    """
    Load pretrained weights into model.
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights
        strict: Whether to strictly enforce that the keys match
    """
    if not os.path.exists(pretrained_path):
        raise ValueError(f"Pretrained weights not found at: {pretrained_path}")
        
    # Load weights
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different formats
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
    # Load weights into model
    model.load_state_dict(state_dict, strict=strict)
    
def save_model(
    model: Transformer,
    save_path: str,
    tokenizer: Optional[Tokenizer] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model weights and configuration.
    
    Args:
        model: Model to save
        save_path: Path to save to
        tokenizer: Optional tokenizer to save
        config: Optional configuration to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model weights
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': config or model.config.__dict__
        },
        save_path
    )
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer_path = os.path.join(
            os.path.dirname(save_path),
            'tokenizer.json'
        )
        tokenizer.save_pretrained(tokenizer_path)
        
def convert_precision(
    model: Transformer,
    precision: str = 'fp16',
    device: Optional[torch.device] = None
) -> Transformer:
    """
    Convert model precision.
    
    Args:
        model: Model to convert
        precision: Target precision (fp16, bf16, or fp8)
        device: Optional target device
        
    Returns:
        Converted model
    """
    if device is not None:
        model = model.to(device)
        
    if precision == 'fp16':
        model = model.half()
    elif precision == 'bf16':
        model = convert_to_bf16(model)
    elif precision == 'fp8':
        model = convert_to_fp8(model)
    else:
        raise ValueError(f"Unsupported precision: {precision}")
        
    return model

def get_embedding_size(model: Transformer) -> int:
    """Get embedding size of model."""
    return model.config.hidden_size

def get_num_parameters(model: Transformer) -> int:
    """Get number of parameters in model."""
    return sum(p.numel() for p in model.parameters())

def get_parameter_size(model: Transformer) -> float:
    """Get size of model parameters in GB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

def model_to_half(model: Transformer) -> Transformer:
    """Convert model to half precision."""
    for param in model.parameters():
        param.data = param.data.half()
    return model

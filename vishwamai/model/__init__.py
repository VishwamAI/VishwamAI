"""Model components for Vishwamai."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List, Tuple

# Import core model components
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.model.transformer.config import VishwamaiConfig
from vishwamai.model.transformer.block import TransformerBlock
from vishwamai.model.transformer.layer import TransformerLayer
from vishwamai.model.transformer.moe_mla_block import MoEMLABlock

# Import attention mechanisms
from vishwamai.model.attention import (
    MultiHeadAttention,
    FlashAttention,
    CrossAttention
)

# Import MoE components
from vishwamai.model.moe import (
    ExpertLayer,
    Router,
    MoELayer,
    GatingNetwork
)

# Import MLA components
from vishwamai.model.mla import (
    MultiLevelAttention,
    MLABlock,
    LevelManager,
    ResidualConnection
)

# Import embedding layers
from vishwamai.model.embeddings import (
    TokenEmbedding,
    PositionalEncoding
)

# Import initialization utilities
from vishwamai.model.initialization import (
    initialize_weights,
    initialize_experts,
    initialize_router
)

def create_model(
    config: Union[Dict, VishwamaiConfig],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> VishwamaiModel:
    """Create a Vishwamai model instance.
    
    Args:
        config: Model configuration dictionary or VishwamaiConfig instance
        device: Target device for the model
        dtype: Data type for model parameters
        
    Returns:
        VishwamaiModel: Initialized model instance
    """
    if isinstance(config, dict):
        config = VishwamaiConfig(**config)
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.float16
        
    model = VishwamaiModel(config)
    model.to(device=device, dtype=dtype)
    
    return model

def load_pretrained(
    path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
) -> VishwamaiModel:
    """Load a pretrained Vishwamai model.
    
    Args:
        path: Path to pretrained model weights
        device: Target device for loaded model
        dtype: Data type for model parameters
        **kwargs: Additional arguments passed to model loading
        
    Returns:
        VishwamaiModel: Loaded model instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.float16
        
    # Load model config and weights
    config = VishwamaiConfig.from_pretrained(path)
    model = create_model(config, device=device, dtype=dtype)
    
    # Load state dict
    state_dict = torch.load(path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    
    return model

def get_model_size(model: nn.Module) -> Tuple[int, int, float]:
    """Get model size statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple containing:
        - Total number of parameters
        - Number of trainable parameters  
        - Model size in GB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    
    return total_params, trainable_params, size_gb

__all__ = [
    # Model creation
    "create_model",
    "load_pretrained",
    "get_model_size",
    
    # Core components
    "VishwamaiModel",
    "VishwamaiConfig",
    "TransformerBlock",
    "TransformerLayer",
    "MoEMLABlock",
    
    # Attention
    "MultiHeadAttention", 
    "FlashAttention",
    "CrossAttention",
    
    # MoE
    "ExpertLayer",
    "Router",
    "MoELayer",
    "GatingNetwork",
    
    # MLA
    "MultiLevelAttention",
    "MLABlock", 
    "LevelManager",
    "ResidualConnection",
    
    # Embeddings
    "TokenEmbedding",
    "PositionalEncoding",
    
    # Initialization
    "initialize_weights",
    "initialize_experts",
    "initialize_router",
]

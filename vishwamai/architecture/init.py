"""
Memory-Efficient Model Initialization
=================================

This module provides utilities for memory-efficient initialization of VishwamAI models.
"""

import os
import math
import torch
import torch.nn as nn
import torch.cuda as cuda
from typing import Optional, Dict, Any, Tuple
from .config import VishwamaiConfig
from .transformer import VishwamaiModel

def get_gpu_memory() -> Tuple[int, int]:
    """Get total and free GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0, 0
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    free = total - (reserved + allocated)
    return total, free

def estimate_model_size(config: VishwamaiConfig) -> int:
    """Estimate model memory requirements in bytes."""
    vocab_size = config.vocab_size
    hidden_size = config.dim
    num_layers = config.depth
    sequence_length = config.max_seq_length
    
    # Estimate parameters size
    params_size = (
        (vocab_size * hidden_size) +  # Embeddings
        (num_layers * (
            (4 * hidden_size * hidden_size) +  # Attention matrices
            (4 * hidden_size)  # Layer norms
        ))
    ) * 2  # Convert to bytes (assuming fp16)
    
    # Estimate activation size
    activation_size = (
        (sequence_length * hidden_size * 4) +  # Key/Query/Value/Output
        (sequence_length * sequence_length)  # Attention weights
    ) * 4  # Convert to bytes (assuming fp32 activations)
    
    return params_size + activation_size

def validate_config(config: VishwamaiConfig) -> VishwamaiConfig:
    """Validate and adjust config based on available memory."""
    total_mem, free_mem = get_gpu_memory()
    estimated_size = estimate_model_size(config)
    
    if total_mem > 0 and estimated_size > free_mem * 0.9:  # Leave 10% buffer
        # Adjust model size to fit in memory
        reduction_factor = math.sqrt(free_mem * 0.9 / estimated_size)
        config.dim = int(config.dim * reduction_factor)
        config.max_seq_length = int(config.max_seq_length * reduction_factor)
    
    return config

def init_weights(module: nn.Module, dtype: Optional[torch.dtype] = None):
    """Initialize weights for transformer modules with optional dtype."""
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    if isinstance(module, nn.Linear):
        module.weight.data = module.weight.data.to(dtype)
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data = module.bias.data.to(dtype)
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        module.weight.data = module.weight.data.to(dtype)
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data = module.weight.data.to(dtype)
        module.bias.data = module.bias.data.to(dtype)
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

def init_model(
    config: Optional[VishwamaiConfig] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    memory_efficient: bool = True
) -> VishwamaiModel:
    """
    Initialize a new VishwamAI model.
    
    Args:
        config: Model configuration. If None, uses default config.
        device: Device to place model on. If None, uses CUDA if available.
        
    Returns:
        Initialized VishwamAI model
    """
    # Initialize with memory constraints in mind
    if config is None:
        config = VishwamaiConfig()
    
    if memory_efficient:
        config = validate_config(config)
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Initialize model with mixed precision
    with torch.cuda.amp.autocast(enabled=True):
        model = VishwamaiModel(config)
        model.apply(lambda m: init_weights(m, dtype))
        
    # Move to device in chunks if needed
    if memory_efficient and device.type == "cuda":
        for name, param in model.named_parameters():
            param.data = param.data.to(device)
            torch.cuda.empty_cache()
    else:
        model = model.to(device)
    
    return model

def load_checkpoint(
    path: str,
    config: Optional[VishwamaiConfig] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    memory_efficient: bool = True
) -> VishwamaiModel:
    """
    Load a model from checkpoint.
    
    Args:
        path: Path to checkpoint file
        config: Model configuration. If None, loads from checkpoint
        device: Device to place model on
        strict: Whether to strictly enforce that the keys in state_dict match
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory-efficient checkpoint loading
    if memory_efficient and device.type == "cuda":
        checkpoint = torch.load(path, map_location="cpu")
    else:
        checkpoint = torch.load(path, map_location=device)
    
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]
    elif config is None:
        config = VishwamaiConfig()
        
    model = init_model(config, device=torch.device("cpu") if memory_efficient else device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict in chunks for memory efficiency
    if memory_efficient and device.type == "cuda":
        for key, value in state_dict.items():
            if key in model.state_dict():
                model.state_dict()[key].copy_(value.to(device))
                del value
                torch.cuda.empty_cache()
    else:
        model.load_state_dict(state_dict, strict=strict)

    if memory_efficient and device.type == "cuda":
        model = model.to(device)
    
    return model

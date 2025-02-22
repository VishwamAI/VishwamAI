"""
Model factory and preset configurations for Vishwamai
"""
from typing import Optional, Dict, Any, Union
from enum import Enum
import warnings

import torch
import torch.distributed as dist

from ..config import (
    ModelArgs,
    UnifiedConfig,
    VISHWAMAI_TINY,
    VISHWAMAI_BASE,
    VISHWAMAI_LARGE,
    VISHWAMAI_EXPERT,
    VISHWAMAI_PARALLEL
)
from .vishwamai_model import VishwamaiModel  # Forward import to avoid circular dependencies
from ..utils.t4_utils import get_device_capabilities

class ModelType(str, Enum):
    """Supported model types"""
    BASE = "base"
    TRANSFORMER = "transformer"
    ENCODER = "encoder"
    DECODER = "decoder"
    ENCODER_DECODER = "encoder_decoder"
    EXPERT = "expert"
    PARALLEL = "parallel"

def initialize_distributed(config: ModelArgs) -> None:
    """Initialize distributed training if needed"""
    if config.unified.parallel.tensor_parallel_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Set device based on local rank
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

def create_model(
    config: Optional[Union[ModelArgs, Dict[str, Any]]] = None,
    model_type: Optional[ModelType] = ModelType.BASE,
    **kwargs
) -> "VishwamaiModel":
    """
    Create a model instance with the specified configuration.
    
    Args:
        config: Model configuration (ModelArgs object or dict)
        model_type: Type of model to create
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured VishwamaiModel instance
    """
    # Handle configuration
    if config is None:
        config = VISHWAMAI_BASE
    elif isinstance(config, dict):
        config = ModelArgs.from_dict(config)
        
    # Apply overrides
    if kwargs:
        config = config.update(**kwargs)
        
    # Initialize distributed training if needed
    if config.unified.parallel.tensor_parallel_size > 1:
        initialize_distributed(config)
            
    # Create model
    if model_type == ModelType.EXPERT:
        if config.num_experts == 0:
            warnings.warn("Creating expert model but num_experts is 0, using default expert config")
            config = VISHWAMAI_EXPERT
    elif model_type == ModelType.PARALLEL:
        if config.unified.parallel.tensor_parallel_size == 1:
            warnings.warn("Creating parallel model but tensor_parallel_size is 1, using default parallel config")
            config = VISHWAMAI_PARALLEL
            
    return VishwamaiModel(config)

def create_expert_model(**kwargs) -> "VishwamaiModel":
    """Create a Mixture of Experts model"""
    return create_model(
        config=VISHWAMAI_EXPERT,
        model_type=ModelType.EXPERT,
        **kwargs
    )

def create_parallel_model(**kwargs) -> "VishwamaiModel":
    """Create a model with tensor parallelism"""
    return create_model(
        config=VISHWAMAI_PARALLEL,
        model_type=ModelType.PARALLEL,
        **kwargs
    )

def create_from_pretrained(
    model_name: str,
    precision: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> "VishwamaiModel":
    """
    Load a pretrained model with optional precision and device settings.
    
    Args:
        model_name: Name or path of pretrained model
        precision: Optional precision mode ("fp16", "fp32", "bf16")
        device: Optional device to load model on
        **kwargs: Additional configuration overrides
        
    Returns:
        Loaded model instance
    """
    # Get device capabilities
    capabilities = get_device_capabilities()
    
    # Determine optimal precision if not specified
    if precision is None:
        if capabilities["bfloat16"]:
            precision = "bf16"
        elif capabilities["amp"]:
            precision = "fp16"
        else:
            precision = "fp32"
            
    # Load configuration
    config = ModelArgs.from_pretrained(model_name)
    
    # Update configuration
    config.update(
        dtype=precision,
        use_mixed_precision=(precision != "fp32"),
        **kwargs
    )
    
    # Create model
    model = create_model(config=config)
    
    # Load weights
    # TODO: Implement weight loading from registry/storage
    raise NotImplementedError("Loading pretrained weights not yet implemented")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        
    return model

# Default model creation functions
def create_tiny_model(**kwargs) -> "VishwamaiModel":
    """Create a tiny model"""
    return create_model(config=VISHWAMAI_TINY, **kwargs)

def create_base_model(**kwargs) -> "VishwamaiModel":
    """Create a base model"""
    return create_model(config=VISHWAMAI_BASE, **kwargs)

def create_large_model(**kwargs) -> "VishwamaiModel":
    """Create a large model"""
    return create_model(config=VISHWAMAI_LARGE, **kwargs)

# Configuration checking utilities
def check_config_compatibility(config: ModelArgs) -> None:
    """Check if configuration is compatible with available hardware"""
    capabilities = get_device_capabilities()
    
    # Check precision compatibility
    if config.dtype == "bf16" and not capabilities["bfloat16"]:
        raise ValueError("BF16 precision requested but not supported by hardware")
        
    # Check Flash Attention compatibility
    if config.use_flash_attention and not capabilities["flash_attention"]:
        warnings.warn("Flash Attention requested but not available, falling back to standard attention")
        config.use_flash_attention = False
        
    # Check parallel compatibility
    if config.unified.parallel.tensor_parallel_size > torch.cuda.device_count():
        raise ValueError(f"Requested {config.unified.parallel.tensor_parallel_size} GPUs but only {torch.cuda.device_count()} available")

def merge_configs(*configs: ModelArgs) -> ModelArgs:
    """Merge multiple configurations, later configs override earlier ones"""
    base_config = configs[0]
    for config in configs[1:]:
        base_config = base_config.update(**config.to_dict())
    return base_config

# Version
__version__ = "0.1.0"

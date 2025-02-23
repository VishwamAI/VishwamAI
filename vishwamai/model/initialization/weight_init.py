"""General weight initialization functions."""
from typing import Union
import math
import torch
import torch.nn as nn

def init_normal(module: nn.Module, mean: float = 0.0, std: float = 1.0) -> None:
    """Initialize weights using normal distribution.
    
    Args:
        module: PyTorch module
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def init_uniform(module: nn.Module, a: float = 0.0, b: float = 1.0) -> None:
    """Initialize weights using uniform distribution.
    
    Args:
        module: PyTorch module
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a=a, b=b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def init_xavier_normal(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights using Xavier normal initialization.
    
    Args:
        module: PyTorch module
        gain: Optional scaling factor
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def init_xavier_uniform(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights using Xavier uniform initialization.
    
    Args:
        module: PyTorch module
        gain: Optional scaling factor
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def init_kaiming_normal(
    module: nn.Module,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    """Initialize weights using Kaiming normal initialization.
    
    Args:
        module: PyTorch module
        a: Negative slope for LeakyReLU
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity after this layer
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.kaiming_normal_(
            module.weight,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity
        )
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def init_kaiming_uniform(
    module: nn.Module,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    """Initialize weights using Kaiming uniform initialization.
    
    Args:
        module: PyTorch module
        a: Negative slope for LeakyReLU
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity after this layer
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.kaiming_uniform_(
            module.weight,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity
        )
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)

def scaled_init(
    module: nn.Module,
    std: float = 0.02,
    bias_const: float = 0.0,
    scale_factor: Union[float, str] = 1.0
) -> None:
    """Initialize weights with scaled normal distribution.
    
    This is commonly used in transformer architectures where the initialization
    scale needs to be adjusted based on number of layers, heads etc.
    
    Args:
        module: PyTorch module
        std: Base standard deviation
        bias_const: Constant for bias initialization
        scale_factor: Factor to scale std by. If 'auto', uses 1/sqrt(2*num_layers)
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if isinstance(scale_factor, str) and scale_factor == 'auto':
            # Assuming module is part of a transformer with this attribute
            if hasattr(module, 'num_layers'):
                scale = 1.0 / math.sqrt(2.0 * module.num_layers)
            else:
                scale = 1.0
        else:
            scale = float(scale_factor)
            
        nn.init.normal_(module.weight, mean=0.0, std=std * scale)
        
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias_const)

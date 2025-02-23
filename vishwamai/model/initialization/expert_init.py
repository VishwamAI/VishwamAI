"""Expert weight initialization functions for MoE layers."""
from typing import Optional, Union
import math
import torch
import torch.nn as nn

from .weight_init import scaled_init

def init_expert_weights(
    expert: nn.Module,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    num_experts: int = 1,
    capacity_factor: float = 1.0,
    expert_scale: Optional[float] = None
) -> None:
    """Initialize expert module weights.
    
    Args:
        expert: Expert module to initialize
        init_method: Initialization method ['scaled_normal', 'kaiming', 'xavier']
        init_std: Standard deviation for normal initialization
        num_experts: Number of experts in MoE layer
        capacity_factor: Expert capacity multiplier
        expert_scale: Optional explicit scaling factor for expert weights
    """
    # Calculate expert-specific scaling based on capacity and number of experts
    if expert_scale is None:
        # Scale based on number of experts and capacity
        expert_scale = math.sqrt(capacity_factor / num_experts)
    
    if init_method == 'scaled_normal':
        # Scale normal initialization by expert-specific factor
        for module in expert.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                scaled_init(
                    module,
                    std=init_std,
                    scale_factor=expert_scale
                )
                
    elif init_method == 'kaiming':
        # Kaiming initialization with expert scaling
        for module in expert.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
                    gain = nn.init.calculate_gain('relu')
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        module.weight.normal_(0, std * expert_scale)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    elif init_method == 'xavier':
        # Xavier initialization with expert scaling
        for module in expert.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    std = expert_scale * math.sqrt(2.0 / (fan_in + fan_out))
                    with torch.no_grad():
                        module.weight.normal_(0, std)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")
        
def init_expert_biases(
    expert: nn.Module,
    bias_init: Union[float, str] = 0.0
) -> None:
    """Initialize expert module biases.
    
    Args:
        expert: Expert module to initialize
        bias_init: Bias initialization value or method
            - float: Use constant initialization
            - 'zero': Initialize to zeros
            - 'normal': Initialize from N(0, 0.02)
    """
    for module in expert.modules():
        if hasattr(module, 'bias') and module.bias is not None:
            if isinstance(bias_init, (int, float)):
                nn.init.constant_(module.bias, bias_init)
            elif bias_init == 'zero':
                nn.init.zeros_(module.bias)
            elif bias_init == 'normal':
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown bias initialization: {bias_init}")
                
def reset_failed_experts(
    expert: nn.Module,
    expert_idx: int,
    init_method: str = 'scaled_normal',
    init_std: float = 0.02,
    num_experts: int = 1,
    capacity_factor: float = 1.0
) -> None:
    """Re-initialize a failed expert's weights.
    
    This is used when an expert is detected to have failed (e.g., due to
    consistently receiving no tokens or producing poor outputs).
    
    Args:
        expert: Expert module to re-initialize
        expert_idx: Index of expert being reset
        init_method: Initialization method
        init_std: Standard deviation for normal initialization
        num_experts: Total number of experts
        capacity_factor: Expert capacity multiplier
    """
    # Calculate new scale factor - may want different scaling for resets
    reset_scale = math.sqrt(capacity_factor / num_experts)
    
    # Re-initialize weights
    init_expert_weights(
        expert,
        init_method=init_method,
        init_std=init_std,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        expert_scale=reset_scale
    )
    
    # Re-initialize biases
    init_expert_biases(expert, bias_init='zero')

"""Initialization utilities for expert networks in MoE layers."""

import math
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

from .weight_init import (
    normal_init_,
    xavier_uniform_init_,
    kaiming_uniform_init_,
    scaled_init_,
    _calculate_fan_in_and_fan_out
)

def initialize_expert_weights(
    expert_layer: nn.Module,
    num_experts: int,
    hidden_size: int,
    expert_size: int,
    method: str = "normal",
    **kwargs
) -> None:
    """Initialize expert network weights.
    
    Args:
        expert_layer: Expert layer module
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        expert_size: Expert FFN dimension size
        method: Weight initialization method
        **kwargs: Additional initialization arguments
    """
    # Up-projection weights (hidden -> expert)
    if hasattr(expert_layer, "w1"):
        if method == "normal":
            std = kwargs.get("std", 0.02)
            normal_init_(expert_layer.w1, std=std)
        elif method == "xavier_uniform":
            xavier_uniform_init_(expert_layer.w1)
        elif method == "kaiming_uniform":
            kaiming_uniform_init_(expert_layer.w1)
        elif method == "scaled":
            scale = kwargs.get("scale", 1.0)
            scaled_init_(expert_layer.w1, scale=scale)
            
    # Down-projection weights (expert -> hidden)
    if hasattr(expert_layer, "w2"):
        if method == "normal":
            std = kwargs.get("std", 0.02 / math.sqrt(2 * num_experts))
            normal_init_(expert_layer.w2, std=std)
        elif method == "xavier_uniform":
            gain = 1.0 / math.sqrt(2 * num_experts)
            xavier_uniform_init_(expert_layer.w2, gain=gain)
        elif method == "kaiming_uniform":
            kaiming_uniform_init_(expert_layer.w2)
        elif method == "scaled":
            scale = kwargs.get("scale", 1.0) / (2 * num_experts)
            scaled_init_(expert_layer.w2, scale=scale)

def initialize_expert_biases(
    expert_layer: nn.Module,
    init_val: float = 0.0
) -> None:
    """Initialize expert network biases.
    
    Args:
        expert_layer: Expert layer module
        init_val: Initial bias value
    """
    if hasattr(expert_layer, "b1") and expert_layer.b1 is not None:
        nn.init.constant_(expert_layer.b1, init_val)
        
    if hasattr(expert_layer, "b2") and expert_layer.b2 is not None:
        nn.init.constant_(expert_layer.b2, init_val)

def initialize_expert_layer_norm(
    layer_norm: nn.LayerNorm,
    eps: float = 1e-5
) -> None:
    """Initialize expert layer normalization.
    
    Args:
        layer_norm: LayerNorm module
        eps: Small constant for numerical stability
    """
    if hasattr(layer_norm, "weight"):
        nn.init.ones_(layer_norm.weight)
        
    if hasattr(layer_norm, "bias"):
        nn.init.zeros_(layer_norm.bias)
        
    layer_norm.eps = eps

def initialize_expert_gate_weights(
    gate_weights: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    init_method: str = "normal",
    **kwargs
) -> None:
    """Initialize expert gating weights.
    
    Args:
        gate_weights: Gating weight tensor
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        init_method: Weight initialization method
        **kwargs: Additional initialization arguments
    """
    if init_method == "normal":
        std = kwargs.get("std", 0.02)
        with torch.no_grad():
            gate_weights.normal_(0.0, std)
    elif init_method == "uniform":
        a = kwargs.get("a", -0.05)
        b = kwargs.get("b", 0.05)
        with torch.no_grad():
            gate_weights.uniform_(a, b)
    elif init_method == "zero":
        with torch.no_grad():
            gate_weights.zero_()
    else:
        raise ValueError(f"Unknown gate initialization method: {init_method}")

def add_jitter_to_expert_weights(
    expert_weights: torch.Tensor,
    jitter_noise: float = 0.1
) -> None:
    """Add random jitter to expert weights.
    
    Args:
        expert_weights: Expert weight tensor
        jitter_noise: Scale of noise to add
    """
    if jitter_noise > 0:
        with torch.no_grad():
            noise = torch.randn_like(expert_weights) * jitter_noise
            expert_weights.add_(noise)

def initialize_expert_dropout(
    expert_layer: nn.Module,
    dropout_prob: float = 0.1
) -> None:
    """Initialize expert dropout layers.
    
    Args:
        expert_layer: Expert layer module
        dropout_prob: Dropout probability
    """
    if hasattr(expert_layer, "dropout1"):
        expert_layer.dropout1.p = dropout_prob
        
    if hasattr(expert_layer, "dropout2"):
        expert_layer.dropout2.p = dropout_prob

def initialize_expert_activation(
    expert_layer: nn.Module,
    activation_type: str = "gelu"
) -> None:
    """Initialize expert activation functions.
    
    Args:
        expert_layer: Expert layer module
        activation_type: Type of activation function
    """
    if activation_type == "gelu":
        expert_layer.activation = nn.GELU()
    elif activation_type == "relu":
        expert_layer.activation = nn.ReLU()
    elif activation_type == "swish":
        expert_layer.activation = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

class ExpertInitializer:
    """Helper class for expert initialization."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        expert_size: int,
        init_method: str = "normal",
        eps: float = 1e-5,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        """Initialize expert initializer.
        
        Args:
            num_experts: Number of experts
            hidden_size: Hidden dimension size
            expert_size: Expert FFN dimension size
            init_method: Weight initialization method
            eps: LayerNorm epsilon
            dropout: Dropout probability
            activation: Activation function type
            **kwargs: Additional initialization arguments
        """
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.init_method = init_method
        self.eps = eps
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs
        
    def __call__(self, expert_layer: nn.Module) -> None:
        """Initialize expert layer components.
        
        Args:
            expert_layer: Expert layer to initialize
        """
        # Initialize weights
        initialize_expert_weights(
            expert_layer,
            self.num_experts,
            self.hidden_size,
            self.expert_size,
            self.init_method,
            **self.kwargs
        )
        
        # Initialize biases
        initialize_expert_biases(expert_layer)
        
        # Initialize layer norm
        if hasattr(expert_layer, "layer_norm"):
            initialize_expert_layer_norm(expert_layer.layer_norm, self.eps)
            
        # Initialize dropout
        initialize_expert_dropout(expert_layer, self.dropout)
        
        # Initialize activation
        initialize_expert_activation(expert_layer, self.activation)

__all__ = [
    "initialize_expert_weights",
    "initialize_expert_biases",
    "initialize_expert_layer_norm",
    "initialize_expert_gate_weights",
    "add_jitter_to_expert_weights",
    "initialize_expert_dropout",
    "initialize_expert_activation",
    "ExpertInitializer",
]

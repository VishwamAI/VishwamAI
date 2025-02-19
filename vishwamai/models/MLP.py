"""
Multi-Layer Perceptron (MLP) implementation for VishwamAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from vishwamai.models.base_layers import Linear, LayerNorm
from vishwamai.utils.parallel import model_parallel_forward

class MLP(nn.Module):
    """
    Multi-Layer Perceptron module with advanced features and optimizations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm: bool = False,
        use_model_parallel: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.use_model_parallel = use_model_parallel
        
        # Up projection
        self.fc1 = Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            use_model_parallel=use_model_parallel
        )
        
        # Down projection
        self.fc2 = Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            use_model_parallel=use_model_parallel
        )
        
        # Optional layer normalization
        self.layer_norm = LayerNorm(hidden_size) if layer_norm else None
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights for linear layers."""
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def get_activation(self, activation_string: str):
        """Get activation function from string name."""
        activation_functions = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "swish": lambda x: x * torch.sigmoid(x),
            "mish": lambda x: x * torch.tanh(F.softplus(x))
        }
        return activation_functions.get(activation_string.lower(), F.gelu)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for MLP.
        
        Args:
            hidden_states: Input tensor
            residual: Optional residual tensor for skip connections
            
        Returns:
            Output tensor after MLP transformation
        """
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        # Up projection with activation
        if self.use_model_parallel:
            intermediate = model_parallel_forward(
                self.fc1,
                hidden_states
            )
        else:
            intermediate = self.fc1(hidden_states)
            
        intermediate = self.get_activation(self.activation)(intermediate)
        intermediate = self.dropout(intermediate)
        
        # Down projection
        if self.use_model_parallel:
            output = model_parallel_forward(
                self.fc2,
                intermediate
            )
        else:
            output = self.fc2(intermediate)
            
        output = self.dropout(output)
        
        # Add residual connection if provided
        if residual is not None:
            output = output + residual
            
        return output
        
    def extra_repr(self) -> str:
        """String representation of module."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"activation={self.activation}, "
            f"use_model_parallel={self.use_model_parallel}"
        )

class GatedMLP(MLP):
    """
    Gated variation of MLP with additional control mechanism.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm: bool = False,
        use_model_parallel: bool = False
    ):
        super().__init__(
            hidden_size,
            intermediate_size,
            activation,
            dropout,
            bias,
            layer_norm,
            use_model_parallel
        )
        
        # Add gating mechanism
        self.gate = Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            use_model_parallel=use_model_parallel
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with gating mechanism."""
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        # Compute gate values
        if self.use_model_parallel:
            gate_vals = model_parallel_forward(
                self.gate,
                hidden_states
            )
        else:
            gate_vals = self.gate(hidden_states)
            
        gate_vals = torch.sigmoid(gate_vals)
        
        # Up projection with activation and gating
        if self.use_model_parallel:
            intermediate = model_parallel_forward(
                self.fc1,
                hidden_states
            )
        else:
            intermediate = self.fc1(hidden_states)
            
        intermediate = self.get_activation(self.activation)(intermediate)
        intermediate = intermediate * gate_vals
        intermediate = self.dropout(intermediate)
        
        # Down projection
        if self.use_model_parallel:
            output = model_parallel_forward(
                self.fc2,
                intermediate
            )
        else:
            output = self.fc2(intermediate)
            
        output = self.dropout(output)
        
        # Add residual connection if provided
        if residual is not None:
            output = output + residual
            
        return output

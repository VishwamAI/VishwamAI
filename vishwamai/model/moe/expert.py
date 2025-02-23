"""Expert network implementation for MoE layers."""

from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initialization import (
    initialize_expert_weights,
    initialize_expert_biases,
    initialize_expert_layer_norm
)

class ExpertNetwork(nn.Module):
    """Expert network for MoE layer."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, Callable] = "gelu",
        dropout_prob: float = 0.1,
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize expert network.
        
        Args:
            hidden_size: Size of hidden dimension
            intermediate_size: Size of intermediate FFN dimension
            activation: Activation function or name
            dropout_prob: Dropout probability
            use_layer_norm: Whether to use layer normalization
            layer_norm_eps: Layer norm epsilon
            bias: Whether to use bias in linear layers
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                hidden_size,
                eps=layer_norm_eps,
                **factory_kwargs
            )
        else:
            self.layer_norm = None
            
        # Up projection
        self.fc1 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
            **factory_kwargs
        )
        
        # Down projection
        self.fc2 = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            **factory_kwargs
        )
        
        # Activation
        if isinstance(activation, str):
            self.activation = self._get_activation(activation)
        else:
            self.activation = activation
            
        # Dropout
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name.
        
        Args:
            name: Name of activation function
            
        Returns:
            Activation function
        """
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "swish": F.silu,  # Alias for SiLU
        }
        
        if name not in activations:
            raise ValueError(
                f"Unknown activation function: {name}. "
                f"Available options are: {list(activations.keys())}"
            )
            
        return activations[name]
        
    def _init_weights(self):
        """Initialize expert weights."""
        # Initialize up projection
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc1.bias, -bound, bound)
            
        # Initialize down projection
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if self.fc2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc2.bias, -bound, bound)
            
        # Initialize layer norm
        if self.layer_norm is not None:
            nn.init.ones_(self.layer_norm.weight)
            nn.init.zeros_(self.layer_norm.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process input through expert network.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        residual = hidden_states
        
        # Apply layer normalization if available
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        # Up projection with dropout
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        
        # Down projection with dropout
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"activation={self.activation.__name__ if hasattr(self.activation, '__name__') else self.activation}, "
            f"layer_norm={self.layer_norm is not None}"
        )

class ParallelExpertNetwork(ExpertNetwork):
    """Expert network optimized for parallel computation."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through expert network with optional masking.
        
        Args:
            hidden_states: Input tensor of shape [num_experts, tokens_per_expert, hidden_size]
            expert_mask: Optional boolean mask of shape [num_experts, tokens_per_expert]
            
        Returns:
            Output tensor of shape [num_experts, tokens_per_expert, hidden_size]
        """
        residual = hidden_states
        
        # Apply layer normalization if available
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        # Up projection with dropout
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.training:
            hidden_states = self.dropout1(hidden_states)
            
        # Down projection with dropout
        hidden_states = self.fc2(hidden_states)
        if self.training:
            hidden_states = self.dropout2(hidden_states)
            
        # Add residual connection
        hidden_states = hidden_states + residual
        
        # Apply expert mask if provided
        if expert_mask is not None:
            hidden_states = hidden_states.masked_fill(~expert_mask.unsqueeze(-1), 0.0)
            
        return hidden_states

def create_experts(
    num_experts: int,
    expert_class: type[ExpertNetwork],
    *args,
    **kwargs
) -> nn.ModuleList:
    """Create a list of expert networks.
    
    Args:
        num_experts: Number of experts to create
        expert_class: Expert network class to use
        *args: Positional arguments passed to expert constructor
        **kwargs: Keyword arguments passed to expert constructor
        
    Returns:
        ModuleList containing expert networks
    """
    return nn.ModuleList([
        expert_class(*args, **kwargs)
        for _ in range(num_experts)
    ])

"""Gating mechanisms for expert networks."""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertGate(nn.Module):
    """Gate for controlling expert activation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        gate_type: str = "multiplicative",
        gate_temperature: float = 1.0,
        gate_noise: float = 0.1,
        gate_dropout: float = 0.1,
        use_softmax: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize expert gate.
        
        Args:
            hidden_size: Size of hidden dimension
            num_experts: Number of experts
            gate_type: Type of gating ('multiplicative' or 'additive')
            gate_temperature: Temperature for gate logits
            gate_noise: Amount of noise to add during training
            gate_dropout: Gate dropout probability
            use_softmax: Whether to apply softmax to gate logits
            bias: Whether to use bias in gate projection
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.gate_type = gate_type
        self.gate_temperature = gate_temperature
        self.gate_noise = gate_noise
        self.use_softmax = use_softmax
        
        # Gate projection
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias, **factory_kwargs)
        
        # Optional components
        if gate_type == "multiplicative":
            self.scale = nn.Parameter(torch.ones(num_experts, **factory_kwargs))
        else:
            self.scale = None
            
        self.dropout = nn.Dropout(gate_dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize gate weights."""
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
            
        if self.scale is not None:
            nn.init.ones_(self.scale)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute gating weights for experts.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            expert_mask: Optional boolean mask for available experts
            
        Returns:
            Tuple containing:
            - Gating weights of shape [batch_size, seq_length, num_experts]
            - Dictionary of auxiliary outputs
        """
        # Compute gate logits
        gate_logits = self.gate(hidden_states)  # [B, L, E]
        
        # Add noise during training
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(gate_logits) * self.gate_noise
            gate_logits = gate_logits + noise
            
        # Apply expert mask if provided
        if expert_mask is not None:
            gate_logits = gate_logits.masked_fill(~expert_mask, float("-inf"))
            
        # Apply temperature scaling
        if self.gate_temperature != 1.0:
            gate_logits = gate_logits / self.gate_temperature
            
        # Apply activation
        if self.use_softmax:
            gate_weights = F.softmax(gate_logits, dim=-1)
        else:
            gate_weights = F.sigmoid(gate_logits)
            
        # Apply multiplicative scaling if enabled
        if self.scale is not None:
            gate_weights = gate_weights * self.scale
            
        # Apply dropout during training
        if self.training:
            gate_weights = self.dropout(gate_weights)
            
        # Collect auxiliary outputs
        aux_outputs = {
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
        }
        
        return gate_weights, aux_outputs

class HierarchicalGate(nn.Module):
    """Hierarchical gating for expert groups."""
    
    def __init__(
        self,
        hidden_size: int,
        num_groups: int,
        experts_per_group: int,
        group_temperature: float = 1.0,
        expert_temperature: float = 1.0,
        gate_noise: float = 0.1,
        gate_dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize hierarchical gate.
        
        Args:
            hidden_size: Size of hidden dimension
            num_groups: Number of expert groups
            experts_per_group: Number of experts per group
            group_temperature: Temperature for group gate
            expert_temperature: Temperature for expert gate
            gate_noise: Amount of noise to add during training
            gate_dropout: Gate dropout probability
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.group_temperature = group_temperature
        self.expert_temperature = expert_temperature
        self.gate_noise = gate_noise
        
        # Group gate
        self.group_gate = nn.Linear(hidden_size, num_groups, **factory_kwargs)
        
        # Expert gates (one per group)
        self.expert_gates = nn.ModuleList([
            nn.Linear(hidden_size, experts_per_group, **factory_kwargs)
            for _ in range(num_groups)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(gate_dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize gate weights."""
        # Initialize group gate
        nn.init.normal_(self.group_gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.group_gate.bias)
        
        # Initialize expert gates
        for gate in self.expert_gates:
            nn.init.normal_(gate.weight, mean=0.0, std=0.02)
            nn.init.zeros_(gate.bias)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hierarchical gating weights.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            group_mask: Optional boolean mask for available groups
            expert_mask: Optional boolean mask for available experts per group
            
        Returns:
            Tuple containing:
            - Group weights of shape [batch_size, seq_length, num_groups]
            - Expert weights of shape [batch_size, seq_length, num_groups, experts_per_group]
            - Dictionary of auxiliary outputs
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Compute group gate logits
        group_logits = self.group_gate(hidden_states)  # [B, L, G]
        
        # Add noise during training
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(group_logits) * self.gate_noise
            group_logits = group_logits + noise
            
        # Apply group mask if provided
        if group_mask is not None:
            group_logits = group_logits.masked_fill(~group_mask, float("-inf"))
            
        # Apply temperature and softmax for group weights
        group_weights = F.softmax(group_logits / self.group_temperature, dim=-1)
        
        # Compute expert gate logits for each group
        expert_logits = torch.stack([
            gate(hidden_states)
            for gate in self.expert_gates
        ], dim=2)  # [B, L, G, E]
        
        # Add noise during training
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(expert_logits) * self.gate_noise
            expert_logits = expert_logits + noise
            
        # Apply expert mask if provided
        if expert_mask is not None:
            expert_logits = expert_logits.masked_fill(~expert_mask, float("-inf"))
            
        # Apply temperature and softmax for expert weights
        expert_weights = F.softmax(expert_logits / self.expert_temperature, dim=-1)
        
        # Apply dropout during training
        if self.training:
            group_weights = self.dropout(group_weights)
            expert_weights = self.dropout(expert_weights)
            
        # Collect auxiliary outputs
        aux_outputs = {
            "group_logits": group_logits,
            "group_weights": group_weights,
            "expert_logits": expert_logits,
            "expert_weights": expert_weights,
        }
        
        return group_weights, expert_weights, aux_outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_groups={self.num_groups}, "
            f"experts_per_group={self.experts_per_group}"
        )

"""Router implementation for MoE layers."""

import math
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initialization import initialize_router_weights

class TopKRouter(nn.Module):
    """Token router using top-k routing strategy."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_selected_experts: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.1,
        use_aux_loss: bool = True,
        expert_capacity: Optional[int] = None,
        exact_capacity: bool = False,
        hidden_dropout_prob: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize router.
        
        Args:
            hidden_size: Size of hidden dimension
            num_experts: Number of experts
            num_selected_experts: Number of experts to route each token to
            capacity_factor: Factor to determine expert capacity
            jitter_noise: Amount of noise to add to routing weights
            use_aux_loss: Whether to compute auxiliary load balancing loss
            expert_capacity: Optional fixed expert capacity
            exact_capacity: Whether to strictly enforce expert capacity
            hidden_dropout_prob: Hidden state dropout probability
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.use_aux_loss = use_aux_loss
        self.expert_capacity = expert_capacity
        self.exact_capacity = exact_capacity
        
        # Routing weights
        self.weight = nn.Parameter(torch.empty(hidden_size, num_experts, **factory_kwargs))
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize router weights."""
        initialize_router_weights(self, self.num_experts, self.hidden_size)
        
    def _compute_capacity(
        self,
        batch_size: int,
        seq_length: int
    ) -> int:
        """Compute expert capacity.
        
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            Expert capacity (tokens per expert)
        """
        if self.expert_capacity is not None:
            capacity = self.expert_capacity
        else:
            tokens_per_expert = batch_size * seq_length * self.num_selected_experts / self.num_experts
            capacity = int(tokens_per_expert * self.capacity_factor)
            
        return capacity
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Route tokens to experts.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            expert_mask: Optional boolean mask for available experts
            importance_scores: Optional token importance scores
            
        Returns:
            Tuple containing:
            - Routing weights of shape [batch_size, seq_length, num_selected_experts]
            - Expert indices of shape [batch_size, seq_length, num_selected_experts]
            - Dictionary of auxiliary outputs including load balancing loss
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Compute routing scores
        routing_logits = torch.matmul(hidden_states, self.weight)  # [B, L, E]
        
        # Apply jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(routing_logits) * self.jitter_noise
            routing_logits = routing_logits + noise
            
        # Apply expert mask if provided
        if expert_mask is not None:
            routing_logits = routing_logits.masked_fill(~expert_mask, float("-inf"))
            
        # Compute softmax probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)  # [B, L, E]
        
        # Apply importance scores if provided
        if importance_scores is not None:
            routing_probs = routing_probs * importance_scores.unsqueeze(-1)
            
        # Get top-k experts and probabilities
        top_k_probs, top_k_indices = torch.topk(
            routing_probs,
            k=self.num_selected_experts,
            dim=-1,
            sorted=True
        )  # [B, L, K]
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute load balancing auxiliary loss
        aux_loss = None
        if self.use_aux_loss:
            # Compute expert load
            expert_load = routing_probs.mean(dim=[0, 1])  # [E]
            
            # Compute load balancing loss
            aux_loss = torch.mean(expert_load * torch.log(expert_load + 1e-9)) * self.num_experts
            
        # Handle capacity constraints
        capacity = self._compute_capacity(batch_size, seq_length)
        expert_counts = torch.zeros(
            (self.num_experts,),
            device=hidden_states.device,
            dtype=torch.long
        )
        
        if self.exact_capacity:
            # Track expert assignment counts
            for expert_idx in top_k_indices.view(-1):
                expert_counts[expert_idx] += 1
                
            # Mask out experts that exceed capacity
            capacity_mask = expert_counts[top_k_indices] <= capacity  # [B, L, K]
            top_k_probs = top_k_probs.masked_fill(~capacity_mask, 0.0)
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
            
        # Collect auxiliary outputs
        aux_outputs = {
            "load_balancing_loss": aux_loss if aux_loss is not None else 0.0,
            "expert_counts": expert_counts,
            "route_probs": routing_probs,
        }
        
        return top_k_probs, top_k_indices, aux_outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"num_selected={self.num_selected_experts}, "
            f"capacity_factor={self.capacity_factor}, "
            f"jitter={self.jitter_noise}"
        )

class DenseRouter(nn.Module):
    """Router that computes dense weighted combinations of experts."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        temperature: float = 1.0,
        hidden_dropout_prob: float = 0.1,
        use_aux_loss: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize dense router.
        
        Args:
            hidden_size: Size of hidden dimension
            num_experts: Number of experts
            temperature: Temperature for attention scores
            hidden_dropout_prob: Hidden state dropout probability
            use_aux_loss: Whether to compute auxiliary load balancing loss
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_aux_loss = use_aux_loss
        
        # Routing weights
        self.weight = nn.Parameter(torch.empty(hidden_size, num_experts, **factory_kwargs))
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize router weights."""
        initialize_router_weights(self, self.num_experts, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute expert combination weights.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            expert_mask: Optional boolean mask for available experts
            
        Returns:
            Tuple containing:
            - Expert weights of shape [batch_size, seq_length, num_experts]
            - Dictionary of auxiliary outputs
        """
        # Compute routing logits
        routing_logits = torch.matmul(hidden_states, self.weight)  # [B, L, E]
        
        # Apply expert mask if provided
        if expert_mask is not None:
            routing_logits = routing_logits.masked_fill(~expert_mask, float("-inf"))
            
        # Apply temperature scaling and compute expert weights
        routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
        
        # Apply dropout during training
        if self.training:
            routing_weights = self.dropout(routing_weights)
            
        # Compute load balancing auxiliary loss
        aux_loss = None
        if self.use_aux_loss:
            # Compute expert load
            expert_load = routing_weights.mean(dim=[0, 1])  # [E]
            
            # Compute load balancing loss
            aux_loss = torch.mean(expert_load * torch.log(expert_load + 1e-9)) * self.num_experts
            
        # Collect auxiliary outputs
        aux_outputs = {
            "load_balancing_loss": aux_loss if aux_loss is not None else 0.0,
            "route_probs": routing_weights,
        }
        
        return routing_weights, aux_outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"temperature={self.temperature}"
        )

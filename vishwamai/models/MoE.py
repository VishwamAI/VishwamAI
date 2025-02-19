"""
Mixture of Experts (MoE) implementation for VishwamAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math

from vishwamai.models.base_layers import Linear, LayerNorm
from vishwamai.utils.parallel import model_parallel_forward, gather_from_model_parallel_region

class ExpertLayer(nn.Module):
    """Individual expert neural network."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_model_parallel: bool = False
    ):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size, use_model_parallel=use_model_parallel)
        self.fc2 = Linear(hidden_size, output_size, use_model_parallel=use_model_parallel)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        h = self.activation(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with dynamic routing and load balancing.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        output_size: int = None,
        num_experts: int = 8,
        expert_capacity: int = 128,
        k: int = 2,  # Number of experts to route to
        activation: str = "gelu",
        dropout: float = 0.1,
        use_model_parallel: bool = False,
        load_balance: bool = True,
        noise_eps: float = 1e-2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size * 4
        self.output_size = output_size or input_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.k = k
        self.use_model_parallel = use_model_parallel
        self.load_balance = load_balance
        self.noise_eps = noise_eps
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                activation=activation,
                dropout=dropout,
                use_model_parallel=use_model_parallel
            ) for _ in range(num_experts)
        ])
        
        # Router parameters
        self.router = Linear(input_size, num_experts, use_model_parallel=use_model_parallel)
        
        # Expert gate bias to encourage sparse routing
        self.gate_bias = nn.Parameter(torch.zeros(num_experts))
        
        # For tracking expert usage
        register_buffer = self.register_buffer
        register_buffer('_expert_counts', torch.zeros(num_experts))
        register_buffer('_total_tokens_routed', torch.scalar_tensor(0))
        
    def _compute_router_probabilities(
        self,
        hidden_states: torch.Tensor,
        noise_eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing probabilities for tokens to experts."""
        if noise_eps is None:
            noise_eps = self.noise_eps
            
        # Get router logits
        router_logits = self.router(hidden_states)
        
        if self.training and noise_eps > 0:
            # Add noise during training for better load balancing
            router_noise = torch.randn_like(router_logits) * noise_eps
            router_logits = router_logits + router_noise
            
        # Add learned bias
        router_logits = router_logits + self.gate_bias
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.k, dim=-1)
        
        # Convert to probabilities
        router_probs = F.softmax(top_k_logits, dim=-1)
        
        return router_probs, top_k_indices
        
    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Compute fraction of tokens routed to each expert
        expert_counts = torch.zeros(
            self.num_experts,
            device=router_probs.device
        )
        expert_counts.scatter_add_(
            0,
            expert_indices.view(-1),
            router_probs.view(-1)
        )
        fraction_per_expert = expert_counts / router_probs.size(0)
        
        # Ideal fraction is uniform distribution
        target_fraction = 1.0 / self.num_experts
        
        # Compute loss as squared difference from ideal fraction
        balance_loss = torch.mean(
            (fraction_per_expert - target_fraction) ** 2
        ) * self.num_experts
        
        return balance_loss
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass routing tokens through experts.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            return_scores: Whether to return routing probabilities
            
        Returns:
            Dict containing:
                output: Combined expert outputs
                balance_loss: Load balancing loss if training
                router_scores: Routing probabilities if return_scores=True
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        tokens_per_batch = batch_size * seq_len
        
        # Get router probabilities
        router_probs, expert_indices = self._compute_router_probabilities(
            hidden_states.view(tokens_per_batch, hidden_size)
        )
        
        # Initialize output tensor
        final_output = torch.zeros(
            (tokens_per_batch, self.output_size),
            device=hidden_states.device
        )
        
        # Dispatch tokens to experts
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert inputs
            expert_input = hidden_states.view(tokens_per_batch, hidden_size)[expert_mask]
            
            # Get routing probabilities for these tokens
            token_probs = router_probs[expert_mask][
                expert_indices[expert_mask] == expert_idx
            ]
            
            # Compute expert output
            if self.use_model_parallel:
                expert_output = model_parallel_forward(
                    self.experts[expert_idx],
                    expert_input
                )
            else:
                expert_output = self.experts[expert_idx](expert_input)
                
            # Combine with routing probabilities
            weighted_output = token_probs.unsqueeze(-1) * expert_output
            
            # Add to final output
            final_output[expert_mask] += weighted_output
            
        # Reshape output back to input shape
        final_output = final_output.view(batch_size, seq_len, self.output_size)
        
        # Compute load balancing loss during training
        outputs = {'output': final_output}
        if self.training and self.load_balance:
            outputs['balance_loss'] = self._compute_load_balancing_loss(
                router_probs,
                expert_indices
            )
            
        if return_scores:
            outputs['router_scores'] = router_probs
            
        # Update usage statistics
        if self.training:
            self._update_usage_stats(expert_indices, tokens_per_batch)
            
        return outputs
        
    def _update_usage_stats(
        self,
        expert_indices: torch.Tensor,
        total_tokens: int
    ):
        """Update expert usage statistics."""
        expert_counts = torch.bincount(
            expert_indices.view(-1),
            minlength=self.num_experts
        )
        self._expert_counts += expert_counts
        self._total_tokens_routed += total_tokens
        
    def get_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        if self._total_tokens_routed == 0:
            return {'uniform_fraction': 1.0, 'overflow_fraction': 0.0}
            
        fractions = self._expert_counts / self._total_tokens_routed
        uniform = torch.ones_like(fractions) / self.num_experts
        
        return {
            'uniform_fraction': (
                torch.min(fractions / uniform)
            ).item(),
            'overflow_fraction': (
                torch.sum(torch.relu(fractions - self.expert_capacity))
                / self.num_experts
            ).item()
        }

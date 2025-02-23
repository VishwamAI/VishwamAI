"""Mixture of Experts layer implementation."""

from typing import Optional, Tuple, Dict, Union, Type

import torch
import torch.nn as nn

from .expert import ExpertNetwork, ParallelExpertNetwork
from .router import TopKRouter, DenseRouter

class MoELayer(nn.Module):
    """Mixture of Experts layer with routing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_class: Type[ExpertNetwork] = ParallelExpertNetwork,
        router_class: Type[Union[TopKRouter, DenseRouter]] = TopKRouter,
        num_selected_experts: int = 2,
        expert_capacity_factor: float = 1.25,
        expert_dropout_prob: float = 0.1,
        router_dropout_prob: float = 0.1,
        jitter_noise: float = 0.1,
        expert_parallel: bool = True,
        use_aux_loss: bool = True,
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **expert_kwargs
    ):
        """Initialize MoE layer.
        
        Args:
            hidden_size: Size of hidden dimension
            num_experts: Number of experts
            expert_class: Expert network class to use
            router_class: Router class to use
            num_selected_experts: Number of experts to route each token to
            expert_capacity_factor: Factor to determine expert capacity
            expert_dropout_prob: Expert network dropout probability
            router_dropout_prob: Router dropout probability
            jitter_noise: Amount of noise to add to routing weights
            expert_parallel: Whether to process experts in parallel
            use_aux_loss: Whether to compute auxiliary losses
            router_z_loss_coef: Coefficient for router z-loss
            router_aux_loss_coef: Coefficient for router auxiliary loss
            device: Device to create tensors on
            dtype: Data type for parameters
            **expert_kwargs: Additional arguments passed to expert constructor
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.expert_parallel = expert_parallel
        self.use_aux_loss = use_aux_loss
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # Create experts
        self.experts = nn.ModuleList([
            expert_class(
                hidden_size=hidden_size,
                dropout_prob=expert_dropout_prob,
                **expert_kwargs,
                **factory_kwargs
            )
            for _ in range(num_experts)
        ])
        
        # Create router
        router_args = {
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "hidden_dropout_prob": router_dropout_prob,
            "use_aux_loss": use_aux_loss,
            **factory_kwargs
        }
        
        if router_class == TopKRouter:
            router_args.update({
                "num_selected_experts": num_selected_experts,
                "capacity_factor": expert_capacity_factor,
                "jitter_noise": jitter_noise,
            })
            
        self.router = router_class(**router_args)
        
    def _process_experts_sequential(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Tuple[torch.Tensor, torch.Tensor],
        expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process tokens through experts sequentially.
        
        Args:
            hidden_states: Input tensor
            router_outputs: Tuple of routing weights and expert indices
            expert_mask: Optional mask for unavailable experts
            
        Returns:
            Output tensor
        """
        route_probs, route_indices = router_outputs
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Initialize output tensor
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get tokens routed to this expert
            tokens_for_expert = (route_indices == expert_idx)
            if not tokens_for_expert.any():
                continue
                
            # Get expert inputs and weights
            expert_inputs = hidden_states[tokens_for_expert]
            expert_weights = route_probs[tokens_for_expert]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Scale by routing weights and accumulate
            output[tokens_for_expert] += expert_output * expert_weights.unsqueeze(-1)
            
        return output
        
    def _process_experts_parallel(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Tuple[torch.Tensor, torch.Tensor],
        expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process tokens through experts in parallel.
        
        Args:
            hidden_states: Input tensor
            router_outputs: Tuple of routing weights and expert indices
            expert_mask: Optional mask for unavailable experts
            
        Returns:
            Output tensor
        """
        route_probs, route_indices = router_outputs
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Reshape inputs for parallel processing
        # [batch*seq, num_selected, hidden]
        tokens_per_expert = hidden_states.view(-1, hidden_size)
        tokens_per_expert = tokens_per_expert.unsqueeze(1).expand(
            -1, self.num_selected_experts, -1
        )
        
        # Create expert mask
        expert_mask = torch.zeros(
            (batch_size * seq_length, self.num_experts),
            dtype=torch.bool,
            device=hidden_states.device
        )
        expert_mask.scatter_(1, route_indices.view(-1, self.num_selected_experts), 1)
        
        # Process through all experts in parallel
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_tokens = tokens_per_expert[expert_mask[:, expert_idx]]
            if len(expert_tokens) == 0:
                continue
                
            # Process tokens
            expert_output = expert(expert_tokens)
            expert_outputs.append((expert_idx, expert_output))
            
        # Combine expert outputs
        output = torch.zeros_like(hidden_states)
        for expert_idx, expert_output in expert_outputs:
            # Get routing weights for this expert
            expert_weights = route_probs[route_indices == expert_idx]
            
            # Scale output by routing weights
            scaled_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Accumulate in output tensor
            output_idx = (route_indices == expert_idx).nonzero()
            output[output_idx[:, 0], output_idx[:, 1]] = scaled_output
            
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            expert_mask: Optional boolean mask for available experts
            importance_scores: Optional token importance scores
            
        Returns:
            Tuple containing:
            - Output tensor of same shape as input
            - Dictionary of auxiliary outputs
        """
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Get routing weights from router
        if isinstance(self.router, TopKRouter):
            route_probs, route_indices, router_aux = self.router(
                hidden_states,
                expert_mask=expert_mask,
                importance_scores=importance_scores
            )
        else:
            route_probs, router_aux = self.router(
                hidden_states,
                expert_mask=expert_mask
            )
            route_indices = None
            
        # Process through experts
        if self.expert_parallel:
            output = self._process_experts_parallel(
                hidden_states,
                (route_probs, route_indices),
                expert_mask=expert_mask
            )
        else:
            output = self._process_experts_sequential(
                hidden_states,
                (route_probs, route_indices),
                expert_mask=expert_mask
            )
            
        # Compute auxiliary losses
        aux_loss = 0.0
        if self.use_aux_loss:
            if "load_balancing_loss" in router_aux:
                aux_loss += router_aux["load_balancing_loss"] * self.router_aux_loss_coef
                
            # Add router z-loss
            if route_probs is not None:
                z_loss = torch.mean(torch.square(torch.logsumexp(route_probs, dim=-1)))
                aux_loss += z_loss * self.router_z_loss_coef
                
        # Collect auxiliary outputs
        aux_outputs = {
            "aux_loss": aux_loss,
            "route_probs": route_probs,
            "route_indices": route_indices,
            **router_aux
        }
        
        return output, aux_outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"num_selected={self.num_selected_experts}, "
            f"parallel={self.expert_parallel}"
        )

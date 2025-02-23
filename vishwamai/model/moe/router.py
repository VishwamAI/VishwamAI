"""Router module for Mixture of Experts."""

from typing import Optional, Tuple, Dict, Any
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

def compute_router_z_loss(router_logits: jnp.ndarray) -> jnp.ndarray:
    """Compute router z-loss to encourage confident routing.
    
    Args:
        router_logits: Raw routing logits [batch, seq_len, num_experts]
        
    Returns:
        Z-loss value
    """
    num_experts = router_logits.shape[-1]
    log_z = jax.nn.logsumexp(router_logits, axis=-1)
    z_loss = jnp.square(log_z)
    return z_loss.mean()

def compute_load_balancing_loss(router_probs: jnp.ndarray,
                              expert_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute load balancing auxiliary loss.
    
    Args:
        router_probs: Router probabilities [batch, seq_len, num_experts]
        expert_mask: Binary mask of selected experts [batch, seq_len, num_experts]
        
    Returns:
        Load balancing loss value
    """
    # Calculate fraction of tokens routed to each expert
    routing_fraction = expert_mask.mean(axis=(0, 1))
    
    # Target uniform routing distribution
    num_experts = router_probs.shape[-1]
    target_fraction = jnp.ones_like(routing_fraction) / num_experts
    
    # Compute load balancing loss using KL divergence
    load_balancing_loss = jnp.sum(
        routing_fraction * jnp.log(routing_fraction / target_fraction)
    )
    
    return load_balancing_loss

class ExpertRouter(nn.Module):
    """Routes tokens to experts using top-k routing."""
    
    num_experts: int
    capacity_factor: float = 1.25
    num_experts_per_token: int = 2
    jitter_noise: float = 0.1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    def setup(self):
        """Initialize router components."""
        # Router projection to compute expert scores
        self.router = nn.Dense(
            self.num_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="router"
        )
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 expert_capacity: Optional[int] = None,
                 deterministic: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Route tokens to experts.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            expert_capacity: Optional override for expert capacity
            deterministic: Whether to apply noise/jitter
            
        Returns:
            Tuple of:
                - Binary dispatch tensor [batch, seq_len, num_experts, expert_capacity]
                - Dict of auxiliary outputs including router probs and losses
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Compute routing logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        if not deterministic and self.jitter_noise > 0:
            # Add multiplicative jitter noise during training
            routing_noise = jax.random.uniform(
                self.make_rng('noise'),
                router_logits.shape,
                minval=1.0 - self.jitter_noise,
                maxval=1.0 + self.jitter_noise
            )
            router_logits = router_logits * routing_noise
            
        # Get router probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Compute expert capacity
        tokens_per_expert = math.ceil(
            seq_length * batch_size * self.capacity_factor / self.num_experts
        )
        expert_capacity = expert_capacity or tokens_per_expert
        
        # Get top-k expert indices and scores
        top_k_probs, top_k_indices = jax.lax.top_k(
            router_probs,
            k=self.num_experts_per_token
        )
        
        # Create binary expert selection mask
        expert_mask = jnp.zeros(
            (batch_size, seq_length, self.num_experts),
            dtype=bool
        )
        expert_mask = expert_mask.at[
            jnp.arange(batch_size)[:, None, None],
            jnp.arange(seq_length)[None, :, None],
            top_k_indices
        ].set(True)
        
        # Create dispatch tensor mapping tokens to expert slots
        position_in_expert = jnp.cumsum(expert_mask, axis=1) - 1
        dispatch_mask = position_in_expert < expert_capacity
        
        # Create final dispatch tensor
        dispatch = jnp.zeros(
            (batch_size, seq_length, self.num_experts, expert_capacity),
            dtype=hidden_states.dtype
        )
        dispatch = dispatch.at[
            jnp.arange(batch_size)[:, None, None, None],
            jnp.arange(seq_length)[None, :, None, None],
            jnp.where(dispatch_mask, top_k_indices, 0),
            jnp.where(dispatch_mask, position_in_expert, 0)
        ].set(
            jnp.where(
                dispatch_mask[..., None],
                top_k_probs[..., None],
                0.0
            )
        )
        
        # Compute auxiliary losses
        aux_losses = {
            'router_z_loss': compute_router_z_loss(router_logits),
            'load_balancing_loss': compute_load_balancing_loss(
                router_probs, expert_mask
            )
        }
        
        # Include auxiliary outputs
        aux_outputs = {
            'router_probs': router_probs,
            'expert_mask': expert_mask,
            'dispatch_mask': dispatch_mask,
            **aux_losses
        }
        
        return dispatch, aux_outputs
        
    def _count_dropped_tokens(self, dispatch_mask: jnp.ndarray) -> jnp.ndarray:
        """Count tokens dropped due to expert overflow.
        
        Args:
            dispatch_mask: Binary mask of successfully routed tokens
            
        Returns:
            Number of dropped tokens per expert
        """
        # Sum tokens routed to each expert
        tokens_per_expert = dispatch_mask.sum(axis=(0, 1))
        
        # Count total tokens that should have been routed
        total_tokens = dispatch_mask.shape[0] * dispatch_mask.shape[1]
        target_tokens = total_tokens // self.num_experts
        
        # Calculate dropped tokens
        dropped_tokens = jnp.maximum(0, target_tokens - tokens_per_expert)
        
        return dropped_tokens

"""Mixture of Experts layers optimized for TPU."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict, List, Union, Tuple
from vishwamai.kernels.moe_dispatch import compute_routing_prob, compute_routing_indices, load_balance_loss

class TPUMoELayer(nn.Module):
    """
    TPU-optimized Mixture of Experts layer.
    
    This implementation is optimized for efficient execution on TPUs.
    """
    num_experts: int = 8
    expert_dim: int = 1024
    output_dim: Optional[int] = None
    num_experts_per_tok: int = 2
    router_bias: bool = False
    router_dtype: Any = jnp.float32
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    expert_dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply Mixture of Experts layer to input tensor."""
        batch_size, seq_len, hidden_dim = x.shape
        output_dim = self.output_dim or hidden_dim
        
        # Router projection
        router_logits = nn.Dense(
            features=self.num_experts,
            use_bias=self.router_bias,
            dtype=self.router_dtype,
            param_dtype=self.param_dtype,
            name="router"
        )(x)
        
        # Reshape for routing
        x_reshaped = x.reshape(-1, hidden_dim)  # [batch*seq, hidden]
        router_logits_reshaped = router_logits.reshape(-1, self.num_experts)  # [batch*seq, num_experts]
        
        # Get routing weights and indices for top-k experts
        route_weights, route_indices = compute_routing_indices(
            router_logits_reshaped, 
            top_k=self.num_experts_per_tok
        )
        
        # Create expert feedforward networks
        experts = [
            ExpertFFN(
                hidden_dim=self.expert_dim,
                output_dim=output_dim,
                dropout_rate=self.expert_dropout,
                name=f"expert_{i}",
                dtype=self.dtype,
                param_dtype=self.param_dtype
            ) 
            for i in range(self.num_experts)
        ]
        
        # Process with experts
        outputs = jnp.zeros((batch_size * seq_len, output_dim), dtype=self.dtype)
        
        # For each token, dispatch to relevant experts
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            # This is inefficient for TPU but kept for clarity
            # In practice, we'd use a batched approach
            token_indices = jnp.where(route_indices == expert_idx)[0]
            if len(token_indices) == 0:
                continue
                
            # Get token inputs for this expert
            expert_inputs = x_reshaped[token_indices]
            
            # Apply expert
            expert_output = experts[expert_idx](expert_inputs, deterministic=deterministic)
            
            # Find weights for this expert
            expert_position = jnp.where(route_indices == expert_idx)[1]
            expert_weights = jnp.take_along_axis(
                route_weights, 
                jnp.expand_dims(expert_position, -1), 
                axis=-1
            ).squeeze(-1)
            
            # Scale output by routing weight
            expert_output = expert_output * expert_weights[:, None]
            
            # Add to output
            outputs = outputs.at[token_indices].add(expert_output)
        
        # Reshape back to original dimensions
        outputs = outputs.reshape(batch_size, seq_len, output_dim)
        
        return outputs

class TPUSparseMoEDispatch(nn.Module):
    """
    TPU-optimized sparse Mixture of Experts dispatch layer.
    
    Implements a more efficient token-to-expert routing mechanism.
    """
    num_experts: int = 8
    capacity_factor: float = 1.25
    num_experts_per_tok: int = 2
    router_bias: bool = False
    router_dtype: Any = jnp.float32
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        expert_fn: Callable[[jnp.ndarray, bool], jnp.ndarray],
        deterministic: bool = True,
        router_weights: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Dispatch tokens to experts and combine outputs.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            expert_fn: Function that applies expert computation to inputs
            deterministic: Whether to run in deterministic mode
            router_weights: Optional pre-computed router weights
            
        Returns:
            Tuple of (output, auxiliary_outputs)
        """
        batch_size, seq_len, hidden_dim = x.shape
        inputs_reshaped = x.reshape(-1, hidden_dim)
        
        # Router projection if weights not provided
        if router_weights is None:
            router_weights = self.param(
                'router_weights',
                nn.initializers.normal(stddev=0.02),
                (hidden_dim, self.num_experts),
                self.param_dtype
            )
            
            if self.router_bias:
                router_bias = self.param(
                    'router_bias',
                    nn.initializers.zeros,
                    (self.num_experts,),
                    self.param_dtype
                )
            else:
                router_bias = 0.0
                
            # Compute routing logits
            router_logits = (
                jnp.matmul(inputs_reshaped, router_weights)
                + router_bias
            )
        else:
            # Use provided router weights
            router_logits = router_weights
        
        # Compute routing probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Find top-k experts per token
        top_k_probs, top_k_indices = jax.lax.top_k(
            router_probs, k=self.num_experts_per_tok
        )
        
        # Normalize top-k probabilities
        top_k_probs_sum = jnp.sum(top_k_probs, axis=-1, keepdims=True)
        top_k_probs = top_k_probs / top_k_probs_sum
        
        # Compute expert capacity
        token_count = batch_size * seq_len
        capacity = int(token_count * self.capacity_factor * self.num_experts_per_tok / self.num_experts)
        capacity = max(capacity, 4)  # Minimum capacity
        
        # Auxiliary outputs for monitoring
        aux_outputs = {
            'router_probs': router_probs,
            'expert_indices': top_k_indices,
            'expert_capacity': capacity,
        }
        
        # Compute load balancing loss
        aux_outputs['balance_loss'] = load_balance_loss(
            router_probs, top_k_indices, self.num_experts
        )
        
        # Initialize output
        final_output = jnp.zeros_like(inputs_reshaped)
        
        # Expert processing
        for expert_idx in range(self.num_experts):
            # Get indices where this expert is in top-k
            expert_mask = (top_k_indices == expert_idx)
            token_indices = jnp.where(expert_mask.any(axis=1))[0]
            
            if len(token_indices) == 0:
                continue
                
            # Get inputs for this expert
            expert_inputs = inputs_reshaped[token_indices]
            
            # Apply expert function
            expert_output = expert_fn(expert_inputs, deterministic)
            
            # Get weights for this expert
            expert_probs = jnp.where(
                expert_mask, 
                top_k_probs,
                jnp.zeros_like(top_k_probs)
            ).sum(axis=1)
            expert_probs = expert_probs[token_indices]
            
            # Scale output by routing probs
            scaled_expert_output = expert_output * expert_probs[:, None]
            
            # Add to final output
            final_output = final_output.at[token_indices].add(scaled_expert_output)
        
        # Reshape output back to input shape
        output = final_output.reshape(batch_size, seq_len, hidden_dim)
        
        return output, aux_outputs

class ExpertFFN(nn.Module):
    """
    Expert feedforward network for MoE layers.
    
    Each expert consists of a two-layer MLP with GELU activation.
    """
    hidden_dim: int
    output_dim: Optional[int] = None
    dropout_rate: float = 0.0
    activation: Callable = jax.nn.gelu
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply expert MLP to input tensor."""
        output_dim = self.output_dim or x.shape[-1]
        
        # First dense layer
        x = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dense1"
        )(x)
        
        # Activation
        x = self.activation(x)
        
        # Dropout
        if not deterministic and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        # Second dense layer
        x = nn.Dense(
            features=output_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dense2"
        )(x)
        
        return x

class TPUBalancedMoE(nn.Module):
    """
    Balanced Mixture of Experts layer for TPU.
    
    Uses a load-balanced routing strategy to ensure even utilization of experts.
    """
    num_experts: int = 8
    expert_dim: int = 1024
    output_dim: Optional[int] = None
    num_experts_per_tok: int = 2
    load_balance_coef: float = 0.01
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    expert_dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply load-balanced MoE layer to input tensor."""
        batch_size, seq_len, hidden_dim = x.shape
        output_dim = self.output_dim or hidden_dim
        
        # Router projection
        router_logits = nn.Dense(
            features=self.num_experts,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="router"
        )(x)
        
        # Reshape for routing
        inputs_reshaped = x.reshape(-1, hidden_dim)
        router_logits_reshaped = router_logits.reshape(-1, self.num_experts)
        
        # Compute router probabilities with auxiliary loss for load balancing
        dispatch_tensor, combine_tensor = compute_routing_prob(
            inputs=x,
            routing_weights=self.param(
                'routing_weights',
                nn.initializers.normal(0.02),
                (hidden_dim, self.num_experts),
                self.param_dtype
            ),
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            deterministic=deterministic
        )
        
        # Create experts
        experts = [
            ExpertFFN(
                hidden_dim=self.expert_dim,
                output_dim=output_dim,
                dropout_rate=self.expert_dropout,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"expert_{i}"
            )
            for i in range(self.num_experts)
        ]
        
        # Initialize output tensor
        outputs = jnp.zeros((batch_size * seq_len, output_dim), dtype=self.dtype)
        
        # Apply each expert to the relevant tokens
        for expert_idx in range(self.num_experts):
            # Get dispatch and combine tensors for this expert
            expert_dispatch = dispatch_tensor[:, expert_idx, :]
            expert_combine = combine_tensor[:, expert_idx, :]
            
            # Find tokens that use this expert
            token_mask = expert_dispatch.sum(axis=1) > 0
            if not jnp.any(token_mask):
                continue
                
            # Get token indices that use this expert
            token_indices = jnp.where(token_mask)[0]
            
            # Get inputs for this expert
            expert_inputs = inputs_reshaped[token_indices]
            
            # Apply expert to inputs
            expert_output = experts[expert_idx](expert_inputs, deterministic=deterministic)
            
            # Weight by combine tensor
            expert_weights = expert_combine[token_indices].sum(axis=1)
            expert_output = expert_output * expert_weights[:, None]
            
            # Add to output
            outputs = outputs.at[token_indices].add(expert_output)
        
        # Reshape back to original dimensions
        final_output = outputs.reshape(batch_size, seq_len, output_dim)
        
        return final_output
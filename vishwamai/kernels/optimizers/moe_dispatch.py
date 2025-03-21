"""Mixture of Experts routing and dispatch kernels for TPU."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

def compute_routing_prob(
    inputs: jnp.ndarray,
    routing_weights: jnp.ndarray,
    num_experts: int,
    top_k: int = 2,
    use_sigmoid: bool = False,
    capacity_factor: float = 1.25,
    noise_std: float = 0.0,
    deterministic: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute routing probabilities for mixture of experts.
    
    Args:
        inputs: Input tensor of shape [batch_size, sequence_length, hidden_dim]
        routing_weights: Router weights of shape [hidden_dim, num_experts]
        num_experts: Number of experts
        top_k: Number of experts to route to for each token
        use_sigmoid: If True, use sigmoid instead of softmax
        capacity_factor: Capacity factor for expert allocation
        noise_std: Standard deviation of noise to add to router logits in training
        deterministic: Whether to run in deterministic mode (no noise)
            
    Returns:
        Tuple of (dispatch_tensor, combine_tensor)
    """
    # Reshape input if needed for routing
    batch_size, seq_len, hidden_dim = inputs.shape
    inputs_reshaped = inputs.reshape(-1, hidden_dim)
    
    # Compute routing logits
    router_logits = jnp.matmul(inputs_reshaped, routing_weights)  # [batch*seq, num_experts]
    
    # Add noise during training for exploration
    if not deterministic and noise_std > 0:
        router_logits = router_logits + jax.random.normal(
            jax.random.PRNGKey(0), router_logits.shape) * noise_std
    
    # Get routing probabilities
    if use_sigmoid:
        # Independent routing with sigmoid
        routing_probs = jax.nn.sigmoid(router_logits)
        
        # Ensure min probability for numerical stability
        routing_probs = jnp.maximum(routing_probs, 1e-6)
    else:
        # Softmax routing (default)
        routing_probs = jax.nn.softmax(router_logits, axis=-1)
    
    # For TPU efficiency, find top-k experts per token
    top_k_probs, top_k_indices = jax.lax.top_k(routing_probs, top_k)
    
    # Set up expert capacity
    tokens_per_batch = batch_size * seq_len
    capacity = int(tokens_per_batch * capacity_factor * top_k / num_experts)
    
    # Create dispatch and combine tensors
    # dispatch_tensor: [batch_size*seq_len, num_experts, capacity]
    # combine_tensor: [batch_size*seq_len, num_experts, capacity]
    dispatch_tensor = jnp.zeros((tokens_per_batch, num_experts, capacity))
    combine_tensor = jnp.zeros((tokens_per_batch, num_experts, capacity))
    
    # Expert counts
    expert_counts = jnp.zeros((num_experts,), dtype=jnp.int32)
    
    # For each token, assign to top-k experts
    for token_idx in range(tokens_per_batch):
        for k in range(top_k):
            expert_idx = top_k_indices[token_idx, k]
            expert_count = expert_counts[expert_idx]
            
            # Check if expert has capacity
            if expert_count < capacity:
                # Assign token to expert
                dispatch_tensor = dispatch_tensor.at[token_idx, expert_idx, expert_count].set(1.0)
                combine_tensor = combine_tensor.at[token_idx, expert_idx, expert_count].set(
                    top_k_probs[token_idx, k])
                
                # Update expert count
                expert_counts = expert_counts.at[expert_idx].set(expert_count + 1)
    
    # Normalize routing probabilities
    token_has_assignment = jnp.sum(dispatch_tensor, axis=(1, 2))
    denom = jnp.sum(combine_tensor, axis=(1, 2))
    # Avoid division by zero
    denom = jnp.where(token_has_assignment, denom, 1.0)
    combine_tensor = combine_tensor / denom[:, None, None]
    
    return dispatch_tensor, combine_tensor

def load_balance_loss(
    router_probs: jnp.ndarray,
    expert_indices: jnp.ndarray,
    num_experts: int
) -> jnp.ndarray:
    """
    Compute load balancing loss for mixture of experts.
    
    This loss encourages balanced expert assignment.
    
    Args:
        router_probs: Router probabilities [batch_size * seq_len, num_experts]
        expert_indices: Expert indices [batch_size * seq_len, top_k]
        num_experts: Number of experts
        
    Returns:
        Load balancing loss
    """
    # Get router probs for selected experts
    batch_size = router_probs.shape[0]
    top_k = expert_indices.shape[1]
    
    # Create mask for selected experts
    mask = jnp.zeros((batch_size, num_experts))
    batch_indices = jnp.arange(batch_size)[:, None]
    mask = mask.at[batch_indices, expert_indices].set(1.0)
    
    # Compute fraction of tokens routed to each expert
    router_prob_per_expert = jnp.mean(router_probs, axis=0)
    
    # Compute fraction of tokens assigned to each expert
    expert_usage = jnp.mean(mask, axis=0)
    
    # Compute auxiliary loss: router_z^2 * fraction_expert_assignment
    aux_loss = jnp.mean(router_prob_per_expert * expert_usage) * num_experts
    
    # Scale loss - higher weight means more balancing
    return aux_loss * 0.01

def dispatch_and_combine(
    inputs: jnp.ndarray,
    dispatch_tensor: jnp.ndarray,
    combine_tensor: jnp.ndarray,
    num_experts: int,
    capacity: int
) -> jnp.ndarray:
    """
    Dispatch tokens to experts and combine their outputs.
    
    Args:
        inputs: Input tensor [batch_size*seq_len, hidden_dim]
        dispatch_tensor: Dispatch tensor [batch_size*seq_len, num_experts, capacity]
        combine_tensor: Combine tensor [batch_size*seq_len, num_experts, capacity]
        num_experts: Number of experts
        capacity: Expert capacity
        
    Returns:
        Combined expert outputs [batch_size*seq_len, hidden_dim]
    """
    batch_size, hidden_dim = inputs.shape
    
    # Reshape for expert processing
    dispatched_inputs = inputs[:, None, None, :] * dispatch_tensor[:, :, :, None]
    
    # Shape: [num_experts, capacity, hidden_dim]
    expert_inputs = jnp.sum(dispatched_inputs, axis=0)
    
    # Process by experts (placeholder - in real impl. would call the experts here)
    # For now, just returning the expert inputs as if they are outputs
    expert_outputs = expert_inputs
    
    # Combine expert outputs
    # Shape: [batch_size*seq_len, num_experts, capacity, hidden_dim]
    combined_outputs = expert_outputs[None, :, :, :] * combine_tensor[:, :, :, None]
    
    # Sum over experts and capacity
    outputs = jnp.sum(combined_outputs, axis=(1, 2))
    
    return outputs

def compute_routing_indices(
    router_logits: jnp.ndarray,
    top_k: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute routing indices from router logits for efficient TPU inference.
    
    Args:
        router_logits: Router logits [batch_size*seq_len, num_experts]
        top_k: Number of experts to route to
        
    Returns:
        Tuple of (routing_weights, routing_indices)
    """
    # Compute softmax for weights
    routing_weights = jax.nn.softmax(router_logits, axis=-1)
    
    # Get top-k experts
    top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, top_k)
    
    # Normalize the weights
    top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)
    
    return top_k_weights, top_k_indices
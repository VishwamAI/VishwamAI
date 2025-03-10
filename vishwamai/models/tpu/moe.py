"""
TPU-optimized Mixture of Experts implementation using JAX/XLA
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Optional, List, Tuple

class OptimizedMoE(hk.Module):
    def __init__(self, num_experts: int, expert_size: int, input_size: int,
                 capacity_factor: float = 1.25,
                 min_expert_capacity: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.input_size = input_size
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        
    def __call__(self, x: jnp.ndarray, expert_mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        
        # Gate computation
        gates = self._compute_gates(x, expert_mask)
        
        # Expert dispatch
        dispatch_tensor = self._create_dispatch_tensor(
            gates, batch_size * seq_len
        )
        
        # Process with experts
        expert_inputs = self._dispatch_to_experts(x.reshape(-1, self.input_size), dispatch_tensor)
        expert_outputs = self._process_with_experts(expert_inputs)
        
        # Combine outputs
        combined_output = self._combine_expert_outputs(
            expert_outputs, dispatch_tensor, batch_size, seq_len
        )
        
        return combined_output
        
    def _compute_gates(self, x: jnp.ndarray,
                      expert_mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Compute load-balanced routing using expert weights"""
        gates = hk.Linear(self.num_experts, with_bias=False)(x)
        
        if expert_mask is not None:
            gates = jnp.where(expert_mask, gates, -1e9)
            
        # Normalize gates
        return jax.nn.softmax(gates, axis=-1)
        
    def _create_dispatch_tensor(self, gates: jnp.ndarray,
                              total_tokens: int) -> jnp.ndarray:
        """Create optimized dispatch tensor for token routing"""
        # Calculate capacity
        capacity = int(total_tokens * self.capacity_factor / self.num_experts)
        capacity = max(capacity, self.min_expert_capacity)
        
        # Get top-k gates
        top_gates, top_indices = jax.lax.top_k(gates, k=2)
        top_gates = top_gates / jnp.sum(top_gates, axis=-1, keepdims=True)
        
        # Create dispatch tensor
        dispatch_tensor = jnp.zeros(
            (self.num_experts, total_tokens),
            dtype=gates.dtype
        )
        
        for i in range(2):
            pos = top_indices[..., i]
            gates_i = top_gates[..., i]
            dispatch_tensor = dispatch_tensor.at[pos].add(gates_i)
            
        return dispatch_tensor
        
    def _dispatch_to_experts(self, inputs: jnp.ndarray,
                           dispatch_tensor: jnp.ndarray) -> List[jnp.ndarray]:
        """Dispatch tokens to experts"""
        expert_inputs = []
        for i in range(self.num_experts):
            expert_mask = dispatch_tensor[i] > 0
            if jnp.any(expert_mask):
                masked_input = inputs[expert_mask]
                expert_inputs.append(masked_input)
            else:
                expert_inputs.append(None)
        return expert_inputs
        
    def _process_with_experts(self, expert_inputs: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Process inputs with each expert"""
        expert_outputs = []
        for i in range(self.num_experts):
            if expert_inputs[i] is not None:
                expert = hk.Linear(self.expert_size)
                output = expert(expert_inputs[i])
                expert_outputs.append(output)
            else:
                expert_outputs.append(None)
        return expert_outputs
        
    def _combine_expert_outputs(self, expert_outputs: List[jnp.ndarray],
                              dispatch_tensor: jnp.ndarray,
                              batch_size: int, seq_len: int) -> jnp.ndarray:
        """Combine expert outputs efficiently"""
        # Stack valid outputs
        valid_outputs = [out for out in expert_outputs if out is not None]
        if not valid_outputs:
            return jnp.zeros((batch_size, seq_len, self.expert_size))
            
        stacked_experts = jnp.concatenate(valid_outputs, axis=0)
        
        # Combine using dispatch tensor
        combined = jnp.matmul(
            dispatch_tensor.transpose(),
            stacked_experts
        )
        
        return combined.reshape(batch_size, seq_len, -1)


class ExpertModule(hk.Module):
    """Individual expert module with TPU optimization"""
    
    def __init__(self, expert_size: int, hidden_size: int,
                 activation=jax.nn.gelu,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.expert_size = expert_size
        self.hidden_size = hidden_size
        self.activation = activation
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.hidden_size)(x)
        x = self.activation(x)
        return hk.Linear(self.expert_size)(x)


class ExpertRouter(hk.Module):
    """Router for selecting experts"""
    
    def __init__(self, num_experts: int,
                 noise_std: float = 1.0,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.noise_std = noise_std
        
    def __call__(self, x: jnp.ndarray,
                 is_training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Route tokens to experts"""
        # Router logits
        router_logits = hk.Linear(self.num_experts, with_bias=False)(x)
        
        if is_training:
            # Add noise during training for better load balancing
            noise = jax.random.normal(hk.next_rng_key(), router_logits.shape) * self.noise_std
            router_logits = router_logits + noise
            
        # Get routing probabilities
        routing_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Get expert assignments
        expert_indices = jnp.argmax(routing_probs, axis=-1)
        expert_weights = jnp.max(routing_probs, axis=-1)
        
        return expert_indices, expert_weights


class ExpertGating(hk.Module):
    """Optimized gating mechanism for expert routing with TPU efficiency"""
    
    def __init__(self, num_experts: int, expert_capacity: int,
                 noise_std: float = 1.0, use_balancing: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.noise_std = noise_std
        self.use_balancing = use_balancing
        
    def __call__(self, x: jnp.ndarray, expert_mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute load-balanced routing"""
        # Router logits
        router_logits = hk.Linear(self.num_experts, with_bias=False)(x)
        
        if is_training and self.noise_std > 0:
            # Add noise during training for better load balancing
            noise = jax.random.normal(hk.next_rng_key(), router_logits.shape) * self.noise_std
            router_logits = router_logits + noise
            
        if expert_mask is not None:
            router_logits = jnp.where(expert_mask, router_logits, -1e9)
            
        # Compute router probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Get expert assignments and compute dispatch
        top_k_gates, top_k_indices = jax.lax.top_k(router_probs, k=2)
        gates_sum = jnp.sum(top_k_gates, axis=-1, keepdims=True)
        top_k_gates = top_k_gates / gates_sum  # Normalize
        
        # Create dispatch tensors
        dispatch_tensor = jnp.zeros(
            (x.shape[0], self.num_experts, self.expert_capacity),
            dtype=x.dtype
        )
        
        # Expert balancing loss if needed
        balancing_loss = None
        if self.use_balancing and is_training:
            # Compute fraction of tokens going to each expert
            expert_usage = jnp.mean(router_probs, axis=0)
            target_usage = jnp.ones_like(expert_usage) / self.num_experts
            balancing_loss = jnp.sum(
                expert_usage * jnp.log(expert_usage / target_usage)
            )
        
        return top_k_gates, top_k_indices, dispatch_tensor, balancing_loss


# Auxiliary load balancing
def compute_load_balancing_loss(gates: jnp.ndarray, num_experts: int) -> jnp.ndarray:
    """Compute load balancing loss for better expert utilization"""
    # Compute fraction of tokens going to each expert
    router_probs = jax.nn.softmax(gates, axis=-1)
    fraction_to_experts = jnp.mean(router_probs, axis=0)
    
    # Target uniform distribution
    target_distribution = jnp.ones_like(fraction_to_experts) / num_experts
    
    # KL divergence loss for load balancing
    kl_loss = jnp.sum(fraction_to_experts * jnp.log(fraction_to_experts / target_distribution))
    
    return kl_loss


# Example usage
if __name__ == "__main__":
    def run_moe(x: jnp.ndarray) -> jnp.ndarray:
        moe = OptimizedMoE(
            num_experts=8,
            expert_size=512,
            input_size=512
        )
        return moe(x, is_training=True)
        
    # Initialize
    batch_size, seq_len, hidden_size = 2, 64, 512
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, seq_len, hidden_size))
    
    # Transform and initialize
    transformed = hk.transform(run_moe)
    params = transformed.init(rng, x)
    
    # Forward pass
    output = transformed.apply(params, rng, x)
    print("MoE Output shape:", output.shape)
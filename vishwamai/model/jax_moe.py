"""JAX implementation of Mixture of Experts layer."""
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax.sharding import PartitionSpec as P

class MoELayer(nn.Module):
    """Mixture of Experts layer implementation in JAX/Flax.
    
    Args:
        num_experts: Number of experts
        hidden_dim: Hidden dimension size
        expert_dim: Expert hidden dimension size
        num_tokens_per_expert: Maximum tokens per expert
        router_dtype: Data type for router computation
        router_bias: Whether to use router bias
        dtype: Data type for computations
        kernel_init: Weight initialization function
        deterministic: Whether to use deterministic dropout
    """
    num_experts: int
    hidden_dim: int
    expert_dim: Optional[int] = None
    num_tokens_per_expert: Optional[int] = None
    router_dtype: Any = jnp.float32
    router_bias: bool = False
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: bool = False

    def setup(self):
        """Initialize MoE components."""
        self.expert_dim = self.expert_dim or self.hidden_dim * 4
        
        # Router
        self.router = nn.Dense(
            self.num_experts,
            use_bias=self.router_bias,
            dtype=self.router_dtype,
            kernel_init=self.kernel_init,
            name='router'
        )
        
        # Experts
        self.experts = [
            Expert(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.expert_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                deterministic=self.deterministic
            )
            for _ in range(self.num_experts)
        ]

    def _compute_router_probabilities(
        self,
        hidden_states: jnp.ndarray,
        input_padding_mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute routing probabilities.
        
        Args:
            hidden_states: Input tensor
            input_padding_mask: Optional padding mask
            
        Returns:
            Tuple of:
            - Router probabilities
            - Expert indices
            - Load balancing loss
        """
        # Get router logits
        router_logits = self.router(hidden_states)
        
        if input_padding_mask is not None:
            router_logits = jnp.where(
                input_padding_mask[:, :, None],
                router_logits,
                jnp.finfo(router_logits.dtype).min
            )
            
        # Compute router probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Get expert assignments
        expert_indices = jnp.argmax(router_probs, axis=-1)
        
        # Compute load balancing loss
        expert_counts = jnp.sum(
            jax.nn.one_hot(expert_indices, self.num_experts),
            axis=(0, 1)
        )
        load_balancing_loss = jnp.mean(
            expert_counts * expert_counts
        ) * (self.num_experts / (jnp.sum(expert_counts) ** 2))
        
        return router_probs, expert_indices, load_balancing_loss

    def _dispatch_to_experts(
        self,
        hidden_states: jnp.ndarray,
        expert_indices: jnp.ndarray,
        router_probs: jnp.ndarray
    ) -> jnp.ndarray:
        """Dispatch tokens to experts.
        
        Args:
            hidden_states: Input tensor
            expert_indices: Expert assignments
            router_probs: Router probabilities
            
        Returns:
            Combined expert outputs
        """
        # Create binary dispatch matrix
        dispatch_mask = jax.nn.one_hot(expert_indices, self.num_experts)
        
        # Scale inputs by router probabilities
        expert_inputs = hidden_states * router_probs[..., None]
        
        # Dispatch to experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = dispatch_mask[..., i:i+1]
            tokens = expert_inputs * expert_mask
            
            # Run expert
            output = expert(tokens)
            expert_outputs.append(output)
            
        # Combine expert outputs
        combined_outputs = sum(expert_outputs)
        return combined_outputs

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_padding_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None
    ) -> Dict[str, jnp.ndarray]:
        """Apply MoE layer.
        
        Args:
            hidden_states: Input tensor
            input_padding_mask: Optional padding mask
            deterministic: Whether to use deterministic dropout
            
        Returns:
            Dict containing:
            - Output tensor
            - Router probabilities
            - Load balancing loss
        """
        # Compute routing probabilities
        router_probs, expert_indices, load_balancing_loss = self._compute_router_probabilities(
            hidden_states,
            input_padding_mask
        )
        
        # Dispatch tokens to experts and combine outputs
        outputs = self._dispatch_to_experts(
            hidden_states,
            expert_indices,
            router_probs
        )
        
        return {
            'hidden_states': outputs,
            'router_probs': router_probs,
            'load_balancing_loss': load_balancing_loss
        }

class Expert(nn.Module):
    """Expert network implementation.
    
    Args:
        hidden_dim: Hidden dimension size
        intermediate_dim: Intermediate dimension size
        activation: Activation function
        dropout_rate: Dropout probability
        dtype: Data type
        kernel_init: Weight initialization function
        deterministic: Whether to use deterministic dropout
    """
    hidden_dim: int
    intermediate_dim: int
    activation: Callable = nn.gelu
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: bool = False

    def setup(self):
        """Initialize expert components."""
        self.wi = nn.Dense(
            self.intermediate_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='wi'
        )
        self.wo = nn.Dense(
            self.hidden_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='wo'
        )
        self.dropout = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=self.deterministic
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        """Apply expert transformation.
        
        Args:
            hidden_states: Input tensor
            deterministic: Whether to use dropout
            
        Returns:
            Transformed tensor
        """
        x = self.wi(hidden_states)
        x = self.activation(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.wo(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

@partial(jax.jit, static_argnums=(1, 2))
def create_expert_partitions(
    hidden_states: jnp.ndarray,
    num_experts: int,
    num_tokens_per_expert: int
) -> Tuple[List[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Create expert partitions from input tensor.
    
    Args:
        hidden_states: Input tensor
        num_experts: Number of experts
        num_tokens_per_expert: Maximum tokens per expert
        
    Returns:
        Tuple of:
        - List of expert inputs
        - Combined expert mask
        - Load balancing loss
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device_count = jax.device_count()
    
    # Reshape for expert parallelism
    reshaped = hidden_states.reshape(-1, hidden_dim)
    total_tokens = reshaped.shape[0]
    
    # Compute tokens per partition
    tokens_per_partition = total_tokens // device_count
    
    # Create partitions
    partitions = []
    partition_masks = []
    start_idx = 0
    
    for i in range(device_count):
        end_idx = start_idx + min(tokens_per_partition, num_tokens_per_expert)
        
        # Get partition tokens
        partition = jax.lax.dynamic_slice(
            reshaped,
            (start_idx, 0),
            (end_idx - start_idx, hidden_dim)
        )
        partitions.append(partition)
        
        # Create mask
        mask = jnp.zeros((total_tokens,))
        mask = mask.at[start_idx:end_idx].set(1)
        partition_masks.append(mask)
        
        start_idx = end_idx
        
    combined_mask = jnp.stack(partition_masks)
    
    # Compute load balancing loss
    expert_counts = jnp.sum(combined_mask, axis=-1)
    load_balancing_loss = jnp.mean(
        expert_counts * expert_counts
    ) * (device_count / (jnp.sum(expert_counts) ** 2))
    
    return partitions, combined_mask, load_balancing_loss

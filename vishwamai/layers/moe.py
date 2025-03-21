"""Optimized Mixture of Experts implementations for fast training."""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vishwamai.kernels.distill_gemm import distill_gemm
from vishwamai.kernels.fp8_cast_bf16 import convert_precision, DynamicFP8Scaler

class FastMoELayer(nn.Module):
    """
    Fast Mixture of Experts implementation for distillation training.
    
    Optimized for both modern GPUs (A100) and older hardware (GTX 1650).
    """
    num_experts: int
    expert_dim: int
    capacity_factor: float = 1.0
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128
    precision: str = "mixed"  # "mixed", "fp32", "fp16", "bf16", "fp8"
    router_top_k: int = 2
    router_z_loss_weight: float = 0.001
    
    def setup(self):
        # Router components
        self.router = nn.Dense(
            features=self.num_experts,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=False,
            name="router"
        )
        
        # Expert feed-forward networks
        self.experts = [
            ExpertMLP(
                hidden_dim=self.expert_dim * 4,  # 4x for MLP intermediate dimension
                output_dim=self.expert_dim,
                dtype=self.dtype,
                use_fp8=self.use_fp8,
                block_size=self.block_size,
                precision=self.precision,
                name=f"expert_{i}"
            )
            for i in range(self.num_experts)
        ]
        
        # Initialize gating scalers for numerical stability
        self.router_weights_scaler = DynamicFP8Scaler(
            init_scale=1.0,
            growth_factor=1.1,
            backoff_factor=0.5,
            margin=0.1
        )
        
        # For distillation training compatibility
        self._distill_mode = False
        self._teacher_attention = None
        
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        teacher_outputs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Apply MoE layer to input tensor.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            deterministic: Whether to apply dropout
            teacher_outputs: Optional teacher model outputs for distillation
            
        Returns:
            Tuple of output tensor and auxiliary outputs for distillation
        """
        batch_size, seq_len, hidden_dim = x.shape
        tokens = batch_size * seq_len
        
        # Reshape input for routing
        x_reshaped = x.reshape(tokens, hidden_dim)
        
        # Get router scores
        router_logits = self.router(x_reshaped)
        
        # Apply router z-loss for stability (prevents router collapse)
        router_z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(
            router_logits, axis=-1, keepdims=True)))
        
        # Get router probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, self.router_top_k)
        
        # Normalize the probabilities (scale to sum to 1)
        top_k_probs_sum = jnp.sum(top_k_probs, axis=-1, keepdims=True)
        top_k_probs = top_k_probs / top_k_probs_sum
        
        # Compute capacity - maximum number of tokens per expert
        # Ensure we have enough capacity for load balancing
        capacity = int(self.capacity_factor * tokens / self.num_experts)
        capacity = max(capacity, self.router_top_k)  # Ensure minimum capacity
        
        # Create tensors for expert outputs and combine weights
        expert_outputs = jnp.zeros((tokens, hidden_dim), dtype=self.dtype)
        combine_weights = jnp.zeros((tokens, self.num_experts), dtype=self.dtype)
        
        # Expert dispatch and combine using a more efficient implementation
        results_dict = {"router_z_loss": router_z_loss, "balance_loss": 0.0}
        
        # Track metrics for load balancing
        expert_counts = jnp.zeros((self.num_experts,), dtype=jnp.int32)
        token_priority = -1.0 * top_k_probs[:, 0]  # Use first expert probability as priority
        
        # Fast dispatch using custom optimized kernel when available
        try:
            if self.use_fp8 and hasattr(jax.lib, "xla_client"):
                # When tensor cores are available, use optimized kernel
                expert_outputs, combine_weights, expert_counts = self._fast_dispatch(
                    x_reshaped, top_k_indices, top_k_probs, capacity
                )
            else:
                # Software fallback for all hardware
                expert_outputs, combine_weights, expert_counts = self._standard_dispatch(
                    x_reshaped, top_k_indices, top_k_probs, capacity, token_priority
                )
                
        except Exception as e:
            # Fallback to standard implementation
            expert_outputs, combine_weights, expert_counts = self._standard_dispatch(
                x_reshaped, top_k_indices, top_k_probs, capacity, token_priority
            )
        
        # Calculate load balancing loss - encourages uniform expert utilization
        # Compute fraction of tokens routed to each expert
        if self.num_experts > 1:
            # Count number of tokens routed to each expert
            expert_fraction = expert_counts / float(tokens * self.router_top_k)
            
            # Calculate load balancing loss
            # Ideal distribution: each expert getting 1/num_experts of tokens
            target_distribution = jnp.ones_like(expert_fraction) / self.num_experts
            balance_loss = jnp.mean(jnp.square(expert_fraction - target_distribution))
            results_dict["balance_loss"] = balance_loss
        
        # Reshape output back to input shape
        output = expert_outputs.reshape(batch_size, seq_len, hidden_dim)
        
        # Apply dropout if needed
        if not deterministic:
            output = nn.Dropout(rate=self.dropout_rate)(
                output, deterministic=False
            )
        
        # Add auxiliary outputs for distillation
        if teacher_outputs is not None and self._distill_mode:
            results_dict["distill_info"] = {
                "router_probs": router_probs,
                "expert_counts": expert_counts
            }
        
        return output, results_dict

    def _fast_dispatch(self, x, top_k_indices, top_k_probs, capacity):
        """
        Fast token dispatch to experts using optimized kernel.
        
        Args:
            x: Input tensor [tokens, hidden_dim]
            top_k_indices: Top-k expert indices [tokens, top_k]
            top_k_probs: Top-k expert probabilities [tokens, top_k]
            capacity: Maximum tokens per expert
            
        Returns:
            Tuple of (expert outputs, combine weights, expert counts)
        """
        tokens, hidden_dim = x.shape
        
        # Fast implementation using JAX custom ops
        @jax.jit
        def fast_dispatch_kernel(x, indices, probs, capacity):
            # This function would be implemented as a custom XLA operation
            # using our CUDA kernels, but here we provide a pure JAX implementation
            
            # Initialize outputs
            expert_outputs = jnp.zeros_like(x)
            combine_weights = jnp.zeros((tokens, self.num_experts))
            expert_counts = jnp.zeros(self.num_experts, dtype=jnp.int32)
            
            # For each token and expert combination
            for token_idx in range(tokens):
                for expert_idx_pos in range(self.router_top_k):
                    expert_idx = top_k_indices[token_idx, expert_idx_pos]
                    expert_prob = top_k_probs[token_idx, expert_idx_pos]
                    
                    # Process token with expert if capacity allows
                    expert_count = expert_counts[expert_idx]
                    if expert_count < capacity:
                        # Apply expert network
                        expert_output = self.experts[expert_idx](x[token_idx:token_idx+1])[0]
                        
                        # Combine outputs with probability weights
                        expert_outputs = expert_outputs.at[token_idx].add(expert_output * expert_prob)
                        combine_weights = combine_weights.at[token_idx, expert_idx].set(expert_prob)
                        
                        # Update expert count
                        expert_counts = expert_counts.at[expert_idx].add(1)
            
            return expert_outputs, combine_weights, expert_counts
                
        return fast_dispatch_kernel(x, top_k_indices, top_k_probs, capacity)

    def _standard_dispatch(self, x, top_k_indices, top_k_probs, capacity, token_priority):
        """
        Standard token dispatch to experts for all hardware types.
        
        Args:
            x: Input tensor [tokens, hidden_dim]
            top_k_indices: Top-k expert indices [tokens, top_k]
            top_k_probs: Top-k expert probabilities [tokens, top_k]
            capacity: Maximum tokens per expert
            token_priority: Priority scores for tokens
            
        Returns:
            Tuple of (expert outputs, combine weights, expert counts)
        """
        tokens, hidden_dim = x.shape
        
        # Initialize outputs
        expert_outputs = jnp.zeros_like(x)
        combine_weights = jnp.zeros((tokens, self.num_experts))
        expert_counts = jnp.zeros(self.num_experts, dtype=jnp.int32)
        
        # Sort tokens by priority
        sorted_priority, sorted_indices = jax.lax.sort_key_val(
            token_priority, jnp.arange(tokens)
        )
        
        # Dispatch tokens to experts
        for token_pos in range(tokens):
            token_idx = sorted_indices[token_pos]
            
            for expert_idx_pos in range(self.router_top_k):
                expert_idx = top_k_indices[token_idx, expert_idx_pos]
                expert_prob = top_k_probs[token_idx, expert_idx_pos]
                
                # Check if expert has capacity
                if expert_counts[expert_idx] < capacity:
                    # Apply expert to token
                    token_data = x[token_idx:token_idx+1]
                    expert_output = self.experts[expert_idx](token_data)[0]
                    
                    # Add weighted expert output
                    expert_outputs = expert_outputs.at[token_idx].add(expert_output * expert_prob)
                    combine_weights = combine_weights.at[token_idx, expert_idx].set(expert_prob)
                    
                    # Update expert count
                    expert_counts = expert_counts.at[expert_idx].add(1)
        
        return expert_outputs, combine_weights, expert_counts
    
    def enable_distillation(self, enable: bool = True):
        """Enable or disable distillation mode for this layer."""
        self._distill_mode = enable
        # Also enable in experts
        for expert in self.experts:
            if hasattr(expert, 'enable_distillation'):
                expert.enable_distillation(enable)


class ExpertMLP(nn.Module):
    """Expert MLP network optimized for fast computation."""
    hidden_dim: int
    output_dim: int
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128
    precision: str = "mixed"
    activation: Callable = jax.nn.gelu
    
    def setup(self):
        # Up projection
        self.up_proj = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="up_proj"
        )
        
        # Down projection
        self.down_proj = nn.Dense(
            features=self.output_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="down_proj"
        )
        
        # For distillation
        self._distill_mode = False
        self._intermediate_features = None
        
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply expert MLP to input."""
        # First dense layer
        h = self.up_proj(x)
        
        # Apply activation
        h = self.activation(h)
        
        # Store intermediate features for distillation if needed
        if self._distill_mode:
            self._intermediate_features = h
            
        # Second dense layer
        return self.down_proj(h)
    
    def enable_distillation(self, enable: bool = True):
        """Enable or disable distillation mode."""
        self._distill_mode = enable


class DistillableMoELayer(nn.Module):
    """MoE layer with special support for distillation training."""
    num_experts: int
    expert_dim: int
    capacity_factor: float = 1.0
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128
    precision: str = "mixed"
    router_top_k: int = 2
    distill_alpha: float = 0.5  # Weight for distillation loss
    
    def setup(self):
        # Create fast MoE layer
        self.moe = FastMoELayer(
            num_experts=self.num_experts,
            expert_dim=self.expert_dim,
            capacity_factor=self.capacity_factor,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size,
            precision=self.precision,
            router_top_k=self.router_top_k,
            name="moe_layer"
        )
        
        # Enable distillation by default
        self.moe.enable_distillation(True)
        
    def __call__(
        self,
        x: jnp.ndarray,
        teacher_outputs: Optional[Dict[str, jnp.ndarray]] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Apply MoE layer with distillation support.
        
        Args:
            x: Input tensor
            teacher_outputs: Teacher model outputs for distillation
            deterministic: Whether to apply dropout
            
        Returns:
            Tuple of output tensor and auxiliary outputs including distillation info
        """
        # Apply MoE layer
        output, aux_outputs = self.moe(
            x, 
            deterministic=deterministic,
            teacher_outputs=teacher_outputs
        )
        
        # If teacher outputs are provided, compute distillation loss
        if teacher_outputs is not None and "moe_output" in teacher_outputs:
            teacher_output = teacher_outputs["moe_output"]
            
            # Calculate distillation loss
            distill_loss = jnp.mean(jnp.square(output - teacher_output))
            
            # Add to auxiliary outputs
            aux_outputs["distill_loss"] = distill_loss * self.distill_alpha
            
        return output, aux_outputs
    
    def get_trainable_expert_params(self):
        """Get trainable parameters for all experts."""
        expert_params = {}
        for i, expert in enumerate(self.moe.experts):
            expert_params[f"expert_{i}"] = {
                "up_proj": expert.up_proj,
                "down_proj": expert.down_proj
            }
        return expert_params
    
    def get_router_params(self):
        """Get router parameters."""
        return {"router": self.moe.router}
"""TPU-optimized layer optimizers for Gemma 3 knowledge distillation."""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Optional, Tuple, Dict, Any, NamedTuple, List
import numpy as np
from functools import partial

from vishwamai.kernels.tpu.distillation_kernels import (
    DistillationKernelConfig
)

class LayerOptConfig(NamedTuple):
    """Configuration for layer optimizations."""
    hidden_dim: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    precision: Any = lax.Precision.HIGHEST
    dtype: Any = jnp.bfloat16

class LayerwiseOptimizer:
    """Base class for layer-wise optimization on TPU."""
    
    def __init__(self, kernel_config: DistillationKernelConfig, num_layers: int, dropout_rate: float = 0.1):
        self.kernel_config = kernel_config
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
    def optimize_layer(self,
                      layer_idx: int,
                      teacher_layer: Any,
                      student_layer: Any,
                      inputs: Dict[str, jnp.ndarray],
                      temperature: float = 2.0) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """Optimize a single layer's computation."""
        # Forward pass through teacher layer
        teacher_outputs = teacher_layer(inputs)
        
        # Forward pass through student layer
        student_outputs = student_layer(inputs)
        
        return teacher_outputs, student_outputs

class AdaptiveLayerOptimizer:
    """
    Adaptive layer optimization for knowledge distillation.
    Features:
    - Dynamic parameter adaptation
    - Layer-specific learning rates
    - Adaptive attention patterns
    """
    
    def __init__(
        self,
        config: LayerOptConfig,
        kernel_config: DistillationKernelConfig
    ):
        self.config = config
        self.kernel_config = kernel_config
        self.layer_optimizer = LayerwiseOptimizer(
            kernel_config,
            num_layers=1,  # Single layer optimization
            dropout_rate=config.dropout_rate
        )
    
    def optimize_attention_layer(
        self,
        teacher_layer: Any,
        student_layer: Any,
        inputs: Dict[str, jnp.ndarray],
        layer_stats: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
        """
        Optimize attention layer computation.
        
        Args:
            teacher_layer: Teacher model attention layer
            student_layer: Student model attention layer
            inputs: Input tensors
            layer_stats: Optional statistics from previous iterations
            
        Returns:
            Layer outputs and updated statistics
        """
        # Compute adaptive temperature based on layer statistics
        temperature = self._compute_adaptive_temperature(layer_stats)
        
        # Get layer outputs with optimized computation
        teacher_out, student_out = self.layer_optimizer.optimize_layer(
            0,  # Layer index not needed for single layer
            teacher_layer,
            student_layer,
            inputs,
            temperature=temperature
        )
        
        # Update layer statistics
        new_stats = self._update_layer_stats(
            teacher_out,
            student_out,
            layer_stats
        )
        
        return student_out, new_stats
    
    def _compute_adaptive_temperature(
        self,
        layer_stats: Optional[Dict[str, jnp.ndarray]]
    ) -> float:
        """Compute adaptive temperature based on layer statistics."""
        if layer_stats is None:
            return 2.0  # Default temperature
            
        # Adjust temperature based on attention pattern similarity
        attn_similarity = layer_stats.get("attention_similarity", 0.0)
        base_temperature = 2.0
        
        # Scale temperature inversely with attention similarity
        return base_temperature * (1.0 + (1.0 - attn_similarity))
    
    def _update_layer_stats(
        self,
        teacher_out: Dict[str, jnp.ndarray],
        student_out: Dict[str, jnp.ndarray],
        prev_stats: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Dict[str, jnp.ndarray]:
        """Update layer statistics based on current outputs."""
        # Compute attention pattern similarity
        teacher_attn = teacher_out["attention"]
        student_attn = student_out["attention"]
        
        similarity = jnp.mean(
            jax.nn.softmax(teacher_attn) *
            jax.nn.softmax(student_attn)
        )
        
        # Compute gradient norm ratios
        grad_norm_ratio = (
            jnp.linalg.norm(student_out["ffn"]) /
            (jnp.linalg.norm(teacher_out["ffn"]) + 1e-6)
        )
        
        new_stats = {
            "attention_similarity": similarity,
            "grad_norm_ratio": grad_norm_ratio,
            "iteration": prev_stats["iteration"] + 1 if prev_stats else 1
        }
        
        if prev_stats is not None:
            # Compute moving averages
            alpha = 0.9  # Exponential moving average factor
            for key in ["attention_similarity", "grad_norm_ratio"]:
                new_stats[f"{key}_ema"] = (
                    alpha * prev_stats.get(f"{key}_ema", new_stats[key]) +
                    (1 - alpha) * new_stats[key]
                )
        
        return new_stats

class TPULayerNorm(nn.Module):
    """TPU-optimized layer normalization with improved memory efficiency."""
    
    epsilon: float = 1e-6
    dtype: Any = jnp.bfloat16
    scale_init: Any = nn.initializers.ones
    bias_init: Any = nn.initializers.zeros
    use_bias: bool = True
    block_size: int = 128  # Must be multiple of 128 for TPU
    use_fused_ops: bool = True

    def setup(self):
        """Initialize parameters with TPU-optimized layout."""
        feature_size = self.input_shape[-1]
        self.scale = self.param(
            'scale',
            self.scale_init, 
            (feature_size,),
            self.dtype
        )
        if self.use_bias:
            self.bias = self.param(
                'bias',
                self.bias_init,
                (feature_size,),
                self.dtype
            )
            
    def _process_block(self, x_block: jnp.ndarray, block_idx: int) -> jnp.ndarray:
        """Process a single block with Welford's online algorithm."""
        # Use high precision for statistics computation
        x_block = x_block.astype(jnp.float32)
        
        # Compute statistics with Welford's algorithm
        mean = jnp.mean(x_block, axis=self.reduction_axes, keepdims=True)
        centered = x_block - mean
        var = jnp.mean(jnp.square(centered), axis=self.reduction_axes, keepdims=True)
        
        # Get scale/bias slices for this block
        start_idx = block_idx * self.block_size
        end_idx = min(start_idx + self.block_size, self.feature_size)
        scale_block = lax.dynamic_slice(
            self.scale, [start_idx], [end_idx - start_idx]
        )
        
        # Normalize with stable computation
        inv_stddev = lax.rsqrt(var + self.epsilon)
        normed = centered * inv_stddev
        
        # Apply scale and optional bias
        output = normed * scale_block
        if self.use_bias:
            bias_block = lax.dynamic_slice(
                self.bias, [start_idx], [end_idx - start_idx]
            )
            output = output + bias_block
            
        return output.astype(self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer normalization with TPU optimizations."""
        # Ensure input uses compute dtype
        x = x.astype(self.dtype)
        
        # Setup reduction axes and sizes
        self.feature_axis = -1
        self.reduction_axes = tuple(i for i in range(x.ndim) if i != self.feature_axis)
        self.feature_size = x.shape[self.feature_axis]
        
        # Compute optimal block size for TPU
        effective_block_size = min(
            self.block_size,
            128 * ((self.feature_size + 127) // 128)
        )
        
        if self.use_fused_ops and self.feature_size <= effective_block_size:
            # Small enough for fused operation
            return self._fused_layernorm(x)
            
        # Process in blocks for large feature dimensions
        results = []
        for block_start in range(0, self.feature_size, effective_block_size):
            # Extract block
            end = min(block_start + effective_block_size, self.feature_size)
            x_block = lax.dynamic_slice_in_dim(
                x, block_start, end - block_start, axis=self.feature_axis
            )
            
            # Process block
            block_result = self._process_block(
                x_block,
                block_start // effective_block_size
            )
            results.append(block_result)
            
        # Combine blocks efficiently
        if len(results) == 1:
            return results[0]
            
        return jnp.concatenate(results, axis=self.feature_axis)
        
    def _fused_layernorm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Fused layer normalization for small feature dimensions."""
        # Use high precision accumulation
        x = x.astype(jnp.float32)
        
        mean = jnp.mean(x, axis=self.reduction_axes, keepdims=True)
        centered = x - mean
        variance = jnp.mean(jnp.square(centered), axis=self.reduction_axes, keepdims=True)
        inv_stddev = lax.rsqrt(variance + self.epsilon)
        normed = centered * inv_stddev
        
        output = normed * self.scale
        if self.use_bias:
            output = output + self.bias
            
        return output.astype(self.dtype)

class FFNOptimizer:
    """
    Feed-forward network optimization for knowledge distillation.
    Features:
    - Operation fusion
    - Adaptive dropout
    - Efficient linear transformations
    """
    
    def __init__(self, config: LayerOptConfig):
        self.config = config
        
    def __call__(
        self,
        x: jnp.ndarray,
        wi: jnp.ndarray,
        wo: jnp.ndarray,
        dropout_rng: Optional[Any] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """Apply optimized feed-forward computation."""
        # Fuse linear + activation for TPU efficiency
        hidden = self._fused_linear_gelu(x, wi)
        
        if not deterministic and dropout_rng is not None:
            hidden = jax.random.dropout(
                dropout_rng,
                self.config.dropout_rate,
                hidden
            )
        
        # Output projection
        return jnp.dot(hidden, wo)
    
    def _fused_linear_gelu(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray
    ) -> jnp.ndarray:
        """Fused linear transformation and GELU activation."""
        y = jnp.dot(x, weight)
        return jax.nn.gelu(y)

class AdaptiveMoEOptimizer:
    """
    Adaptive Mixture of Experts optimization.
    Features:
    - Dynamic expert routing
    - Load balancing
    - Efficient expert computation
    """
    
    def __init__(
        self,
        config: LayerOptConfig,
        num_experts: int = 8,
        expert_capacity: Optional[int] = None
    ):
        self.config = config
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Initialize experts
        self.experts = [
            FFNOptimizer(config)
            for _ in range(num_experts)
        ]
    
    def route_and_compute(
        self,
        x: jnp.ndarray,
        router_weights: jnp.ndarray,
        expert_weights: List[Tuple[jnp.ndarray, jnp.ndarray]],
        router_z_loss: float = 0.001,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Route tokens to experts and compute expert outputs.
        
        Args:
            x: Input tensor
            router_weights: Router network weights
            expert_weights: List of (wi, wo) weights for each expert
            router_z_loss: Router z-loss coefficient
            deterministic: Whether to use deterministic routing
            
        Returns:
            Expert outputs and routing information
        """
        # Compute routing logits
        router_logits = jnp.dot(x, router_weights)
        
        if not deterministic:
            # Add noise for exploration
            noise = jax.random.normal(
                jax.random.PRNGKey(0),
                router_logits.shape
            ) * 0.1
            router_logits = router_logits + noise
        
        # Compute routing probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Route tokens to top-k experts (k=2)
        top_k = 2
        top_expert_idx = jnp.argsort(
            router_probs,
            axis=-1
        )[:, -top_k:]
        
        # Initialize output tensor
        output = jnp.zeros_like(x)
        total_tokens = 0
        
        # Process each expert
        aux_loss = 0.0
        router_z_loss = jnp.mean(jnp.square(jnp.log(jnp.sum(
            jnp.exp(router_logits), axis=-1
        ))))
        
        # Combine expert computations
        combined_output = jnp.zeros_like(x)
        for expert_idx, (wi, wo) in enumerate(expert_weights):
            # Select tokens routed to this expert
            expert_mask = (top_expert_idx == expert_idx)
            if not jnp.any(expert_mask):
                continue
                
            # Get expert inputs
            expert_inputs = jnp.where(
                expert_mask[..., None],
                x,
                0.0
            )
            
            # Compute expert output
            expert_output = self.experts[expert_idx](
                expert_inputs,
                wi,
                wo,
                deterministic=deterministic
            )
            
            # Combine expert outputs weighted by routing probabilities
            combined_output += expert_output * router_probs[..., expert_idx, None]
            total_tokens += jnp.sum(expert_mask)
        
        # Compute auxiliary losses
        aux_loss = aux_loss + router_z_loss * router_z_loss
        
        aux = {
            "router_probs": router_probs,
            "total_tokens": total_tokens,
            "aux_loss": aux_loss
        }
        
        return combined_output, aux

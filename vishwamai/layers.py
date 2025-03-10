import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable
from .kernel import fp8_gemm_optimized, act_quant

class MLABlock(nn.Module):
    """Multi-head Linear Attention Block with TPU optimization"""
    num_heads: int
    head_dim: int = 128
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        batch_size, seq_len, hidden_dim = x.shape
        head_dim = hidden_dim // self.num_heads
        
        # Project queries, keys, and values
        qkv = nn.Dense(3 * hidden_dim, use_bias=False)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        queries, keys, values = jnp.split(qkv, 3, axis=2)
        
        # Reshape for attention computation
        queries = queries.transpose((0, 2, 1, 3))  # (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose((0, 2, 1, 3))
        values = values.transpose((0, 2, 1, 3))
        
        # Optimized attention using FP8 GEMM
        # Quantize for TPU
        queries_quant, q_scale = act_quant(queries)
        keys_quant, k_scale = act_quant(keys)
        
        # Compute attention scores with FP8 precision
        attention_scores = fp8_gemm_optimized(
            queries_quant, q_scale,
            keys_quant.transpose((0, 1, 3, 2)), k_scale
        )
        
        # Scale attention scores
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        if training:
            attention_weights = nn.Dropout(rate=self.dropout_rate)(
                attention_weights, deterministic=not training
            )
        
        # Apply attention to values
        values_quant, v_scale = act_quant(values)
        attention_output = fp8_gemm_optimized(attention_weights, jnp.ones_like(q_scale),
                                            values_quant, v_scale)
        
        # Reshape and project output
        attention_output = attention_output.transpose((0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_dim)
        
        return nn.Dense(hidden_dim)(attention_output)

class MoELayer(nn.Module):
    """Mixture of Experts layer with dynamic routing"""
    num_experts: int
    ffn_dim: int
    capacity_factor: float = 1.2
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Router dense layer
        router_logits = nn.Dense(self.num_experts)(x)
        
        # Calculate routing probabilities
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts (k=2 for standard MoE)
        k = 2
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, k)
        
        # Normalize probabilities
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
        
        # Initialize expert outputs
        expert_outputs = []
        
        # Process inputs through each expert
        for expert_idx in range(self.num_experts):
            # Expert FFN
            expert_fn = nn.Sequential([
                nn.Dense(self.ffn_dim),
                nn.gelu,
                nn.Dropout(rate=self.dropout_rate)(deterministic=not training),
                nn.Dense(hidden_dim)
            ])
            
            # Get inputs routed to this expert
            expert_mask = (top_k_indices == expert_idx)
            expert_mask = jnp.any(expert_mask, axis=-1)
            
            if jnp.any(expert_mask):
                # Process expert inputs
                expert_input = x[expert_mask]
                expert_output = expert_fn(expert_input)
                
                # Store output with routing weights
                expert_outputs.append((expert_output, expert_mask, top_k_probs[..., 0]))
        
        # Combine expert outputs
        combined_output = jnp.zeros((batch_size, seq_len, hidden_dim))
        for expert_output, mask, prob in expert_outputs:
            update = jnp.where(mask[..., None], expert_output * prob[..., None], 0)
            combined_output = combined_output + update
            
        return combined_output
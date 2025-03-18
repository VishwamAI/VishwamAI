"""TPU-optimized attention layers for VishwamAI."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict, List, Union, Tuple
from vishwamai.kernels.sparse import sparse_attention, block_sparse_attention

class FlashAttention(nn.Module):
    """
    TPU-optimized Flash Attention implementation.
    
    A more efficient attention implementation that avoids materializing 
    the full attention matrix for better memory usage.
    """
    num_heads: int
    head_dim: Optional[int] = None
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal"
    )
    bias_init: Callable = nn.initializers.zeros
    use_rope: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    block_size: int = 64
    
    @nn.compact
    def __call__(
        self,
        inputs_q: jnp.ndarray,
        inputs_kv: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Apply Flash Attention to inputs."""
        if inputs_kv is None:
            inputs_kv = inputs_q
            
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        head_dim = self.head_dim or qkv_features // self.num_heads
        
        # Project inputs to queries, keys, and values
        query = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="query"
        )(inputs_q)
        
        key = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="key"
        )(inputs_kv)
        
        value = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="value"
        )(inputs_kv)
        
        # Reshape to multi-head attention
        batch, q_len, _ = inputs_q.shape
        _, kv_len, _ = inputs_kv.shape
        
        # Reshape heads
        query = query.reshape(batch, q_len, self.num_heads, head_dim)
        key = key.reshape(batch, kv_len, self.num_heads, head_dim)
        value = value.reshape(batch, kv_len, self.num_heads, head_dim)
        
        # Transpose for batched matrix multiplication
        query = query.transpose((0, 2, 1, 3))  # [batch, heads, q_len, head_dim]
        key = key.transpose((0, 2, 1, 3))      # [batch, heads, kv_len, head_dim]
        value = value.transpose((0, 2, 1, 3))  # [batch, heads, kv_len, head_dim]
        
        # Apply rotary embeddings if needed
        if self.use_rope:
            from vishwamai.layers.rotary import TPURotaryEmbedding, apply_rotary_pos_emb
            
            rotary = TPURotaryEmbedding(
                dim=head_dim,
                max_seq_len=max(q_len, kv_len)
            )
            cos, sin = rotary(
                jnp.zeros((1, max(q_len, kv_len), head_dim)),
                seq_len=max(q_len, kv_len)
            )
            query, key = apply_rotary_pos_emb(query, key, cos[:q_len], sin[:q_len])
        
        # Flash attention implementation
        # Compute attention in blocks to avoid materializing full attention matrix
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Split processing into chunks for TPU efficiency
        def process_query_block(query_block, key_block, value_block, mask_block=None):
            # Compute scaled attention scores
            attention_scores = jnp.matmul(query_block, jnp.transpose(key_block, (0, 1, 3, 2))) * scale
            
            # Apply mask if provided
            if mask_block is not None:
                attention_scores = jnp.where(
                    mask_block > 0,
                    attention_scores,
                    jnp.full_like(attention_scores, -1e10)
                )
                
            # Apply attention with stable softmax
            attention_weights = nn.softmax(attention_scores, axis=-1)
            
            # Apply attention dropout if training
            if not deterministic and self.dropout_rate > 0:
                attention_weights = nn.Dropout(
                    rate=self.dropout_rate, 
                    deterministic=deterministic
                )(attention_weights)
                
            # Compute weighted sum
            return jnp.matmul(attention_weights, value_block)
            
        # Process using block algorithm based on sequence length
        if max(q_len, kv_len) > 1024:
            # Use block-sparse attention for very long sequences
            output = block_sparse_attention(
                query, key, value,
                block_size=self.block_size,
                causal=mask is not None
            )
        else:
            # Standard attention with TPU optimization
            output = process_query_block(query, key, value, mask)
                
        # Transpose back and combine heads
        output = output.transpose((0, 2, 1, 3))  # [batch, q_len, heads, head_dim]
        output = output.reshape(batch, q_len, qkv_features)
        
        # Final projection
        output = nn.Dense(
            features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output"
        )(output)
        
        return output

class ChunkwiseCausalAttention(nn.Module):
    """
    Chunk-wise causal attention optimized for TPU.
    
    Processes sequences in chunks for better TPU utilization.
    """
    num_heads: int
    head_dim: Optional[int] = None
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    chunk_size: int = 128
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal"
    )
    bias_init: Callable = nn.initializers.zeros
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
        cached_attention: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Apply chunkwise causal attention to inputs."""
        features = self.out_features or inputs.shape[-1]
        qkv_features = self.qkv_features or inputs.shape[-1]
        head_dim = self.head_dim or qkv_features // self.num_heads
        
        # Project inputs to queries, keys, and values
        qkv_projection = nn.Dense(
            qkv_features * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="qkv"
        )(inputs)
        
        # Split into q, k, v and reshape to multihead
        batch, seq_len, _ = inputs.shape
        qkv = qkv_projection.reshape(batch, seq_len, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Handle cached attention for autoregressive inference
        if cached_attention is not None:
            cache_k, cache_v = cached_attention["cached_key"], cached_attention["cached_value"]
            
            # Update key, value caches for incremental decoding
            one_token = seq_len == 1
            if one_token:
                key = jnp.concatenate([cache_k, key], axis=2)
                value = jnp.concatenate([cache_v, value], axis=2)
                
            # Update cache
            cached_attention = {"cached_key": key, "cached_value": value}
        
        # Process attention in chunks
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Chunk the sequences
        num_chunks = seq_len // self.chunk_size + (1 if seq_len % self.chunk_size > 0 else 0)
        chunk_outputs = []
        
        for i in range(num_chunks):
            # Get current chunk indices
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            
            # Extract current query chunk
            query_chunk = jax.lax.dynamic_slice(
                query,
                (0, 0, start_idx, 0),
                (batch, self.num_heads, end_idx - start_idx, head_dim)
            )
            
            # Process chunks
            if cached_attention is not None and one_token:
                # For incremental decoding: attend to full key/value
                # This is the typical case for autoregressive generation
                attention_scores = jnp.matmul(
                    query_chunk, 
                    jnp.transpose(key, (0, 1, 3, 2))
                ) * scale
                
                # Apply causal mask for the single new token
                seq_idx = cached_attention.get("seq_idx", seq_len - 1)
                causal_mask = jnp.greater_equal(
                    jnp.arange(key.shape[2])[None, None, None, :],
                    seq_idx
                )
                
                attention_scores = jnp.where(
                    causal_mask,
                    jnp.full_like(attention_scores, -1e10),
                    attention_scores
                )
            else:
                # Standard chunked attention
                # Create causal mask for this chunk
                q_indices = jnp.arange(start_idx, end_idx)
                k_indices = jnp.arange(seq_len)
                causal_mask = k_indices[None, :] <= q_indices[:, None]
                causal_mask = causal_mask[None, None, :, :]
                
                # Compute attention scores
                attention_scores = jnp.matmul(
                    query_chunk, 
                    jnp.transpose(key, (0, 1, 3, 2))
                ) * scale
                
                # Apply causal mask
                attention_scores = jnp.where(
                    causal_mask,
                    attention_scores,
                    jnp.full_like(attention_scores, -1e10)
                )
            
            # Apply softmax
            attention_weights = nn.softmax(attention_scores, axis=-1)
            
            # Apply dropout if training
            if not deterministic and self.dropout_rate > 0:
                attention_weights = nn.Dropout(
                    rate=self.dropout_rate
                )(attention_weights, deterministic=False)
            
            # Apply attention to values
            chunk_output = jnp.matmul(attention_weights, value)
            chunk_outputs.append(chunk_output)
        
        # Concatenate chunk outputs
        if len(chunk_outputs) > 1:
            output = jnp.concatenate(chunk_outputs, axis=2)
        else:
            output = chunk_outputs[0]
        
        # Transpose back and combine heads
        output = output.transpose((0, 2, 1, 3))  # [batch, seq_len, heads, head_dim]
        output = output.reshape(batch, -1, qkv_features)
        
        # Final projection
        output = nn.Dense(
            features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output"
        )(output)
        
        if cached_attention is not None:
            return output, cached_attention
        else:
            return output
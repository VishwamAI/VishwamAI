"""
TPU-optimized Transformer implementation for VishwamAI
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Dict, Tuple, Generator
from .layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention
import optax
import flax
import time
import jax.lax as lax
from flax.training import train_state
from vishwamai.layers.attention import FlashAttention
from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant, optimize_kernel_layout

class TPUGEMMLinear(nn.Module):
    """TPU v2-optimized linear layer using FP8 GEMM operations"""
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel',
                          self.kernel_init,
                          (inputs.shape[-1], self.features))

        # TPU v2 optimized quantization with proper axis handling
        inputs_quant, input_scale = act_quant(
            inputs, 
            num_bits=8,
            axis=None  # Let act_quant determine proper axes based on input shape
        )
        kernel_quant, kernel_scale = act_quant(
            kernel, 
            num_bits=8,
            axis=None
        )
        
        # Optimize memory layout for TPU
        inputs_quant = optimize_kernel_layout(inputs_quant)
        kernel_quant = optimize_kernel_layout(kernel_quant)

        # Process in chunks for TPU v2 memory efficiency
        chunk_size = 32  # Optimal for TPU v2
        num_chunks = max(1, inputs.shape[0] // chunk_size)
        outputs = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, inputs.shape[0])
            
            # Get current chunk
            chunk_inputs = lax.dynamic_slice(
                inputs_quant,
                (start_idx,) + (0,) * (inputs_quant.ndim - 1),
                (end_idx - start_idx,) + inputs_quant.shape[1:]
            )
            chunk_scale = lax.dynamic_slice(
                input_scale,
                (start_idx,) + (0,) * (input_scale.ndim - 1),
                (end_idx - start_idx,) + input_scale.shape[1:]
            )
            
            # Perform FP8 matrix multiplication for chunk
            chunk_output = fp8_gemm_optimized(
                chunk_inputs, 
                kernel_quant,
                dtype=self.dtype
            )
            outputs.append(chunk_output)
        
        y = jnp.concatenate(outputs, axis=0) if num_chunks > 1 else outputs[0]
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
            
        return y

class TPULayerNorm(nn.Module):
    """TPU-optimized Layer Normalization"""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feat_shape = (x.shape[-1],)
        scale = self.param('scale', self.scale_init, feat_shape)
        bias = self.param('bias', self.bias_init, feat_shape)
        
        # Compute statistics
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        
        # Normalize and apply scale/bias
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        return x_norm * scale + bias

class MultiHeadAttention(nn.Module):
    """Multi-head attention with TPU optimizations and relative position encoding"""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    max_relative_position: int = 32
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self,
                 inputs_q: jnp.ndarray,
                 inputs_kv: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        
        # Project inputs to Q, K, V
        query = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='query'
        )(inputs_q)
        
        key = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='key'
        )(inputs_kv)
        
        value = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='value'
        )(inputs_kv)

        batch_size = inputs_q.shape[0]
        seq_len_q = inputs_q.shape[1]
        seq_len_kv = inputs_kv.shape[1]

        # Relative position embeddings
        rel_pos_emb = self.param(
            'rel_pos_embedding',
            nn.initializers.normal(0.02),
            (2 * self.max_relative_position + 1, self.head_dim),
            self.dtype
        )

        # Create relative position matrix
        positions = jnp.arange(seq_len_q)[:, None] - jnp.arange(seq_len_kv)[None, :]
        positions = jnp.clip(
            positions + self.max_relative_position,
            0,
            2 * self.max_relative_position
        )

        # Reshape for attention computation
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for batched matrix multiplication
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))

        # Get relative position embeddings for current sequence
        rel_pos_bias = jnp.take(rel_pos_emb, positions, axis=0)
        rel_pos_bias = rel_pos_bias.transpose((2, 0, 1))[None, :, :, :]

        # Compute content-based attention with FP8 precision
        key_quant, key_scale = act_quant(key.transpose((0, 1, 3, 2)))
        query_quant, query_scale = act_quant(query)
        
        content_scores = fp8_gemm_optimized(
            query_quant, query_scale,
            key_quant, key_scale
        )

        # Add relative position bias
        attention_scores = content_scores + rel_pos_bias
        
        # Adaptive attention scaling
        scale_factor = jnp.sqrt(query.shape[-1]) * jnp.power(
            jnp.minimum(seq_len_q, seq_len_kv), 0.25
        )
        attention_scores = attention_scores / scale_factor

        # Apply mask if provided
        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, -1e10)

        # Compute attention weights with stable softmax
        max_score = jnp.max(attention_scores, axis=-1, keepdims=True)
        exp_weights = jnp.exp(attention_scores - max_score)
        attention_weights = exp_weights / (jnp.sum(exp_weights, axis=-1, keepdims=True) + 1e-6)

        if not deterministic:
            attention_weights = nn.Dropout(rate=self.dropout_rate)(
                attention_weights, deterministic=False
            )

        # Apply attention to values using FP8 precision
        value_quant, value_scale = act_quant(value)
        attention_output = fp8_gemm_optimized(
            attention_weights, jnp.ones_like(query_scale),
            value_quant, value_scale
        )

        # Reshape and project output
        attention_output = attention_output.transpose((0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final projection with skip connection
        output = TPUGEMMLinear(
            features=inputs_q.shape[-1],
            dtype=self.dtype,
            name='output'
        )(attention_output)

        return output

class TransformerBlock(nn.Module):
    """Transformer block with TPU optimizations"""
    num_heads: int
    head_dim: int 
    mlp_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self,
                 inputs: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        # Detect hardware for optimal kernel selection
        platform = jax.devices()[0].platform
        
        # Layer normalization before attention (Pre-LN architecture)
        if platform == 'tpu':
            # TPU-optimized layer norm
            x = TPULayerNorm(dtype=self.dtype)(inputs)
        else:
            # Standard layer norm for other platforms
            x = nn.LayerNorm(dtype=self.dtype, epsilon=1e-6)(inputs)
        
        # Multi-head attention with hardware-specific optimizations
        if platform == 'tpu':
            # TPU-optimized attention with FP8 precision
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(x, x, mask=mask, deterministic=deterministic)
        elif platform == 'gpu' and hasattr(jax.lib, 'xla_bridge'):
            # Use FlashAttention if available for GPU
            try:
                attention_output = FlashAttention(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype
                )(x, x, mask=mask, deterministic=deterministic)
            except:
                # Fall back to standard attention
                attention_output = MultiHeadAttention(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype
                )(x, x, mask=mask, deterministic=deterministic)
        else:
            # CPU-optimized attention with memory-efficient implementation
            attention_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(x, x, mask=mask, deterministic=deterministic)
        
        # Residual connection after attention
        x = attention_output + inputs
        
        # Layer norm before MLP
        if platform == 'tpu':
            y = TPULayerNorm(dtype=self.dtype)(x)
        else:
            y = nn.LayerNorm(dtype=self.dtype, epsilon=1e-6)(x)
        
        # MLP with hardware-specific optimizations
        if platform == 'tpu':
            # TPU-optimized MLP with FP8 precision
            mlp_output = self._mlp_tpu(y, deterministic=deterministic)
        elif platform == 'gpu':
            # GPU-optimized MLP with kernel fusion
            mlp_output = self._mlp_gpu(y, deterministic=deterministic)
        else:
            # CPU-optimized MLP
            mlp_output = self._mlp_cpu(y, deterministic=deterministic)
        
        # Final residual connection
        return mlp_output + x
    
    def _mlp_tpu(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """TPU-optimized MLP with FP8 precision and kernel fusion."""
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self,
                 inputs: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        
        # Token and position embeddings
        x = TokenEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_dim,
            dtype=self.dtype
        )(inputs)
        
        positions = jnp.arange(inputs.shape[1])[None, :]
        position_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_seq_len, self.hidden_dim),
            self.dtype
        )
        x = x + position_embedding[:, :inputs.shape[1]]

        if not deterministic:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name=f'transformer_block_{i}'
            )(x, mask, deterministic)

        # Output layer normalization
        x = TPULayerNorm(dtype=self.dtype)(x)
        
        # Final projection to vocabulary
        logits = TPUGEMMLinear(
            features=self.vocab_size,
            dtype=self.dtype,
            name='output'
        )(x)

        return logits

class TransformerComputeLayerTPU(nn.Module):
    """TPU-optimized compute layer for transformer training"""
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        # Pre-normalize for better TPU performance
        x = TPULayerNorm(dtype=self.dtype)(inputs)

        # Self-attention block with TPU optimizations
        attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            name='self_attention'
        )(x, x, mask=mask, deterministic=deterministic)
        
        if not deterministic:
            attention = nn.Dropout(rate=self.dropout_rate)(
                attention, deterministic=deterministic
            )

        # Residual connection
        x = attention + inputs

        # Feed-forward block with TPU optimizations
        y = TPULayerNorm(dtype=self.dtype)(x)
        
        # Two-layer MLP with GELU activation
        y = TPUGEMMLinear(
            features=self.ff_dim,
            dtype=self.dtype,
            name='ff_1'
        )(y)
        y = jax.nn.gelu(y)
        
        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(
                y, deterministic=deterministic
            )
            
        y = TPUGEMMLinear(
            features=self.embed_dim,
            dtype=self.dtype,
            name='ff_2'
        )(y)
        
        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(
                y, deterministic=deterministic
            )

        # Final residual connection
        return y + x

# Enhanced components for VishwamAI

class FlashAttention(nn.Module):
    """TPU-optimized Flash Attention implementation."""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.bfloat16  # Change default to bfloat16

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        # Cast inputs to correct dtype at the start
        query = query.astype(self.dtype)
        key = key.astype(self.dtype)
        value = value.astype(self.dtype)
        
        batch_size, seq_len_q = query.shape[0], query.shape[2]
        _, _, seq_len_k, _ = key.shape
        
        # TPU-optimized block sizes 
        block_k = min(128, seq_len_k)
        
        # Initialize accumulators with correct dtype
        O = jnp.zeros((batch_size, self.num_heads, seq_len_q, self.head_dim), dtype=self.dtype)
        L = jnp.ones((batch_size, self.num_heads, seq_len_q, 1), dtype=self.dtype) * -1e4
        m = jnp.ones((batch_size, self.num_heads, seq_len_q, 1), dtype=self.dtype) * -1e4
        
        def process_block(carry, block_idx):
            O, L, m = carry
            start_idx = block_idx * block_k
            
            k_block = jax.lax.dynamic_slice(
                key,
                (0, 0, start_idx, 0),
                (batch_size, self.num_heads, block_k, self.head_dim)
            )
            v_block = jax.lax.dynamic_slice(
                value,
                (0, 0, start_idx, 0),
                (batch_size, self.num_heads, block_k, self.head_dim)
            )
            
            # Compute attention scores
            S = jnp.einsum('bhqd,bhkd->bhqk', query, k_block).astype(self.dtype)
            S = S / jnp.sqrt(self.head_dim).astype(self.dtype)
            
            if mask is not None:
                mask_block = jax.lax.dynamic_slice(
                    mask,
                    (0, 0, 0, start_idx),
                    (batch_size, self.num_heads, seq_len_q, block_k)
                )
                S = jnp.where(mask_block, S, jnp.full_like(S, -1e10, dtype=self.dtype))
            
            # Update max scores and ensure dtype consistency
            m_block = jnp.max(S, axis=-1, keepdims=True)
            m_new = jnp.maximum(m, m_block).astype(self.dtype)
            
            # Compute exp(S) with proper scaling
            exp_S = jnp.exp(S - m_new).astype(self.dtype)
            
            # Update normalization term
            L_new = (L * jnp.exp(m - m_new) + jnp.sum(exp_S, axis=-1, keepdims=True)).astype(self.dtype)
            
            # Update output accumulator
            O_new = (O * jnp.exp(m - m_new) + jnp.einsum('bhqk,bhkd->bhqd', exp_S, v_block)).astype(self.dtype)
            
            return (O_new, L_new, m_new), None
        
        # Scan over blocks
        num_blocks = (seq_len_k + block_k - 1) // block_k
        (O_final, L_final, _), _ = jax.lax.scan(
            process_block,
            (O, L, m),
            jnp.arange(num_blocks)
        )
        
        # Final rescaling
        output = (O_final / L_final).astype(self.dtype)
        
        return output

class RMSNorm(nn.Module):
    """TPU-optimized RMSNorm for better performance than LayerNorm"""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    scale_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feat_shape = (x.shape[-1],)
        scale = self.param('scale', self.scale_init, feat_shape)
        
        # Compute RMS
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon)
        
        # Normalize and apply scale
        return (x / rms) * scale

class TPURotaryEmbedding(nn.Module):
    """Rotary Position Embedding optimized for TPU"""
    dim: int
    max_seq_len: int = 2048
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, positions: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        if positions is None:
            positions = jnp.arange(seq_len)
            
        # Generate frequency bases
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        # Precompute freqs for efficiency
        freqs = jnp.outer(positions, inv_freq)
        
        # Create rotary embeddings
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        cos = cos.reshape(1, seq_len, 1, self.dim // 2)
        sin = sin.reshape(1, seq_len, 1, self.dim // 2)
        
        # Apply rotation
        x_rot = x.reshape(batch_size, seq_len, num_heads, -1, 2)
        x1, x2 = x_rot[..., 0], x_rot[..., 1]
        
        # Efficient complex number multiplication
        result_1 = x1 * cos - x2 * sin
        result_2 = x1 * sin + x2 * cos
        
        # Reshape and return
        return jnp.stack([result_1, result_2], axis=-1).reshape(
            batch_size, seq_len, num_heads, head_dim
        )

class EnhancedTransformerBlock(nn.Module):
    """Enhanced Transformer block with optimizations for TPU"""
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    use_rotary: bool = True
    use_flash_attn: bool = True
    use_rms_norm: bool = True
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self,
                inputs: jnp.ndarray,
                mask: Optional[jnp.ndarray] = None,
                deterministic: bool = True) -> jnp.ndarray:
        
        # Pre-normalization: Choose between RMSNorm and LayerNorm
        if self.use_rms_norm:
            x_norm = RMSNorm(dtype=self.dtype)(inputs)
        else:
            x_norm = TPULayerNorm(dtype=self.dtype)(inputs)
        
        # Project inputs to Q, K, V
        query = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='query'
        )(x_norm)
        
        key = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='key'
        )(x_norm)
        
        value = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            name='value'
        )(x_norm)

        # Reshape for attention
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            rotary = TPURotaryEmbedding(dim=self.head_dim, dtype=self.dtype)
            positions = jnp.arange(seq_len)
            query = rotary(query, positions)
            key = rotary(key, positions)
        
        # Choose between Flash Attention and standard attention
        if self.use_flash_attn:
            # Reshape for flash attention
            query = query.transpose((0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
            key = key.transpose((0, 2, 1, 3))
            value = value.transpose((0, 2, 1, 3))
            
            # Apply flash attention
            attention_output = FlashAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(query, key, value, mask, deterministic)
            
            # Reshape back
            attention_output = attention_output.transpose((0, 2, 1, 3))
            attention_output = attention_output.reshape(batch_size, seq_len, -1)
        else:
            # Standard attention from MultiHeadAttention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(x_norm, x_norm, mask, deterministic)
        
        # Project attention output
        attention_output = TPUGEMMLinear(
            features=inputs.shape[-1],
            dtype=self.dtype,
            name='attention_output'
        )(attention_output)
        
        if not deterministic:
            attention_output = nn.Dropout(rate=self.dropout_rate)(
                attention_output, deterministic=False
            )
        
        # First residual connection
        x = attention_output + inputs

        # Pre-normalization for FFN
        if self.use_rms_norm:
            y_norm = RMSNorm(dtype=self.dtype)(x)
        else:
            y_norm = TPULayerNorm(dtype=self.dtype)(x)
            
        # FFN with SwiGLU activation for better performance
        y1 = TPUGEMMLinear(features=self.mlp_dim, dtype=self.dtype, name='ff_gate')(y_norm)
        y2 = TPUGEMMLinear(features=self.mlp_dim, dtype=self.dtype, name='ff_proj')(y_norm)
        
        # SwiGLU activation
        y = jax.nn.silu(y1) * y2
        
        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)
            
        # Final projection
        y = TPUGEMMLinear(features=inputs.shape[-1], dtype=self.dtype, name='ff_output')(y)
        
        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)
        
        # Second residual connection
        return y + x

class EnhancedTransformerModel(nn.Module):
    """Enhanced Transformer model with TPU optimizations"""
    vocab_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_dim: int
    mlp_dim: int
    max_seq_len: int
    dropout_rate: float = 0.1
    use_rotary: bool = True
    use_flash_attn: bool = True
    use_rms_norm: bool = True
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self,
                inputs: jnp.ndarray,
                mask: Optional[jnp.ndarray] = None,
                deterministic: bool = True) -> jnp.ndarray:
        
        # Token embeddings
        x = TokenEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_dim,
            dtype=self.dtype
        )(inputs)
        
        # Add position embeddings if not using rotary
        if not self.use_rotary:
            positions = jnp.arange(inputs.shape[1])[None, :]
            position_embedding = self.param(
                'pos_embedding',
                nn.initializers.normal(stddev=0.02),
                (1, self.max_seq_len, self.hidden_dim),
                self.dtype
            )
            x = x + position_embedding[:, :inputs.shape[1]]

        if not deterministic:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        # Enhanced transformer blocks
        for i in range(self.num_layers):
            x = EnhancedTransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                use_rotary=self.use_rotary,
                use_flash_attn=self.use_flash_attn,
                use_rms_norm=self.use_rms_norm,
                dtype=self.dtype,
                name=f'transformer_block_{i}'
            )(x, mask, deterministic)

        # Final normalization
        if self.use_rms_norm:
            x = RMSNorm(dtype=self.dtype)(x)
        else:
            x = TPULayerNorm(dtype=self.dtype)(x)
        
        # Final projection to vocabulary
        logits = TPUGEMMLinear(
            features=self.vocab_size,
            dtype=self.dtype,
            name='output'
        )(x)

        return logits

# Configuration for TPU v2 mixed precision training
DTYPE_CONFIG = {
    'param_dtype': jnp.float32,
    'compute_dtype': jnp.float32,  # TPU v2 performs better with float32 for compute
    'output_dtype': jnp.float32,
    'embedding_dtype': jnp.int32
}

def create_vishwamai_transformer(config):
    """Create a VishwamAI transformer model with TPU v2 optimized configuration."""
    # Handle TPUTrainingConfig object
    if hasattr(config, 'model_config'):
        model_config = config.model_config
    else:
        model_config = config
        
    return EnhancedTransformerModel(
        vocab_size=model_config['vocab_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        head_dim=model_config['head_dim'],
        hidden_dim=model_config['hidden_dim'],
        mlp_dim=model_config['mlp_dim'],
        max_seq_len=model_config['max_seq_len'],
        dropout_rate=model_config.get('dropout_rate', 0.1),
        use_flash_attn=model_config.get('use_flash_attn', True),
        use_rotary=model_config.get('use_rotary', True),
        use_rms_norm=model_config.get('use_rms_norm', False),
        dtype=getattr(config, 'dtype', jnp.bfloat16)
    )

# Training utilities
def create_learning_rate_schedule(
    base_learning_rate: float,
    warmup_steps: int,
    decay_steps: int
) -> Callable[[int], float]:
    """Create learning rate schedule with warmup and cosine decay."""
    
    def schedule(step):
        # Linear warmup
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        
        # Cosine decay
        if step <= warmup_steps:
            return base_learning_rate * warmup_factor
        else:
            decay_ratio = (step - warmup_steps) / decay_steps
            return base_learning_rate * jnp.maximum(
                0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(decay_ratio, 1.0)))
            )
    
    return schedule

# TPU-optimized loss function
def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0
) -> jnp.ndarray:
    """
    Compute weighted cross entropy with label smoothing for TPU efficiency.
    
    Args:
        logits: Model output logits
        targets: Target labels
        weights: Optional masking weights
        label_smoothing: Label smoothing factor
    """
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = label_smoothing / (vocab_size - 1)
    
    # Create smoothed targets
    soft_targets = jax.nn.one_hot(targets, vocab_size)
    soft_targets = soft_targets * confidence + low_confidence * (1.0 - soft_targets)
    
    # Compute cross entropy
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(soft_targets * log_probs, axis=-1)
    
    if weights is not None:
        loss = loss * weights
        
    return jnp.mean(loss)

def create_train_state(
    rng: Any,
    config: Dict[str, Any],
    learning_rate_schedule: Callable[[int], float]
) -> Any:
    """
    Create training state with model parameters and optimizer state.
    
    Args:
        rng: PRNG key
        config: Model configuration
        learning_rate_schedule: Learning rate schedule function
    """
    model = create_vishwamai_transformer(config)
    
    # Create sample input for parameter initialization
    dummy_input = jnp.ones((2, config['max_seq_len']), dtype=jnp.int32)
    
    # Initialize parameters
    variables = model.init(
        rng,
        dummy_input,
        deterministic=False
    )
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.get('max_grad_norm', 1.0)),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            b1=config.get('beta1', 0.9),
            b2=config.get('beta2', 0.98),
            eps=config.get('epsilon', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

@jax.jit
def train_step(
    state: Any,
    batch: Dict[str, jnp.ndarray],
    dropout_rng: Any,
    label_smoothing: float = 0.0
) -> Tuple[Any, Dict[str, float]]:
    """
    Single training step with TPU optimization.
    
    Args:
        state: Current training state
        batch: Batch of training data
        dropout_rng: PRNG key for dropout
        label_smoothing: Label smoothing factor
    """
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            deterministic=False,
            rngs={'dropout': dropout_rng}
        )
        loss = compute_weighted_cross_entropy(
            logits=logits,
            targets=batch['labels'],
            weights=batch.get('attention_mask'),
            label_smoothing=label_smoothing
        )
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update state
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'learning_rate': state.opt_state.hyperparams['learning_rate']
    }
    
    return new_state, metrics

@jax.jit
def evaluate_step(
    state: Any,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, float]:
    """
    Single evaluation step with TPU optimization.
    
    Args:
        state: Current training state
        batch: Batch of evaluation data
    """
    logits = state.apply_fn(
        {'params': state.params},
        batch['input_ids'],
        deterministic=True
    )
    
    loss = compute_weighted_cross_entropy(
        logits=logits,
        targets=batch['labels'],
        weights=batch.get('attention_mask')
    )
    
    return {'loss': loss}

def generate_text(
    state: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9
) -> str:
    """
    Generate text using the trained model.
    
    Args:
        state: Training state containing model parameters
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input prompt text
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array(input_ids)[None, :]
    
    # Initialize sequence
    cur_len = input_ids.shape[1]
    
    @jax.jit
    def sample_next_token(params, inputs, temperature, prng_key):
        logits = state.apply_fn(
            {'params': params},
            inputs,
            deterministic=True
        )
        next_token_logits = logits[0, -1, :] / temperature
        
        # Apply top-k
        if top_k > 0:
            next_token_logits = top_k_logits(next_token_logits, top_k)
        
        # Apply top-p
        if top_p < 1.0:
            next_token_logits = nucleus_sampling(next_token_logits, top_p)
        
        # Sample
        next_token = jax.random.categorical(prng_key, next_token_logits)
        return next_token
    
    # Generate tokens
    generated = input_ids
    for _ in range(max_length - cur_len):
        next_token = sample_next_token(
            state.params,  
            generated,
            temperature,
            jax.random.PRNGKey(int(time.time()))
        )
        generated = jnp.concatenate([generated, next_token[None, None]], axis=1)
        
        # Stop if EOS token is generated
        if next_token == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated[0])

def top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Apply top-k filtering to logits."""
    values, _ = jax.lax.top_k(logits, k)
    min_values = values[:, -1, None]
    return jnp.where(logits < min_values, -1e10, logits)

def nucleus_sampling(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """Apply nucleus (top-p) sampling to logits."""
    sorted_logits = jnp.sort(logits)[::-1]
    cumsum_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    cutoff_index = jnp.sum(cumsum_probs < p)
    cutoff_logit = sorted_logits[cutoff_index]
    return jnp.where(logits < cutoff_logit, -1e10, logits)

# Model saving and loading utilities
def save_model_checkpoint(state: Any, path: str):
    """Save model checkpoint."""
    with open(path, 'wb') as f:
        f.write(flax.serialization.to_bytes(state))

def load_model_checkpoint(path: str, state: Any) -> Any:
    """Load model checkpoint."""
    with open(path, 'rb') as f:
        return flax.serialization.from_bytes(state, f.read())

# Distributed training utilities
def setup_distributed_training(num_devices: int) -> Tuple[Any, Any]:
    """Setup distributed training across TPU devices."""
    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ('batch',))
    return devices, mesh

def data_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True
) -> Generator:
    """TPU-optimized data loader."""
    num_samples = len(dataset)
    indices = jnp.arange(num_samples)
    
    if shuffle:
        indices = jax.random.permutation(
            jax.random.PRNGKey(int(time.time())),
            indices
        )
    
    for i in range(0, num_samples, batch_size):
        if drop_last and i + batch_size > num_samples:
            break
            
        batch_indices = indices[i:i + batch_size]
        yield {k: v[batch_indices] for k, v in dataset.items()}
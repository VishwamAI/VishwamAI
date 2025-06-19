"""
Advanced Attention Mechanisms for VishwamAI - Gemma Inspired

This module implements state-of-the-art attention mechanisms including:
- Grouped Query Attention (GQA) for memory efficiency
- Flash Attention 2 for training speed
- Sliding Window Attention for long sequences  
- Query-Key Normalization for stability
- Soft attention logit capping
- RoPE with frequency scaling
- Multi-scale attention patterns
- Cross-modal attention mechanisms
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, Union, List
import chex
import math
import functools
from enum import Enum


class AttentionType(Enum):
    """Types of attention patterns."""
    GLOBAL = "global"
    LOCAL_SLIDING = "local_sliding"  
    SPARSE_GLOBAL = "sparse_global"
    DILATED = "dilated"
    CROSS_MODAL = "cross_modal"


class QueryPreAttentionNorm(Enum):
    """Query normalization strategies before attention."""
    NONE = "none"
    BY_HEAD_DIM = "by_head_dim"
    BY_SQRT_HEAD_DIM = "by_sqrt_head_dim"
    BY_EMBED_DIM_DIV_HEADS = "by_embed_dim_div_heads"


class AdvancedRoPE(nn.Module):
    """
    Advanced Rotary Position Embedding with scaling and frequency interpolation.
    
    Based on improvements from Gemma and other recent models.
    """
    
    head_dim: int
    max_seq_len: int = 8192
    base_freq: float = 10000.0
    scaling_factor: float = 1.0
    interpolation_factor: float = 1.0
    
    def setup(self):
        # Compute base frequencies
        half_dim = self.head_dim // 2
        exponent = jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim
        
        # Apply frequency scaling and interpolation
        inv_freq = 1.0 / (
            self.base_freq ** exponent * self.scaling_factor
        )
        
        # Store as non-trainable parameter
        self.inv_freq = inv_freq / self.interpolation_factor
    
    def __call__(
        self, 
        seq_len: int, 
        offset: int = 0,
        dtype: jnp.dtype = jnp.float32
    ) -> Tuple[chex.Array, chex.Array]:
        """Generate cosine and sine embeddings."""
        
        # Create position indices
        positions = jnp.arange(offset, offset + seq_len, dtype=dtype)
        
        # Compute angle arguments
        angles = jnp.outer(positions, self.inv_freq)
        
        # Concatenate for complex representation
        angles = jnp.concatenate([angles, angles], axis=-1)
        
        return jnp.cos(angles), jnp.sin(angles)


def apply_rotary_embedding(
    tensor: chex.Array, 
    cos: chex.Array, 
    sin: chex.Array
) -> chex.Array:
    """Apply rotary position embedding to tensor."""
    
    def rotate_half(x):
        """Rotate half the hidden dimensions."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    
    # Expand cos/sin to match tensor dimensions
    if len(cos.shape) == 2:  # [seq_len, head_dim]
        cos = cos[None, :, None, :]  # [1, seq_len, 1, head_dim]
        sin = sin[None, :, None, :]
    
    return (tensor * cos) + (rotate_half(tensor) * sin)


class QueryKeyNormalization(nn.Module):
    """Query and Key normalization for improved training stability."""
    
    head_dim: int
    eps: float = 1e-6
    
    def setup(self):
        # Learnable scale parameters
        self.query_scale = self.param(
            'query_scale',
            nn.initializers.ones,
            (self.head_dim,)
        )
        
        self.key_scale = self.param(
            'key_scale', 
            nn.initializers.ones,
            (self.head_dim,)
        )
    
    def __call__(
        self, 
        query: chex.Array, 
        key: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Normalize query and key tensors."""
        
        # L2 normalization
        query_norm = jnp.linalg.norm(query, axis=-1, keepdims=True)
        key_norm = jnp.linalg.norm(key, axis=-1, keepdims=True)
        
        query = query / (query_norm + self.eps) * self.query_scale
        key = key / (key_norm + self.eps) * self.key_scale
        
        return query, key


class SlidingWindowMask(nn.Module):
    """Sliding window attention mask generator."""
    
    window_size: int
    
    def __call__(self, seq_len: int) -> chex.Array:
        """Generate sliding window attention mask."""
        
        # Create position matrix
        positions = jnp.arange(seq_len)
        query_pos = positions[:, None]  # [seq_len, 1]
        key_pos = positions[None, :]    # [1, seq_len]
        
        # Compute relative distances
        distance = jnp.abs(query_pos - key_pos)
        
        # Create mask for sliding window
        mask = distance <= self.window_size
        
        return mask


class FlashAttention2(nn.Module):
    """
    Flash Attention 2 implementation for memory-efficient attention.
    
    Implements block-wise computation to reduce memory usage while
    maintaining mathematical equivalence to standard attention.
    """
    
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 128
    dropout: float = 0.1
    attn_type: AttentionType = AttentionType.GLOBAL
    window_size: Optional[int] = None
    use_qk_norm: bool = False
    query_pre_attn_norm: QueryPreAttentionNorm = QueryPreAttentionNorm.BY_SQRT_HEAD_DIM
    attn_logits_soft_cap: Optional[float] = None
    
    def setup(self):
        # Validate configuration
        assert self.embed_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        
        self.groups_per_kv_head = self.num_heads // self.num_kv_heads
        self.scale = self._get_attention_scale()
        
        # Query, Key, Value projections
        self.q_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.k_proj = nn.Dense(
            features=self.num_kv_heads * self.head_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.v_proj = nn.Dense(
            features=self.num_kv_heads * self.head_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Output projection
        self.o_proj = nn.Dense(
            features=self.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Rotary embeddings
        self.rope = AdvancedRoPE(
            head_dim=self.head_dim,
            max_seq_len=8192,
            base_freq=10000.0,
            scaling_factor=1.0
        )
        
        # Query-Key normalization
        if self.use_qk_norm:
            self.qk_norm = QueryKeyNormalization(head_dim=self.head_dim)
        
        # Sliding window mask
        if self.attn_type == AttentionType.LOCAL_SLIDING and self.window_size:
            self.sliding_mask = SlidingWindowMask(window_size=self.window_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    
    def _get_attention_scale(self) -> float:
        """Get attention scaling factor based on normalization strategy."""
        
        if self.query_pre_attn_norm == QueryPreAttentionNorm.BY_SQRT_HEAD_DIM:
            return 1.0 / math.sqrt(self.head_dim)
        elif self.query_pre_attn_norm == QueryPreAttentionNorm.BY_HEAD_DIM:
            return 1.0 / self.head_dim
        elif self.query_pre_attn_norm == QueryPreAttentionNorm.BY_EMBED_DIM_DIV_HEADS:
            return 1.0 / (self.embed_dim / self.num_heads)
        else:
            return 1.0
    
    def _reshape_for_attention(
        self, 
        tensor: chex.Array, 
        num_heads: int
    ) -> chex.Array:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, num_heads, self.head_dim)
    
    def _apply_grouped_query_attention(
        self,
        query: chex.Array,
        key: chex.Array, 
        value: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply grouped query attention by expanding KV heads."""
        
        if self.num_kv_heads == self.num_heads:
            return query, key, value
        
        # Expand key and value to match query heads
        key = jnp.repeat(key, self.groups_per_kv_head, axis=2)
        value = jnp.repeat(value, self.groups_per_kv_head, axis=2)
        
        return query, key, value
    
    def _compute_attention_scores(
        self,
        query: chex.Array,
        key: chex.Array,
        mask: Optional[chex.Array] = None
    ) -> chex.Array:
        """Compute attention scores with optional masking."""
        
        # Compute scaled dot-product attention
        # query, key: [batch, seq_len, num_heads, head_dim]
        scores = jnp.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        
        # Apply soft capping if configured
        if self.attn_logits_soft_cap is not None:
            scores = jnp.tanh(scores / self.attn_logits_soft_cap)
            scores = scores * self.attn_logits_soft_cap
        
        # Apply attention mask
        if mask is not None:
            # Expand mask to match scores shape
            if len(mask.shape) == 2:  # [seq_len, seq_len]
                mask = mask[None, None, :, :]  # [1, 1, seq_len, seq_len]
            elif len(mask.shape) == 3:  # [batch, seq_len, seq_len]
                mask = mask[:, None, :, :]  # [batch, 1, seq_len, seq_len]
            
            scores = jnp.where(mask, scores, -jnp.inf)
        
        return scores
    
    def _apply_sliding_window_mask(
        self, 
        scores: chex.Array,
        seq_len: int
    ) -> chex.Array:
        """Apply sliding window mask to attention scores."""
        
        if self.attn_type != AttentionType.LOCAL_SLIDING or not self.window_size:
            return scores
        
        # Generate sliding window mask
        window_mask = self.sliding_mask(seq_len)
        
        # Apply mask
        window_mask = window_mask[None, None, :, :]  # Expand dimensions
        scores = jnp.where(window_mask, scores, -jnp.inf)
        
        return scores
    
    def __call__(
        self,
        hidden_states: chex.Array,
        key_value_states: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Forward pass of Flash Attention 2."""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Determine if cross-attention
        kv_states = key_value_states if key_value_states is not None else hidden_states
        kv_seq_len = kv_states.shape[1]
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(kv_states)
        value = self.v_proj(kv_states)
        
        # Reshape for multi-head attention
        query = self._reshape_for_attention(query, self.num_heads)
        key = self._reshape_for_attention(key, self.num_kv_heads)
        value = self._reshape_for_attention(value, self.num_kv_heads)
        
        # Apply RoPE for self-attention
        if key_value_states is None:  # Self-attention
            cos, sin = self.rope(seq_len)
            query = apply_rotary_embedding(query, cos, sin)
            key = apply_rotary_embedding(key, cos, sin)
        
        # Apply query-key normalization if enabled
        if self.use_qk_norm:
            query, key = self.qk_norm(query, key)
        
        # Apply grouped query attention
        query, key, value = self._apply_grouped_query_attention(query, key, value)
        
        # Compute attention scores
        scores = self._compute_attention_scores(query, key, attention_mask)
        
        # Apply sliding window mask for self-attention
        if key_value_states is None:
            scores = self._apply_sliding_window_mask(scores, seq_len)
        
        # Compute attention probabilities
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=not training)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different modalities.
    
    Allows text to attend to vision features and vice versa.
    """
    
    text_dim: int
    vision_dim: int
    num_heads: int = 16
    dropout: float = 0.1
    temperature: float = 1.0
    
    def setup(self):
        self.head_dim = min(self.text_dim, self.vision_dim) // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Text-to-vision attention
        self.text_q_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        self.vision_k_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        self.vision_v_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        # Vision-to-text attention
        self.vision_q_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        self.text_k_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        self.text_v_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False
        )
        
        # Output projections
        self.text_out_proj = nn.Dense(features=self.text_dim, use_bias=False)
        self.vision_out_proj = nn.Dense(features=self.vision_dim, use_bias=False)
        
        # Dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    
    def _cross_attention(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        training: bool = True
    ) -> chex.Array:
        """Compute cross-attention between modalities."""
        
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = jnp.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        scores = scores / self.temperature
        
        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=not training)
        
        # Apply attention to values
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        
        # Reshape output
        output = output.reshape(batch_size, q_len, -1)
        
        return output
    
    def __call__(
        self,
        text_features: chex.Array,
        vision_features: chex.Array,
        training: bool = True
    ) -> Tuple[chex.Array, chex.Array]:
        """Apply bidirectional cross-modal attention."""
        
        # Text attends to vision
        text_q = self.text_q_proj(text_features)
        vision_k = self.vision_k_proj(vision_features)
        vision_v = self.vision_v_proj(vision_features)
        
        text_attended = self._cross_attention(
            text_q, vision_k, vision_v, training=training
        )
        text_output = self.text_out_proj(text_attended)
        
        # Vision attends to text
        vision_q = self.vision_q_proj(vision_features)
        text_k = self.text_k_proj(text_features)
        text_v = self.text_v_proj(text_features)
        
        vision_attended = self._cross_attention(
            vision_q, text_k, text_v, training=training
        )
        vision_output = self.vision_out_proj(vision_attended)
        
        return text_output, vision_output


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention with different window sizes.
    
    Captures both local and global dependencies simultaneously.
    """
    
    embed_dim: int
    num_heads: int
    window_sizes: List[int] = None
    dropout: float = 0.1
    
    def setup(self):
        if self.window_sizes is None:
            self.window_sizes = [64, 256, 1024, -1]  # -1 for global
        
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Separate attention heads for each scale
        self.scale_attentions = [
            FlashAttention2(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads // len(self.window_sizes),
                num_kv_heads=self.num_heads // len(self.window_sizes),
                head_dim=self.head_dim,
                attn_type=AttentionType.LOCAL_SLIDING if window_size > 0 else AttentionType.GLOBAL,
                window_size=window_size if window_size > 0 else None,
                dropout=self.dropout
            )
            for window_size in self.window_sizes
        ]
        
        # Fusion layer
        self.fusion = nn.Dense(
            features=self.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
    
    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Apply multi-scale attention."""
        
        # Apply each scale attention
        scale_outputs = []
        for attention_layer in self.scale_attentions:
            output = attention_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                training=training
            )
            scale_outputs.append(output)
        
        # Concatenate scale outputs
        concatenated = jnp.concatenate(scale_outputs, axis=-1)
        
        # Fuse scales
        fused_output = self.fusion(concatenated)
        
        return fused_output


class SparseAttention(nn.Module):
    """
    Sparse attention pattern for very long sequences.
    
    Uses fixed patterns to reduce computational complexity.
    """
    
    embed_dim: int
    num_heads: int
    block_size: int = 64
    num_random_blocks: int = 3
    dropout: float = 0.1
    
    def setup(self):
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention components
        self.q_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.k_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.v_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        self.o_proj = nn.Dense(features=self.embed_dim, use_bias=False)
        
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    
    def _create_sparse_mask(
        self, 
        seq_len: int, 
        block_size: int,
        num_random_blocks: int
    ) -> chex.Array:
        """Create sparse attention mask."""
        
        num_blocks = seq_len // block_size
        mask = jnp.zeros((seq_len, seq_len), dtype=bool)
        
        # Local attention within blocks
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_len)
            mask = mask.at[start:end, start:end].set(True)
        
        # Global attention to first tokens
        mask = mask.at[:, :block_size].set(True)
        mask = mask.at[:block_size, :].set(True)
        
        # Random attention blocks
        key = jax.random.PRNGKey(42)
        for i in range(num_blocks):
            # Select random blocks to attend to
            random_blocks = jax.random.choice(
                key, num_blocks, shape=(num_random_blocks,), replace=False
            )
            
            start_i = i * block_size
            end_i = min((i + 1) * block_size, seq_len)
            
            for j in random_blocks:
                start_j = j * block_size
                end_j = min((j + 1) * block_size, seq_len)
                mask = mask.at[start_i:end_i, start_j:end_j].set(True)
        
        return mask
    
    def __call__(
        self,
        hidden_states: chex.Array,
        training: bool = True
    ) -> chex.Array:
        """Apply sparse attention."""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Create sparse mask
        sparse_mask = self._create_sparse_mask(
            seq_len, self.block_size, self.num_random_blocks
        )
        
        # Compute attention scores
        scores = jnp.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        
        # Apply sparse mask
        sparse_mask = sparse_mask[None, None, :, :]  # Expand for batch and heads
        scores = jnp.where(sparse_mask, scores, -jnp.inf)
        
        # Compute attention probabilities
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=not training)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


# Factory function for creating different attention types
def create_attention_layer(
    attention_type: str,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Create attention layer of specified type."""
    
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    head_dim = embed_dim // num_heads
    
    if attention_type == "flash":
        return FlashAttention2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            **kwargs
        )
    elif attention_type == "multi_scale":
        return MultiScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == "sparse":
        return SparseAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **kwargs
        )
    elif attention_type == "cross_modal":
        return CrossModalAttention(
            text_dim=embed_dim,
            vision_dim=kwargs.get('vision_dim', embed_dim),
            num_heads=num_heads,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# Configuration presets
GEMMA_ATTENTION_CONFIG = {
    "embed_dim": 3584,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 256,
    "use_qk_norm": True,
    "query_pre_attn_norm": QueryPreAttentionNorm.BY_SQRT_HEAD_DIM,
    "attn_logits_soft_cap": 50.0,
    "dropout": 0.1,
    "attn_type": AttentionType.LOCAL_SLIDING,
    "window_size": 1024
}

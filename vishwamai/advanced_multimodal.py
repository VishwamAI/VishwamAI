"""
Advanced Multimodal Architecture with Gemma-inspired Q, K, V Attention.

This module implements state-of-the-art multimodal transformer architecture
with advanced attention mechanisms, vision encoders, and cross-modal fusion
inspired by Google DeepMind's Gemma architecture.

Key Features:
- Grouped Query Attention (GQA) for efficient memory usage
- SigLIP-style vision encoder with attention pooling
- Cross-modal attention and fusion mechanisms
- RoPE (Rotary Position Embedding) for better positional understanding
- Soft attention logit capping for stable training
- Sliding window attention for long sequences
- Multi-scale patch embeddings for vision
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Any, Union, Tuple, List
import chex
import math
import functools
from dataclasses import dataclass


@dataclass
class MultimodalConfig:
    """Configuration for multimodal transformer."""
    
    # Model dimensions
    embed_dim: int = 3584
    vocab_size: int = 262144
    max_seq_len: int = 8192
    
    # Attention configuration
    num_heads: int = 16
    num_kv_heads: int = 8  # For Grouped Query Attention
    head_dim: int = 256
    attn_dropout: float = 0.1
    use_flash_attention: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 1024
    attn_logits_soft_cap: Optional[float] = 50.0
    
    # Vision configuration
    vision_embed_dim: int = 1024
    image_size: int = 800
    patch_size: int = 14
    vision_layers: int = 24
    vision_heads: int = 16
    num_vision_tokens: int = 256
    
    # Cross-modal configuration
    cross_attn_layers: List[int] = None  # Layers with cross-attention
    fusion_type: str = "adaptive_gate"  # "concat", "cross_attn", "adaptive_gate"
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    
    # Training configuration
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_post_layer_norm: bool = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """Apply RMS normalization."""
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x / jnp.sqrt(variance + self.eps)
        scale = self.param('scale', nn.initializers.ones, x.shape[-1:])
        return x * scale


class RotaryPositionalEmbedding(nn.Module):
    """Advanced Rotary Position Embedding with scaling support."""
    
    head_dim: int
    max_seq_len: int = 8192
    base: float = 10000.0
    scaling_factor: float = 1.0
    
    def setup(self):
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (
            jnp.arange(0, self.head_dim, 2).astype(jnp.float32) / self.head_dim
        ))
        self.inv_freq = inv_freq * self.scaling_factor
    
    def __call__(self, seq_len: int, offset: int = 0) -> Tuple[chex.Array, chex.Array]:
        """Generate rotary embeddings for given sequence length."""
        
        # Create position indices
        positions = jnp.arange(offset, offset + seq_len, dtype=jnp.float32)
        
        # Compute frequencies
        freqs = jnp.outer(positions, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        
        return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: chex.Array, cos: chex.Array, sin: chex.Array) -> chex.Array:
    """Apply rotary position embedding to input tensor."""
    
    def rotate_half(tensor):
        """Rotate half the hidden dims of the input."""
        x1, x2 = jnp.split(tensor, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    
    return (x * cos) + (rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention as used in Gemma models.
    
    This reduces memory usage by sharing key/value heads across multiple
    query heads while maintaining performance.
    """
    
    config: MultimodalConfig
    layer_idx: int = 0
    is_cross_attention: bool = False
    
    def setup(self):
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.num_kv_heads = self.config.num_kv_heads
        self.head_dim = self.config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query projection (full heads)
        self.q_proj = nn.Dense(
            features=self.num_heads * self.head_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Key and Value projections (grouped heads)
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
        
        # Rotary embeddings (only for self-attention)
        if not self.is_cross_attention:
            self.rope = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_seq_len=self.config.max_seq_len,
                base=self.config.rope_theta,
                scaling_factor=self.config.rope_scaling
            )
        
        # Dropout
        self.dropout = nn.Dropout(rate=self.config.attn_dropout)
    
    def _reshape_for_attention(self, tensor: chex.Array, num_heads: int) -> chex.Array:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, num_heads, self.head_dim)
    
    def _apply_sliding_window_mask(self, attn_weights: chex.Array, seq_len: int) -> chex.Array:
        """Apply sliding window attention mask."""
        if not self.config.use_sliding_window or self.is_cross_attention:
            return attn_weights
        
        window_size = self.config.sliding_window_size
        
        # Create sliding window mask
        positions = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
        mask = jnp.abs(positions) <= window_size
        
        # Apply mask
        mask = mask[None, None, :, :]  # Add batch and head dimensions
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
        
        return attn_weights
    
    def __call__(
        self,
        hidden_states: chex.Array,
        key_value_states: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        position_ids: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Forward pass of grouped query attention."""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Determine if this is cross-attention
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
        if not self.is_cross_attention and hasattr(self, 'rope'):
            cos, sin = self.rope(seq_len)
            query = apply_rope(query, cos, sin)
            key = apply_rope(key, cos, sin)
        
        # Expand key/value heads to match query heads (for GQA)
        if self.num_kv_heads != self.num_heads:
            key = jnp.repeat(key, self.num_heads // self.num_kv_heads, axis=2)
            value = jnp.repeat(value, self.num_heads // self.num_kv_heads, axis=2)
        
        # Compute attention scores
        # query: [batch, seq_len, num_heads, head_dim]
        # key: [batch, kv_seq_len, num_heads, head_dim]
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        
        # Apply soft capping if configured
        if self.config.attn_logits_soft_cap is not None:
            attn_weights = jnp.tanh(attn_weights / self.config.attn_logits_soft_cap)
            attn_weights = attn_weights * self.config.attn_logits_soft_cap
        
        # Apply sliding window mask for self-attention
        if not self.is_cross_attention:
            attn_weights = self._apply_sliding_window_mask(attn_weights, seq_len)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask to match attention weights shape
            mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
        
        # Compute attention probabilities
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=not training)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_probs, value)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP-style vision encoder with attention pooling.
    
    Based on Google's SigLIP architecture used in Gemma models.
    """
    
    config: MultimodalConfig
    
    def setup(self):
        self.embed_dim = self.config.vision_embed_dim
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Positional embeddings
        self.position_embedding = self.param(
            'position_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.num_patches, self.embed_dim)
        )
        
        # Vision transformer layers
        self.layers = [
            VisionTransformerLayer(
                config=self.config,
                layer_idx=i
            )
            for i in range(self.config.vision_layers)
        ]
        
        # Attention pooling head
        self.attention_pool = AttentionPoolingHead(
            embed_dim=self.embed_dim,
            num_queries=self.config.num_vision_tokens,
            num_heads=self.config.vision_heads
        )
        
        # Layer normalization
        self.layer_norm = RMSNorm(eps=self.config.layer_norm_eps)
    
    def __call__(self, images: chex.Array, training: bool = True) -> chex.Array:
        """Encode images to vision tokens."""
        
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(images)  # [batch, h_patches, w_patches, embed_dim]
        h_patches, w_patches = x.shape[1], x.shape[2]
        x = x.reshape(batch_size, h_patches * w_patches, self.embed_dim)
        
        # Add positional embeddings
        x = x + self.position_embedding[None, :, :]
        
        # Apply vision transformer layers
        for layer in self.layers:
            x = layer(x, training=training)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Attention pooling to fixed number of tokens
        vision_tokens = self.attention_pool(x, training=training)
        
        return vision_tokens


class VisionTransformerLayer(nn.Module):
    """Single layer of vision transformer."""
    
    config: MultimodalConfig
    layer_idx: int
    
    def setup(self):
        # Multi-head attention
        self.attention = GroupedQueryAttention(
            config=self.config,
            layer_idx=self.layer_idx,
            is_cross_attention=False
        )
        
        # Feed-forward network
        self.mlp = FeedForwardNetwork(
            embed_dim=self.config.vision_embed_dim,
            hidden_dim=self.config.vision_embed_dim * 4,
            dropout=self.config.dropout
        )
        
        # Layer normalizations
        self.input_layernorm = RMSNorm(eps=self.config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(eps=self.config.layer_norm_eps)
    
    def __call__(self, hidden_states: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass of vision transformer layer."""
        
        # Pre-attention layer norm
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_output = self.attention(
            hidden_states=normed_hidden_states,
            training=training
        )
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Pre-MLP layer norm
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward network
        mlp_output = self.mlp(normed_hidden_states, training=training)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states


class AttentionPoolingHead(nn.Module):
    """Attention pooling to convert variable-length sequence to fixed-length."""
    
    embed_dim: int
    num_queries: int
    num_heads: int
    
    def setup(self):
        # Learnable query tokens
        self.query_tokens = self.param(
            'query_tokens',
            nn.initializers.normal(stddev=0.02),
            (self.num_queries, self.embed_dim)
        )
        
        # Multi-head attention for pooling
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=0.1,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Layer normalization
        self.layer_norm = RMSNorm()
    
    def __call__(self, hidden_states: chex.Array, training: bool = True) -> chex.Array:
        """Pool hidden states using attention."""
        
        batch_size = hidden_states.shape[0]
        
        # Expand query tokens for batch
        queries = jnp.broadcast_to(
            self.query_tokens[None, :, :],
            (batch_size, self.num_queries, self.embed_dim)
        )
        
        # Apply attention pooling
        pooled_output = self.attention(
            queries,
            hidden_states,
            deterministic=not training
        )
        
        # Layer normalization
        pooled_output = self.layer_norm(pooled_output)
        
        return pooled_output


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    embed_dim: int
    hidden_dim: int
    dropout: float = 0.1
    
    def setup(self):
        # SwiGLU requires two linear projections for gating
        self.gate_proj = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.up_proj = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.down_proj = nn.Dense(
            features=self.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.dropout = nn.Dropout(rate=self.dropout)
    
    def __call__(self, x: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass with SwiGLU activation."""
        
        # SwiGLU: swish(gate) * up
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = jax.nn.swish(gate) * up
        
        # Apply dropout
        hidden = self.dropout(hidden, deterministic=not training)
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output


class MultimodalTransformerLayer(nn.Module):
    """
    Transformer layer with cross-modal attention capabilities.
    
    Supports both self-attention and cross-attention between modalities.
    """
    
    config: MultimodalConfig
    layer_idx: int
    
    def setup(self):
        # Self-attention
        self.self_attention = GroupedQueryAttention(
            config=self.config,
            layer_idx=self.layer_idx,
            is_cross_attention=False
        )
        
        # Cross-attention (if this layer supports it)
        self.has_cross_attention = (
            self.config.cross_attn_layers is not None and 
            self.layer_idx in self.config.cross_attn_layers
        )
        
        if self.has_cross_attention:
            self.cross_attention = GroupedQueryAttention(
                config=self.config,
                layer_idx=self.layer_idx,
                is_cross_attention=True
            )
            
            self.cross_attn_layernorm = RMSNorm(eps=self.config.layer_norm_eps)
        
        # Feed-forward network
        self.mlp = FeedForwardNetwork(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.embed_dim * 4,
            dropout=self.config.dropout
        )
        
        # Layer normalizations
        self.input_layernorm = RMSNorm(eps=self.config.layer_norm_eps)
        
        if self.config.use_post_layer_norm:
            self.post_attention_layernorm = RMSNorm(eps=self.config.layer_norm_eps)
    
    def __call__(
        self,
        hidden_states: chex.Array,
        vision_states: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Forward pass of multimodal transformer layer."""
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            training=training
        )
        
        hidden_states = residual + attn_output
        
        # Cross-attention with vision (if applicable)
        if self.has_cross_attention and vision_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layernorm(hidden_states)
            
            cross_attn_output = self.cross_attention(
                hidden_states=hidden_states,
                key_value_states=vision_states,
                training=training
            )
            
            hidden_states = residual + cross_attn_output
        
        # Feed-forward network
        residual = hidden_states
        
        if self.config.use_post_layer_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states, training=training)
        hidden_states = residual + mlp_output
        
        return hidden_states


class AdaptiveModalityFusion(nn.Module):
    """Adaptive fusion mechanism for combining text and vision representations."""
    
    embed_dim: int
    vision_dim: int
    
    def setup(self):
        # Projection layers
        self.text_proj = nn.Dense(
            features=self.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        self.vision_proj = nn.Dense(
            features=self.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Gating mechanism
        self.gate = nn.Dense(
            features=1,
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )
        
        # Layer normalization
        self.layer_norm = RMSNorm()
    
    def __call__(
        self,
        text_features: chex.Array,
        vision_features: chex.Array,
        training: bool = True
    ) -> chex.Array:
        """Adaptively fuse text and vision features."""
        
        # Project to same dimension
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        
        # Compute fusion weights
        combined = text_proj + vision_proj
        gate_weights = jax.nn.sigmoid(self.gate(combined))
        
        # Weighted combination
        fused = gate_weights * text_proj + (1 - gate_weights) * vision_proj
        
        # Layer normalization
        fused = self.layer_norm(fused)
        
        return fused


class GemmaInspiredMultimodalTransformer(nn.Module):
    """
    Complete multimodal transformer inspired by Gemma architecture.
    
    Features:
    - Grouped Query Attention for efficiency
    - SigLIP-style vision encoder
    - Cross-modal attention layers
    - Adaptive modality fusion
    - RoPE positional embeddings
    - SwiGLU feed-forward networks
    """
    
    config: MultimodalConfig
    
    def setup(self):
        # Text embedding
        self.text_embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Vision encoder
        self.vision_encoder = SigLIPVisionEncoder(config=self.config)
        
        # Vision-to-text projection
        self.vision_projection = nn.Dense(
            features=self.config.embed_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Transformer layers
        self.layers = [
            MultimodalTransformerLayer(
                config=self.config,
                layer_idx=i
            )
            for i in range(24)  # Number of layers
        ]
        
        # Final layer normalization
        self.final_layer_norm = RMSNorm(eps=self.config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Adaptive fusion (if configured)
        if self.config.fusion_type == "adaptive_gate":
            self.modality_fusion = AdaptiveModalityFusion(
                embed_dim=self.config.embed_dim,
                vision_dim=self.config.vision_embed_dim
            )
    
    def __call__(
        self,
        input_ids: chex.Array,
        images: Optional[chex.Array] = None,
        attention_mask: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Forward pass of multimodal transformer."""
        
        batch_size, seq_len = input_ids.shape
        
        # Text embeddings
        hidden_states = self.text_embedding(input_ids)
        
        # Vision encoding (if images provided)
        vision_states = None
        if images is not None:
            vision_features = self.vision_encoder(images, training=training)
            vision_states = self.vision_projection(vision_features)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                vision_states=vision_states,
                attention_mask=attention_mask,
                training=training
            )
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits


def create_multimodal_model(
    vocab_size: int = 262144,
    embed_dim: int = 3584,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    image_size: int = 800,
    patch_size: int = 14,
    **kwargs
) -> GemmaInspiredMultimodalTransformer:
    """Create a Gemma-inspired multimodal transformer model."""
    
    config = MultimodalConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        image_size=image_size,
        patch_size=patch_size,
        **kwargs
    )
    
    return GemmaInspiredMultimodalTransformer(config=config)


# Example usage and configuration presets
GEMMA_4B_MULTIMODAL_CONFIG = MultimodalConfig(
    embed_dim=2560,
    vocab_size=262144,
    num_heads=8,
    num_kv_heads=4,
    head_dim=256,
    vision_embed_dim=1024,
    image_size=800,
    patch_size=14,
    vision_layers=24,
    vision_heads=16,
    num_vision_tokens=256,
    cross_attn_layers=[4, 8, 12, 16, 20],  # Cross-attention every 4 layers
    fusion_type="adaptive_gate",
    use_sliding_window=True,
    sliding_window_size=1024,
    attn_logits_soft_cap=50.0
)

GEMMA_12B_MULTIMODAL_CONFIG = MultimodalConfig(
    embed_dim=3840,
    vocab_size=262144,
    num_heads=16,
    num_kv_heads=8,
    head_dim=256,
    vision_embed_dim=1280,
    image_size=800,
    patch_size=14,
    vision_layers=32,
    vision_heads=20,
    num_vision_tokens=512,
    cross_attn_layers=[6, 12, 18, 24, 30],
    fusion_type="adaptive_gate",
    use_sliding_window=True,
    sliding_window_size=1024,
    attn_logits_soft_cap=50.0
)

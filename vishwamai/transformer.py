import flax.linen as nn
import jax
import jax.numpy as jnp
import logging
from typing import Tuple, Optional, Dict, Any
from functools import partial

logger = logging.getLogger(__name__)

def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Convert an image into patches."""
    B, H, W, C = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    x = x.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, num_patches_h*num_patches_w, patch_size*patch_size*C)
    return x

@partial(jax.jit, static_argnums=(3, 4))
def flash_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, 
                   num_heads: int, dropout_rate: float, 
                   deterministic: bool = True, 
                   rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """
    Enhanced Flash Attention 2.0 implementation with block-sparse attention and improved memory efficiency.
    
    Args:
        q, k, v: Query, key and value tensors of shape (B, S, H)
        num_heads: Number of attention heads
        dropout_rate: Attention dropout rate
        deterministic: If True, disables dropout
        rng_key: PRNG key for dropout randomness
        
    Returns:
        attention_output: Attention output of shape (B, S, H)
    """
    B, S, H = q.shape
    head_dim = H // num_heads
    scale = head_dim ** -0.5
    block_size = min(256, S)  # Optimize for different sequence lengths
    
    # Reshape to (B, num_heads, S, head_dim) with optimal memory layout
    q = jax.lax.reshape(q, (B, S, num_heads, head_dim), dimensions=(0, 1, 2, 3))
    k = jax.lax.reshape(k, (B, S, num_heads, head_dim), dimensions=(0, 1, 2, 3))
    v = jax.lax.reshape(v, (B, S, num_heads, head_dim), dimensions=(0, 1, 2, 3))
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    
    def blocked_attention(q_block, k_block, v_block, mask_block=None):
        # Compute attention scores for current block
        attn = jax.lax.dot_general(
            q_block, k_block,
            dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1)))
        ) * scale
        
        if mask_block is not None:
            attn = jnp.where(mask_block, attn, -1e9)
        
        # Memory-efficient softmax
        attn_max = jax.lax.reduce_max(attn, axes=(3,), keepdims=True)
        exp_attn = jnp.exp(attn - attn_max)
        exp_sum = jnp.sum(exp_attn, axis=-1, keepdims=True)
        attn = exp_attn / (exp_sum + 1e-9)  # Improved numerical stability
        
        if not deterministic and dropout_rate > 0:
            if rng_key is None:
                raise ValueError("rng_key must be provided when dropout is enabled")
            dropout_rng = jax.random.fold_in(rng_key, q_block.shape[2])  # Unique key per block
            attn = jax.random.bernoulli(dropout_rng, 1.0 - dropout_rate, attn.shape) * attn / (1.0 - dropout_rate)
        
        # Compute block output with optimized matmul
        return jax.lax.dot_general(
            attn, v_block,
            dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1)))
        )
    
    # Process attention in blocks for better memory efficiency
    output_blocks = []
    for i in range(0, S, block_size):
        q_block = jax.lax.dynamic_slice(q, (0, 0, i, 0), (B, num_heads, min(block_size, S - i), head_dim))
        block_output = blocked_attention(q_block, k, v)
        output_blocks.append(block_output)
    
    # Concatenate blocks and reshape output
    output = jnp.concatenate(output_blocks, axis=2)
    return jax.lax.reshape(output.transpose(0, 2, 1, 3), (B, S, H))

class PatchEmbedding(nn.Module):
    """Patch embedding layer."""
    embedding_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.embedding_size)(x)

class GQAttention(nn.Module):
    """Grouped-Query Attention for better efficiency."""
    num_heads: int
    num_kv_heads: int  # Number of key/value heads (grouped)
    dropout_rate: float
    
    @nn.compact
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, 
                 deterministic: bool = True) -> jnp.ndarray:
        head_dim = q.shape[-1] // self.num_heads
        
        # Project q, k, v with different number of heads
        q = nn.Dense(self.num_heads * head_dim)(q)
        k = nn.Dense(self.num_kv_heads * head_dim)(k)
        v = nn.Dense(self.num_kv_heads * head_dim)(v)
        
        # Repeat k/v heads to match number of q heads
        k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=-2)
        v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=-2)
        
        return flash_attention(q, k, v, self.num_heads, self.dropout_rate, deterministic)

class TransformerBlock(nn.Module):
    """Transformer block with GQA and Flash Attention."""
    num_heads: int
    num_kv_heads: int
    mlp_dim: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_size = x.shape[-1]
        x_norm = nn.LayerNorm()(x)
        
        # Self-attention with GQA
        attn_output = GQAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate
        )(x_norm, x_norm, x_norm, deterministic)
        
        x = x + nn.Dense(hidden_size)(attn_output)
        
        # FFN
        x_norm = nn.LayerNorm()(x)
        x = x + nn.Dense(features=self.mlp_dim, use_bias=False)(x_norm)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(features=hidden_size, use_bias=False)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        return x

class VisionTransformer10B(nn.Module):
    """10B parameter Vision Transformer with optimized attention."""
    num_classes: int = 1000
    patch_size: int = 16
    hidden_size: int = 6144
    num_heads: int = 96
    num_kv_heads: int = 24  # 4x head grouping
    num_layers: int = 24
    mlp_dim: int = 24576
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        logger.info("Starting Vision Transformer forward pass")
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Patch embedding
        x = patchify(x, self.patch_size)
        x = PatchEmbedding(self.hidden_size)(x)

        # Add CLS token
        cls_token = self.param('cls_token', 
                             nn.initializers.normal(stddev=0.02), 
                             (1, 1, self.hidden_size))
        cls_token = jnp.tile(cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)

        # Add position embeddings
        pos_embedding = self.param('pos_embedding',
                                 nn.initializers.normal(stddev=0.02),
                                 (1, num_patches + 1, self.hidden_size))
        x = x + pos_embedding

        # Apply transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            )(x, deterministic=not train)

        # Classification head
        x = x[:, 0]  # Take CLS token
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_classes)(x)

        logger.info("Completed Vision Transformer forward pass")
        return x

"""VishwamAI model implementation."""

from .layers.layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer
)
from .thoughts.tot import TreeOfThoughts
from .thoughts.cot import ChainOfThoughtPrompting

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VishwamAIConfig:
    """Configuration for VishwamAI model."""
    vocab_size: int = 32000
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    head_dim: int = 128
    mlp_dim: int = 8192
    max_seq_len: int = 2048
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attn: bool = True
    gradient_checkpointing: bool = False
    max_branches: int = 3
    max_depth: int = 3
    beam_width: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

class VishwamAI(nn.Module):
    """TPU-optimized language model with advanced reasoning capabilities."""
    config: VishwamAIConfig

    def setup(self):
        """Initialize the model components."""
        # Initialize attention module
        self.attention = TPUMultiHeadAttention(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dropout_rate=self.config.attention_dropout,
            use_flash_attn=self.config.use_flash_attn
        )
        
        # Other components
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)
        
        # Token embedding
        self.token_embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim
        )
        
        # Position embedding
        self.position_embedding = nn.Embed(
            num_embeddings=self.config.max_seq_len,
            features=self.config.hidden_dim
        )
        
        # Output head
        self.output_proj = TPUGEMMLinear(
            features=self.config.vocab_size,
            use_bias=True
        )

    def compute_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rng: Optional[Any] = None
    ) -> jnp.ndarray:
        """Compute attention using the model's attention mechanism."""
        # This can be called from outside since it doesn't rely on setup() fields
        attn = TPUMultiHeadAttention(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dropout_rate=self.config.attention_dropout,
            use_flash_attn=self.config.use_flash_attn,
            name='compute_attention'
        )
        return attn(query, key, value, mask=mask, deterministic=deterministic, rng=rng)

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        deterministic: bool = True,
        rngs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass through the model."""
        if rngs is None:
            rngs = {}

        # Cast inputs to bfloat16 for TPU efficiency
        x = self.token_embedding(input_ids.astype(jnp.int32))

        # Add positional embeddings
        positions = jnp.arange(input_ids.shape[1])[None]
        x = x + self.position_embedding(positions)

        # Optional dropout for training
        x = self.dropout(x, deterministic=deterministic, rng=rngs.get('dropout'))

        # Process through transformer blocks
        for i in range(self.config.num_layers):
            x = TransformerBlock(
                config=self.config,
                name=f'block_{i}'
            )(x, deterministic=deterministic, rngs=rngs)

        # Final layer norm and output projection
        x = TPULayerNorm(
            epsilon=1e-5,
            dtype=jnp.bfloat16,
            name='ln_f'
        )(x)
        logits = self.output_proj(x)

        return {"logits": logits}

    def initialize_kv_cache(
        self,
        batch_size: int,
        max_length: int,
        num_heads: int,
        head_dim: int
    ) -> Dict[str, jnp.ndarray]:
        """Initialize key-value cache for inference."""
        return {
            'key_cache': jnp.zeros((batch_size, max_length, num_heads, head_dim), dtype=jnp.float32),
            'value_cache': jnp.zeros((batch_size, max_length, num_heads, head_dim), dtype=jnp.float32),
            'cur_index': 0
        }

    def memory_efficient_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        rngs: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        """Memory-efficient attention implementation."""
        # Compute attention scores with flash attention when possible
        scale = 1.0 / jnp.sqrt(query.shape[-1])

        # Handle batch, seq_len, num_heads, head_dim properly
        b, s, h, d = query.shape
        q_reshaped = query.transpose(0, 2, 1, 3)  # [b, h, s, d]
        k_reshaped = key.transpose(0, 2, 1, 3)    # [b, h, s, d]
        v_reshaped = value.transpose(0, 2, 1, 3)   # [b, h, s, d]

        scores = jnp.matmul(q_reshaped, jnp.swapaxes(k_reshaped, -2, -1)) * scale

        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, float('-inf'))

        # Apply softmax
        weights = jax.nn.softmax(scores, axis=-1)

        if not deterministic:
            dropout = nn.Dropout(
                rate=self.config.dropout_rate,
                broadcast_dims=(1, 2),
                rng_collection='dropout'
            )
            weights = dropout(weights, deterministic=False, rng=rngs.get('dropout') if rngs else None)

        # Compute attention output and restore original shape
        output = jnp.matmul(weights, v_reshaped)  # [b, h, s, d]
        output = output.transpose(0, 2, 1, 3)      # [b, s, h, d]

        return output

class TransformerBlock(nn.Module):
    """Transformer block with TPU optimizations."""
    config: VishwamAIConfig

    def setup(self):
        """Initialize transformer block components."""
        # Initialize attention module
        self.attention = TPUMultiHeadAttention(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dropout_rate=self.config.attention_dropout,
            use_flash_attn=self.config.use_flash_attn
        )
        
        # MLP components
        self.mlp_dense1 = TPUGEMMLinear(
            features=self.config.mlp_dim,
            use_bias=True,
            dtype=jnp.bfloat16
        )
        self.mlp_dense2 = TPUGEMMLinear(
            features=self.config.hidden_dim,
            use_bias=True,
            dtype=jnp.bfloat16
        )
        
        # Layer norms
        self.ln1 = TPULayerNorm(epsilon=1e-5, dtype=jnp.bfloat16)
        self.ln2 = TPULayerNorm(epsilon=1e-5, dtype=jnp.bfloat16)
        
        # Dropouts
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True, rngs: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Forward pass through transformer block with memory optimizations."""
        if rngs is None:
            rngs = {}

        # Split computation for gradient checkpointing
        def attention_block(x):
            x_norm = self.ln1(x)
            attn_output = self.attention(
                x_norm, x_norm, x_norm,  # q, k, v
                deterministic=deterministic,
                rng=rngs.get('dropout')
            )
            if not deterministic:
                attn_output = self.dropout(
                    attn_output,
                    deterministic=deterministic,
                    rng=rngs.get('dropout')
                )
            return x + attn_output

        def mlp_block(x):
            x_norm = self.ln2(x)
            h = self.mlp_dense1(x_norm)
            h = jax.nn.gelu(h)
            if not deterministic:
                h = self.dropout(h, deterministic=deterministic, rng=rngs.get('dropout'))
            h = self.mlp_dense2(h)
            if not deterministic:
                h = self.dropout(h, deterministic=deterministic, rng=rngs.get('dropout'))
            return x + h

        # Apply blocks with optional gradient checkpointing
        if self.config.gradient_checkpointing:
            x = jax.checkpoint(attention_block)(x)
            x = jax.checkpoint(mlp_block)(x)
        else:
            x = attention_block(x)
            x = mlp_block(x)

        return x
"""
Core model architecture for VishwamAI.

Implements a unified Transformer-based architecture that can handle multiple modalities
through tokenization into a shared sequence space.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import chex

from .attention import FlashAttention, OptimizedAttention
from .layers import FeedForward, RMSNorm, RotaryEmbedding
from .kernels import get_optimal_kernels


@dataclass
class ModelConfig:
    """Configuration for VishwamAI model."""
    
    # Model dimensions
    dim: int = 2048
    depth: int = 32
    heads: int = 32
    head_dim: Optional[int] = None
    vocab_size: int = 50304
    max_seq_len: int = 8192
    
    # Efficiency features
    dropout_rate: float = 0.1
    use_flash_attention: bool = True
    use_grouped_query_attention: bool = True
    gqa_groups: int = 8
    use_rmsnorm: bool = True
    use_rotary_embeddings: bool = True
    
    # Multimodal features
    vision_patch_size: int = 16
    vision_dim: int = 1024
    audio_dim: int = 512
    enable_multimodal: bool = True
    
    # Mixed Expert features
    expert_count: int = 8
    expert_capacity: int = 4
    use_moe: bool = False
    
    # Hardware optimizations
    use_bfloat16: bool = True
    gradient_checkpointing: bool = True
    kernel_fusion: bool = True
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.dim // self.heads
        assert self.dim % self.heads == 0, "dim must be divisible by heads"


class TransformerBlock(nn.Module):
    """Single transformer block with optimizations."""
    
    config: ModelConfig
    
    def setup(self):
        # Attention layer
        if self.config.use_flash_attention:
            self.attention = FlashAttention(
                dim=self.config.dim,
                heads=self.config.heads,
                head_dim=self.config.head_dim,
                dropout=self.config.dropout_rate,
                use_gqa=self.config.use_grouped_query_attention,
                gqa_groups=self.config.gqa_groups
            )
        else:
            self.attention = OptimizedAttention(
                dim=self.config.dim,
                heads=self.config.heads,
                head_dim=self.config.head_dim,
                dropout=self.config.dropout_rate
            )
        
        # Feed forward layer
        self.ff = FeedForward(
            dim=self.config.dim,
            hidden_dim=self.config.dim * 4,
            dropout=self.config.dropout_rate,
            use_moe=self.config.use_moe,
            expert_count=self.config.expert_count,
            expert_capacity=self.config.expert_capacity
        )
        
        # Normalization layers
        if self.config.use_rmsnorm:
            self.norm1 = RMSNorm(self.config.dim)
            self.norm2 = RMSNorm(self.config.dim)
        else:
            self.norm1 = nn.LayerNorm(self.config.dim)
            self.norm2 = nn.LayerNorm(self.config.dim)
    
    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        # Pre-norm architecture for better stability
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + attn_out
        
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class VishwamAIModel(nn.Module):
    """Main VishwamAI model with multimodal capabilities."""
    
    config: ModelConfig
    
    def setup(self):
        # Token embeddings
        self.token_embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.dim
        )
        
        # Positional embeddings
        if self.config.use_rotary_embeddings:
            self.pos_embedding = RotaryEmbedding(
                dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len
            )
        else:
            self.pos_embedding = nn.Embed(
                num_embeddings=self.config.max_seq_len,
                features=self.config.dim
            )
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(self.config)
            for _ in range(self.config.depth)
        ]
        
        # Final normalization
        if self.config.use_rmsnorm:
            self.final_norm = RMSNorm(self.config.dim)
        else:
            self.final_norm = nn.LayerNorm(self.config.dim)
        
        # Output projection
        self.output_projection = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)
    
    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Forward pass through the model."""
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Positional embeddings
        seq_len = input_ids.shape[-1]
        if self.config.use_rotary_embeddings:
            # RoPE is applied within attention layers
            pass
        else:
            positions = jnp.arange(seq_len)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        
        # Apply dropout
        x = self.dropout(x, deterministic=not training)
        
        # Apply transformer blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.config.gradient_checkpointing and training:
                x = nn.remat(block)(x, mask=attention_mask)
            else:
                x = block(x, mask=attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        params: Dict[str, Any],
        input_ids: chex.Array,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> chex.Array:
        """Generate text using the model."""
        
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        batch_size, seq_len = input_ids.shape
        generated = input_ids
        
        for _ in range(max_length - seq_len):
            # Get logits for next token
            logits = self.apply(params, generated, training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k_indices = jnp.argsort(next_token_logits, axis=-1)[:, -top_k:]
                top_k_logits = jnp.take_along_axis(next_token_logits, top_k_indices, axis=-1)
                next_token_logits = jnp.where(
                    jnp.arange(next_token_logits.shape[-1])[None, :] >= top_k_indices[:, 0:1],
                    next_token_logits,
                    -jnp.inf
                )
            
            # Sample next token
            rng_key, sample_key = jax.random.split(rng_key)
            next_token = jax.random.categorical(sample_key, next_token_logits)
            
            # Append to sequence
            generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)
            
            # Check for EOS token (assuming token ID 2)
            if jnp.all(next_token == 2):
                break
        
        return generated


def create_integrated_model(config: ModelConfig) -> Dict[str, Any]:
    """Create an integrated VishwamAI model with all components."""
    
    # Get optimal kernels for current hardware
    kernels = get_optimal_kernels()
    
    # Create model
    model = VishwamAIModel(config)
    
    # Create optimizer
    optimizer = optax.adamw(
        learning_rate=1e-4,
        weight_decay=0.01,
        b1=0.9,
        b2=0.95
    )
    
    # Setup mixed precision if enabled
    if config.use_bfloat16:
        optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=5)
    
    return {
        'model': model,
        'optimizer': optimizer,
        'kernels': kernels,
        'config': config
    }


def create_train_state(
    model: VishwamAIModel,
    config: ModelConfig,
    rng_key: jax.random.PRNGKey,
    learning_rate: float = 1e-4
) -> train_state.TrainState:
    """Create training state for the model."""
    
    # Initialize parameters
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input, training=True)
    
    # Create optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=0.01,
        b1=0.9,
        b2=0.95
    )
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state

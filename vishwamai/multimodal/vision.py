"""Vision components of VishwamAI multimodal models."""

from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple

from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention
from vishwamai.multimodal.vision_utils import MAPHead, _posemb_sincos_2d

class ViTBlock(nn.Module):
    """Transformer encoder block for ViT with memory optimizations."""
    
    hidden_dim: int
    num_heads: int
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Memory-efficient layer norm
        y = TPULayerNorm(dtype=self.dtype)(x)
        
        # FlashAttention for better memory efficiency
        y = TPUMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            use_flash_attention=True,  # Enable FlashAttention
            block_size=64  # Optimize for TPU memory access
        )(y, mask=None, deterministic=not training)
        
        # Gradient checkpointing for memory efficiency
        if training:
            y = nn.remat(lambda x, y: x + y)(x, y)
        else:
            x = x + y

        # Memory-efficient MLP with kernel fusion
        y = TPULayerNorm(dtype=self.dtype)(x)
        mlp_dim = self.mlp_dim or 4 * self.hidden_dim
        
        # Fused MLP operations
        y = nn.Sequential([
            TPUGEMMLinear(features=mlp_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            TPUGEMMLinear(features=self.hidden_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout_rate)
        ])(y, deterministic=not training)
        
        return x + y

class ViTEncoder(nn.Module):
    """Vision Transformer encoder."""
    
    hidden_dim: int  # Size of hidden dimension
    num_layers: int  # Number of transformer blocks
    num_heads: int   # Number of attention heads
    mlp_dim: Optional[int] = None  # Size of MLP dimension
    patch_size: int = 16  # Size of image patches
    image_size: int = 224  # Input image size
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    pos_embedding_init: str = 'sincos'  # Position embedding type
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        pixel_values: jnp.ndarray,
        training: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Apply vision transformer encoder.
        
        Args:
            pixel_values: Image tensor [batch, height, width, channels]
            training: Whether in training mode
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing:
            - last_hidden_state: Final hidden state [batch, sequence_length, hidden_dim]
            - hidden_states: All hidden states (if output_hidden_states=True)
        """
        # Check image size
        batch_size, height, width, num_channels = pixel_values.shape
        
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f'Input image size ({height}x{width}) must match model image_size ({self.image_size})'
            )
            
        # Patch embedding
        patch_embedder = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            dtype=self.dtype
        )
        
        x = patch_embedder(pixel_values)
        
        # Reshape to (batch, sequence_length, channels)
        batch_size, h, w, c = x.shape
        sequence_length = h * w
        x = x.reshape(batch_size, sequence_length, c)

        # Add position embeddings
        if self.pos_embedding_init == 'sincos':
            pos_embed = _posemb_sincos_2d(
                h=h,
                w=w,
                width=c,
                temperature=10000.0,
                dtype=self.dtype
            )
            x = x + pos_embed[None, :, :]
        else:
            pos_embed = self.param(
                'pos_embedding',
                nn.initializers.normal(0.02),
                (1, sequence_length, self.hidden_dim)
            )
            x = x + pos_embed
        
        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Store hidden states if needed
        hidden_states = [] if output_hidden_states else None
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            if output_hidden_states:
                hidden_states.append(x)
                
            x = ViTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name=f'block_{i}'
            )(x, training=training)

        # Final layer norm
        x = TPULayerNorm(dtype=self.dtype, name='final_layernorm')(x)
        
        # Handle output
        outputs = {'last_hidden_state': x}
        if output_hidden_states:
            hidden_states.append(x)
            outputs['hidden_states'] = tuple(hidden_states)
            
        return outputs

class CLIPAdapter(nn.Module):
    """Adapter for CLIP vision encoder."""
    
    hidden_dim: int
    projection_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Project vision features to joint embedding space.
        
        Args:
            hidden_states: Vision features (batch_size, seq_len, hidden_dim)
            deterministic: Whether to use deterministic behavior
            
        Returns:
            Projected features (batch_size, seq_len, projection_dim)
        """
        x = TPULayerNorm()(hidden_states)
        x = TPUGEMMLinear(features=self.projection_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x
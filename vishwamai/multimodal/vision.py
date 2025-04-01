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
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """Process inputs through ViT block."""
        # Layer norm and attention
        y = TPULayerNorm(dtype=self.dtype)(x)
        attention = TPUMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype
        )(y, deterministic=not training)
        
        # Residual with gradient checkpointing
        if training:
            x = nn.remat(lambda x, y: x + y)(x, attention)
        else:
            x = x + attention
            
        # MLP block with memory optimization
        y = TPULayerNorm(dtype=self.dtype)(x)
        mlp_dim = self.mlp_dim or self.hidden_dim * 4
        y = nn.Sequential([
            TPUGEMMLinear(features=mlp_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            TPUGEMMLinear(features=self.hidden_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout_rate)
        ])(y, deterministic=not training)
        
        x = x + y
        return x

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
        """Process image through Vision Transformer.
        
        Args:
            pixel_values: Input images [batch, height, width, channels]
            training: Whether in training mode
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Dict with encoded features and optional hidden states
        """
        # Efficient patch embedding with TPU-optimized convolution
        patch_embedder = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            dtype=self.dtype
        )
        
        x = patch_embedder(pixel_values)
        
        # Reshape to sequence
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
                dtype=self.dtype,
                scale=1.0  # Added scaling factor
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
        
        # Process with transformer blocks
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
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
    def setup(self):
        # Import CLIP locally to avoid startup dependency
        try:
            from transformers import CLIPVisionModel
            self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self._has_clip = True
        except ImportError:
            print("Warning: transformers CLIP model not found")
            self._has_clip = False
            
        self.projection = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype
        )
    
    def __call__(
        self,
        pixel_values: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Process images through CLIP vision encoder.
        
        Args:
            pixel_values: Image tensor [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Dictionary with image features and optionally pooled output
        """
        if not self._has_clip:
            raise ValueError("CLIP model not available")
            
        # Convert to CLIP expected format
        if pixel_values.shape[-1] == 3:
            pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
            
        # Get CLIP features
        outputs = self.clip(pixel_values)
        
        # Project to model hidden dimension
        features = self.projection(outputs.last_hidden_state)
        
        if not training:
            features = nn.Dropout(rate=self.dropout_rate)(
                features, deterministic=True
            )
        
        return {
            "last_hidden_state": features,
            "pooled_output": outputs.pooler_output
        }
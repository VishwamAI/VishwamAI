"""Fusion modules for multimodal processing."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional
from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm
from vishwamai.layers.attention import FlashAttention

class CrossAttentionFuser(nn.Module):
    """Cross-attention based fusion of multimodal features."""
    
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        """Initialize cross attention."""
        self.vision_attention = FlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            causal=False,
            dtype=self.dtype
        )
        self.text_attention = FlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            causal=False,
            dtype=self.dtype
        )
        self.vision_ln = TPULayerNorm()
        self.text_ln = TPULayerNorm()
        self.vision_proj = TPUGEMMLinear(features=self.hidden_dim)
        self.text_proj = TPUGEMMLinear(features=self.hidden_dim)
    
    def __call__(
        self,
        vision_features: jnp.ndarray,
        text_features: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Cross attend between vision and text features.
        
        Args:
            vision_features: Vision features [batch, vision_seq_len, hidden_dim]
            text_features: Text features [batch, text_seq_len, hidden_dim]
            training: Whether in training mode
            
        Returns:
            Dictionary with processed vision and text features
        """
        # Vision attending to text
        vision_x = self.vision_ln(vision_features)
        vision_x = self.vision_attention(vision_x, mask=None, deterministic=not training)
        vision_x = vision_features + vision_x
        vision_x = self.vision_proj(vision_x)
        
        # Text attending to vision  
        text_x = self.text_ln(text_features)
        text_x = self.text_attention(text_x, mask=None, deterministic=not training)
        text_x = text_features + text_x
        text_x = self.text_proj(text_x)
        
        return {
            'vision_output': vision_x,
            'text_output': text_x
        }

class MultimodalProjector(nn.Module):
    """Project multimodal features to joint space."""
    
    hidden_dim: int
    projection_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        """Initialize projections."""
        self.vision_proj = nn.Sequential([
            TPULayerNorm(),
            TPUGEMMLinear(features=self.projection_dim),
            nn.Dropout(rate=self.dropout_rate)
        ])
        self.text_proj = nn.Sequential([
            TPULayerNorm(),
            TPUGEMMLinear(features=self.projection_dim),
            nn.Dropout(rate=self.dropout_rate)
        ])
    
    def __call__(
        self,
        vision_features: jnp.ndarray,
        text_features: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Project features to joint space.
        
        Args:
            vision_features: Vision features [batch, seq_len, hidden_dim]
            text_features: Text features [batch, seq_len, hidden_dim]
            training: Whether in training mode
            
        Returns:
            Dictionary with projected features
        """
        vision_proj = self.vision_proj(vision_features, deterministic=not training)
        text_proj = self.text_proj(text_features, deterministic=not training)
        
        return {
            'vision_projected': vision_proj,
            'text_projected': text_proj
        }
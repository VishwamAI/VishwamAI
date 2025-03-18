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

    @nn.compact
    def __call__(
        self,
        vision_features: jnp.ndarray,
        text_features: jnp.ndarray,
        deterministic: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """Cross attend between vision and text features.
        
        Args:
            vision_features: Vision features [batch, vision_seq_len, hidden_dim]
            text_features: Text features [batch, text_seq_len, hidden_dim]
            deterministic: Whether in inference mode (no dropout)
            
        Returns:
            Dictionary with processed vision and text features
        """
        # Vision attending to text
        vision_attention = FlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            causal=False,
            dtype=self.dtype
        )
        text_attention = FlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            causal=False,
            dtype=self.dtype
        )
        
        vision_x = TPULayerNorm()(vision_features)
        vision_x = vision_attention(vision_x, deterministic=deterministic)
        vision_x = vision_features + vision_x
        vision_x = TPUGEMMLinear(features=self.hidden_dim)(vision_x)
        
        # Text attending to vision  
        text_x = TPULayerNorm()(text_features)
        text_x = text_attention(text_x, deterministic=deterministic)
        text_x = text_features + text_x
        text_x = TPUGEMMLinear(features=self.hidden_dim)(text_x)
        
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

    @nn.compact
    def __call__(
        self,
        vision_features: jnp.ndarray,
        text_features: jnp.ndarray,
        deterministic: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """Project features to joint embedding space.
        
        Args:
            vision_features: Vision features [batch, seq_len, hidden_dim]
            text_features: Text features [batch, seq_len, hidden_dim]
            deterministic: Whether in inference mode (no dropout)
            
        Returns:
            Dictionary with projected features
        """
        # Vision projection
        vision_x = TPULayerNorm()(vision_features)
        vision_x = TPUGEMMLinear(features=self.projection_dim)(vision_x)
        if not deterministic:
            vision_x = nn.Dropout(rate=self.dropout_rate)(vision_x, deterministic=False)
            
        # Text projection
        text_x = TPULayerNorm()(text_features)
        text_x = TPUGEMMLinear(features=self.projection_dim)(text_x)
        if not deterministic:
            text_x = nn.Dropout(rate=self.dropout_rate)(text_x, deterministic=False)
            
        return {
            'vision_embedding': vision_x,
            'text_embedding': text_x
        }
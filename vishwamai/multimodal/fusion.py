"""Fusion modules for multimodal processing."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple
from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm
from vishwamai.layers.attention import FlashAttention

class CrossAttentionFuser(nn.Module):
    """Cross-attention based fusion of multimodal features with optimized memory usage."""
    
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_flash_attention: bool = True
    block_size: int = 64
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        vision_features: jnp.ndarray,
        text_features: jnp.ndarray,
        audio_features: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Fuse multimodal features using cross-attention with memory optimizations.
        
        Args:
            vision_features: Visual features [batch, vis_seq, hidden]
            text_features: Text features [batch, text_seq, hidden]
            audio_features: Optional audio features [batch, audio_seq, hidden]
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Tuple of:
            - Fused features [batch, total_seq, hidden]
            - Attention statistics for analysis
        """
        # Project modalities to common space using memory-efficient GEMM
        vision_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(0.02)
        )(vision_features)
        
        text_proj = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(0.02)
        )(text_features)

        # Handle optional audio input
        if audio_features is not None:
            audio_proj = TPUGEMMLinear(
                features=self.hidden_dim,
                dtype=self.dtype,
                kernel_init=nn.initializers.truncated_normal(0.02)
            )(audio_features)
            # Concatenate all modalities
            concat_features = jnp.concatenate([vision_proj, text_proj, audio_proj], axis=1)
        else:
            # Just vision and text
            concat_features = jnp.concatenate([vision_proj, text_proj], axis=1)
        
        # FlashAttention for better memory efficiency 
        attention = FlashAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            use_causal_mask=False,
            block_size=self.block_size
        )(concat_features, mask=attention_mask, deterministic=not training)

        # Gradient checkpointing for memory efficiency
        if training:
            x = nn.remat(lambda x, y: x + y)(concat_features, attention)
        else:
            x = concat_features + attention
        
        # Memory-efficient MLP with kernel fusion
        y = TPULayerNorm(dtype=self.dtype)(x)
        mlp_dim = self.hidden_dim * 4

        y = nn.Sequential([
            TPUGEMMLinear(features=mlp_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            TPUGEMMLinear(features=self.hidden_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout_rate)
        ])(y, deterministic=not training)

        # Compute attention statistics for analysis
        attn_stats = {
            'cross_attention_scores': jnp.mean(attention, axis=(0,1)),
            'feature_norms': {
                'vision': jnp.mean(jnp.linalg.norm(vision_proj, axis=-1)),
                'text': jnp.mean(jnp.linalg.norm(text_proj, axis=-1)),
                'fused': jnp.mean(jnp.linalg.norm(x, axis=-1))
            }
        }
        if audio_features is not None:
            attn_stats['feature_norms']['audio'] = jnp.mean(jnp.linalg.norm(audio_proj, axis=-1))
        
        return x, attn_stats

class MultimodalProjector(nn.Module):
    """Projects fused features to task-specific outputs."""
    
    output_dim: int
    hidden_dim: int = 1024
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """
        Project fused features to output space.
        
        Args:
            features: Fused multimodal features [batch, seq, hidden]
            training: Whether in training mode
            
        Returns:
            Output projections [batch, output_dim]
        """
        # Memory-efficient pooling and projection
        x = jnp.mean(features, axis=1)  # Global average pooling
        
        x = TPULayerNorm(dtype=self.dtype)(x)
        x = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = TPUGEMMLinear(
            features=self.output_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(0.02)
        )(x)
        
        return x
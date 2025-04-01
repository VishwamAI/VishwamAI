"""Multimodal encoder implementation for VishwamAI."""
from vishwamai.layers.layers import TPUMultiHeadAttention
from vishwamai.transformer import TransformerBlock

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple

from vishwamai.kernels.core.kernel import fp8_gemm_optimized
from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm
from vishwamai.layers.attention import FlashAttention
from vishwamai.multimodal.vision import VisionEncoder
from vishwamai.multimodal.sonar import SonarEncoder

def sinusoidal_position_embedding(
    positions: jnp.ndarray,
    dim: int,
    dtype: Any = jnp.float32,
    min_scale: float = 1.0,
    max_scale: float = 10000.0
) -> jnp.ndarray:
    """Compute sinusoidal position embeddings efficiently."""
    log_timescale_increment = jnp.log(max_scale / min_scale) / (dim // 2 - 1)
    inv_timescales = min_scale * jnp.exp(jnp.arange(0, dim // 2) * -log_timescale_increment)
    
    scaled_time = jnp.expand_dims(positions, axis=1) * jnp.expand_dims(inv_timescales, axis=0)
    
    signal = jnp.concatenate([
        jnp.sin(scaled_time),
        jnp.cos(scaled_time)
    ], axis=1)

    if dim % 2 == 1:
        signal = jnp.pad(signal, [[0, 0], [0, 1]])
        
    return signal.astype(dtype)

class AudioEncoder(nn.Module):
    """Audio encoder for processing spectrogram inputs with memory optimizations."""
    
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    use_flash_attention: bool = True
    block_size: int = 64

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """
        Process audio spectrogram inputs.
        
        Args:
            inputs: Input spectrograms [batch, time, freq]
            training: Whether in training mode
        
        Returns:
            Encoded audio features
        """
        # Project inputs to hidden dimension with memory-efficient GEMM
        x = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(0.02)
        )(inputs)

        # Add sinusoidal position embeddings
        positions = jnp.arange(inputs.shape[1])[None]
        x = x + nn.remat(sinusoidal_position_embedding)(
            positions, self.hidden_dim, dtype=self.dtype
        )
        
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # Process with transformer layers using memory optimizations
        for _ in range(self.num_layers):
            # Layer norm with TPU optimization
            y = TPULayerNorm(dtype=self.dtype)(x)
            
            # FlashAttention for better memory efficiency
            attention = TPUMultiHeadAttention(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                block_size=self.block_size
            )(y, deterministic=not training)
            
            # Gradient checkpointing for memory efficiency
            if training:
                x = nn.remat(lambda x, y: x + y)(x, attention)
            else:
                x = x + attention
                
            # Memory-efficient MLP with kernel fusion
            y = TPULayerNorm(dtype=self.dtype)(x)
            y = nn.Sequential([
                TPUGEMMLinear(features=self.mlp_dim, dtype=self.dtype),
                nn.gelu,
                nn.Dropout(rate=self.dropout_rate),
                TPUGEMMLinear(features=self.hidden_dim, dtype=self.dtype),
                nn.Dropout(rate=self.dropout_rate)
            ])(y, deterministic=not training)
            
            x = x + y

        # Final layer norm
        return TPULayerNorm(dtype=self.dtype)(x)

class MultimodalEncoder(nn.Module):
    """Multimodal encoder that combines vision, text and audio."""
    
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    def setup(self):
        # Modality-specific encoders
        self.vision_encoder = VisionEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype
        )
        
        self.audio_encoder = AudioEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.hidden_dim * 4,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            block_size=64
        )
        
        # Cross-modal attention
        self.cross_attention = FlashAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            use_causal_mask=False,
            block_size=64
        )
        
        # Output projection
        self.layer_norm = TPULayerNorm(dtype=self.dtype)
        self.output_projection = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(0.02)
        )
    
    def __call__(
        self,
        vision_inputs: jnp.ndarray,
        text_inputs: jnp.ndarray,
        audio_inputs: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Process multimodal inputs.
        
        Args:
            vision_inputs: Visual features [batch, vis_seq, vis_dim]
            text_inputs: Text features [batch, text_seq, text_dim]
            audio_inputs: Optional audio features [batch, audio_seq, audio_dim]
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Dictionary containing encoded features and intermediate states
        """
        # Encode each modality
        vision_features = self.vision_encoder(
            vision_inputs,
            training=training
        )['last_hidden_state']
        
        # Process audio if provided
        audio_features = None
        if audio_inputs is not None:
            audio_features = self.audio_encoder(
                audio_inputs,
                training=training
            )

        # Fuse modalities with cross attention
        if audio_features is not None:
            # Concatenate all modalities
            combined_features = jnp.concatenate([
                vision_features,
                text_inputs,
                audio_features
            ], axis=1)
        else:
            # Just vision and text
            combined_features = jnp.concatenate([
                vision_features,
                text_inputs
            ], axis=1)

        # Cross-modal attention with memory optimizations
        attention_output = self.cross_attention(
            combined_features,
            mask=attention_mask,
            deterministic=not training
        )
        
        # Gradient checkpointing
        if training and self.use_gradient_checkpointing:
            x = nn.remat(lambda x, y: x + y)(combined_features, attention_output)
        else:
            x = combined_features + attention_output
            
        # Final processing
        x = self.layer_norm(x)
        outputs = self.output_projection(x)
        
        return {
            'last_hidden_state': outputs,
            'vision_features': vision_features,
            'text_features': text_inputs,
            'audio_features': audio_features,
            'attention_output': attention_output
        }

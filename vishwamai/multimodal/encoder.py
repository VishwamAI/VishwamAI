"""Multimodal encoder implementation for VishwamAI."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple
from ..layers import TPUGEMMLinear, TPULayerNorm
from ..transformer import TransformerBlock

class AudioEncoder(nn.Module):
    """Audio encoder for processing spectrogram inputs."""
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Project spectrogram features to hidden dimension
        x = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            name='audio_proj'
        )(x)
        
        # Add learned position embeddings
        positions = self.param('audio_pos_embed',
                          nn.initializers.normal(0.02),
                          (1, x.shape[1], self.hidden_dim))
        x = x + positions
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(x, deterministic=deterministic)
            
        return x

class VisionEncoder(nn.Module):
    """Vision encoder with patch embedding and transformer blocks."""
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    patch_size: int = 14
    image_size: int = 896
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Patch embedding
        batch_size, height, width, channels = x.shape
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        
        # Create patch embedding layer
        patch_embed = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            name='patch_embed'
        )
        
        # Convert image to patches
        x = jnp.reshape(x, (batch_size, num_patches, -1))
        x = patch_embed(x)
        
        # Add position embeddings
        pos_embed = self.param('pos_embed',
                             nn.initializers.normal(0.02),
                             (1, num_patches, self.hidden_dim))
        x = x + pos_embed
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype
            )(x, deterministic=deterministic)
        
        return x

class MultimodalEncoder(nn.Module):
    """Combined vision, audio, and text encoder."""
    config: Dict[str, Any]

    def setup(self):
        # Vision encoder
        if self.config.get('vision_config'):
            self.vision_encoder = VisionEncoder(
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['vision_layers'],
                num_heads=self.config['num_heads'],
                mlp_dim=self.config['mlp_dim'],
                patch_size=self.config.get('patch_size', 14),
                image_size=self.config.get('image_size', 896),
                dropout_rate=self.config.get('dropout_rate', 0.1),
                dtype=self.config.get('dtype', jnp.float32)
            )
            
        # Audio encoder
        if self.config.get('audio_config'):
            self.audio_encoder = AudioEncoder(
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['audio_layers'],
                num_heads=self.config['num_heads'],
                mlp_dim=self.config['mlp_dim'],
                dropout_rate=self.config.get('dropout_rate', 0.1),
                dtype=self.config.get('dtype', jnp.float32)
            )
        
        # Cross-attention for modality fusion
        self.cross_attention = TransformerBlock(
            num_heads=self.config['num_heads'],
            head_dim=self.config['hidden_dim'] // self.config['num_heads'],
            mlp_dim=self.config['mlp_dim'],
            dropout_rate=self.config.get('dropout_rate', 0.1),
            dtype=self.config.get('dtype', jnp.float32)
        )
        
        # Final layer norm
        self.norm = TPULayerNorm(dtype=self.config.get('dtype', jnp.float32))

    def __call__(
        self,
        image_input: Optional[jnp.ndarray] = None,
        audio_input: Optional[jnp.ndarray] = None,
        text_input: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        outputs = {}
        features_to_fuse = []
        
        # Process vision input if provided
        if image_input is not None and hasattr(self, 'vision_encoder'):
            vision_features = self.vision_encoder(
                image_input,
                deterministic=not training
            )
            outputs['vision_features'] = vision_features
            features_to_fuse.append(vision_features)
        
        # Process audio input if provided
        if audio_input is not None and hasattr(self, 'audio_encoder'):
            audio_features = self.audio_encoder(
                audio_input,
                deterministic=not training
            )
            outputs['audio_features'] = audio_features
            features_to_fuse.append(audio_features)
        
        # Process text input if provided
        if text_input is not None:
            outputs['text_features'] = text_input
            features_to_fuse.append(text_input)
        
        # Fuse modalities if multiple are present
        if len(features_to_fuse) > 1:
            # Concatenate features along sequence dimension
            fused_features = jnp.concatenate(features_to_fuse, axis=1)
            
            # Apply cross-attention for modality fusion
            fused_features = self.cross_attention(
                fused_features,
                mask=None,
                deterministic=not training
            )
            fused_features = self.norm(fused_features)
            outputs['fused_features'] = fused_features
            return fused_features, outputs
        
        # Return single modality features if only one is provided
        elif len(features_to_fuse) == 1:
            return features_to_fuse[0], outputs
        else:
            raise ValueError("At least one input modality must be provided")
"""
Multimodal processing capabilities for VishwamAI.

Implements vision, audio, and text processing components that can be
unified into a single transformer architecture.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Any, Union, Tuple
import chex

from .layers import RMSNorm, TokenEmbedding
from .attention import OptimizedAttention


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings for transformer processing."""
    
    patch_size: int = 16
    embed_dim: int = 768
    in_channels: int = 3
    
    def setup(self):
        self.projection = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )
    
    def __call__(self, images: chex.Array) -> chex.Array:
        """Convert images to patch embeddings.
        
        Args:
            images: [batch, height, width, channels]
            
        Returns:
            patches: [batch, num_patches, embed_dim]
        """
        
        batch_size, height, width, channels = images.shape
        
        # Ensure image dimensions are divisible by patch size
        assert height % self.patch_size == 0, f"Height {height} not divisible by patch size {self.patch_size}"
        assert width % self.patch_size == 0, f"Width {width} not divisible by patch size {self.patch_size}"
        
        # Project patches
        patches = self.projection(images)  # [batch, h_patches, w_patches, embed_dim]
        
        # Flatten spatial dimensions
        h_patches, w_patches = patches.shape[1], patches.shape[2]
        patches = patches.reshape(batch_size, h_patches * w_patches, self.embed_dim)
        
        return patches


class VisionEncoder(nn.Module):
    """Vision encoder using Vision Transformer (ViT) architecture."""
    
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_cls_token: bool = True
    
    def setup(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        
        # Class token (optional)
        if self.use_cls_token:
            self.cls_token = self.param(
                'cls_token',
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )
        
        # Positional embeddings
        pos_embed_shape = (1, self.num_patches + (1 if self.use_cls_token else 0), self.embed_dim)
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            pos_embed_shape
        )
        
        # Transformer blocks
        self.blocks = [
            VisionTransformerBlock(
                dim=self.embed_dim,
                heads=self.heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
            for _ in range(self.depth)
        ]
        
        # Final norm
        self.norm = nn.LayerNorm()
        
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    
    def __call__(self, images: chex.Array, training: bool = True) -> chex.Array:
        """Encode images to features.
        
        Args:
            images: [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            features: [batch, seq_len, embed_dim]
        """
        
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [batch, num_patches, embed_dim]
        
        # Add class token
        if self.use_cls_token:
            cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout_layer(x, deterministic=not training)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision encoder."""
    
    dim: int
    heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    def setup(self):
        # Attention
        self.attention = OptimizedAttention(
            dim=self.dim,
            heads=self.heads,
            head_dim=self.dim // self.heads,
            dropout=self.dropout
        )
        
        # MLP
        mlp_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(mlp_dim),
            nn.gelu,
            nn.Dropout(rate=self.dropout),
            nn.Dense(self.dim),
            nn.Dropout(rate=self.dropout)
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
    
    def __call__(self, x: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass through transformer block."""
        
        # Attention with residual connection
        attn_out = self.attention(self.norm1(x), training=training)
        x = x + attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x


class AudioEncoder(nn.Module):
    """Audio encoder for processing audio spectrograms."""
    
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    embed_dim: int = 512
    depth: int = 6
    heads: int = 8
    
    def setup(self):
        # Convolutional frontend
        self.conv_layers = [
            nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
        ]
        
        # Projection to embedding dimension
        self.projection = nn.Dense(self.embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Embed(num_embeddings=5000, features=self.embed_dim)
        
        # Transformer blocks
        self.blocks = [
            VisionTransformerBlock(  # Reuse vision transformer block
                dim=self.embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(self.depth)
        ]
        
        # Final norm
        self.norm = nn.LayerNorm()
    
    def __call__(self, audio: chex.Array, training: bool = True) -> chex.Array:
        """Encode audio spectrograms to features.
        
        Args:
            audio: [batch, time, freq] spectrogram
            training: Whether in training mode
            
        Returns:
            features: [batch, seq_len, embed_dim]
        """
        
        # Add channel dimension if needed
        if len(audio.shape) == 3:
            audio = audio[..., None]  # [batch, time, freq, 1]
        
        x = audio
        
        # Convolutional frontend
        for conv in self.conv_layers:
            x = conv(x)
            x = jax.nn.relu(x)
        
        # Flatten spatial dimensions
        batch_size, time_dim, freq_dim, channels = x.shape
        x = x.reshape(batch_size, time_dim * freq_dim, channels)
        
        # Project to embedding dimension
        x = self.projection(x)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)
        pos_emb = self.pos_embed(positions)
        x = x + pos_emb[None, :, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class MultimodalProcessor(nn.Module):
    """Unified processor for multiple modalities."""
    
    vocab_size: int = 50304
    embed_dim: int = 768
    vision_config: Optional[Dict[str, Any]] = None
    audio_config: Optional[Dict[str, Any]] = None
    use_modality_tokens: bool = True
    
    def setup(self):
        # Text embedding
        self.text_embed = TokenEmbedding(
            vocab_size=self.vocab_size,
            dim=self.embed_dim
        )
        
        # Vision encoder
        if self.vision_config is not None:
            self.vision_encoder = VisionEncoder(
                embed_dim=self.embed_dim,
                **self.vision_config
            )
            self.vision_projection = nn.Dense(self.embed_dim)
        
        # Audio encoder
        if self.audio_config is not None:
            self.audio_encoder = AudioEncoder(
                embed_dim=self.embed_dim,
                **self.audio_config
            )
            self.audio_projection = nn.Dense(self.embed_dim)
        
        # Modality tokens
        if self.use_modality_tokens:
            self.text_token = self.param(
                'text_token',
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )
            if self.vision_config is not None:
                self.vision_token = self.param(
                    'vision_token',
                    nn.initializers.normal(stddev=0.02),
                    (1, 1, self.embed_dim)
                )
            if self.audio_config is not None:
                self.audio_token = self.param(
                    'audio_token',
                    nn.initializers.normal(stddev=0.02),
                    (1, 1, self.embed_dim)
                )
    
    def process_text(self, text_ids: chex.Array) -> chex.Array:
        """Process text tokens."""
        
        embeddings = self.text_embed(text_ids)
        
        if self.use_modality_tokens:
            batch_size = text_ids.shape[0]
            text_tokens = jnp.broadcast_to(self.text_token, (batch_size, 1, self.embed_dim))
            embeddings = jnp.concatenate([text_tokens, embeddings], axis=1)
        
        return embeddings
    
    def process_vision(self, images: chex.Array, training: bool = True) -> chex.Array:
        """Process images."""
        
        if not hasattr(self, 'vision_encoder'):
            raise ValueError("Vision encoder not configured")
        
        features = self.vision_encoder(images, training=training)
        embeddings = self.vision_projection(features)
        
        if self.use_modality_tokens:
            batch_size = images.shape[0]
            vision_tokens = jnp.broadcast_to(self.vision_token, (batch_size, 1, self.embed_dim))
            embeddings = jnp.concatenate([vision_tokens, embeddings], axis=1)
        
        return embeddings
    
    def process_audio(self, audio: chex.Array, training: bool = True) -> chex.Array:
        """Process audio spectrograms."""
        
        if not hasattr(self, 'audio_encoder'):
            raise ValueError("Audio encoder not configured")
        
        features = self.audio_encoder(audio, training=training)
        embeddings = self.audio_projection(features)
        
        if self.use_modality_tokens:
            batch_size = audio.shape[0]
            audio_tokens = jnp.broadcast_to(self.audio_token, (batch_size, 1, self.embed_dim))
            embeddings = jnp.concatenate([audio_tokens, embeddings], axis=1)
        
        return embeddings
    
    def __call__(
        self,
        text_ids: Optional[chex.Array] = None,
        images: Optional[chex.Array] = None,
        audio: Optional[chex.Array] = None,
        training: bool = True
    ) -> chex.Array:
        """Process multiple modalities and concatenate."""
        
        embeddings_list = []
        
        # Process each modality if provided
        if text_ids is not None:
            text_emb = self.process_text(text_ids)
            embeddings_list.append(text_emb)
        
        if images is not None:
            vision_emb = self.process_vision(images, training=training)
            embeddings_list.append(vision_emb)
        
        if audio is not None:
            audio_emb = self.process_audio(audio, training=training)
            embeddings_list.append(audio_emb)
        
        if not embeddings_list:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all modalities
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        else:
            return jnp.concatenate(embeddings_list, axis=1)


class ModalityAdapter(nn.Module):
    """Adapter layer for aligning different modalities."""
    
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    
    def setup(self):
        if self.hidden_dim is None:
            self.hidden_dim = max(self.input_dim, self.output_dim)
        
        self.layers = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dense(self.output_dim),
            nn.LayerNorm()
        ])
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Adapt modality features to target dimension."""
        return self.layers(x)

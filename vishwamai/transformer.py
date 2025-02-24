import flax.linen as nn
import jax.numpy as jnp
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """
    Convert an image into patches.

    Args:
        x (jnp.ndarray): Input image tensor of shape (B, H, W, C).
        patch_size (int): Size of each patch.

    Returns:
        jnp.ndarray: Patchified image tensor of shape (B, num_patches, patch_size*patch_size*C).
    """
    B, H, W, C = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    x = x.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, num_patches_h*num_patches_w, patch_size*patch_size*C)
    return x

class PatchEmbedding(nn.Module):
    embedding_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds flattened image patches into a learned embedding space.
        
        Transforms each flattened patch into a dense embedding by applying a linear projection.
        
        Args:
            x (jnp.ndarray): Input tensor of shape (B, num_patches, patch_size*patch_size*C) representing flattened image patches.
        
        Returns:
            jnp.ndarray: Tensor of shape (B, num_patches, embedding_size) containing the embedded patches.
        """
        return nn.Dense(self.embedding_size)(x)

class TransformerBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Applies a transformer block to a sequence.
        
        This block performs self-attention followed by a feedforward network, each with a residual
        connection and preceded by layer normalization. Dropout is applied during training mode.
        
        Args:
            x (jnp.ndarray): Input tensor with shape (B, seq_len, hidden_size).
            train (bool): Indicates if the block is operating in training mode, affecting dropout.
        
        Returns:
            jnp.ndarray: Output tensor with shape (B, seq_len, hidden_size) after transformation.
        """
        x_norm = nn.LayerNorm()(x)
        q = nn.Dense(self.hidden_size)(x_norm)
        k = nn.Dense(self.hidden_size)(x_norm)
        v = nn.Dense(self.hidden_size)(x_norm)
        attn_output = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate)(q, k, v, deterministic=not train)
        attn_output = nn.Dense(self.hidden_size)(attn_output)  # Output projection
        x = x + attn_output
        x_norm = nn.LayerNorm()(x)
        ffn_output = nn.MLP([self.mlp_dim, self.hidden_size], activation=nn.relu, dropout_rate=self.dropout_rate)(x_norm, deterministic=not train)
        x = x + ffn_output
        return x

class VisionTransformer10B(nn.Module):
    num_classes: int = 1000
    patch_size: int = 16
    hidden_size: int = 6144
    num_heads: int = 96
    num_layers: int = 24
    mlp_dim: int = 24576
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Apply the Vision Transformer model.

        Args:
            x (jnp.ndarray): Input image tensor of shape (B, H, W, C).
            train (bool): Whether the model is in training mode.

        Returns:
            jnp.ndarray: Output logits of shape (B, num_classes).
        """
        logger.info("Starting Vision Transformer forward pass")
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        x = patchify(x, self.patch_size)  # (B, num_patches, P*P*C)
        x = PatchEmbedding(self.hidden_size)(x)  # (B, num_patches, hidden_size)
        cls_token = self.param('cls_token', nn.initializers.normal(stddev=0.02), (1, 1, self.hidden_size))
        cls_token = jnp.tile(cls_token, (B, 1, 1))
        x = jnp.concat([cls_token, x], axis=1)  # (B, num_patches+1, hidden_size)
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, num_patches+1, self.hidden_size))
        x = x + pos_embedding
        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, self.mlp_dim, self.dropout_rate)(x, train)
        x = x[:, 0]  # Take [CLS] token
        x = nn.Dense(self.num_classes)(x)
        logger.info("Completed Vision Transformer forward pass")
        return x

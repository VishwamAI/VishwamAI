"""Utility functions for vision models."""

from __future__ import annotations
from collections.abc import Sequence
from flax import linen as nn
import jax
from jax import numpy as jnp
from kauldron import typing
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, TypeVar, Literal
from numpy.typing import NDArray, ArrayLike

# Define type variables for better type hinting
B = TypeVar('B')  # Batch dimension
M = TypeVar('M')  # Sequence length dimension
D = TypeVar('D')  # Embedding dimension
N = TypeVar('N')  # Image height dimension
P = TypeVar('P')  # Image width dimension


def _posemb_sincos_2d(
    h: int,
    w: int,
    width: int,
    temperature: float = 10000.0,
    dtype: Any = jnp.float32
) -> jnp.ndarray:
    """Generate 2D sinusoidal position embeddings.
    
    Args:
        h: Height (patches)
        w: Width (patches)
        width: Hidden dimension size
        temperature: Temperature for frequencies
        dtype: Data type for outputs
        
    Returns:
        Position embeddings [h*w, width]
    """
    y, x = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing='ij'
    )
    
    if width % 4 != 0:
        raise ValueError(f"Width {width} must be divisible by 4 for 2D position embeddings")
        
    omega = jnp.arange(width // 4, dtype=jnp.float32) / (width // 4 - 1)
    omega = 1. / (temperature ** omega)
    
    y = y.reshape(-1)[:, None] * omega[None, :]
    x = x.reshape(-1)[:, None] * omega[None, :]
    
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    
    return pe.astype(dtype)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply MLP block.
        
        Args:
            x: Input tensor
            deterministic: Whether in inference mode
            
        Returns:
            Transformed tensor
        """
        actual_mlp_dim = self.mlp_dim or x.shape[-1] * 4
        
        y = nn.Dense(features=actual_mlp_dim, dtype=self.dtype)(x)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(features=x.shape[-1], dtype=self.dtype)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        
        return y


class MAPHead(nn.Module):
    """Multi-head Attention Pooling."""
    
    block_id: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply multi-head attention pooling.
        
        Args:
            x: Input tensor [batch, length, channels]
            deterministic: Whether in inference mode
            
        Returns:
            Pooled features [batch, channels]  
        """
        n, l, d = x.shape
        probe = self.param(
            'probe',
            nn.initializers.xavier_uniform(),
            (1, 1, d)
        )
        probe = jnp.tile(probe, [n, 1, 1])

        # Self attention
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform()
        )(probe, x)

        # Layer norm and MLP
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
        
        return x[:, 0]  # Return CLS token only


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  block_id: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(
      self, x: jax.Array, deterministic: bool = True
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)

    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlock(
        block_id=self.block_id,
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  depth: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(2,),  # 0=self, 2=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      x = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth,
      )(
          block_id=0,
          name="encoderblock",
          dtype_mm=self.dtype_mm,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
      )(
          x, deterministic
      )
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            block_id=lyr,
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        x = block_cur(x, deterministic)
    x: jax.Array = nn.LayerNorm(name="encoder_norm")(x)
    return x


class ViTModel(nn.Module):
  """ViT model.

  Attributes:
    compression_type: The compression type.
    width: The model dimension of the vision encoder.
    mlp_dim: The hidden dimension in the ffw layers.
    num_heads: The number of the heads.
    depth: The number of the layers.
    patch_size: The size to patchify images.
    posemb: The position embedding type.
    dropout: The dropout rate.
    scan: Whether to scan the layers (layer stacking).
    remat_policy: The remat policy.
    dtype_mm: The dtype to convert the input to.
    output_length: Number of soft tokens per image.
  """

  patch_size: Sequence[int] = (14, 14)
  width: int = 1152
  depth: int = 27
  mlp_dim: int | None = 4304  # Defaults to 4x input dim
  num_heads: int = 16
  posemb: str = "learn"  # Can also be "sincos2d"
  dropout: float = 0.0
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  def _get_posemb(
      self,
      typ: str,
      *,
      seqshape: tuple[int, int],
      width: int,
      name: str,
      dtype: jnp.dtype = jnp.float32,
  ) -> jnp.ndarray:  # Changed from string literal to jnp.ndarray
    """Returns the position embedding."""
    if typ == "learn":
      return self.param(
          name,
          nn.initializers.normal(stddev=1 / np.sqrt(width)),
          (1, np.prod(seqshape), width),
          dtype,
      )
    elif typ == "sincos2d":
      return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
    else:
      raise ValueError(f"Unknown posemb type: {typ}")

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,  # Changed from string literal to jnp.ndarray
      *,
      train: bool = False,
  ):
    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding="VALID",
        name="embedding",
        dtype=self.dtype_mm,
    )(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = x + self._get_posemb(
        self.posemb,
        seqshape=(h, w),
        width=c,
        name="pos_embedding",
        dtype=x.dtype,
    )

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        name="Transformer",
    )(x, deterministic=not train)

    return x


def convert_to_rgb(
    image: NDArray,
    input_format: str = "BGR"
) -> NDArray:
    """Convert image to RGB format.
    
    Args:
        image: Input image array of shape (height, width, 3)
        input_format: Current color format ('BGR', 'RGB', etc)
        
    Returns:
        RGB image array
    """
    if input_format == "BGR":
        return image[..., ::-1]
    elif input_format == "RGB":
        return image
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

def center_crop(
    image: NDArray,
    target_height: int,
    target_width: int
) -> NDArray:
    """Center crop image to target size.
    
    Args:
        image: Image array of shape (height, width, channels)
        target_height: Height to crop to
        target_width: Width to crop to
        
    Returns:
        Cropped image array
    """
    height, width = image.shape[:2]
    
    start_h = (height - target_height) // 2
    start_w = (width - target_width) // 2
    
    return image[start_h:start_h + target_height,
                start_w:start_w + target_width]

def apply_image_transforms(
    image: NDArray,
    do_resize: bool = True,
    do_center_crop: bool = True,
    size: Union[int, Tuple[int, int]] = 224,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
    do_normalize: bool = True,
) -> NDArray:
    """Apply standard image transformations.
    
    Args:
        image: Input image array of shape (height, width, channels)
        do_resize: Whether to resize image
        do_center_crop: Whether to center crop
        size: Target size for resize/crop
        mean: Optional mean for normalization 
        std: Optional std for normalization
        do_normalize: Whether to normalize
        
    Returns:
        Transformed image array
    """
    if isinstance(size, int):
        size = (size, size)
        
    if do_resize:
        image = jax.image.resize(
            image,
            shape=(*size, image.shape[-1]),
            method="bilinear",
            antialias=True
        )
        
    if do_center_crop:
        image = center_crop(image, size[0], size[1])
        
    if do_normalize:
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
            
        mean = jnp.array(mean).reshape(1, 1, -1)
        std = jnp.array(std).reshape(1, 1, -1)
        
        image = (image - mean) / std
        
    return image

def extract_patches(
    images: NDArray,
    patch_size: int,
    stride: Optional[int] = None,
) -> NDArray:
    """Extract image patches.
    
    Args:
        images: Batch of images of shape (batch_size, height, width, channels)
        patch_size: Size of patches
        stride: Optional stride for patch extraction
        
    Returns:
        Array of patches of shape (batch_size, num_patches, patch_dim)
    """
    stride = stride or patch_size
    batch_size, height, width, channels = images.shape
    
    # Extract patches using convolution operation
    patch_dim = patch_size * patch_size * channels
    
    # Create identity filter for patch extraction
    identity_filter = jnp.eye(patch_dim).reshape(
        patch_size, patch_size, channels, patch_dim
    )
    
    # Extract patches with convolution
    patches = jax.lax.conv_general_dilated(
        images,
        identity_filter, 
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Reshape to (batch_size, num_patches, patch_dim)
    patches = patches.reshape(batch_size, -1, patch_dim)
    
    return patches
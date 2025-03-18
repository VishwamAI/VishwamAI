# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma implementation of the vision encoders."""

from __future__ import annotations

import dataclasses
import functools
from typing import cast, Any, Dict, Optional, Tuple, Union

import chex
import einops
from etils import epy
import flax
from flax import linen as nn
from vishwamai.multimodal import vision_utils
import jax
from jax import numpy as jnp
from kauldron.typing import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from numpy.typing import NDArray, ArrayLike

with epy.lazy_imports():
    # TODO(epot): Refactor to move everything inside gemma/gm/
    from gemma.gm.vision import _preprocess  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

# Constants
BEGIN_IMAGE_TOKEN = 255999
END_IMAGE_TOKEN = 262144
NEW_LINE_TOKEN = 108
TOKEN_PLACEHOLDER = -2
NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
NUM_TOKENS_PER_MEDIA = NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


@typechecked
def check_mask(input_data: Float["L"]) -> tuple[Bool["1"], Int["L"]]:  # type: ignore
    """Checks that the mask contains the correct number of blocks.

    Args:
        input_data: The input data to check.

    Returns:
        A boolean indicating whether the mask contains the correct number of blocks
        and their starting positions (filled with 0 for jit compatibility).
    """
    is_mask = input_data == TOKEN_PLACEHOLDER
    start_idx = (
        jnp.where(
            jnp.logical_and(is_mask[:-1] != is_mask[1:], ~is_mask[:-1]),
            size=is_mask.shape[0],
            fill_value=-1,
        )[0]
        + 1
    )
    end_idx = (
        jnp.where(
            jnp.logical_and(is_mask[:-1] != is_mask[1:], is_mask[:-1]),
            size=is_mask.shape[0],
            fill_value=NUM_PLACEHOLDER_TOKENS_PER_IMAGE - 1,
        )[0]
        + 1
    )
    all_blocks = end_idx - start_idx
    is_valid = jnp.all(all_blocks == NUM_PLACEHOLDER_TOKENS_PER_IMAGE)
    is_valid = jnp.logical_and(~is_mask[0], is_valid)
    is_valid = jnp.logical_and(~is_mask[-1], is_valid)
    return is_valid, start_idx


@typechecked
def check_special_vision_token(
    input_data: Float["B L"],  # type: ignore
    *,
    start_positions: Int["B L"],  # type: ignore
    special_token: int,
    position_offset: int,
) -> Bool["1"]:  # type: ignore
    """Checks that the input data contains the correct special vision tokens.

    Args:
        input_data: The input data to check.
        start_positions: The starting positions of the blocks (filled with 0 for jit
          compatibility).
        special_token: The mask token.
        position_offset: The position offset.

    Returns:
        A boolean indicating whether the input data contains the correct special
        vision tokens.
    """
    dummy_data = jnp.copy(input_data)
    # Fix the zero-filled array
    dummy_data = dummy_data.at[jnp.arange(len(dummy_data)), position_offset].set(
        special_token
    )
    return jnp.all(
        dummy_data[jnp.arange(len(input_data)), start_positions + position_offset]
        == special_token
    )


@flax.struct.dataclass
class VisionInitEmbeddings:
    """Container for vision encoder output."""

    patches: Float["B N P D"] | None  # type: ignore
    token_buffer: Int["B NEW_BUFFER"]  # type: ignore
    num_input_tokens: Int["B"]  # type: ignore


@typechecked
def initialize_vision_tokens(
    patches: Float["B N P D"] | None,  # type: ignore
    token_buffer: Int["B BUFFER"],  # type: ignore
    num_input_tokens: Int["B"],  # type: ignore
) -> VisionInitEmbeddings:
    """Initializes vision embeddings.

    Vision data initialization wrapper for sampling.

    Example (text only inference):
    ```python
    Input:
    {
        "patches": None,
        "token_buffer": [
            [255999, 108, 262144, 108, ...],
            ],
        "num_input_tokens": [100,]
    }
    Output = Input
    ```
    Example (images):
    ```python
    Input:
    {
        "patches": [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
            ],
        "token_buffer": [
            [255999, 108, 262144, 108, ...],
            ],
        "num_input_tokens": [100,]
    }
    Output = {
        "patches": [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
            ],
        "token_buffer": [
            [255999, DoubleNewLine, BeginImageToken, -2, -2, ..., -2,
            EndImageToken, DoubleNewLine,
            108, 262144, 108, ...],
            ],
        "num_input_tokens": [360,]
    }
    ```
    where -2 is the image token placeholder.

    NOTE: all images are inserted at the beginning of the token buffer and not
    interleaved with text tokens.

    Args:
        patches: patched images of shape BxNxPxD.
        token_buffer: The token buffer to initialize.
        num_input_tokens: The number of input tokens.

    Returns:
        The vision embeddings, the token buffer, and the number of input tokens.
    """
    if patches is not None:
        # First create an array filled with the vision token placeholder [-2,...,-2]
        base_mm_tokens = jnp.full(
            shape=(token_buffer.shape[0], NUM_TOKENS_PER_MEDIA),
            fill_value=TOKEN_PLACEHOLDER,
            dtype=jnp.int32,
        )
        # Then edit the ends with double new line, begin image and end image tokens
        base_mm_tokens = base_mm_tokens.at[:, 0].set(NEW_LINE_TOKEN)
        base_mm_tokens = base_mm_tokens.at[:, 1].set(BEGIN_IMAGE_TOKEN)
        base_mm_tokens = base_mm_tokens.at[:, -2].set(END_IMAGE_TOKEN)
        base_mm_tokens = base_mm_tokens.at[:, -1].set(NEW_LINE_TOKEN)
        # Then we repeat this tensor for each image in the batch
        mm_tokens = jnp.repeat(base_mm_tokens, patches.shape[1], axis=0)
        # Insert images after the first token, which is BOS as expected in gemma v3
        token_buffer = jnp.concatenate(
            [token_buffer[:, :1], mm_tokens, token_buffer[:, 1:]], axis=1
        )
        num_input_tokens += NUM_TOKENS_PER_MEDIA * patches.shape[1]
    return VisionInitEmbeddings(patches, token_buffer, num_input_tokens)


class VisionExit(nn.Module):
    """The vision exit layer.

    Possibly downsample the soft tokens to a required output length.

    Attributes:
        output_length: The embed will be spatially avg-pooled to this output length.
    """

    output_length: int = 256

    @typechecked
    def __call__(
        self, x: Float["B INPUT_LENGTH D"]  # type: ignore
    ) -> Float["B OUTPUT_LENGTH D"]:  # type: ignore
        cur_length = x.shape[1]
        if cur_length == self.output_length:
            return x
        cur_width = int(cur_length**0.5)
        assert cur_width**2 == cur_length
        output_width = int(self.output_length**0.5)
        assert (
            output_width**2 == self.output_length
        ), f"Cannot pool {x.shape=} to {self.output_length}=!"
        x = einops.rearrange(x, " b (h w) d -> b h w d", h=cur_width, w=cur_width)
        assert not cur_width % output_width, f"{cur_width=} {output_width=}"
        window = cur_width // output_width
        window_shape = (window, window)
        x = nn.avg_pool(x, window_shape=window_shape, strides=window_shape)
        return einops.rearrange(x, "b h w d -> b (h w) d")


class SigLiPFromPatches(nn.Module):
    """SigLIP vision encoder forward pass from PatchifiedMedia."""

    siglip_encoder: vision_utils.ViTModel = dataclasses.field(
        default_factory=vision_utils.ViTModel
    )
    siglip_exit: VisionExit = dataclasses.field(default_factory=VisionExit)
    num_mm_tokens_per_image_prepool: int = 4096
    num_mm_tokens_per_image: int = 256
    image_height: int = 896
    image_width: int = 896
    image_channels: int = 3
    apply_stop_gradient: bool = True

    @functools.partial(nn.jit, static_argnames=("self", "is_training"))
    @nn.compact
    def __call__(
        self,
        *,
        patches: Float["B N P D"],  # type: ignore
        is_training: bool,
    ) -> Float["B N siglip_embed_dim"]:  # type: ignore
        chex.assert_rank(patches, 4)
        batch_size, num_frames, num_patches, num_channels = patches.shape
        num_patches_one_side = (
            self.image_height // self.siglip_encoder.patch_size[0]
        )
        chex.assert_equal(num_channels, 3 * self.siglip_encoder.patch_size[0] ** 2)
        chex.assert_equal(num_patches, num_patches_one_side**2)
        flattened_images = einops.rearrange(
            patches,
            "b n (h w) c -> (b n) h w c",
            h=num_patches_one_side,
            w=num_patches_one_side,
            c=num_channels,
        )
        flattened_images = einops.rearrange(
            flattened_images,
            "b h w (p q c) -> b (h p) (w q) c",
            h=num_patches_one_side,
            w=num_patches_one_side,
            p=self.siglip_encoder.patch_size[0],
            q=self.siglip_encoder.patch_size[0],
            c=3,
        )

        soft_tokens = self.siglip_encoder(flattened_images)

        if self.num_mm_tokens_per_image_prepool != self.num_mm_tokens_per_image:
            soft_tokens = self.siglip_exit(soft_tokens)
            assert soft_tokens.shape[-2] == self.siglip_exit.output_length

        soft_tokens = einops.rearrange(
            soft_tokens, "(b n) ... -> b n ...", b=batch_size, n=num_frames
        )
        soft_tokens = cast(jax.Array, soft_tokens)

        if self.apply_stop_gradient:
            soft_tokens = jax.lax.stop_gradient(soft_tokens)
        return soft_tokens

    def patchify_images(self, images: Float["*B H W C"]) -> Float["*B P D"]:  # type: ignore
        """Patchify images.

        Args:
            images: The images to patchify.

        Returns:
            The patches of the images of shape (*batch, num_patches, patch_size *
            patch_size * channels)
        """
        *batch_dims, _, _, _ = images.shape
        images = einops.rearrange(images, "... h w c -> (...) h w c")

        preprocess_fn = functools.partial(
            _preprocess.pre_process_image,
            image_shape=(self.image_height, self.image_width, self.image_channels),
        )
        images = jax.vmap(preprocess_fn)(images)

        patches = _preprocess.patchify_images(
            images,
            patch_size=self.siglip_encoder.patch_size,
        )
        patches = patches.reshape((*batch_dims,) + patches.shape[1:])
        return patches


class PatchEmbedding(nn.Module):
    """Image to patch embedding."""

    patch_size: int = 16
    hidden_size: int = 768
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images: NDArray) -> NDArray:
        """Convert images to patch embeddings.

        Args:
            images: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, hidden_size)
        """
        batch_size, height, width, channels = images.shape

        # Linear projection of flattened patches
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype,
            name="patch_proj",
        )(images)

        # Reshape to (batch_size, num_patches, hidden_size)
        return x.reshape(batch_size, -1, self.hidden_size)


class VisionEncoder(nn.Module):
    """Vision encoder with patch embedding and transformer."""

    num_layers: int
    num_heads: int
    mlp_dim: int
    hidden_size: int = 768
    patch_size: int = 16
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values: NDArray,
        deterministic: bool = True,
    ) -> Tuple[NDArray, Dict[str, NDArray]]:
        """Process image input through vision encoder.

        Args:
            pixel_values: Image tensor of shape (batch_size, height, width, channels)
            deterministic: Whether to apply dropout

        Returns:
            Tuple of:
            - Final hidden states of shape (batch_size, sequence_length, hidden_size)
            - Dictionary with intermediate activations
        """
        # Extract patches and project to hidden size
        x = PatchEmbedding(
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
        )(pixel_values)

        # Add position embeddings
        num_patches = x.shape[1]
        position_embeddings = self.param(
            "position_embeddings",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.hidden_size),
        )
        x = x + position_embeddings

        # Dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Store intermediate activations
        activations = {}

        # Process through transformer blocks
        for i in range(self.num_layers):
            # Layer normalization
            x = nn.LayerNorm()(x)

            # Multi-head attention
            attention_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                dropout_rate=self.attention_dropout,
                deterministic=deterministic,
            )(x, x)

            # Residual connection
            x = x + attention_output

            # MLP block
            y = nn.LayerNorm()(x)
            y = nn.Dense(features=self.mlp_dim, dtype=self.dtype)(y)
            y = nn.gelu(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(features=self.hidden_size, dtype=self.dtype)(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

            # Residual connection
            x = x + y

            # Store activation
            activations[f"layer_{i}"] = x

        # Final layer norm
        x = nn.LayerNorm()(x)
        activations["last_hidden_state"] = x

        return x, activations


class VisionProjection(nn.Module):
    """Project vision features to text space."""

    hidden_size: int
    projection_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: NDArray,
        deterministic: bool = True,
    ) -> NDArray:
        """Project vision features.

        Args:
            hidden_states: Vision features of shape (batch_size, sequence_length, hidden_size)
            deterministic: Whether to apply dropout

        Returns:
            Projected features of shape (batch_size, sequence_length, projection_dim)
        """
        x = nn.LayerNorm()(hidden_states)
        x = nn.Dense(features=self.projection_dim, dtype=self.dtype)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x
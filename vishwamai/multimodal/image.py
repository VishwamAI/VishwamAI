from __future__ import annotations
from collections.abc import Sequence
import einops
from etils import epath
import jax
from jax import numpy as jnp
from kauldron import typing
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Optional, Tuple, Union, Any
from numpy.typing import NDArray, ArrayLike

_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896  # SigLip expected input image size
_DEFAULT_PATCH_SIZE = 14  # SigLip expected patch size


def normalize_images(images: NDArray) -> NDArray:
    """Normalize the image to zero mean and unit variance.

    Args:
        images: Image array of shape (height, width, channels)

    Returns:
        Normalized image array
    """
    mean = np.asarray(_IMAGE_MEAN).reshape(1, 1, -1)
    std = np.asarray(_IMAGE_STD).reshape(1, 1, -1)
    return (images - mean) / std


def pre_process_image(
    image: NDArray,
    *,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
) -> NDArray:
    """Pre-process image.

    Args:
        image: Image array of shape (height, width, channels)
        image_height: Target height
        image_width: Target width

    Returns:
        Processed image array
    """
    # all inputs are expected to have been jpeg compressed.
    # TODO(eyvinec): we should remove tf dependency.
    image = jnp.asarray(
        tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)
    )
    image = jax.image.resize(
        image,
        shape=(image_height, image_width, 3),
        method="bilinear",
        antialias=True,
    )
    image = normalize_images(image)
    image = jnp.clip(image, -1, 1)
    return image


def patchify_images(
    images: NDArray,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    padding: str = "VALID",
) -> NDArray:
    """Convert images into patches.
    
    Args:
        images: Image array of shape (batch_size, height, width, channels)
        patch_size: Size of patches
        padding: Padding mode for patching

    Returns:
        Array of patches
    """
    batch_size, height, width, channels = images.shape
    
    # Extract patches
    patches = jax.lax.conv_general_dilated(
        images,
        jnp.eye(patch_size * patch_size * channels).reshape(
            patch_size, patch_size, channels, -1
        ),
        window_strides=(patch_size, patch_size),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Reshape to (batch_size, num_patches, patch_dim)
    num_patches = (height // patch_size) * (width // patch_size)
    patch_dim = patch_size * patch_size * channels
    patches = patches.reshape(batch_size, num_patches, patch_dim)
    
    return patches


def load_image_files(
    img_paths: Sequence[Sequence[str | None]],
    patch_size: int = _DEFAULT_PATCH_SIZE,
) -> Optional[NDArray]:
    """Load and process image files.

    Args:
        img_paths: Sequence of image file paths
        patch_size: Size of patches

    Returns:
        Array of processed image patches
    """
    if len(img_paths) == 1 and len(img_paths[0]) == 1 and img_paths[0][0] is None:
        return None
        
    patches = []
    for imgs_path in img_paths:
        tmp = []
        for img_path in imgs_path:
            if img_path is None:
                raise ValueError(
                    "some img_paths are None and some are not. we only support all None"
                    " or all not None for now."
                )
            with epath.Path(img_path).open("rb") as f:
                img = pre_process_image(np.array(Image.open(f).convert("RGB")))
            tmp.append(patchify_images(img[None, ...], patch_size))
        patches.append(jnp.asarray(tmp))
    patches = jnp.asarray(patches)
    return patches

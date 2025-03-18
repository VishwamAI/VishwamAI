"""Image processing utilities for multimodal inputs."""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
import numpy as np

# Standard normalization constants
IMAGE_MEAN = (127.5, 127.5, 127.5)
IMAGE_STD = (127.5, 127.5, 127.5)

def normalize_image(
    image: jnp.ndarray,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD
) -> jnp.ndarray:
    """Normalize image with mean and standard deviation."""
    mean = jnp.array(mean).reshape(1, 1, -1)
    std = jnp.array(std).reshape(1, 1, -1)
    return (image - mean) / std

def resize_image(
    image: jnp.ndarray,
    height: int,
    width: int,
    method: str = 'bilinear'
) -> jnp.ndarray:
    """Resize image to target dimensions."""
    return jax.image.resize(
        image,
        shape=(height, width, image.shape[-1]),
        method=method,
        antialias=True
    )

def create_patches(
    image: jnp.ndarray,
    patch_size: int,
    stride: Optional[int] = None
) -> jnp.ndarray:
    """Convert image into patches.
    
    Args:
        image: Image tensor of shape [H, W, C]
        patch_size: Size of patches to extract
        stride: Stride for patch extraction, defaults to patch_size
        
    Returns:
        Patches tensor of shape [N, P*P*C] where N is number of patches
    """
    if stride is None:
        stride = patch_size
        
    height, width, channels = image.shape
    patches = jax.lax.conv_general_dilated(
        image[None, ...],  # Add batch dimension
        jnp.eye(patch_size * patch_size * channels).reshape(
            patch_size, patch_size, channels, -1
        ),
        window_strides=[stride, stride],
        padding='VALID',
        dimension_numbers=('NHWC', 'HWCO', 'NHWC')
    )
    
    # Reshape to [num_patches, patch_dim]
    patches = patches[0]  # Remove batch dimension
    num_patches = patches.shape[0] * patches.shape[1]
    patch_dim = patch_size * patch_size * channels
    patches = patches.reshape(num_patches, patch_dim)
    
    return patches

class ImageProcessor:
    """Processor for preparing images for the vision encoder."""
    
    def __init__(
        self,
        image_size: int = 896,
        patch_size: int = 14,
        mean: Tuple[float, float, float] = IMAGE_MEAN,
        std: Tuple[float, float, float] = IMAGE_STD
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        
    def __call__(
        self,
        images: Union[jnp.ndarray, np.ndarray],
        return_tensors: bool = True
    ) -> jnp.ndarray:
        """Process images for model input.
        
        Args:
            images: Single image or batch of images
            return_tensors: Whether to return jax tensors
            
        Returns:
            Processed image patches
        """
        # Handle single image vs batch
        if len(images.shape) == 3:
            images = images[None]
            
        if not isinstance(images, jnp.ndarray):
            images = jnp.array(images)
            
        processed_images = []
        for image in images:
            # Resize if needed
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                image = resize_image(image, self.image_size, self.image_size)
                
            # Normalize
            image = normalize_image(image, self.mean, self.std)
            
            # Create patches
            patches = create_patches(image, self.patch_size)
            processed_images.append(patches)
            
        # Stack batch
        processed_images = jnp.stack(processed_images)
        
        return processed_images if return_tensors else np.array(processed_images)

    def prepare_images(
        self,
        pixel_values: Union[jnp.ndarray, np.ndarray],
        return_tensors: bool = True
    ) -> dict:
        """Prepare images with additional metadata.
        
        Args:
            pixel_values: Image pixel values
            return_tensors: Whether to return jax tensors
            
        Returns:
            Dict containing processed images and metadata
        """
        processed_images = self(pixel_values, return_tensors=return_tensors)
        
        num_patches = processed_images.shape[1]
        patch_dim = processed_images.shape[2]
        
        return {
            'pixel_values': processed_images,
            'num_patches': num_patches,
            'patch_dim': patch_dim,
            'image_size': (self.image_size, self.image_size),
            'patch_size': self.patch_size
        }
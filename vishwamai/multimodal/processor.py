"""Image and batch processing utilities for multimodal inputs."""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union, Tuple
import albumentations as A
from albumentations.core.composition import Compose
from audio_processor import AudioProcessor  # Assuming this is in a separate file

class ImageProcessor:
    """Process images for vision models."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    ):
        """Initialize image processor.

        Args:
            image_size: Target image size.
            mean: RGB mean values for normalization.
            std: RGB standard deviation values for normalization.
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.transform = Compose([
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std)
        ])

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        return_tensors: bool = True
    ) -> Union[np.ndarray, jnp.ndarray]:
        """Process images.

        Args:
            images: Single image or list of images.
            return_tensors: Whether to return JAX tensors.

        Returns:
            Processed image(s) as NumPy array or JAX tensor.
        """
        if not isinstance(images, list):
            images = [images]

        processed = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[-1] == 1:
                image = np.concatenate([image] * 3, axis=-1)

            # Apply transforms
            transformed = self.transform(image=image)['image']
            processed.append(transformed)

        processed = np.stack(processed)
        return jnp.array(processed) if return_tensors else processed

class MultimodalBatchProcessor:
    """Process multimodal batches including images, text, and audio."""

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[Any] = None,  # Tokenizer for text (e.g., from transformers)
        image_size: int = 224,
        audio_config: Optional[Dict[str, Any]] = None,
        max_text_length: int = 512,
        **kwargs
    ):
        """Initialize batch processor.

        Args:
            image_processor: Optional image processor instance.
            audio_processor: Optional audio processor instance.
            tokenizer: Optional tokenizer for text processing.
            image_size: Default image size if no image_processor provided.
            audio_config: Configuration for AudioProcessor if not provided.
            max_text_length: Maximum token length for text inputs.
            **kwargs: Additional processor configs.
        """
        self.image_processor = image_processor or ImageProcessor(image_size=image_size, **kwargs)
        self.audio_processor = audio_processor or AudioProcessor(**(audio_config or {}))
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for text processing (e.g., from transformers).")

    def __call__(
        self,
        images: Optional[List[Union[Image.Image, np.ndarray]]] = None,
        texts: Optional[List[str]] = None,
        audios: Optional[List[Union[np.ndarray, jnp.ndarray]]] = None,
        audio_sample_rates: Optional[List[int]] = None,
        return_tensors: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        """Process multimodal inputs.

        Args:
            images: Optional list of images (PIL Image or NumPy array).
            texts: Optional list of text inputs.
            audios: Optional list of audio waveforms (NumPy or JAX arrays).
            audio_sample_rates: Optional list of sample rates for each audio.
            return_tensors: Whether to return JAX tensors.
            return_dict: Whether to return a dictionary or tuple.
            **kwargs: Additional processing options.

        Returns:
            Dictionary of processed inputs or tuple of tensors:
                - pixel_values: Processed images [batch, channels, height, width]
                - input_ids: Tokenized text [batch, sequence_length]
                - audio_features: Processed audio [batch, frames, n_mels]
                - Additional metadata (e.g., masks, dimensions)
        """
        outputs = {}

        # Process images
        if images is not None:
            pixel_values = self.image_processor(
                images,
                return_tensors=return_tensors
            )
            if return_dict:
                outputs.update({
                    "pixel_values": pixel_values,
                    "pixel_mask": jnp.ones(
                        (len(images), self.image_processor.image_size, self.image_processor.image_size),
                        dtype=jnp.int32 if return_tensors else np.int32
                    ),
                    "height": self.image_processor.image_size,
                    "width": self.image_processor.image_size
                })
            else:
                outputs["pixel_values"] = pixel_values

        # Process text
        if texts is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for text processing.")
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="jax" if return_tensors else "np"
            )
            if return_dict:
                outputs.update({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"]
                })
            else:
                outputs["input_ids"] = tokenized["input_ids"]

        # Process audio
        if audios is not None:
            if audio_sample_rates is None:
                audio_sample_rates = [self.audio_processor.sample_rate] * len(audios)
            elif len(audio_sample_rates) != len(audios):
                raise ValueError("audio_sample_rates must match the number of audios.")

            audio_features = []
            for audio, sample_rate in zip(audios, audio_sample_rates):
                processed = self.audio_processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors=return_tensors
                )
                audio_features.append(processed["input_features"][0])  # Remove batch dim from each

            audio_features = jnp.stack(audio_features) if return_tensors else np.stack(audio_features)
            if return_dict:
                outputs.update({
                    "audio_features": audio_features,
                    "audio_mask": jnp.ones(
                        (len(audios), audio_features.shape[1]),
                        dtype=jnp.int32 if return_tensors else np.int32
                    ),
                    "num_frames": audio_features.shape[1],
                    "num_mel_bins": audio_features.shape[2]
                })
            else:
                outputs["audio_features"] = audio_features

        if not return_dict:
            # Return tuple of main features in order: images, text, audio
            return tuple(
                outputs.get(key) for key in ["pixel_values", "input_ids", "audio_features"]
                if key in outputs
            )

        return outputs

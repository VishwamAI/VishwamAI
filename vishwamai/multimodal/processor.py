"""Image and batch processing utilities for multimodal inputs."""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union, Tuple
import albumentations as A
from albumentations.core.composition import Compose
from vishwamai.multimodal.audio_processor import AudioProcessor

class ImageProcessor:
    """Process images for vision models."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        advanced_augmentations: bool = False
    ):
        """Initialize image processor.

        Args:
            image_size: Target image size.
            mean: RGB mean values for normalization.
            std: RGB standard deviation values for normalization.
            advanced_augmentations: Enable advanced data augmentation
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std

        if advanced_augmentations:
            self.transform = Compose([
                A.SmallestMaxSize(max_size=image_size),
                A.RandomResizedCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.OneOf([
                    A.GaussianBlur(p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.3),
                A.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = Compose([
                A.SmallestMaxSize(max_size=image_size),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std)
            ])

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        return_tensors: bool = True,
        memory_efficient: bool = True
    ) -> Union[np.ndarray, jnp.ndarray]:
        """Process images with memory efficiency.

        Args:
            images: Single image or list of images.
            return_tensors: Whether to return JAX tensors.
            memory_efficient: Use memory efficient processing

        Returns:
            Processed image(s) as NumPy array or JAX tensor.
        """
        if not isinstance(images, list):
            images = [images]

        processed = []
        for image in images:
            if isinstance(image, Image.Image):
                if memory_efficient:
                    # Convert to numpy array with uint8 to save memory
                    image = np.array(image, dtype=np.uint8)
                else:
                    image = np.array(image)

            # Convert grayscale to RGB efficiently
            if len(image.shape) == 2:
                image = image[..., None]
                image = np.repeat(image, 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)

            # Apply transforms with error handling
            try:
                transformed = self.transform(image=image)['image']
                processed.append(transformed)
            except Exception as e:
                print(f"Warning: Error processing image: {e}")
                # Return zeros as fallback
                processed.append(np.zeros((self.image_size, self.image_size, 3)))

            # Clear memory
            if memory_efficient:
                del image
                import gc
                gc.collect()

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

class MultimodalProcessor:
    """Processor for preparing multimodal inputs for VishwamAI."""
    
    def __init__(
        self,
        tokenizer: Any,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        sonar_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize processor.
        
        Args:
            tokenizer: The tokenizer for text processing
            image_processor: Optional custom image processor
            audio_processor: Optional custom audio processor
            sonar_config: Optional SONAR configuration for multilingual
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor or ImageProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        
        if sonar_config:
            from vishwamai.multimodal.sonar import SonarEncoder
            self.sonar_encoder = SonarEncoder(sonar_config)
            self._has_sonar = True
        else:
            self._has_sonar = False
            
    def prepare_vision_inputs(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """Prepare vision inputs.
        
        Args:
            images: Single image or batch of images
            return_tensors: Whether to return JAX tensors
            
        Returns:
            Dict with processed image inputs
        """
        return self.image_processor.prepare_images(
            images,
            return_tensors=return_tensors
        )
        
    def prepare_audio_inputs(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """Prepare audio inputs.
        
        Args:
            audio: Single audio or batch of audio waveforms
            sampling_rate: Sample rate(s) of audio
            return_tensors: Whether to return JAX tensors
            
        Returns:
            Dict with processed audio inputs
        """
        return self.audio_processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors
        )
        
    def prepare_text_inputs(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """Prepare text inputs.
        
        Args:
            texts: Text or batch of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Max sequence length
            return_tensors: Whether to return JAX tensors
            
        Returns:
            Dict with processed text inputs
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="jax" if return_tensors else None
        )
        
    def prepare_multilingual_inputs(
        self,
        text: Optional[Union[str, List[str]]] = None,
        speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """Prepare multilingual inputs using SONAR.
        
        Args:
            text: Optional text input
            speech: Optional speech input
            src_lang: Source language code
            tgt_lang: Target language code
            return_tensors: Whether to return JAX tensors
            
        Returns:
            Dict with processed multilingual inputs
        """
        if not self._has_sonar:
            raise ValueError("SONAR configuration required for multilingual processing")
            
        inputs = {}
        if text is not None:
            text_inputs = self.sonar_encoder.embed_text(
                text if isinstance(text, list) else [text],
                src_lang=src_lang
            )
            inputs.update(text_inputs)
            
        if speech is not None:
            speech_inputs = self.sonar_encoder.embed_speech(
                speech if isinstance(speech, list) else [speech],
                src_lang=src_lang
            )
            inputs.update(speech_inputs)
            
        if tgt_lang is not None:
            inputs["target_lang"] = tgt_lang
            
        return inputs

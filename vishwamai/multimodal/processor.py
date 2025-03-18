"""Unified processor for multimodal inputs."""

from typing import Optional, Union, Dict, Any, List
import jax.numpy as jnp
import numpy as np
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor

class MultimodalProcessor:
    """Unified processor for handling multiple input modalities."""
    
    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048
    ):
        self.image_processor = image_processor or ImageProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        audio: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process multimodal inputs.
        
        Args:
            text: Optional text input
            images: Optional image input
            audio: Optional audio input
            sampling_rate: Optional audio sampling rate
            return_tensors: Whether to return jax tensors
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing processed features for each modality
        """
        outputs = {}
        
        # Process text if provided
        if text is not None and self.tokenizer is not None:
            text_features = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="jax" if return_tensors else "np"
            )
            outputs.update(text_features)
        
        # Process images if provided
        if images is not None:
            image_features = self.image_processor(
                images,
                return_tensors=return_tensors
            )
            outputs.update({
                f"image_{k}": v 
                for k, v in image_features.items()
            })
        
        # Process audio if provided
        if audio is not None:
            audio_features = self.audio_processor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors=return_tensors
            )
            outputs.update({
                f"audio_{k}": v 
                for k, v in audio_features.items()
            })
        
        return outputs
    
    def batch_process(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Process a batch of multimodal inputs.
        
        Args:
            batch: Dictionary containing text, image, and/or audio inputs
            **kwargs: Additional arguments passed to __call__
            
        Returns:
            Processed batch
        """
        return self(
            text=batch.get('text'),
            images=batch.get('images'),
            audio=batch.get('audio'),
            sampling_rate=batch.get('sampling_rate'),
            **kwargs
        )

    def prepare_inputs(
        self,
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for model inference.
        
        Args:
            text: Optional text input
            image: Optional single image
            audio: Optional single audio
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary with processed inputs ready for model
        """
        inputs = {}
        
        # Process each modality if provided
        if text is not None:
            text_inputs = self(text=text, **kwargs)
            inputs.update(text_inputs)
            
        if image is not None:
            image_inputs = self(images=image, **kwargs)
            inputs.update(image_inputs)
            
        if audio is not None:
            audio_inputs = self(audio=audio, **kwargs)
            inputs.update(audio_inputs)
            
        return inputs
    
    def prepare_vision_inputs(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare vision inputs specifically.
        
        Args:
            images: Image or list of images
            **kwargs: Additional arguments for image processor
            
        Returns:
            Processed vision inputs
        """
        return self(images=images, **kwargs)
    
    def prepare_audio_inputs(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare audio inputs specifically.
        
        Args:
            audio: Audio or list of audio inputs
            sampling_rate: Optional sampling rate
            **kwargs: Additional arguments for audio processor
            
        Returns:
            Processed audio inputs
        """
        return self(
            audio=audio,
            sampling_rate=sampling_rate,
            **kwargs
        )
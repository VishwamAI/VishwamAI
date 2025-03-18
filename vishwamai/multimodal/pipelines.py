"""Pipelines for multimodal tasks."""

from typing import Optional, List, Dict, Any, Union
import jax.numpy as jnp
import numpy as np
from .processor import MultimodalProcessor
from .config import MultimodalConfig, create_default_multimodal_config
from ..model import VishwamAI
from ..tokenizer import VishwamAITokenizer

class ImageCaptioningPipeline:
    """Pipeline for generating image captions."""
    
    def __init__(
        self,
        model: VishwamAI,
        processor: MultimodalProcessor,
        config: Optional[MultimodalConfig] = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.config = config or create_default_multimodal_config()
    
    def __call__(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate captions for images."""
        # Process images
        inputs = self.processor.prepare_vision_inputs(images)
        
        # Add caption prompt
        prompt = "Describe this image in detail:"
        text_inputs = self.processor.tokenizer(
            prompt,
            return_tensors="jax",
            add_special_tokens=False
        )
        inputs.update(text_inputs)
        
        # Generate caption
        outputs = self.model.generate_chat(
            [{"role": "user", "content": prompt}],
            image_input=inputs["image_pixel_values"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        return outputs

class VisualQuestionAnswering:
    """Pipeline for visual question answering."""
    
    def __init__(
        self,
        model: VishwamAI,
        processor: MultimodalProcessor,
        config: Optional[MultimodalConfig] = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.config = config or create_default_multimodal_config()
    
    def __call__(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        questions: Union[str, List[str]],
        max_new_tokens: int = 50,
        **kwargs
    ) -> Union[str, List[str]]:
        """Answer questions about images."""
        # Process inputs
        vision_inputs = self.processor.prepare_vision_inputs(images)
        
        # Format each question
        if isinstance(questions, str):
            questions = [questions]
            
        answers = []
        for question in questions:
            # Create chat format
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about images."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            # Generate answer
            answer = self.model.generate_chat(
                messages,
                image_input=vision_inputs["image_pixel_values"],
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            answers.append(answer)
            
        return answers[0] if len(answers) == 1 else answers

class AudioCaptioningPipeline:
    """Pipeline for generating audio descriptions."""
    
    def __init__(
        self,
        model: VishwamAI,
        processor: MultimodalProcessor,
        config: Optional[MultimodalConfig] = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.config = config or create_default_multimodal_config()
    
    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: int = 16000,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate descriptions for audio clips."""
        # Process audio
        inputs = self.processor.prepare_audio_inputs(
            audio,
            sampling_rate=sampling_rate
        )
        
        # Add description prompt
        prompt = "Describe what you hear in this audio:"
        messages = [{"role": "user", "content": prompt}]
        
        # Generate description
        outputs = self.model.generate_chat(
            messages,
            audio_input=inputs["audio_input_features"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return outputs

class MultimodalChatPipeline:
    """Pipeline for multimodal chat interactions."""
    
    def __init__(
        self,
        model: VishwamAI,
        processor: MultimodalProcessor,
        config: Optional[MultimodalConfig] = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.config = config or create_default_multimodal_config()
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        audio: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        sampling_rate: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Process multimodal chat conversation."""
        # Process any provided modalities
        inputs = {}
        
        if images is not None:
            vision_inputs = self.processor.prepare_vision_inputs(images)
            inputs["image_input"] = vision_inputs["image_pixel_values"]
            
        if audio is not None:
            audio_inputs = self.processor.prepare_audio_inputs(
                audio,
                sampling_rate=sampling_rate
            )
            inputs["audio_input"] = audio_inputs["audio_input_features"]
        
        # Generate response
        response = self.model.generate_chat(
            messages,
            image_input=inputs.get("image_input"),
            audio_input=inputs.get("audio_input"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response

class MultilingualPipeline:
    """Pipeline for multilingual text and speech processing using SONAR."""
    
    def __init__(
        self,
        model: VishwamAI,
        processor: MultimodalProcessor,
        config: Optional[MultimodalConfig] = None,
        **kwargs
    ):
        self.model = model
        self.processor = processor
        self.config = config or create_default_multimodal_config(include_sonar=True)
        
        if not self.config.sonar_config:
            raise ValueError("SONAR configuration is required for MultilingualPipeline")
    
    def translate(
        self,
        text: Optional[Union[str, List[str]]] = None,
        speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        src_lang: str = "eng",
        tgt_lang: str = "fra",
        **kwargs
    ) -> Union[str, List[str]]:
        """Translate text or speech between languages.
        
        Args:
            text: Optional text input in source language
            speech: Optional speech input in source language
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional generation arguments
            
        Returns:
            Translated text in target language
        """
        # Process inputs
        inputs = self.processor.prepare_multilingual_inputs(
            text=text,
            speech=speech,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        # Generate translation
        outputs = self.model.generate_chat(
            [{"role": "user", "content": f"Translate to {tgt_lang}:"}],
            **inputs,
            **kwargs
        )
        
        return outputs
    
    def embed(
        self,
        text: Optional[Union[str, List[str]]] = None,
        speech: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        src_lang: Optional[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Get SONAR embeddings for text or speech input.
        
        Args:
            text: Optional text input
            speech: Optional speech input
            src_lang: Source language code
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary with embeddings
        """
        return self.processor.prepare_multilingual_inputs(
            text=text,
            speech=speech,
            src_lang=src_lang,
            **kwargs
        )
    
    def compute_similarity(
        self,
        texts1: List[str],
        texts2: List[str],
        lang1: str = "eng",
        lang2: str = "eng"
    ) -> np.ndarray:
        """Compute similarity scores between text pairs.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts 
            lang1: Language code for first texts
            lang2: Language code for second texts
            
        Returns:
            Array of similarity scores
        """
        # Get embeddings
        emb1 = self.embed(text=texts1, src_lang=lang1)["text_embeddings"]
        emb2 = self.embed(text=texts2, src_lang=lang2)["text_embeddings"]
        
        # Compute similarity
        return self.processor.sonar_encoder.get_similarity_score(emb1, emb2)
    
    def batch_translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
        **kwargs
    ) -> List[str]:
        """Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            List of translated texts
        """
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = self.translate(
                text=batch,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                **kwargs
            )
            translations.extend(batch_translations)
        return translations

def load_multimodal_pipeline(
    model_name_or_path: str,
    task: str = "chat",
    **kwargs
) -> Union[MultimodalChatPipeline, ImageCaptioningPipeline, 
           VisualQuestionAnswering, AudioCaptioningPipeline,
           MultilingualPipeline]:
    """Load a multimodal pipeline.
    
    Args:
        model_name_or_path: Path to pretrained model
        task: Pipeline task (chat, image-captioning, vqa, audio-captioning,
              multilingual)
        **kwargs: Additional arguments
        
    Returns:
        Appropriate pipeline for the task
    """
    # Load model and processor
    model = VishwamAI.from_pretrained(model_name_or_path)
    tokenizer = VishwamAITokenizer.from_pretrained(model_name_or_path)
    processor = MultimodalProcessor(tokenizer=tokenizer)
    
    # Create pipeline based on task
    if task == "chat":
        return MultimodalChatPipeline(model, processor, **kwargs)
    elif task == "image-captioning":
        return ImageCaptioningPipeline(model, processor, **kwargs)
    elif task == "vqa":
        return VisualQuestionAnswering(model, processor, **kwargs)
    elif task == "audio-captioning":
        return AudioCaptioningPipeline(model, processor, **kwargs)
    elif task == "multilingual":
        config = create_default_multimodal_config(include_sonar=True)
        return MultilingualPipeline(model, processor, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
        
__all__ = [
    "ImageCaptioningPipeline",
    "VisualQuestionAnswering",
    "AudioCaptioningPipeline",
    "MultimodalChatPipeline",
    "MultilingualPipeline",
    "load_multimodal_pipeline"
]
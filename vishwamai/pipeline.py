"""
Text generation and model inference pipeline for VishwamAI.

Provides high-level interfaces for text generation, chat completion,
and multimodal inference tasks.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Dict, Any, Optional, List, Union, Callable
import chex
import time

from .model import VishwamAIModel, ModelConfig
from .multimodal import MultimodalProcessor


class TextGenerator:
    """High-level text generation interface."""
    
    def __init__(
        self,
        model: VishwamAIModel,
        params: Dict[str, Any],
        tokenizer: Any = None,
        config: Optional[ModelConfig] = None
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.config = config
        
        # JIT compile generation function
        self.generate_fn = jax.jit(self._generate_step)
    
    def _generate_step(
        self,
        input_ids: chex.Array,
        cache: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> chex.Array:
        """Single generation step."""
        
        # Forward pass
        logits = self.model.apply(
            self.params,
            input_ids,
            training=False
        )
        
        # Get logits for next token
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_indices = jnp.argsort(next_token_logits, axis=-1)[:, -top_k:]
            min_top_k = jnp.take_along_axis(
                next_token_logits, top_k_indices[:, 0:1], axis=-1
            )
            next_token_logits = jnp.where(
                next_token_logits < min_top_k,
                -jnp.inf,
                next_token_logits
            )
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits = jnp.sort(next_token_logits, axis=-1)[:, ::-1]
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff index
            cutoff_mask = cumulative_probs <= top_p
            cutoff_indices = jnp.sum(cutoff_mask, axis=-1, keepdims=True)
            
            # Apply cutoff
            sorted_indices = jnp.argsort(next_token_logits, axis=-1)[:, ::-1]
            cutoff_logits = jnp.take_along_axis(sorted_logits, cutoff_indices, axis=-1)
            
            next_token_logits = jnp.where(
                next_token_logits < cutoff_logits,
                -jnp.inf,
                next_token_logits
            )
        
        # Sample next token
        if rng_key is not None:
            next_token = jax.random.categorical(rng_key, next_token_logits)
        else:
            # Greedy sampling
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        return next_token
    
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_length: int = 512,
        min_length: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Union[str, List[int]]:
        """Generate text from prompt."""
        
        # Setup RNG
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            rng_key = jax.random.PRNGKey(int(time.time()))
        
        # Tokenize prompt if string
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            input_ids = self.tokenizer.encode(prompt)
            return_string = True
        else:
            input_ids = prompt
            return_string = False
        
        # Convert to JAX array
        input_ids = jnp.array([input_ids], dtype=jnp.int32)
        batch_size, prompt_length = input_ids.shape
        
        # Generation loop
        generated = input_ids
        
        for step in range(max_length - prompt_length):
            # Generate next token
            rng_key, step_key = jax.random.split(rng_key)
            next_token = self.generate_fn(
                generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng_key=step_key
            )
            
            # Add to sequence
            generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)
            
            # Check for EOS
            if eos_token_id is not None and jnp.all(next_token == eos_token_id):
                break
            
            # Check minimum length
            if step + prompt_length >= min_length and eos_token_id is not None:
                if jnp.any(next_token == eos_token_id):
                    break
        
        # Extract generated tokens (remove prompt)
        generated_tokens = generated[0, prompt_length:].tolist()
        
        # Convert back to string if needed
        if return_string:
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string output")
            return self.tokenizer.decode(generated_tokens)
        else:
            return generated_tokens
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Chat completion interface."""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for chat interface")
        
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        response = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        return response
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt."""
        
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n"
        
        formatted_prompt += "Assistant: "
        
        return formatted_prompt


class MultimodalGenerator:
    """Generator for multimodal inputs."""
    
    def __init__(
        self,
        model: VishwamAIModel,
        params: Dict[str, Any],
        processor: MultimodalProcessor,
        tokenizer: Any = None
    ):
        self.model = model
        self.params = params
        self.processor = processor
        self.tokenizer = tokenizer
        
        # JIT compile functions
        self.generate_fn = jax.jit(self._generate_step)
    
    def _generate_step(
        self,
        embeddings: chex.Array,
        temperature: float = 1.0,
        top_k: int = 50,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> chex.Array:
        """Generate next token from embeddings."""
        
        # Forward pass
        logits = self.model.apply(
            self.params,
            embeddings,
            training=False,
            method=self.model.forward_embeddings  # Assume we have this method
        )
        
        # Sample next token
        next_token_logits = logits[:, -1, :] / temperature
        
        if top_k > 0:
            top_k_indices = jnp.argsort(next_token_logits, axis=-1)[:, -top_k:]
            min_top_k = jnp.take_along_axis(
                next_token_logits, top_k_indices[:, 0:1], axis=-1
            )
            next_token_logits = jnp.where(
                next_token_logits < min_top_k,
                -jnp.inf,
                next_token_logits
            )
        
        if rng_key is not None:
            next_token = jax.random.categorical(rng_key, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        return next_token
    
    def generate_from_multimodal(
        self,
        text: Optional[str] = None,
        images: Optional[chex.Array] = None,
        audio: Optional[chex.Array] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        seed: Optional[int] = None
    ) -> str:
        """Generate text from multimodal inputs."""
        
        # Setup RNG
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            rng_key = jax.random.PRNGKey(int(time.time()))
        
        # Process inputs
        text_ids = None
        if text is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for text input")
            text_ids = jnp.array([self.tokenizer.encode(text)], dtype=jnp.int32)
        
        # Get multimodal embeddings
        embeddings = self.processor(
            text_ids=text_ids,
            images=images,
            audio=audio,
            training=False
        )
        
        # Generate response tokens
        generated_tokens = []
        current_embeddings = embeddings
        
        for _ in range(max_length):
            rng_key, step_key = jax.random.split(rng_key)
            
            next_token = self.generate_fn(
                current_embeddings,
                temperature=temperature,
                rng_key=step_key
            )
            
            generated_tokens.append(int(next_token[0]))
            
            # Convert token to embedding and append
            if self.tokenizer.eos_token_id and next_token[0] == self.tokenizer.eos_token_id:
                break
            
            # TODO: Implement token-to-embedding conversion for continuation
            # This would require extending the model to handle mixed embeddings/tokens
        
        # Decode tokens to text
        if self.tokenizer is None:
            return " ".join(map(str, generated_tokens))
        else:
            return self.tokenizer.decode(generated_tokens)


def pipeline(
    task: str,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    device_map: str = "auto",
    **kwargs
) -> Union[TextGenerator, MultimodalGenerator]:
    """Create a pipeline for various tasks.
    
    Args:
        task: Task type ("text-generation", "chat", "multimodal-generation")
        model: Model name or path
        model_kwargs: Additional model arguments
        device_map: Device mapping strategy
        **kwargs: Additional pipeline arguments
    
    Returns:
        Appropriate generator instance
    """
    
    if model_kwargs is None:
        model_kwargs = {}
    
    # For now, return a basic text generator
    # In a full implementation, this would load models from HuggingFace or local paths
    
    if task == "text-generation":
        # Create basic configuration
        config = ModelConfig(
            dim=2048,
            depth=24,
            heads=32,
            vocab_size=50304
        )
        
        # Create model (would normally load from checkpoint)
        model_instance = VishwamAIModel(config)
        
        # Initialize dummy parameters (would normally load from checkpoint)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        params = model_instance.init(jax.random.PRNGKey(0), dummy_input)
        
        return TextGenerator(model_instance, params, config=config)
    
    elif task == "multimodal-generation":
        # Create multimodal configuration
        config = ModelConfig(
            dim=2048,
            depth=24,
            heads=32,
            vocab_size=50304,
            enable_multimodal=True
        )
        
        # Create model and processor
        model_instance = VishwamAIModel(config)
        processor = MultimodalProcessor(
            vocab_size=config.vocab_size,
            embed_dim=config.dim,
            vision_config={"image_size": 224, "patch_size": 16},
            audio_config={"n_mels": 80}
        )
        
        # Initialize parameters
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        params = model_instance.init(jax.random.PRNGKey(0), dummy_input)
        
        return MultimodalGenerator(model_instance, params, processor)
    
    else:
        raise ValueError(f"Unsupported task: {task}")


# Utility functions for common inference patterns

def generate_text(
    prompt: str,
    model_path: Optional[str] = None,
    max_length: int = 512,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Quick text generation function."""
    
    generator = pipeline("text-generation", model=model_path)
    return generator.generate(
        prompt,
        max_length=max_length,
        temperature=temperature,
        **kwargs
    )


def chat_completion(
    messages: List[Dict[str, str]],
    model_path: Optional[str] = None,
    **kwargs
) -> str:
    """Quick chat completion function."""
    
    generator = pipeline("text-generation", model=model_path)
    return generator.chat(messages, **kwargs)


def multimodal_completion(
    prompt: str,
    images: Optional[chex.Array] = None,
    audio: Optional[chex.Array] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> str:
    """Quick multimodal completion function."""
    
    generator = pipeline("multimodal-generation", model=model_path)
    return generator.generate_from_multimodal(
        text=prompt,
        images=images,
        audio=audio,
        **kwargs
    )

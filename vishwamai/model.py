import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, List, Dict, Union
from safetensors import safe_open
import os
from huggingface_hub import snapshot_download
from .layers import MLABlock, MoELayer
import json
import jax
class VishwamAI(nn.Module):
    vocab_size: int
    hidden_size: int = 4096  # Scale for 14B params
    num_layers: int = 32  # Based on Phi-4 architecture
    num_heads: int = 32  # Increased attention heads
    num_experts: int = 8  # MoE layer experts
    ffn_dim: int = 16384  # 4x hidden size
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        # Token embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, 
                    features=self.hidden_size)(x)
        
        # Training-specific dropout
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            
        # Process through transformer layers
        for _ in range(self.num_layers):
            # Layer normalization and residual connections
            residual = x
            x = nn.LayerNorm()(x)
            x = MLABlock(num_heads=self.num_heads)(x)
            x = x + residual
            
            # Feed-forward network with MoE
            residual = x
            x = nn.LayerNorm()(x)
            x = MoELayer(num_experts=self.num_experts, 
                        ffn_dim=self.ffn_dim)(x)
            x = x + residual
            
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                
        # Final layer norm
        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size)(x)
        
    def get_distillation_outputs(self, x, temperature=2.0, training=True):
        """Get logits and hidden states for knowledge distillation"""
        logits = self(x, training=training)
        # Scale logits by temperature
        return logits / temperature
        
    def generate_chat(self, messages: List[Dict[str, str]], 
                     max_new_tokens: int = 128,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> str:
        """Generate chat responses in Phi-4 style
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            
        Returns:
            str: Generated response text
        """
        # Format chat context
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n"
                
        formatted_prompt += "Assistant: "
        
        # Convert prompt to token ids
        input_ids = self.tokenizer.encode(formatted_prompt, 
                                        add_special_tokens=True,
                                        return_tensors="jax")
        
        # Generate with nucleus sampling
        output_ids = self.generate(input_ids,
                                 max_length=len(input_ids[0]) + max_new_tokens,
                                 temperature=temperature,
                                 top_p=top_p,
                                 do_sample=True)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = generated_text[len(formatted_prompt):]
        
        return response.strip()
        
    def generate(self, input_ids: jnp.ndarray,
                max_length: int,
                temperature: float = 1.0,
                top_p: float = 0.9,
                do_sample: bool = True):
        """Auto-regressive text generation with nucleus sampling"""
        # Ensure input has correct shape
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            
        batch_size = input_ids.shape[0]
        generated = input_ids
        
        # Generate tokens auto-regressively
        for _ in range(max_length - input_ids.shape[1]):
            # Get logits for next token
            logits = self(generated)[:, -1, :]
            
            if do_sample:
                # Temperature scaling
                logits = logits / temperature
                
                # Nucleus sampling
                sorted_logits, sorted_indices = jax.lax.top_k(logits, k=self.vocab_size)
                cumulative_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.concatenate(
                    [jnp.zeros_like(sorted_indices_to_remove[:,:1]), 
                     sorted_indices_to_remove[:,:-1]], axis=1)
                
                # Sample from the filtered distribution
                filtered_logits = jnp.where(sorted_indices_to_remove, -1e10, sorted_logits)
                next_token = jax.random.categorical(self.make_rng('sampling'), filtered_logits)
                next_token = jnp.take_along_axis(sorted_indices, next_token[:, None], axis=1)
            else:
                # Greedy decoding
                next_token = jnp.argmax(logits, axis=-1)[:, None]
                
            # Append new token
            generated = jnp.concatenate([generated, next_token], axis=1)
            
        return generated

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from HuggingFace Hub or local directory"""
        # Download model if not local
        if not os.path.isdir(model_name_or_path):
            model_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
                local_files_only=False
            )
        else:
            model_path = model_name_or_path
            
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Create model instance
        model = cls(
            vocab_size=config["vocab_size"],
            hidden_size=config.get("hidden_size", cls.hidden_size),
            num_layers=config.get("num_layers", cls.num_layers),
            num_heads=config.get("num_heads", cls.num_heads),
            num_experts=config.get("num_experts", cls.num_experts),
            ffn_dim=config.get("ffn_dim", cls.ffn_dim),
            dropout_rate=config.get("dropout_rate", cls.dropout_rate),
            **kwargs
        )
        
        # Initialize with dummy input to create parameter shapes
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        init_variables = model.init(key, dummy_input)
        
        # Load weights
        params = {}
        weight_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        for weight_file in weight_files:
            file_path = os.path.join(model_path, weight_file)
            with safe_open(file_path, framework="flax") as f:
                for param_name in f.keys():
                    params[param_name] = jax.device_put(f.get_tensor(param_name))
                    
        # Assign loaded parameters
        model.params = params
        return model

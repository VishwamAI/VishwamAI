"""
Hugging Face integration for VishwamAI models.

This module provides Hugging Face-compatible wrappers for VishwamAI models,
enabling seamless integration with the transformers ecosystem, including
tokenizers, datasets, and training utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoTokenizer,
    PretrainedConfig,
    modeling_outputs
)
from transformers.modeling_utils import ModuleUtilsMixin
from flax.core import freeze, unfreeze
import chex

from .model import VishwamAIModel, ModelConfig
from .pipeline import pipeline


class VishwamAIConfig(PretrainedConfig):
    """
    Configuration class for VishwamAI models compatible with Hugging Face.
    
    This class stores the configuration of a VishwamAI model and provides
    compatibility with Hugging Face's configuration system.
    """
    
    model_type = "vishwamai"
    
    def __init__(
        self,
        vocab_size: int = 50304,
        dim: int = 2048,
        depth: int = 24,
        heads: int = 32,
        head_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout_rate: float = 0.1,
        use_flash_attention: bool = True,
        use_grouped_query_attention: bool = True,
        gqa_groups: int = 8,
        use_rmsnorm: bool = True,
        use_rotary_embeddings: bool = True,
        vision_patch_size: int = 16,
        vision_dim: int = 1024,
        audio_dim: int = 512,
        enable_multimodal: bool = True,
        expert_count: int = 8,
        expert_capacity: int = 4,
        use_moe: bool = False,
        use_bfloat16: bool = True,
        gradient_checkpointing: bool = True,
        kernel_fusion: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim or dim // heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.use_flash_attention = use_flash_attention
        self.use_grouped_query_attention = use_grouped_query_attention
        self.gqa_groups = gqa_groups
        self.use_rmsnorm = use_rmsnorm
        self.use_rotary_embeddings = use_rotary_embeddings
        self.vision_patch_size = vision_patch_size
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.enable_multimodal = enable_multimodal
        self.expert_count = expert_count
        self.expert_capacity = expert_capacity
        self.use_moe = use_moe
        self.use_bfloat16 = use_bfloat16
        self.gradient_checkpointing = gradient_checkpointing
        self.kernel_fusion = kernel_fusion
        
        super().__init__(**kwargs)
    
    def to_model_config(self) -> ModelConfig:
        """Convert to VishwamAI ModelConfig."""
        return ModelConfig(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            dropout_rate=self.dropout_rate,
            use_flash_attention=self.use_flash_attention,
            use_grouped_query_attention=self.use_grouped_query_attention,
            gqa_groups=self.gqa_groups,
            use_rmsnorm=self.use_rmsnorm,
            use_rotary_embeddings=self.use_rotary_embeddings,
            vision_patch_size=self.vision_patch_size,
            vision_dim=self.vision_dim,
            audio_dim=self.audio_dim,
            enable_multimodal=self.enable_multimodal,
            expert_count=self.expert_count,
            expert_capacity=self.expert_capacity,
            use_moe=self.use_moe,
            use_bfloat16=self.use_bfloat16,
            gradient_checkpointing=self.gradient_checkpointing,
            kernel_fusion=self.kernel_fusion
        )


class VishwamAIForCausalLM(PreTrainedModel):
    """
    Hugging Face compatible wrapper for VishwamAI models.
    
    This class provides a PyTorch-compatible interface for VishwamAI models,
    enabling use with Hugging Face trainers, pipelines, and other ecosystem tools.
    """
    
    config_class = VishwamAIConfig
    base_model_prefix = "vishwamai"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    
    def __init__(self, config: VishwamAIConfig):
        super().__init__(config)
        
        # Convert to VishwamAI config
        self.model_config = config.to_model_config()
        
        # Initialize JAX model
        self.vishwamai_model = VishwamAIModel(self.model_config)
        
        # Initialize parameters
        self.rng_key = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        self.params = self.vishwamai_model.init(
            self.rng_key, dummy_input, training=False
        )
        
        # Parameter count for compatibility
        self.num_parameters = sum(
            x.size for x in jax.tree_util.tree_leaves(self.params)
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """Forward pass compatible with Hugging Face interface."""
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        # Convert PyTorch tensors to JAX arrays
        input_ids_jax = jnp.array(input_ids.cpu().numpy())
        attention_mask_jax = None
        if attention_mask is not None:
            attention_mask_jax = jnp.array(attention_mask.cpu().numpy())
        
        # Forward pass through VishwamAI model
        logits_jax = self.vishwamai_model.apply(
            self.params,
            input_ids_jax,
            attention_mask=attention_mask_jax,
            training=self.training
        )
        
        # Convert back to PyTorch
        logits = torch.from_numpy(np.array(logits_jax))
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using VishwamAI model."""
        
        # Convert to JAX
        input_ids_jax = jnp.array(input_ids.cpu().numpy())
        
        # Generate
        generated_jax = self.vishwamai_model.generate(
            self.params,
            input_ids_jax,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=self.rng_key
        )
        
        # Convert back to PyTorch
        return torch.from_numpy(np.array(generated_jax))
    
    def get_input_embeddings(self):
        """Get input embeddings layer."""
        # Return a dummy embedding layer for compatibility
        return torch.nn.Embedding(self.config.vocab_size, self.config.dim)
    
    def set_input_embeddings(self, embeddings):
        """Set input embeddings layer."""
        # JAX models don't support direct embedding replacement
        pass
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        # Would need to modify JAX parameters
        pass
    
    def get_output_embeddings(self):
        """Get output embeddings layer."""
        return None
    
    def set_output_embeddings(self, embeddings):
        """Set output embeddings layer."""
        pass


class VishwamAITokenizer:
    """
    Simple tokenizer wrapper for VishwamAI models.
    
    This provides basic tokenization functionality and can be extended
    to use more sophisticated tokenizers like SentencePiece or BPE.
    """
    
    def __init__(self, vocab_size: int = 50304):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        
        # Create basic vocab
        self.vocab = {f"<token_{i}>": i for i in range(vocab_size)}
        self.vocab.update({
            "<pad>": self.pad_token_id,
            "<eos>": self.eos_token_id,
            "<bos>": self.bos_token_id,
            "<unk>": self.unk_token_id,
        })
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        # Simple character-level tokenization for demo
        # In practice, use a proper tokenizer
        tokens = [self.bos_token_id]
        
        for char in text:
            token_name = f"<char_{ord(char)}>"
            if token_name in self.vocab:
                tokens.append(self.vocab[token_name])
            else:
                tokens.append(self.unk_token_id)
        
        tokens.append(self.eos_token_id)
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token.startswith("<char_"):
                    # Extract character from token name
                    char_code = int(token[6:-1])
                    tokens.append(chr(char_code))
                elif token not in ["<pad>", "<eos>", "<bos>"]:
                    tokens.append(token)
        
        return "".join(tokens)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with HuggingFace-compatible interface."""
        
        if isinstance(text, str):
            text = [text]
        
        all_input_ids = []
        all_attention_masks = []
        
        for t in text:
            input_ids = self.encode(t, max_length)
            attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }


def load_vishwamai_model(
    model_name_or_path: str,
    config: Optional[VishwamAIConfig] = None,
    **kwargs
) -> VishwamAIForCausalLM:
    """
    Load a VishwamAI model with Hugging Face compatibility.
    
    Args:
        model_name_or_path: Path to model or model identifier
        config: Model configuration
        **kwargs: Additional arguments
    
    Returns:
        VishwamAI model wrapped for Hugging Face compatibility
    """
    
    if config is None:
        # Try to load config from path
        try:
            config = VishwamAIConfig.from_pretrained(model_name_or_path)
        except:
            # Use default config
            config = VishwamAIConfig()
    
    model = VishwamAIForCausalLM(config)
    
    return model


def save_vishwamai_model(
    model: VishwamAIForCausalLM,
    save_directory: str,
    push_to_hub: bool = False,
    **kwargs
):
    """
    Save a VishwamAI model in Hugging Face format.
    
    Args:
        model: VishwamAI model to save
        save_directory: Directory to save model
        push_to_hub: Whether to push to Hugging Face Hub
        **kwargs: Additional arguments
    """
    
    import os
    from pathlib import Path
    
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    model.config.save_pretrained(save_directory)
    
    # Save model parameters (JAX format)
    import pickle
    with open(save_path / "vishwamai_params.pkl", "wb") as f:
        pickle.dump(model.params, f)
    
    # Save model info
    model_info = {
        "model_type": "vishwamai",
        "architecture": "VishwamAIForCausalLM",
        "num_parameters": model.num_parameters,
        "framework": "jax"
    }
    
    import json
    with open(save_path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    if push_to_hub:
        # Push to Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=save_directory,
                repo_id=kwargs.get("repo_id", "VishwamAI/vishwamai-base"),
                **kwargs
            )
        except ImportError:
            print("huggingface_hub not installed. Cannot push to hub.")


# Register the model for auto-loading
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("vishwamai", VishwamAIConfig)
AutoModel.register(VishwamAIConfig, VishwamAIForCausalLM)
AutoModelForCausalLM.register(VishwamAIConfig, VishwamAIForCausalLM)

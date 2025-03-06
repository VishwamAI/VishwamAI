import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
import os
import gc
import json
import logging
import random

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from functools import partial
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
import safetensors.flax as stf

from google.cloud import storage

@dataclass(frozen=True)  # Make it immutable and hashable
class ModelConfig:
    """Enhanced model configuration with TPU optimizations."""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 32
    num_attention_heads: int = 12
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    use_rope: bool = True
    use_alibi: bool = False
    use_gqa: bool = True
    num_key_value_heads: int = 8
    dtype: str = "bfloat16"  # Default to bfloat16 for TPU
    quantization: Optional[str] = None
    use_dualpipe: bool = True
    use_eplb: bool = True
    use_deepgemm: bool = True
    eplb_window_size: int = 100
    eplb_threshold: float = 0.8

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

    @classmethod
    def map_config_params(cls, config_dict: Dict) -> Dict:
        mapped_dict = config_dict.copy()
        if 'attention_dropout' in mapped_dict:
            mapped_dict['attention_dropout_prob'] = mapped_dict.pop('attention_dropout')
        if 'dropout' in mapped_dict:
            mapped_dict['hidden_dropout_prob'] = mapped_dict.pop('dropout')
        if 'hidden_size' not in mapped_dict and 'dim' in mapped_dict:
            mapped_dict['hidden_size'] = mapped_dict.pop('dim')
        if 'num_attention_heads' not in mapped_dict and 'num_heads' in mapped_dict:
            mapped_dict['num_attention_heads'] = mapped_dict.pop('num_heads')
        if 'num_layers' not in mapped_dict and 'n_layers' in mapped_dict:
            mapped_dict['num_layers'] = mapped_dict.pop('n_layers')
        if 'intermediate_size' not in mapped_dict and 'intermediate_dim' in mapped_dict:
            mapped_dict['intermediate_size'] = mapped_dict.pop('intermediate_dim')
        mapped_dict.pop('attention_bias', None)
        return {k: v for k, v in mapped_dict.items() if k in cls.__dataclass_fields__}

class LayerNorm(nn.Module):
    """TPU-optimized Layer Normalization."""
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16
    scale_init: callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)  # Higher precision for stability
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.epsilon)
        scale = self.param('scale', self.scale_init, (x.shape[-1],))
        x = (x - mean) * inv_std * scale
        return x.astype(self.dtype)

class Dense(nn.Module):
    """TPU-optimized Dense layer."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    kernel_init: callable = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y

class TransformerBlock(nn.Module):
    """TPU-optimized Transformer block."""
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    dropout_rate: float
    attention_dropout: float
    dtype: str = "bfloat16"
    use_dualpipe: bool = True
    partition_idx: int = 0
    
    def setup(self):
        self.attention = nn.SelfAttention(
            num_heads=self.num_attention_heads,
            dtype=jnp.dtype(self.dtype),
            dropout_rate=self.attention_dropout,
            deterministic=True
        )
        self.layer_norm1 = LayerNorm(dtype=jnp.dtype(self.dtype))
        self.layer_norm2 = LayerNorm(dtype=jnp.dtype(self.dtype))
        self.mlp = Dense(features=self.intermediate_size, dtype=jnp.dtype(self.dtype))
        
    def __call__(self, x, attention_mask, deterministic):
        y = self.layer_norm1(x)
        y = self.attention(y, attention_mask, deterministic=deterministic)
        x = x + y
        z = self.layer_norm2(x)
        z = self.mlp(z)
        return x + z

class VishwamAIModel(nn.Module):
    """Main model class with TPU optimizations."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype
        )

        # Create transformer blocks
        self.encoder = [
            TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.config.hidden_dropout_prob,
                attention_dropout=self.config.attention_dropout_prob,
                dtype=self.config.dtype,
                use_dualpipe=self.config.use_dualpipe,
                partition_idx=i
            ) for i in range(self.config.num_layers)
        ]

        self.final_layer_norm = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=dtype)
        self.lm_head = Dense(features=self.config.vocab_size, use_bias=False, dtype=dtype)

    def __call__(self, input_ids, attention_mask=None, deterministic=True):
        hidden_states = self.embeddings(input_ids)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.triu(
                jnp.full((input_ids.shape[1], input_ids.shape[1]), -1e9), 
                k=1
            )

        # Process through transformer layers
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states, attention_mask, deterministic)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            'logits': logits,
            'hidden_states': hidden_states
        }

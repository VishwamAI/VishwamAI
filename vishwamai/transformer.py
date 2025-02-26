import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from .model import ModelConfig

logger = logging.getLogger(__name__)

class VishwamAIModel(nn.Module):
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
        self.lm_head = nn.Dense(features=self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, attention_mask=None, deterministic=True, use_tot=False, tot_rng_key=None):
        hidden_states = self.embeddings(input_ids)
        outputs = {'hidden_states': hidden_states, 'logits': self.lm_head(hidden_states)}
        if use_tot and hasattr(self, 'tot_model') and tot_rng_key is not None:
            thought = self.tot_model(hidden_states, tot_rng_key)
            outputs['tot_outputs'] = {'thought': thought}
        return outputs

    def init(self, rng, input_ids):
        return self.init_weights(rng, input_ids.shape)

class VisionTransformer10B(VishwamAIModel):
    """10B parameter Vision Transformer model implementing the VishwamAI architecture."""
    
    @staticmethod
    def get_default_config():
        return ModelConfig(
            vocab_size=50304,
            hidden_size=6144,
            num_layers=44,
            num_attention_heads=48,
            intermediate_size=24576,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            gradient_checkpointing=True,
            use_flash_attention=True,
            use_rope=True,
            use_alibi=False,
            use_gqa=True,
            num_key_value_heads=4,
            dtype="bfloat16"
        )
    
    def setup(self):
        config = self.get_default_config()
        self.config = config
        
        # Initialize layers
        self.embeddings = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=jnp.dtype(config.dtype)
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embed(
            num_embeddings=config.max_position_embeddings,
            features=config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=jnp.dtype(config.dtype)
        )
        
        # Transformer layers
        self.layers = [
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout_rate=config.hidden_dropout_prob,
                attention_dropout=config.attention_dropout_prob,
                dtype=config.dtype,
                name=f'layer_{i}'
            ) for i in range(config.num_layers)
        ]
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(
            epsilon=config.layer_norm_eps,
            dtype=config.dtype,
            name='ln_f'
        )
        
        # Output head
        self.lm_head = nn.Dense(
            features=config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            name='lm_head'
        )

    def __call__(self, input_ids, attention_mask=None, deterministic=True, use_tot=False, tot_rng_key=None):
        b, s = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Add position embeddings
        position_ids = jnp.arange(s)[None, :].repeat(b, axis=0)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, deterministic)
            
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        outputs = {'hidden_states': hidden_states, 'logits': logits}
        
        # Add ToT outputs if requested
        if use_tot and hasattr(self, 'tot_model') and tot_rng_key is not None:
            thought = self.tot_model(hidden_states, tot_rng_key)
            outputs['tot_outputs'] = {'thought': thought}
            
        return outputs

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    dropout_rate: float
    attention_dropout: float
    dtype: str = "bfloat16"
    
    def setup(self):
        self.attention = SelfAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dropout_rate=self.attention_dropout,
            dtype=self.dtype
        )
        self.ln_1 = nn.LayerNorm(dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(dtype=self.dtype)
        self.mlp = FeedForward(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )
    
    def __call__(self, x, attention_mask=None, deterministic=True):
        # Self attention
        attn_output = self.attention(
            self.ln_1(x),
            attention_mask,
            deterministic=deterministic
        )
        x = x + attn_output
        
        # Feed-forward
        mlp_output = self.mlp(
            self.ln_2(x),
            deterministic=deterministic
        )
        return x + mlp_output

class SelfAttention(nn.Module):
    """Multi-head self attention."""
    hidden_size: int
    num_attention_heads: int
    dropout_rate: float
    dtype: str = "bfloat16"
    
    def setup(self):
        self.head_size = self.hidden_size // self.num_attention_heads
        self.q = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.k = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.v = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.o = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    def __call__(self, x, attention_mask=None, deterministic=True):
        b, s, h = x.shape
        
        # Project queries, keys, and values
        q = self.q(x).reshape(b, s, self.num_attention_heads, self.head_size)
        k = self.k(x).reshape(b, s, self.num_attention_heads, self.head_size)
        v = self.v(x).reshape(b, s, self.num_attention_heads, self.head_size)
        
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(self.head_size)
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask[:, None, None, :]
            
        # Get attention weights
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, deterministic=deterministic)
        
        # Apply attention to values
        attn = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        attn = attn.reshape(b, s, h)
        
        return self.o(attn)

class FeedForward(nn.Module):
    """Feed-forward neural network."""
    hidden_size: int
    intermediate_size: int
    dropout_rate: float
    dtype: str = "bfloat16"
    
    def setup(self):
        self.fc1 = nn.Dense(features=self.intermediate_size, dtype=self.dtype)
        self.fc2 = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    def __call__(self, x, deterministic=True):
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x
